import modal
import os
import re
import subprocess
from pathlib import Path
from typing import Literal, Optional

# --- 1. CONFIGURATION ---
APP_NAME = "verl-grpo-qwen-7b"
# We use Qwen 2.5 7B. You can switch to "Qwen/Qwen2.5-3B-Instruct" for faster iteration.
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
HF_REPO_NAME = "Qwen2.5-7B-GRPO-GSM8K"  # Will be pushed as <your-hf-username>/<this>
# Persistent storage for dataset and checkpoints
VOLUME_NAME = "verl-data-storage"

# Define the Modal App
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# --- 2. THE IMAGE ---
# We use the official VerL image (pinned version) and clone the repo for examples/scripts.
VERL_REPO_PATH = Path("/app/verl")
image = (
    modal.Image.from_registry("verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1")
    .run_commands(
        f"git clone https://github.com/volcengine/verl.git {VERL_REPO_PATH}",
    
    )
    .env({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
    .workdir(str(VERL_REPO_PATH))
)

# --- 3. REWARD FUNCTION ---
# GRPO on GSM8K needs a custom reward function that checks if the model's
# final answer (after "####") matches the ground truth.
PATH_TO_REWARD_FUNCTION = Path("/root/reward_fn.py")
REWARD_FUNCTION_NAME = "compute_reward"


def extract_solution(
    solution_str: str, method: Literal["strict", "flexible"] = "strict"
) -> Optional[str]:
    assert method in ["strict", "flexible"]

    if method == "strict":
        solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = (
                final_answer.split("#### ")[1].replace(",", "").replace("$", "")
            )
    elif method == "flexible":
        answer = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            pass
        else:
            invalid_str = ["", "."]
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_reward(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict
) -> float:
    """Return 1.0 for correct GSM8K answer, 0.0 otherwise."""
    answer = extract_solution(solution_str=solution_str, method="strict")
    if answer is None:
        return 0.0
    if answer == ground_truth:
        return 1.0
    return 0.0

# --- 4. DATA PREPARATION ---
@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def prep_data():
    """
    Downloads GSM8K and converts it to the Parquet format VerL expects.
    This runs once and saves data to the Volume.
    """
    print("Dataset: Preparing GSM8K...")
    
    # We use the built-in GSM8K preprocess script from VerL
    # We save it to /data/gsm8k which is backed by the persistent Volume
    cmd = [
        "python3",
        "examples/data_preprocess/gsm8k.py",
        "--local_dir", "/data/gsm8k"
    ]
    subprocess.run(cmd, check=True)
    print("Dataset: Ready at /data/gsm8k")

# --- 5. GRPO TRAINING ---
@app.function(
    image=image,
    gpu="H100:8",
    timeout=60 * 60 * 2,  # 2 hour limit for this proof-of-concept
    volumes={"/data": vol},
    
)
def train():
    import inspect

    DATA_DIR = "/data/gsm8k"
    MODEL_DIR = "/data/checkpoints/qwen-grpo"

    # Write the reward function to a file inside the container so VerL can import it
    reward_source = "\n".join([
        "import re",
        "from typing import Literal, Optional",
        "",
        inspect.getsource(extract_solution),
        "",
        inspect.getsource(compute_reward),
    ])
    PATH_TO_REWARD_FUNCTION.parent.mkdir(parents=True, exist_ok=True)
    PATH_TO_REWARD_FUNCTION.write_text(reward_source)

    print("Starting GRPO Training on 8x H100s...")

    # VerL uses Hydra/OmegaConf -- all config is passed as CLI dot-notation args
    cmd = [
        "python3",
        "-m", "verl.trainer.main_ppo",
        # --- Algorithm: switch from PPO (gae) to GRPO ---
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        # --- Data ---
        f"data.train_files={DATA_DIR}/train.parquet",
        f"data.val_files={DATA_DIR}/test.parquet",
        "data.train_batch_size=256",
        "data.max_prompt_length=512",
        "data.max_response_length=512",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        # --- Actor / Model ---
        f"actor_rollout_ref.model.path={MODEL_NAME}",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=256",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        # --- Rollout (vLLM generation) ---
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16",
        # --- Reference model ---
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        # --- Trainer ---
        "trainer.critic_warmup=0",
        "trainer.total_epochs=2",
        "trainer.project_name=verl-grpo-proof",
        "trainer.experiment_name=qwen-7b-run",
        "trainer.logger=['console']",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=8",
        "trainer.save_freq=10",
        "trainer.test_freq=10",
        f"trainer.default_local_dir={MODEL_DIR}",
        # --- Custom reward function for GSM8K ---
        f"custom_reward_function.path={PATH_TO_REWARD_FUNCTION}",
        f"custom_reward_function.name={REWARD_FUNCTION_NAME}",
    ]

    subprocess.run(cmd, check=True)

    print(f"Training complete. Checkpoints saved to {MODEL_DIR}")

# --- 6. PUSH TO HUGGING FACE ---
@app.function(
    image=image,
    timeout=60 * 60,  # 1 hour for upload
    volumes={"/data": vol},
)
def push_to_hub():
    from huggingface_hub import HfApi

    MODEL_DIR = Path("/data/checkpoints/qwen-grpo")

    # Reload volume to pick up freshly written checkpoints from training
    vol.reload()

    # Find the latest checkpoint
    iteration_file = MODEL_DIR / "latest_checkpointed_iteration.txt"
    if not iteration_file.exists():
        raise FileNotFoundError(
            f"No checkpoints found at {MODEL_DIR}. Did training complete successfully?"
        )

    latest_step = iteration_file.read_text().strip()
    checkpoint_path = MODEL_DIR / f"global_step_{latest_step}" / "actor" / "huggingface"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_path}"
        )

    # Auto-detect HF username from the token
    api = HfApi()
    hf_username = api.whoami()["name"]
    repo_id = f"{hf_username}/{HF_REPO_NAME}"

    print(f"Pushing checkpoint (step {latest_step}) to https://huggingface.co/{repo_id}")

    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        commit_message=f"GRPO training step {latest_step} on GSM8K",
    )

    print(f"Done! Model pushed to https://huggingface.co/{repo_id}")

# --- 7. LOCAL ENTRYPOINT ---
@app.local_entrypoint()
def main():
    # 1. Prepare Data
    print("Step 1: Preparing Data...")
    prep_data.remote()

    # 2. Run Training
    print("Step 2: Launching Training...")
    train.remote()

    # 3. Push final model to Hugging Face
    print("Step 3: Pushing model to Hugging Face Hub...")
    push_to_hub.remote()