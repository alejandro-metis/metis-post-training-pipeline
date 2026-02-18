"""
Train a model on Mercor ACE tasks using GRPO and verl on Modal.

Uses OpenAI gpt-4o-mini as LLM judge for the reward function.

Usage:
    modal run ace_grpo_modal.py::prep_dataset
    modal run --detach ace_grpo_modal.py::train
"""

import shutil
import subprocess
from pathlib import Path

import modal

app = modal.App("ace-grpo-verl")

# Image: verl base + openai SDK for reward function
VERL_REPO_PATH: Path = Path("/root/verl")
image = (
    modal.Image.from_registry("verlai/verl:vllm011.latest", setup_dockerfile_commands=["ENTRYPOINT []"])
    .env({"VLLM_TARGET_DEVICE": "cuda"})
    .apt_install("git")
    .run_commands(f"git clone https://github.com/volcengine/verl {VERL_REPO_PATH}")
    .uv_pip_install("verl[vllm]==0.7.0", "openai")
)

# Volumes
DATA_PATH: Path = Path("/data")
MODELS_PATH: Path = Path("/models")
MINUTES: int = 60

data_volume = modal.Volume.from_name("ace-grpo-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(
    "ace-grpo-checkpoints", create_if_missing=True
)

# Where files land inside the container
STAGED_DATA_DIR = Path("/root/ace_verl_data")
PATH_TO_REWARD_FUNCTION: Path = Path("/root/ace_reward.py")
REWARD_FUNCTION_NAME: str = "compute_reward"

# Bake local files into the images via add_local_dir / add_local_file
data_image = image.add_local_dir("ace_verl_data", remote_path=str(STAGED_DATA_DIR))
train_image = image.add_local_file("ace_reward.py", remote_path=str(PATH_TO_REWARD_FUNCTION))


@app.function(image=data_image, volumes={DATA_PATH: data_volume})
def prep_dataset() -> None:
    """Copy ACE parquet files into Modal Volume."""
    for split in ["train.parquet", "test.parquet"]:
        src = STAGED_DATA_DIR / split
        dst = DATA_PATH / split
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")

    data_volume.commit()
    print("Dataset ready.")


@app.function(
    image=train_image,
    gpu="H100:2",
    volumes={
        MODELS_PATH: checkpoints_volume,
        DATA_PATH: data_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("openai-secret"),
    ],
    timeout=24 * 60 * MINUTES,
)
def train(*arglist) -> None:
    """Run GRPO training on ACE tasks."""
    data_volume.reload()

    cmd: list[str] = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        # Data
        f"data.train_files={DATA_PATH / 'train.parquet'}",
        f"data.val_files={DATA_PATH / 'test.parquet'}",
        "data.train_batch_size=32",
        "data.max_prompt_length=512",
        "data.max_response_length=2048",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        # Model
        "actor_rollout_ref.model.path=Qwen/Qwen2-0.5B",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.actor.checkpoint.save_contents='model,optimizer,extra,hf_model'",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        # Rollout
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.n=5",
        # Reference
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        # Algorithm
        "algorithm.use_kl_in_reward=False",
        # Trainer
        "trainer.critic_warmup=0",
        "trainer.logger=['console', 'wandb']",
        "trainer.project_name=ace_grpo_qwen2-0.5b",
        "trainer.experiment_name=ace_qwen2-0.5b",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        "trainer.test_freq=5",
        f"trainer.default_local_dir={MODELS_PATH}",
        "trainer.resume_mode=auto",
        "trainer.save_freq=5",
        "trainer.total_training_steps=20",
        "trainer.total_epochs=1",
        # Custom reward function
        f"custom_reward_function.path={str(PATH_TO_REWARD_FUNCTION)}",
        f"custom_reward_function.name={REWARD_FUNCTION_NAME}",
    ]
    if arglist:
        cmd.extend(arglist)

    subprocess.run(cmd, check=True)


# Inference on trained model

VLLM_PORT: int = 8000


def get_latest_checkpoint_file_path():
    with open(MODELS_PATH / "latest_checkpointed_iteration.txt") as f:
        latest_checkpoint_index = int(f.read())
    return str(
        MODELS_PATH
        / f"global_step_{latest_checkpoint_index}"
        / "actor"
        / "huggingface"
    )


vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "vllm==0.9.1",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    .env({"VLLM_USE_V1": "1"})
)

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="H100:2",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={"/root/.cache/vllm": vllm_cache_vol, MODELS_PATH: checkpoints_volume},
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    latest_checkpoint_file_path = get_latest_checkpoint_file_path()

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        latest_checkpoint_file_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        "2",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
