"""
Self-contained Modal script for mass ACE baseline evaluation.

Spins up vLLM on Modal GPUs, runs eval with tools (SearchAPI.io + Jina),
produces autograder-compatible output. Models run in parallel.

Usage:
    # Single model (use full HuggingFace model ID)
    modal run ace_eval_modal.py --model Qwen/Qwen2.5-7B-Instruct

    # Multiple baselines (run in parallel on separate GPU containers)
    modal run ace_eval_modal.py --models "Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct"

    # Specific domains
    modal run ace_eval_modal.py --model Qwen/Qwen2.5-7B-Instruct --domains "shopping,food"

    # Download results
    modal volume get ace-eval-results / ./eval_results/
"""

import json
import os

import modal

app = modal.App("ace-eval")

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def _validate_model(name: str):
    """Verify model is a valid HuggingFace model ID by checking the Hub API."""
    import re

    if not re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", name):
        raise ValueError(
            f"Model name must be a full HuggingFace ID (e.g. Qwen/Qwen2.5-7B-Instruct), "
            f"got: {name!r}"
        )
    from huggingface_hub import model_info
    try:
        model_info(name)
    except Exception:
        raise ValueError(f"Model {name!r} not found on HuggingFace Hub. Check the name.")


def _get_param_count(model_name: str) -> float:
    """Extract parameter count (in billions) from HF model name. E.g. 'Qwen2.5-72B-Instruct' -> 72."""
    import re
    match = re.search(r"(\d+(?:\.\d+)?)[bB]", model_name)
    assert match, f"Could not parse parameter count from model name: {model_name!r}"
    return float(match.group(1))


def _is_large_model(model_name: str) -> bool:
    return _get_param_count(model_name) > 16


def _get_tool_parser(model_name: str) -> str:
    lower = model_name.lower()
    if "llama" in lower:
        return "llama3_json"
    if "mistral" in lower:
        return "mistral"
    return "hermes"


# ---------------------------------------------------------------------------
# Modal image & resources
# ---------------------------------------------------------------------------
eval_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install("vllm>=0.6.4", "openai", "httpx", "pyarrow")
    .env({"VLLM_TARGET_DEVICE": "cuda"})
    .add_local_file("ace_eval.py", "/root/ace_eval.py")
    .add_local_file("ace_grounding_wrapper.py", "/root/ace_grounding_wrapper.py")
    .add_local_dir("ace_scoring", "/root/ace_scoring")
)

results_vol = modal.Volume.from_name("ace-eval-results", create_if_missing=True)
model_cache = modal.Volume.from_name("model-cache", create_if_missing=True)

SECRETS = [
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("searchapi-secret"),
    modal.Secret.from_name("openai-secret"),
]

VOLUMES = {
    "/results": results_vol,
    "/root/.cache/huggingface": model_cache,
}


# ---------------------------------------------------------------------------
# Core eval logic (runs inside Modal container)
# ---------------------------------------------------------------------------
def _run_eval_inner(
    model_name, tasks_json, tp_size, workers, max_turns, max_searches, max_browses,
    judge_model="",
):
    import subprocess
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import httpx
    from openai import OpenAI

    if judge_model:
        os.environ["ACE_JUDGE_MODEL"] = judge_model

    sys.path.insert(0, "/root")
    from ace_eval import _log, eval_single_task

    tasks = json.loads(tasks_json)
    tool_parser = _get_tool_parser(model_name)

    # --- Start vLLM OpenAI-compatible server ---
    _log(f"Starting vLLM: {model_name} (tp={tp_size}, parser={tool_parser})")

    proc = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--tensor-parallel-size", str(tp_size),
            "--port", "8000",
            "--trust-remote-code",
            "--enable-auto-tool-choice",
            "--tool-call-parser", tool_parser,
            "--max-model-len", "16384",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for health (up to 10 min for model download + load)
    for i in range(600):
        try:
            r = httpx.get("http://localhost:8000/health", timeout=2.0)
            if r.status_code == 200:
                _log(f"vLLM ready after {i}s")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        out = proc.stdout.read().decode() if proc.stdout else ""
        raise RuntimeError(f"vLLM failed to start. Last output:\n{out[-3000:]}")

    # --- Run eval ---
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    searchapi_key = os.environ.get("SEARCHAPI_IO_KEY", "")
    model_dir = model_name.replace("/", "_")
    output_dir = "/results"
    total = len(tasks)

    _log(f"Running {total} tasks with {workers} workers")

    results = []
    start_all = time.time()

    if workers <= 1:
        for i, task in enumerate(tasks):
            result = eval_single_task(
                client=client, model=model_name, task=task,
                searchapi_key=searchapi_key, output_dir=output_dir,
                model_dir=model_dir, task_idx=i, total_tasks=total,
                max_turns=max_turns, max_searches=max_searches,
                max_browses=max_browses,
            )
            results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for i, task in enumerate(tasks):
                fut = executor.submit(
                    eval_single_task,
                    client=client, model=model_name, task=task,
                    searchapi_key=searchapi_key, output_dir=output_dir,
                    model_dir=model_dir, task_idx=i, total_tasks=total,
                    max_turns=max_turns, max_searches=max_searches,
                    max_browses=max_browses,
                )
                futures[fut] = task["task_id"]
            for fut in as_completed(futures):
                results.append(fut.result())

    total_time = time.time() - start_all
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = total - ok

    # --- Save summary with per-domain breakdown ---
    ok_results = [r for r in results if r["status"] == "ok"]
    domain_stats = {}
    for r in results:
        d = r.get("domain", "unknown")
        if d not in domain_stats:
            domain_stats[d] = {
                "completed": 0, "failed": 0,
                "total_searches": 0, "total_browses": 0,
                "total_score": 0, "total_hurdle_score": 0, "total_criteria": 0,
            }
        if r["status"] == "ok":
            domain_stats[d]["completed"] += 1
            domain_stats[d]["total_searches"] += r.get("searches", 0)
            domain_stats[d]["total_browses"] += r.get("browses", 0)
            domain_stats[d]["total_score"] += r.get("total_score", 0)
            domain_stats[d]["total_hurdle_score"] += r.get("total_hurdle_score", 0)
            domain_stats[d]["total_criteria"] += r.get("num_criteria", 0)
        else:
            domain_stats[d]["failed"] += 1

    summary = {
        "model": model_name,
        "total_tasks": total,
        "completed": ok,
        "failed": failed,
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_task": round(sum(r["elapsed"] for r in ok_results) / ok, 2) if ok else 0,
        "total_searches": sum(r.get("searches", 0) for r in ok_results),
        "total_browses": sum(r.get("browses", 0) for r in ok_results),
        "domain_stats": domain_stats,
        "results": results,
    }

    summary_path = Path(output_dir) / model_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    results_vol.commit()
    proc.terminate()

    _log(f"\n{model_name}: {ok}/{total} completed, {failed} failed, {total_time:.1f}s")
    return json.dumps(summary)


# ---------------------------------------------------------------------------
# Two GPU tiers (Modal needs static GPU specs at decoration time)
# ---------------------------------------------------------------------------
@app.function(
    image=eval_image, gpu="H100", timeout=7200,
    secrets=SECRETS, volumes=VOLUMES,
)
def eval_1gpu(
    model_name: str, tasks_json: str, workers: int = 4,
    max_turns: int = 10, max_searches: int = 5, max_browses: int = 5,
    judge_model: str = "",
) -> str:
    return _run_eval_inner(
        model_name, tasks_json, tp_size=1, workers=workers,
        max_turns=max_turns, max_searches=max_searches, max_browses=max_browses,
        judge_model=judge_model,
    )


@app.function(
    image=eval_image, gpu="H100:2", timeout=7200,
    secrets=SECRETS, volumes=VOLUMES,
)
def eval_2gpu(
    model_name: str, tasks_json: str, workers: int = 4,
    max_turns: int = 10, max_searches: int = 5, max_browses: int = 5,
    judge_model: str = "",
) -> str:
    return _run_eval_inner(
        model_name, tasks_json, tp_size=2, workers=workers,
        max_turns=max_turns, max_searches=max_searches, max_browses=max_browses,
        judge_model=judge_model,
    )


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    model: str = "",
    models: str = "",
    parquet: str = "ace_verl_data/test.parquet",
    domains: str = "",
    task_ids: str = "",
    workers: int = 4,
    max_turns: int = 10,
    max_searches: int = 5,
    max_browses: int = 5,
    judge_model: str = "",
):
    import pyarrow.parquet as pq

    # Parse model names
    if models:
        model_list = [m.strip() for m in models.split(",")]
    elif model:
        model_list = [model]
    else:
        print("Error: specify --model or --models")
        return

    for m in model_list:
        _validate_model(m)

    # Load tasks from local parquet
    table = pq.read_table(parquet)
    tasks = []
    for i in range(len(table)):
        prompt_list = table.column("prompt")[i].as_py()
        prompt_text = prompt_list[0]["content"] if prompt_list else ""
        reward_model = table.column("reward_model")[i].as_py()
        criteria = json.loads(reward_model.get("ground_truth", "[]"))
        extra_info = table.column("extra_info")[i].as_py()
        tasks.append({
            "task_id": extra_info.get("task_id", str(i)),
            "prompt": prompt_text,
            "criteria": criteria,
            "domain": extra_info.get("domain", "unknown"),
            "shop_vs_product": extra_info.get("shop_vs_product"),
        })

    if domains:
        domain_list = [d.strip() for d in domains.split(",")]
        tasks = [t for t in tasks if t["domain"] in domain_list]
    if task_ids:
        id_list = [tid.strip() for tid in task_ids.split(",")]
        tasks = [t for t in tasks if str(t["task_id"]) in id_list]

    print(f"Tasks: {len(tasks)}")
    print(f"Models: {[m.split('/')[-1] for m in model_list]}")

    tasks_json = json.dumps(tasks)

    # Dispatch all models in parallel (each gets its own GPU container)
    handles = {}
    for m in model_list:
        large = _is_large_model(m)
        fn = eval_2gpu if large else eval_1gpu
        gpu_desc = "2xH100" if large else "1xH100"
        print(f"  Dispatching {m} ({gpu_desc})...")
        handles[m] = fn.spawn(
            model_name=m, tasks_json=tasks_json, workers=workers,
            max_turns=max_turns, max_searches=max_searches, max_browses=max_browses,
            judge_model=judge_model,
        )

    # Collect results as they finish
    all_summaries = {}
    for m, handle in handles.items():
        print(f"\nWaiting for {m}...")
        try:
            summary_json = handle.get()
            summary = json.loads(summary_json)
            all_summaries[m] = summary
            print(f"  {summary['completed']}/{summary['total_tasks']} completed "
                  f"in {summary['total_time_seconds']}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            all_summaries[m] = {
                "model": m, "total_tasks": len(tasks),
                "completed": 0, "failed": len(tasks),
                "total_time_seconds": 0, "results": [],
            }

    print("\nDone.")
