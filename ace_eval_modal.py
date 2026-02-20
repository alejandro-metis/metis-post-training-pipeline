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
        raise ValueError(
            f"Model {name!r} not found on HuggingFace Hub. Check the name."
        )


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
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm>=0.6.4", "openai", "httpx", "pyarrow")
    .env({"VLLM_TARGET_DEVICE": "cuda"})
    .add_local_file("ace_eval.py", "/root/ace_eval.py")
    .add_local_file("ace_eval_prompts.py", "/root/ace_eval_prompts.py")
    .add_local_dir("ace_scoring", "/root/ace_scoring")
)

results_vol = modal.Volume.from_name("ace-eval-results", create_if_missing=True)
model_cache = modal.Volume.from_name("model-cache", create_if_missing=True)

SECRETS = [
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("searchapi-secret"),
    modal.Secret.from_name("openai-secret"),
    modal.Secret.from_name("jina-secret"),
]

VOLUMES = {
    "/results": results_vol,
    "/root/.cache/huggingface": model_cache,
}


# ---------------------------------------------------------------------------
# Core eval logic (runs inside Modal container)
# ---------------------------------------------------------------------------
def _run_eval_inner(
    model_name,
    tasks_json,
    tp_size,
    workers,
    max_turns,
    max_searches,
    max_browses,
    judge_model="",
    prompt_preset="",
    runs=1,
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
    from ace_eval import _log, build_eval_summary, eval_single_task
    from ace_eval_prompts import DEFAULT_PROMPT, get_prompt

    system_prompt = get_prompt(prompt_preset or DEFAULT_PROMPT)
    _log(f"Prompt preset: {prompt_preset or DEFAULT_PROMPT}")

    tasks = json.loads(tasks_json)
    tool_parser = _get_tool_parser(model_name)

    # --- Start vLLM OpenAI-compatible server ---
    _log(f"Starting vLLM: {model_name} (tp={tp_size}, parser={tool_parser})")

    proc = subprocess.Popen(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_name,
            "--tensor-parallel-size",
            str(tp_size),
            "--port",
            "8000",
            "--trust-remote-code",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            tool_parser,
            "--max-model-len",
            "16384",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for health (up to 10 min for model download + load)
    try:
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
        if prompt_preset:
            model_dir = f"{model_dir}_{prompt_preset}"
        output_dir = "/results"
        total = len(tasks)

        # Expand tasks × runs into work items
        use_run_id = runs > 1
        work_items = []
        for task in tasks:
            for run_id in range(1, runs + 1):
                work_items.append((task, run_id if use_run_id else None))
        total_items = len(work_items)

        _log(
            f"Running {total} tasks × {runs} runs = {total_items} work items with {workers} workers"
        )

        results = []
        start_all = time.time()

        if workers <= 1:
            for i, (task, run_id) in enumerate(work_items):
                result = eval_single_task(
                    client=client,
                    model=model_name,
                    task=task,
                    searchapi_key=searchapi_key,
                    output_dir=output_dir,
                    model_dir=model_dir,
                    task_idx=i,
                    total_tasks=total_items,
                    system_prompt=system_prompt,
                    max_turns=max_turns,
                    max_searches=max_searches,
                    max_browses=max_browses,
                    run_id=run_id,
                )
                results.append(result)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for i, (task, run_id) in enumerate(work_items):
                    fut = executor.submit(
                        eval_single_task,
                        client=client,
                        model=model_name,
                        task=task,
                        searchapi_key=searchapi_key,
                        output_dir=output_dir,
                        model_dir=model_dir,
                        task_idx=i,
                        total_tasks=total_items,
                        system_prompt=system_prompt,
                        max_turns=max_turns,
                        max_searches=max_searches,
                        max_browses=max_browses,
                        run_id=run_id,
                    )
                    futures[fut] = (task["task_id"], run_id)
                for fut in as_completed(futures):
                    results.append(fut.result())

        total_time = time.time() - start_all
        summary = build_eval_summary(model_name, results, total_time)
        ok = summary["completed"]
        failed = summary["failed"]

        summary_path = Path(output_dir) / model_dir / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        results_vol.commit()

        _log(
            f"\n{model_name}: {ok}/{total} completed, {failed} failed, {total_time:.1f}s"
        )
        return json.dumps(summary)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Two GPU tiers (Modal needs static GPU specs at decoration time)
# ---------------------------------------------------------------------------
@app.function(
    image=eval_image,
    gpu="H100",
    timeout=7200,
    secrets=SECRETS,
    volumes=VOLUMES,
)
def eval_1gpu(
    model_name: str,
    tasks_json: str,
    workers: int = 4,
    max_turns: int = 10,
    max_searches: int = 5,
    max_browses: int = 5,
    judge_model: str = "",
    prompt_preset: str = "",
    runs: int = 1,
) -> str:
    return _run_eval_inner(
        model_name,
        tasks_json,
        tp_size=1,
        workers=workers,
        max_turns=max_turns,
        max_searches=max_searches,
        max_browses=max_browses,
        judge_model=judge_model,
        prompt_preset=prompt_preset,
        runs=runs,
    )


@app.function(
    image=eval_image,
    gpu="H100:2",
    timeout=7200,
    secrets=SECRETS,
    volumes=VOLUMES,
)
def eval_2gpu(
    model_name: str,
    tasks_json: str,
    workers: int = 4,
    max_turns: int = 10,
    max_searches: int = 5,
    max_browses: int = 5,
    judge_model: str = "",
    prompt_preset: str = "",
    runs: int = 1,
) -> str:
    return _run_eval_inner(
        model_name,
        tasks_json,
        tp_size=2,
        workers=workers,
        max_turns=max_turns,
        max_searches=max_searches,
        max_browses=max_browses,
        judge_model=judge_model,
        prompt_preset=prompt_preset,
        runs=runs,
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
    prompt: str = "",
    runs: int = 1,
    shards: int = 1,
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
        tasks.append(
            {
                "task_id": extra_info.get("task_id", str(i)),
                "prompt": prompt_text,
                "criteria": criteria,
                "domain": extra_info.get("domain", "unknown"),
                "shop_vs_product": extra_info.get("shop_vs_product"),
            }
        )

    if domains:
        domain_list = [d.strip() for d in domains.split(",")]
        tasks = [t for t in tasks if t["domain"] in domain_list]
    if task_ids:
        id_list = [tid.strip() for tid in task_ids.split(",")]
        tasks = [t for t in tasks if str(t["task_id"]) in id_list]

    print(f"Tasks: {len(tasks)}")
    print(f"Models: {[m.split('/')[-1] for m in model_list]}")
    if shards > 1:
        print(f"Shards: {shards} (each model gets {shards} GPU containers)")

    # Dispatch all models in parallel (each gets its own GPU container)
    # With shards > 1, each model gets N containers with disjoint task subsets
    spawn_kwargs = dict(
        workers=workers,
        max_turns=max_turns,
        max_searches=max_searches,
        max_browses=max_browses,
        judge_model=judge_model,
        prompt_preset=prompt,
        runs=runs,
    )

    # handles: {model: [(shard_idx, handle), ...]}
    handles: dict[str, list] = {}
    for m in model_list:
        large = _is_large_model(m)
        fn = eval_2gpu if large else eval_1gpu
        gpu_desc = "2xH100" if large else "1xH100"
        handles[m] = []

        if shards > 1:
            chunks = [(i, tasks[i::shards]) for i in range(shards) if tasks[i::shards]]
            for shard_idx, chunk in chunks:
                print(
                    f"  Dispatching {m} shard {shard_idx + 1}/{shards} ({len(chunk)} tasks, {gpu_desc})..."
                )
                h = fn.spawn(model_name=m, tasks_json=json.dumps(chunk), **spawn_kwargs)
                handles[m].append((shard_idx, h))
        else:
            print(f"  Dispatching {m} ({gpu_desc})...")
            h = fn.spawn(model_name=m, tasks_json=json.dumps(tasks), **spawn_kwargs)
            handles[m].append((0, h))

    # Collect results as they finish, merging shards per model
    from ace_eval import build_eval_summary

    all_summaries = {}
    for m, shard_handles in handles.items():
        n_shards = len(shard_handles)
        print(f"\nWaiting for {m} ({n_shards} shard(s))...")

        all_results = []
        max_time = 0.0
        for shard_idx, handle in sorted(shard_handles):
            try:
                summary_json = handle.get()
                summary = json.loads(summary_json)
                all_results.extend(summary.get("results", []))
                max_time = max(max_time, summary.get("total_time_seconds", 0))
                print(
                    f"  Shard {shard_idx + 1}: {summary['completed']}/{summary['total_tasks']} completed "
                    f"in {summary['total_time_seconds']}s"
                )
            except Exception as e:
                print(f"  Shard {shard_idx + 1}: FAILED: {e}")

        # Build proper merged summary with domain stats, task aggregation, etc.
        merged = build_eval_summary(m, all_results, max_time)
        merged["shards"] = n_shards
        all_summaries[m] = merged

        ok = merged["completed"]
        failed = merged["failed"]
        print(
            f"  Total: {ok}/{len(all_results)} completed, {failed} failed, {max_time:.1f}s wall time"
        )

        # Write merged summary to volume (overwrites partial per-shard summaries)
        if n_shards > 1:
            model_dir = m.replace("/", "_")
            if prompt:
                model_dir = f"{model_dir}_{prompt}"
            summary_path = f"eval_results/{model_dir}/summary.json"
            try:
                with open(summary_path, "w") as f:
                    json.dump(merged, f, indent=2)
                print(f"  Merged summary written to {summary_path}")
            except Exception:
                pass  # local dir may not exist; volume has per-shard copies

    print("\nDone.")
