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

# Lightweight image for coordinator (no GPU, no vLLM)
# pip_install must come before add_local_* per Modal requirements
coord_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai", "httpx", "pyarrow")
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
# Required env vars (set by Modal secrets)
# ---------------------------------------------------------------------------
REQUIRED_ENV_VARS = {
    "SEARCHAPI_IO_KEY": "searchapi-secret",
    "OPENAI_API_KEY": "openai-secret",
    "JINA_API_KEY": "jina-secret",
    "HF_TOKEN": "huggingface-secret",
}


def _check_env_vars():
    """Validate all required API keys are set. Raises RuntimeError with details."""
    missing = []
    empty = []
    for var, secret_name in REQUIRED_ENV_VARS.items():
        val = os.environ.get(var)
        if val is None:
            missing.append(f"  {var} (Modal secret: {secret_name})")
        elif not val.strip():
            empty.append(f"  {var} (Modal secret: {secret_name}) — set but empty")
    if missing or empty:
        parts = []
        if missing:
            parts.append("Missing env vars:\n" + "\n".join(missing))
        if empty:
            parts.append("Empty env vars:\n" + "\n".join(empty))
        parts.append(
            "Fix: create/update Modal secrets with `modal secret create <name> VAR=value`"
        )
        raise RuntimeError("\n".join(parts))


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

    # Validate all API keys before starting expensive GPU work
    _check_env_vars()

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
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Two GPU tiers (Modal needs static GPU specs at decoration time)
# ---------------------------------------------------------------------------
@app.function(
    image=eval_image,
    gpu="H100",
    timeout=14400,
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
    timeout=14400,
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
# Coordinator: runs on Modal (no GPU), spawns GPU shards, collects results.
# Survives --detach so shards keep running when local process disconnects.
# ---------------------------------------------------------------------------
@app.function(
    image=coord_image,
    timeout=21600,  # 6 hours
    secrets=SECRETS,
    volumes={"/results": results_vol},
)
def coordinate_model_eval(
    model_name: str,
    tasks_json: str,
    large: bool,
    shards: int,
    workers: int = 4,
    max_turns: int = 10,
    max_searches: int = 5,
    max_browses: int = 5,
    judge_model: str = "",
    prompt_preset: str = "",
    runs: int = 1,
) -> str:
    """Spawn GPU shards for a single model and collect results."""
    import sys

    sys.path.insert(0, "/root")
    from ace_eval import build_eval_summary

    tasks = json.loads(tasks_json)
    fn = eval_2gpu if large else eval_1gpu
    gpu_desc = "2xH100" if large else "1xH100"

    spawn_kwargs = dict(
        workers=workers,
        max_turns=max_turns,
        max_searches=max_searches,
        max_browses=max_browses,
        judge_model=judge_model,
        prompt_preset=prompt_preset,
        runs=runs,
    )

    # shard_tasks tracks which tasks each shard got (for failure reporting)
    handles = []
    shard_tasks: dict[int, list[dict]] = {}
    if shards > 1:
        chunks = [(i, tasks[i::shards]) for i in range(shards) if tasks[i::shards]]
        for shard_idx, chunk in chunks:
            print(
                f"  Dispatching shard {shard_idx + 1}/{shards} "
                f"({len(chunk)} tasks, {gpu_desc})",
                flush=True,
            )
            h = fn.spawn(
                model_name=model_name, tasks_json=json.dumps(chunk), **spawn_kwargs
            )
            handles.append((shard_idx, h))
            shard_tasks[shard_idx] = chunk
    else:
        print(f"  Dispatching {model_name} ({gpu_desc})", flush=True)
        h = fn.spawn(
            model_name=model_name, tasks_json=json.dumps(tasks), **spawn_kwargs
        )
        handles.append((0, h))
        shard_tasks[0] = tasks

    # Wait for all shards
    all_results = []
    max_time = 0.0
    for shard_idx, handle in sorted(handles):
        try:
            summary_json = handle.get()
            summary = json.loads(summary_json)
            all_results.extend(summary.get("results", []))
            max_time = max(max_time, summary.get("total_time_seconds", 0))
            print(
                f"  Shard {shard_idx + 1}: {summary['completed']}/{summary['total_tasks']} "
                f"completed in {summary['total_time_seconds']}s",
                flush=True,
            )
        except Exception as e:
            print(f"  Shard {shard_idx + 1}: FAILED: {e}", flush=True)
            # Add error entries for all tasks in this failed shard
            for task in shard_tasks.get(shard_idx, []):
                for run_id in range(1, runs + 1):
                    entry = {
                        "task_id": task["task_id"],
                        "domain": task.get("domain", "unknown"),
                        "status": f"error: shard {shard_idx + 1} failed: {e}",
                        "elapsed": 0,
                    }
                    if runs > 1:
                        entry["run_id"] = run_id
                    all_results.append(entry)

    # Build merged summary
    from pathlib import Path

    merged = build_eval_summary(model_name, all_results, max_time)
    merged["shards"] = len(handles)

    model_dir = model_name.replace("/", "_")
    if prompt_preset:
        model_dir = f"{model_dir}_{prompt_preset}"
    summary_path = Path("/results") / model_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(merged, f, indent=2)
    results_vol.commit()

    ok = merged["completed"]
    failed = merged["failed"]
    print(
        f"\n{model_name}: {ok}/{len(all_results)} completed, {failed} failed, "
        f"{max_time:.1f}s wall time",
        flush=True,
    )
    return json.dumps(merged)


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
    workers: int = 0,
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

    # Auto-size workers based on model size if not specified.
    # Small models (≤16B) leave lots of VRAM headroom → more concurrent sequences.
    # Large models (>16B) fill VRAM → fewer concurrent sequences.
    if workers <= 0:
        any_large = any(_is_large_model(m) for m in model_list)
        workers = 12 if any_large else 16
        print(
            f"Auto-sized workers: {workers} ({'large' if any_large else 'small'} model)"
        )

    print(f"Tasks: {len(tasks)}")
    print(f"Models: {[m.split('/')[-1] for m in model_list]}")
    if shards > 1:
        print(f"Shards: {shards} (each model gets {shards} GPU containers)")

    # Dispatch via coordinator (runs on Modal, survives --detach)
    handles: dict[str, object] = {}
    for m in model_list:
        large = _is_large_model(m)
        gpu_desc = "2xH100" if large else "1xH100"
        print(f"  Dispatching {m} via coordinator ({shards} shard(s), {gpu_desc})...")
        h = coordinate_model_eval.spawn(
            model_name=m,
            tasks_json=json.dumps(tasks),
            large=large,
            shards=shards,
            workers=workers,
            max_turns=max_turns,
            max_searches=max_searches,
            max_browses=max_browses,
            judge_model=judge_model,
            prompt_preset=prompt,
            runs=runs,
        )
        handles[m] = h

    # Collect results (fails gracefully in detach mode — coordinator keeps running)
    for m, h in handles.items():
        print(f"\nWaiting for {m}...")
        try:
            summary_json = h.get()
            summary = json.loads(summary_json)
            ok = summary["completed"]
            failed = summary["failed"]
            print(
                f"  {ok}/{summary['total_tasks']} completed, {failed} failed, "
                f"{summary['total_time_seconds']}s"
            )
        except Exception as e:
            print(f"  Local collection failed: {e}")
            print("  (coordinator still running on Modal — results go to volume)")

    print("\nDone.")
