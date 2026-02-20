"""
Baseline evaluation script for ACE tasks with tool use.

Runs models on ACE tasks with web_search + browse_page tools,
then produces autograder-compatible output for scoring.

Works with any OpenAI-compatible endpoint (vLLM on Modal, together.ai, OpenAI, etc.).

Usage:
    # Single model eval
    python ace_eval.py --model Qwen/Qwen2.5-7B-Instruct --api_base http://localhost:8000/v1

    # Multiple baselines (sequentially per model, parallel per task)
    python ace_eval.py --models gpt-4o-mini gpt-4o Qwen/Qwen2.5-72B-Instruct --workers 4

    # Parallel workers for faster eval
    python ace_eval.py --model gpt-4o-mini --workers 8

    # Run on specific domains
    python ace_eval.py --model gpt-4o-mini --domains shopping food
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import httpx
import pyarrow.parquet as pq
from openai import OpenAI

from ace_eval_prompts import DEFAULT_PROMPT, get_prompt, list_prompts
from ace_scoring.jina import extract_title, fetch_page
from ace_scoring.product import extract_products, map_products_to_sources
from ace_scoring.scorer import grade_task
from ace_scoring.sources import enrich_snippet_sources, sources_from_tool_history
from ace_scoring.types import format_criteria_for_autograder

# Tool definitions for the model (OpenAI function calling format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information about products, prices, "
                "reviews, availability, and purchase links."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_page",
            "description": (
                "Fetch and read the full content of a web page. Use after "
                "searching to get detailed product info, exact prices, and links."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the page to read.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

SEARCHAPI_BASE = "https://www.searchapi.io/api/v1/search"

_print_lock = Lock()


def _log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def execute_web_search(
    query: str, api_key: str, max_results: int = 5
) -> tuple[str, dict]:
    """Execute a web search via SearchAPI.io.

    Returns (formatted_text, raw_data).
    """
    try:
        resp = httpx.get(
            SEARCHAPI_BASE,
            params={
                "q": query,
                "engine": "google",
                "api_key": api_key,
                "num": max_results,
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()

        organic = data.get("organic_results", [])
        if not organic:
            return "No results found.", data

        parts = []
        for i, r in enumerate(organic, 1):
            parts.append(
                f"{i}. {r.get('title', '')}\n   {r.get('snippet', '')}\n   URL: {r.get('link', '')}"
            )

        return "Search Results:\n" + "\n\n".join(parts), data

    except Exception as e:
        return f"Search failed: {e}", {}


def execute_browse_page(url: str, max_chars: int = 8000) -> tuple[str, dict]:
    """Fetch page content via Jina Reader.

    Returns (formatted_text, page_data).
    """
    markdown, err = fetch_page(url, timeout=20.0, max_chars=200000)
    if err:
        return f"Failed to load page: {err}", {}

    title = extract_title(markdown, fallback=url)
    truncated = markdown[:max_chars]
    if len(markdown) > max_chars:
        truncated += "\n\n[Content truncated]"

    page_data = {"url": url, "title": title, "markdown": markdown}
    return f"Page content from {url}:\n\n{truncated}", page_data


def run_agent_loop(
    client: OpenAI,
    model: str,
    prompt: str,
    searchapi_key: str,
    system_prompt: str = "",
    max_turns: int = 10,
    max_searches: int = 5,
    max_browses: int = 5,
) -> tuple[str, dict]:
    """Run a model with tools until it produces a final response.

    Returns (response_text, tool_history).
    """
    if not system_prompt:
        system_prompt = get_prompt(DEFAULT_PROMPT)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    tool_history = {"searches": [], "browsed_pages": []}
    search_count = 0
    browse_count = 0

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]

        # Model produced a final response (no tool calls)
        if choice.finish_reason == "stop" or not choice.message.tool_calls:
            return choice.message.content or "", tool_history

        # Process tool calls
        messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            fn_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if fn_name == "web_search" and search_count < max_searches:
                result_text, raw_data = execute_web_search(
                    args.get("query", ""), searchapi_key
                )
                if raw_data:
                    tool_history["searches"].append(
                        {
                            "query": args.get("query", ""),
                            "results": raw_data,
                        }
                    )
                    search_count += 1
                _log(f"  [search #{search_count}] {args.get('query', '')}")

            elif fn_name == "browse_page" and browse_count < max_browses:
                result_text, page_data = execute_browse_page(args.get("url", ""))
                if page_data:
                    tool_history["browsed_pages"].append(page_data)
                    browse_count += 1
                _log(f"  [browse #{browse_count}] {args.get('url', '')}")

            elif fn_name == "web_search":
                result_text = "Search limit reached."
            elif fn_name == "browse_page":
                result_text = "Browse limit reached."
            else:
                result_text = f"Unknown tool: {fn_name}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                }
            )

    # Max turns reached — return whatever we have
    last_msg = messages[-1]
    if last_msg["role"] == "assistant" and isinstance(last_msg.get("content"), str):
        return last_msg["content"], tool_history

    return "", tool_history


def load_tasks_from_parquet(parquet_path: str) -> list[dict]:
    """Load ACE tasks from verl-format parquet."""
    table = pq.read_table(parquet_path)

    tasks = []
    for i in range(len(table)):
        prompt_list = table.column("prompt")[i].as_py()
        prompt_text = prompt_list[0]["content"] if prompt_list else ""

        reward_model = table.column("reward_model")[i].as_py()
        ground_truth = reward_model.get("ground_truth", "[]")
        criteria = json.loads(ground_truth)

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

    return tasks


def eval_single_task(
    client: OpenAI,
    model: str,
    task: dict,
    searchapi_key: str,
    output_dir: str,
    model_dir: str,
    task_idx: int,
    total_tasks: int,
    system_prompt: str = "",
    max_turns: int = 10,
    max_searches: int = 5,
    max_browses: int = 5,
    run_id: int | None = None,
) -> dict:
    """Evaluate a single ACE task. Thread-safe.

    Returns a result dict with task_id, domain, status, elapsed, etc.
    If run_id is set, results go into task_{id}/run_{run_id}/ subdirectory.
    """
    task_id = task["task_id"]
    domain = task["domain"]
    run_label = f" run {run_id}" if run_id is not None else ""
    _log(f"\n[{task_idx + 1}/{total_tasks}] Task {task_id} ({domain}){run_label}")

    start = time.time()

    try:
        response_text, tool_history = run_agent_loop(
            client=client,
            model=model,
            prompt=task["prompt"],
            searchapi_key=searchapi_key,
            system_prompt=system_prompt,
            max_turns=max_turns,
            max_searches=max_searches,
            max_browses=max_browses,
        )

        elapsed = time.time() - start
        n_searches = len(tool_history["searches"])
        n_browses = len(tool_history["browsed_pages"])
        _log(
            f"  Task {task_id}: {len(response_text)} chars, {n_searches} searches, {n_browses} browses ({elapsed:.1f}s)"
        )

        # Build autograder-compatible JSON (inlined from ace_grounding_wrapper)
        from datetime import datetime

        sources = sources_from_tool_history(tool_history)
        product_names = extract_products(response_text, task["prompt"])
        product_source_map = map_products_to_sources(product_names, sources)
        formatted_criteria = format_criteria_for_autograder(task["criteria"])

        autograder_input = {
            "task_id": task_id,
            "query": task["prompt"],
            "responseText": response_text,
            "provider": "custom",
            "productSourceMap": product_source_map,
            "criteria": formatted_criteria,
            "sources": sources,
            "failed_grounded_sites": [],
            "metadata": {
                "total_sources": len(sources),
                "scraped_at": datetime.now().isoformat(),
                "failed_scrapes": 0,
            },
            "pipeline_timing": {
                "total_seconds": 0.0,
                "scraping_seconds": 0.0,
                "processing_seconds": 0.0,
            },
        }
        if task.get("shop_vs_product") and domain.lower() == "shopping":
            autograder_input["shop_vs_product"] = task["shop_vs_product"]

        # Save all artifacts per task (nested under run_id when multiple runs)
        task_dir = Path(output_dir) / model_dir / domain / f"task_{task_id}"
        out_dir = task_dir / f"run_{run_id}" if run_id is not None else task_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Autograder input (what the ACE autograder expects)
        with open(out_dir / "2_scraped_sources.json", "w") as f:
            json.dump(autograder_input, f, indent=2, ensure_ascii=False)

        # 2. Save trace immediately (before scoring) so model response is never lost
        trace = {
            "task_id": task_id,
            "domain": domain,
            "model": model,
            "prompt": task["prompt"],
            "response_text": response_text,
            "tool_history": tool_history,
            "num_searches": n_searches,
            "num_browses": n_browses,
            "num_criteria": len(task["criteria"]),
            "elapsed_seconds": round(elapsed, 2),
            "scoring": None,
        }
        with open(out_dir / "trace.json", "w") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)

        # 3. Enrich sources + score (both can fail on external APIs)
        existing_products = [
            p["product_name"] for p in autograder_input.get("productSourceMap", [])
        ]
        try:
            # Enrich snippet-only sources with full page content via Jina
            enrich_snippet_sources(autograder_input["sources"])

            # Rewrite 2_scraped_sources.json with enriched content
            with open(out_dir / "2_scraped_sources.json", "w") as f:
                json.dump(autograder_input, f, indent=2, ensure_ascii=False)

            # Score: run the scorer to produce 3_autograder_results.json
            score_result = grade_task(
                task_id=task_id,
                response_text=response_text,
                criteria=task["criteria"],
                sources=autograder_input["sources"],
                product_source_map=autograder_input.get("productSourceMap"),
                products=existing_products,
                query=task["prompt"],
                domain=domain,
                shop_vs_product=task.get("shop_vs_product", "Product"),
            )
        except Exception as score_err:
            _log(
                f"  Task {task_id}: SCORING FAILED ({score_err}) — trace saved, skipping score"
            )
            result = {
                "task_id": task_id,
                "domain": domain,
                "status": f"score_error: {score_err}",
                "elapsed": elapsed,
                "response_len": len(response_text),
                "searches": n_searches,
                "browses": n_browses,
            }
            if run_id is not None:
                result["run_id"] = run_id
            return result

        with open(out_dir / "3_autograder_results.json", "w") as f:
            json.dump(score_result.to_dict(), f, indent=2, ensure_ascii=False)

        # Update trace with scoring results
        trace["scoring"] = {
            "total_score": score_result.total_score,
            "total_hurdle_score": score_result.total_hurdle_score,
            "num_criteria": score_result.num_criteria,
            "products": score_result.products,
            "summary": score_result.summary,
            "per_criterion": [
                {
                    "criterion_id": r.criterion_id,
                    "description": r.description,
                    "type": r.type,
                    "score": r.score,
                    "hurdle_tag": r.hurdle_tag,
                    "stage_reached": r.stage_reached,
                    "reasoning": r.reasoning,
                }
                for r in score_result.detailed_results
            ],
        }
        with open(out_dir / "trace.json", "w") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)

        _log(
            f"  Task {task_id}: score={score_result.total_score}/{score_result.num_criteria} "
            f"hurdle={score_result.total_hurdle_score} | saved {out_dir}"
        )

        result = {
            "task_id": task_id,
            "domain": domain,
            "status": "ok",
            "elapsed": elapsed,
            "response_len": len(response_text),
            "searches": n_searches,
            "browses": n_browses,
            "total_score": score_result.total_score,
            "total_hurdle_score": score_result.total_hurdle_score,
            "num_criteria": score_result.num_criteria,
            "criteria_scores": score_result.criteria_scores,
        }
        if run_id is not None:
            result["run_id"] = run_id
        return result

    except Exception as e:
        elapsed = time.time() - start
        _log(f"  Task {task_id}: FAILED ({e})")
        result = {
            "task_id": task_id,
            "domain": domain,
            "status": f"error: {e}",
            "elapsed": elapsed,
        }
        if run_id is not None:
            result["run_id"] = run_id
        return result


def build_eval_summary(model: str, results: list[dict], total_time: float) -> dict:
    """Build eval summary with per-domain breakdown. Shared by local and Modal eval.

    When results include run_id (multiple runs per task), adds per-task
    aggregation with mean/std scores.

    Computes ACE-style metrics (matching paper Tables 2 & 3):
    - ace_scores: overall and per-domain percentage scores
    - criteria_type_scores: per criteria-type pass rates per domain
    """
    from collections import defaultdict
    import statistics

    ok_results = [r for r in results if r["status"] == "ok"]
    ok = len(ok_results)
    failed = len(results) - ok

    domain_stats: dict[str, dict] = {}
    for r in results:
        d = r.get("domain", "unknown")
        if d not in domain_stats:
            domain_stats[d] = {
                "completed": 0,
                "failed": 0,
                "total_searches": 0,
                "total_browses": 0,
                "total_score": 0,
                "total_hurdle_score": 0,
                "total_criteria": 0,
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
        "model": model,
        "total_tasks": len(results),
        "completed": ok,
        "failed": failed,
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_task": round(sum(r["elapsed"] for r in ok_results) / ok, 2)
        if ok
        else 0,
        "total_searches": sum(r.get("searches", 0) for r in ok_results),
        "total_browses": sum(r.get("browses", 0) for r in ok_results),
        "domain_stats": domain_stats,
        "results": results,
    }

    # -- Per-task aggregation across runs (mean/std) --
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in ok_results:
        by_task[str(r["task_id"])].append(r)

    task_agg = {}
    for tid, runs in sorted(by_task.items()):
        scores = [r["total_score"] for r in runs]
        hurdle_scores = [r["total_hurdle_score"] for r in runs]
        nc = runs[0]["num_criteria"]
        score_pcts = [s / nc * 100 if nc > 0 else 0.0 for s in scores]
        hurdle_pcts = [s / nc * 100 if nc > 0 else 0.0 for s in hurdle_scores]
        task_agg[tid] = {
            "domain": runs[0]["domain"],
            "num_criteria": nc,
            "num_runs": len(runs),
            "scores": scores,
            "mean_score": round(statistics.mean(scores), 2),
            "std_score": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0,
            "mean_hurdle_score": round(statistics.mean(hurdle_scores), 2),
            "score_pct": round(statistics.mean(score_pcts), 1),
            "hurdle_score_pct": round(statistics.mean(hurdle_pcts), 1),
        }
    summary["task_aggregation"] = task_agg

    # -- ACE-style percentage scores (paper Table 2) --
    # Per-task pct = mean(hurdle_score / num_criteria * 100) across runs
    # Per-domain pct = mean of per-task pcts
    # Overall pct = mean of all per-task pcts
    domain_task_pcts: dict[str, list[float]] = defaultdict(list)
    domain_task_raw_pcts: dict[str, list[float]] = defaultdict(list)
    for tid, tagg in task_agg.items():
        d = tagg["domain"]
        domain_task_pcts[d].append(tagg["hurdle_score_pct"])
        domain_task_raw_pcts[d].append(tagg["score_pct"])

    ace_domain_scores = {}
    all_task_pcts: list[float] = []
    all_task_raw_pcts: list[float] = []
    for d in sorted(domain_task_pcts.keys()):
        pcts = domain_task_pcts[d]
        raw_pcts = domain_task_raw_pcts[d]
        ace_domain_scores[d] = {
            "score_pct": round(statistics.mean(pcts), 1),
            "raw_score_pct": round(statistics.mean(raw_pcts), 1),
            "num_tasks": len(pcts),
        }
        if len(pcts) > 1:
            ace_domain_scores[d]["std_pct"] = round(statistics.stdev(pcts), 1)
        all_task_pcts.extend(pcts)
        all_task_raw_pcts.extend(raw_pcts)

    ace_scores: dict = {
        "overall": round(statistics.mean(all_task_pcts), 1) if all_task_pcts else 0.0,
        "overall_raw": round(statistics.mean(all_task_raw_pcts), 1)
        if all_task_raw_pcts
        else 0.0,
        "num_tasks": len(all_task_pcts),
        "domains": ace_domain_scores,
    }
    if len(all_task_pcts) > 1:
        ace_scores["overall_std"] = round(statistics.stdev(all_task_pcts), 1)
    summary["ace_scores"] = ace_scores

    # -- Per criteria-type scores (paper Table 3) --
    # Group criterion scores by (domain, type), compute mean score * 100
    # Note: score_pct can be negative for grounded criteria (Shopping/Gaming)
    # since hallucination scores -1. For non-grounded (DIY/Food) it equals pass rate.
    has_criteria_scores = any(r.get("criteria_scores") for r in ok_results)
    if has_criteria_scores:
        # criteria_scores: [[score, type, hurdle_tag], ...]
        type_scores: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in ok_results:
            d = r.get("domain", "unknown")
            for cs in r.get("criteria_scores", []):
                score, ctype, _hurdle = cs
                type_scores[d][ctype].append(score)

        criteria_type_scores = {}
        for d in sorted(type_scores.keys()):
            criteria_type_scores[d] = {}
            for ctype in sorted(type_scores[d].keys()):
                scores = type_scores[d][ctype]
                criteria_type_scores[d][ctype] = {
                    "score_pct": round(statistics.mean(scores) * 100, 1),
                    "count": len(scores),
                }
        summary["criteria_type_scores"] = criteria_type_scores

    return summary


def run_eval_for_model(
    model: str,
    tasks: list[dict],
    api_base: str,
    api_key: str,
    searchapi_key: str,
    output_dir: str,
    workers: int,
    max_turns: int,
    max_searches: int,
    max_browses: int,
    system_prompt: str = "",
    prompt_name: str = "",
    runs: int = 1,
) -> list[dict]:
    """Run eval for a single model across all tasks, with parallel workers.

    When runs > 1, each task is evaluated multiple times. Results are stored
    in task_{id}/run_{n}/ subdirectories and the summary includes per-task
    mean/std aggregation.
    """
    model_dir = model.replace("/", "_")
    if prompt_name:
        model_dir = f"{model_dir}_{prompt_name}"
    client = OpenAI(api_key=api_key, base_url=api_base)

    # Expand tasks × runs into work items
    use_run_id = runs > 1
    work_items = []
    for task in tasks:
        for run_id in range(1, runs + 1):
            work_items.append((task, run_id if use_run_id else None))

    total = len(work_items)

    _log(f"\n{'=' * 60}")
    _log(f"Model: {model}")
    _log(
        f"Tasks: {len(tasks)}, Runs: {runs}, Total work items: {total}, Workers: {workers}"
    )
    _log(f"{'=' * 60}")

    results = []
    start_all = time.time()

    if workers <= 1:
        # Sequential
        for i, (task, run_id) in enumerate(work_items):
            result = eval_single_task(
                client=client,
                model=model,
                task=task,
                searchapi_key=searchapi_key,
                output_dir=output_dir,
                model_dir=model_dir,
                task_idx=i,
                total_tasks=total,
                system_prompt=system_prompt,
                max_turns=max_turns,
                max_searches=max_searches,
                max_browses=max_browses,
                run_id=run_id,
            )
            results.append(result)
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for i, (task, run_id) in enumerate(work_items):
                fut = executor.submit(
                    eval_single_task,
                    client=client,
                    model=model,
                    task=task,
                    searchapi_key=searchapi_key,
                    output_dir=output_dir,
                    model_dir=model_dir,
                    task_idx=i,
                    total_tasks=total,
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
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = total - ok

    _log(f"\n--- {model} Summary ---")
    _log(f"  {ok}/{total} tasks completed ({failed} failed)")
    _log(f"  Total time: {total_time:.1f}s")
    if ok:
        avg_time = sum(r["elapsed"] for r in results if r["status"] == "ok") / ok
        _log(f"  Avg time per task: {avg_time:.1f}s")

    summary = build_eval_summary(model, results, total_time)
    summary_path = Path(output_dir) / model_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"  Summary: {summary_path}")

    return results


def rebuild_summary(results_dir: str) -> None:
    """Rebuild summary.json from on-disk autograder results.

    Reads 3_autograder_results.json + trace.json from each task/run directory,
    reconstructs the results list with criteria_scores, and regenerates
    summary.json with ACE-style metrics.
    """
    from pathlib import Path

    results_path = Path(results_dir)
    if not results_path.is_dir():
        print(f"Error: {results_dir} is not a directory")
        return

    # Read existing summary for model name and timing info
    summary_file = results_path / "summary.json"
    old_summary = {}
    if summary_file.exists():
        with open(summary_file) as f:
            old_summary = json.load(f)

    model_name = old_summary.get("model", results_path.name)
    total_time = old_summary.get("total_time_seconds", 0.0)

    # Scan for autograder result files
    results = []
    for ag_file in sorted(results_path.rglob("3_autograder_results.json")):
        # Extract domain, task_id, run_id from path
        # Pattern: {domain}/task_{id}/run_{n}/3_autograder_results.json
        # or:      {domain}/task_{id}/3_autograder_results.json
        parts = ag_file.relative_to(results_path).parts
        if len(parts) < 2:
            continue

        domain = parts[0]
        task_dir = parts[1]
        if not task_dir.startswith("task_"):
            continue
        task_id = task_dir.replace("task_", "")

        run_id = None
        if len(parts) >= 3 and parts[2].startswith("run_"):
            run_id = int(parts[2].replace("run_", ""))

        with open(ag_file) as f:
            ag_data = json.load(f)

        # Read trace for timing/search/browse counts
        trace_file = ag_file.parent / "trace.json"
        elapsed = 0.0
        response_len = 0
        n_searches = 0
        n_browses = 0
        if trace_file.exists():
            with open(trace_file) as f:
                trace = json.load(f)
            elapsed = trace.get("elapsed_seconds", 0.0)
            response_len = len(trace.get("response_text", ""))
            n_searches = trace.get("num_searches", 0)
            n_browses = trace.get("num_browses", 0)

        result: dict = {
            "task_id": task_id,
            "domain": domain,
            "status": "ok",
            "elapsed": elapsed,
            "response_len": response_len,
            "searches": n_searches,
            "browses": n_browses,
            "total_score": ag_data.get("total_score", 0),
            "total_hurdle_score": ag_data.get("total_hurdle_score", 0),
            "num_criteria": ag_data.get("num_criteria", 0),
            "criteria_scores": ag_data.get("criteria_scores", []),
        }
        if run_id is not None:
            result["run_id"] = run_id
        results.append(result)

    if not results:
        print(f"No autograder results found in {results_dir}")
        return

    print(f"Found {len(results)} result files across {results_dir}")

    summary = build_eval_summary(model_name, results, total_time)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    ace = summary.get("ace_scores", {})
    print(f"\nACE Scores for {model_name}:")
    print(f"  Overall: {ace.get('overall', 0)}%")
    for d, ds in ace.get("domains", {}).items():
        print(f"  {d:>10}: {ds['score_pct']}%")

    if "criteria_type_scores" in summary:
        print("\nCriteria Type Breakdown:")
        for d, types in summary["criteria_type_scores"].items():
            print(f"  {d}:")
            for ctype, info in types.items():
                print(f"    {ctype:45} {info['score_pct']:>6.1f}%  (n={info['count']})")

    print(f"\nSaved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Run ACE baseline eval with tools")

    # Model(s) — support both --model and --models
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument("--model", help="Single model name")
    model_group.add_argument(
        "--models", nargs="+", help="Multiple model names for baseline comparison"
    )

    parser.add_argument(
        "--rebuild-summary",
        metavar="DIR",
        help="Rebuild summary.json from existing results directory (no eval run needed)",
    )

    parser.add_argument(
        "--api_base",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api_key", default=None, help="API key (defaults to OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--searchapi_key",
        default=None,
        help="SearchAPI.io key (defaults to SEARCHAPI_IO_KEY env var)",
    )
    parser.add_argument(
        "--parquet",
        default="ace_verl_data/test.parquet",
        help="Path to ACE parquet file",
    )
    parser.add_argument("--output_dir", default="eval_results", help="Output directory")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Filter by domain (e.g. shopping food)",
    )
    parser.add_argument("--task_ids", nargs="+", default=None, help="Filter by task ID")
    parser.add_argument(
        "--max_turns", type=int, default=10, help="Max agent loop turns per task"
    )
    parser.add_argument(
        "--max_searches", type=int, default=5, help="Max searches per task"
    )
    parser.add_argument(
        "--max_browses", type=int, default=5, help="Max browses per task"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers per model (default: 1 = sequential)",
    )
    parser.add_argument(
        "--judge_model",
        default=None,
        help="OpenAI model for autograder judge (default: gpt-4o, or ACE_JUDGE_MODEL env)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"System prompt preset ({', '.join(list_prompts())}). Default: {DEFAULT_PROMPT}",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per task (default: 1). Use >1 for variance estimation.",
    )
    args = parser.parse_args()

    # Rebuild summary mode — no model needed
    if args.rebuild_summary:
        rebuild_summary(args.rebuild_summary)
        return

    # Require model for eval mode
    if not args.model and not args.models:
        parser.error("--model or --models is required (unless using --rebuild-summary)")

    # Set judge model via env var so ace_scoring.llm picks it up
    if args.judge_model:
        os.environ["ACE_JUDGE_MODEL"] = args.judge_model

    system_prompt = get_prompt(args.prompt)
    _log(f"Prompt preset: {args.prompt}")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    searchapi_key = args.searchapi_key or os.environ.get("SEARCHAPI_IO_KEY", "")

    models = args.models if args.models else [args.model]

    # Load tasks
    tasks = load_tasks_from_parquet(args.parquet)
    _log(f"Loaded {len(tasks)} tasks from {args.parquet}")

    # Filter
    if args.domains:
        tasks = [t for t in tasks if t["domain"] in args.domains]
    if args.task_ids:
        tasks = [t for t in tasks if str(t["task_id"]) in args.task_ids]

    _log(f"Evaluating {len(tasks)} tasks across {len(models)} model(s)")

    all_results = {}
    for model in models:
        results = run_eval_for_model(
            model=model,
            tasks=tasks,
            api_base=args.api_base,
            api_key=api_key,
            searchapi_key=searchapi_key,
            output_dir=args.output_dir,
            workers=args.workers,
            max_turns=args.max_turns,
            max_searches=args.max_searches,
            max_browses=args.max_browses,
            system_prompt=system_prompt,
            prompt_name=args.prompt,
            runs=args.runs,
        )
        all_results[model] = results

    _log("\nDone.")


if __name__ == "__main__":
    main()
