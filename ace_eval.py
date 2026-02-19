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

from ace_grounding_wrapper import build_autograder_input
from ace_scoring.scorer import grade_task
from ace_scoring.sources import enrich_snippet_sources

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

SYSTEM_PROMPT = """You are a helpful shopping and recommendation assistant with access to web search and page browsing tools.

You have two tools:
- web_search(query): Search the web. Returns titles, snippets, and URLs.
- browse_page(url): Read the full content of a web page.

CRITICAL RULES:
1. You MUST browse at least 2-3 pages before giving recommendations. Search snippets are NOT enough — they are often outdated or incomplete.
2. NEVER include a link you haven't verified by browsing. If browse_page fails or returns an error/login wall, do NOT use that link.
3. NEVER invent or guess prices, features, or availability. Only state facts you read directly from a browsed page.
4. Prefer direct product/purchase links (e.g., Amazon, store pages). Avoid linking to review articles unless the user asked for reviews.
5. If you cannot find a working link for a product, say so — do not fabricate one.

WORKFLOW for every request:
1. Search the web to find relevant results
2. Browse the most promising pages (at least 2-3) to read actual content, verify links work, and get real details
3. If a page fails to load or requires login, try a different result
4. Provide recommendations using ONLY facts from pages you successfully browsed — with exact product names, real prices from the page, and direct working links

Here is an example of a good interaction:

User: I need a budget mechanical keyboard for programming under $50.
Assistant: [calls web_search(query="best budget mechanical keyboard under $50 2024")]
Tool result: 1. Best Budget Mechanical Keyboards - RTINGS.com
   The Redragon K552 and Royal Kludge RK61 are top picks under $50...
   URL: https://www.rtings.com/keyboard/reviews/best/budget
   2. Redragon K552 on Amazon - $29.99
   URL: https://www.amazon.com/dp/B016MAK38U
Assistant: [calls browse_page(url="https://www.amazon.com/dp/B016MAK38U")]
Tool result: Page content: Redragon K552 KUMARA Mechanical Gaming Keyboard - $29.99 - Outemu Red switches, TKL layout, RGB backlighting, 4.5 stars...
Assistant: [calls browse_page(url="https://www.rtings.com/keyboard/reviews/best/budget")]
Tool result: Page content: ...The Redragon K552 is our top budget pick. Royal Kludge RK61 ($39.99) is best for compact wireless...
Assistant: Based on my research, here are my top recommendations:

1. **Redragon K552 KUMARA** - $29.99
   - Outemu Red mechanical switches
   - Compact TKL (tenkeyless) layout, RGB backlighting
   - 4.5/5 stars on Amazon
   - Buy link: https://www.amazon.com/dp/B016MAK38U

2. **Royal Kludge RK61** - $39.99
   ...

Now respond to the user's actual request below."""

SEARCHAPI_BASE = "https://www.searchapi.io/api/v1/search"
JINA_READER_BASE = "https://r.jina.ai"

_print_lock = Lock()


def _log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def execute_web_search(query: str, api_key: str, max_results: int = 5) -> tuple[str, dict]:
    """Execute a web search via SearchAPI.io.

    Returns (formatted_text, raw_data).
    """
    try:
        resp = httpx.get(
            SEARCHAPI_BASE,
            params={"q": query, "engine": "google", "api_key": api_key, "num": max_results},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()

        organic = data.get("organic_results", [])
        if not organic:
            return "No results found.", data

        parts = []
        for i, r in enumerate(organic, 1):
            parts.append(f"{i}. {r.get('title', '')}\n   {r.get('snippet', '')}\n   URL: {r.get('link', '')}")

        return "Search Results:\n" + "\n\n".join(parts), data

    except Exception as e:
        return f"Search failed: {e}", {}


def execute_browse_page(url: str, max_chars: int = 8000) -> tuple[str, dict]:
    """Fetch page content via Jina Reader.

    Returns (formatted_text, page_data).
    """
    try:
        resp = httpx.get(
            f"{JINA_READER_BASE}/{url}",
            headers={"Accept": "text/markdown"},
            timeout=20.0,
        )
        resp.raise_for_status()
        markdown = resp.text

        title = url
        for line in markdown.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                break

        truncated = markdown[:max_chars]
        if len(markdown) > max_chars:
            truncated += "\n\n[Content truncated]"

        page_data = {"url": url, "title": title, "markdown": markdown}
        return f"Page content from {url}:\n\n{truncated}", page_data

    except Exception as e:
        return f"Failed to load page: {e}", {}


def run_agent_loop(
    client: OpenAI,
    model: str,
    prompt: str,
    searchapi_key: str,
    max_turns: int = 10,
    max_searches: int = 5,
    max_browses: int = 5,
) -> tuple[str, dict]:
    """Run a model with tools until it produces a final response.

    Returns (response_text, tool_history).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
                    tool_history["searches"].append({
                        "query": args.get("query", ""),
                        "results": raw_data,
                    })
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

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            })

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

        tasks.append({
            "task_id": extra_info.get("task_id", str(i)),
            "prompt": prompt_text,
            "criteria": criteria,
            "domain": extra_info.get("domain", "unknown"),
            "shop_vs_product": extra_info.get("shop_vs_product"),
        })

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
    max_turns: int = 10,
    max_searches: int = 5,
    max_browses: int = 5,
) -> dict:
    """Evaluate a single ACE task. Thread-safe.

    Returns a result dict with task_id, domain, status, elapsed, etc.
    """
    task_id = task["task_id"]
    domain = task["domain"]
    _log(f"\n[{task_idx+1}/{total_tasks}] Task {task_id} ({domain})")

    start = time.time()

    try:
        response_text, tool_history = run_agent_loop(
            client=client,
            model=model,
            prompt=task["prompt"],
            searchapi_key=searchapi_key,
            max_turns=max_turns,
            max_searches=max_searches,
            max_browses=max_browses,
        )

        elapsed = time.time() - start
        n_searches = len(tool_history["searches"])
        n_browses = len(tool_history["browsed_pages"])
        _log(f"  Task {task_id}: {len(response_text)} chars, {n_searches} searches, {n_browses} browses ({elapsed:.1f}s)")

        # Build autograder-compatible JSON
        autograder_input = build_autograder_input(
            task_id=task_id,
            query=task["prompt"],
            response_text=response_text,
            criteria=task["criteria"],
            tool_history=tool_history,
            domain=domain,
            shop_vs_product=task.get("shop_vs_product"),
        )

        # Save all artifacts per task
        out_dir = Path(output_dir) / model_dir / domain / f"task_{task_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Autograder input (what the ACE autograder expects)
        with open(out_dir / "2_scraped_sources.json", "w") as f:
            json.dump(autograder_input, f, indent=2, ensure_ascii=False)

        # 2. Raw trace: response + full tool history for debugging
        # (scoring fields added after grade_task below)

        # 2.5. Enrich snippet-only sources with full page content via Jina
        enrich_snippet_sources(autograder_input["sources"])

        # 3. Score: run the scorer to produce 3_autograder_results.json
        # Pass products from autograder_input to avoid double LLM extraction
        existing_products = [
            p["product_name"]
            for p in autograder_input.get("productSourceMap", [])
        ]
        score_result = grade_task(
            task_id=task_id,
            response_text=response_text,
            criteria=task["criteria"],
            sources=autograder_input["sources"],
            product_source_map=autograder_input.get("productSourceMap"),
            products=existing_products or None,
            query=task["prompt"],
            domain=domain,
            shop_vs_product=task.get("shop_vs_product", "Product"),
        )
        with open(out_dir / "3_autograder_results.json", "w") as f:
            json.dump(score_result.to_dict(), f, indent=2, ensure_ascii=False)

        # Write enriched trace with tool usage stats + scoring results
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
            "scoring": {
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
            },
        }
        with open(out_dir / "trace.json", "w") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)

        _log(f"  Task {task_id}: score={score_result.total_score}/{score_result.num_criteria} "
             f"hurdle={score_result.total_hurdle_score} | saved {out_dir}")

        return {
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
        }

    except Exception as e:
        elapsed = time.time() - start
        _log(f"  Task {task_id}: FAILED ({e})")
        return {
            "task_id": task_id,
            "domain": domain,
            "status": f"error: {e}",
            "elapsed": elapsed,
        }


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
) -> list[dict]:
    """Run eval for a single model across all tasks, with parallel workers."""
    model_dir = model.replace("/", "_")
    client = OpenAI(api_key=api_key, base_url=api_base)
    total = len(tasks)

    _log(f"\n{'='*60}")
    _log(f"Model: {model}")
    _log(f"Tasks: {total}, Workers: {workers}")
    _log(f"{'='*60}")

    results = []
    start_all = time.time()

    if workers <= 1:
        # Sequential
        for i, task in enumerate(tasks):
            result = eval_single_task(
                client=client, model=model, task=task,
                searchapi_key=searchapi_key, output_dir=output_dir,
                model_dir=model_dir, task_idx=i, total_tasks=total,
                max_turns=max_turns, max_searches=max_searches,
                max_browses=max_browses,
            )
            results.append(result)
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for i, task in enumerate(tasks):
                fut = executor.submit(
                    eval_single_task,
                    client=client, model=model, task=task,
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

    _log(f"\n--- {model} Summary ---")
    _log(f"  {ok}/{total} tasks completed ({failed} failed)")
    _log(f"  Total time: {total_time:.1f}s")
    if ok:
        avg_time = sum(r["elapsed"] for r in results if r["status"] == "ok") / ok
        _log(f"  Avg time per task: {avg_time:.1f}s")

    # Write summary JSON with per-domain breakdown
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

    summary_path = Path(output_dir) / model_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": model,
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
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"  Summary: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ACE baseline eval with tools")

    # Model(s) — support both --model and --models
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Single model name")
    model_group.add_argument("--models", nargs="+", help="Multiple model names for baseline comparison")

    parser.add_argument("--api_base", default="https://api.openai.com/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key", default=None, help="API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--searchapi_key", default=None, help="SearchAPI.io key (defaults to SEARCHAPI_IO_KEY env var)")
    parser.add_argument("--parquet", default="ace_verl_data/test.parquet", help="Path to ACE parquet file")
    parser.add_argument("--output_dir", default="eval_results", help="Output directory")
    parser.add_argument("--domains", nargs="+", default=None, help="Filter by domain (e.g. shopping food)")
    parser.add_argument("--task_ids", nargs="+", default=None, help="Filter by task ID")
    parser.add_argument("--max_turns", type=int, default=10, help="Max agent loop turns per task")
    parser.add_argument("--max_searches", type=int, default=5, help="Max searches per task")
    parser.add_argument("--max_browses", type=int, default=5, help="Max browses per task")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers per model (default: 1 = sequential)")
    parser.add_argument("--judge_model", default=None, help="OpenAI model for autograder judge (default: gpt-4o, or ACE_JUDGE_MODEL env)")
    args = parser.parse_args()

    # Set judge model via env var so ace_scoring.llm picks it up
    if args.judge_model:
        os.environ["ACE_JUDGE_MODEL"] = args.judge_model

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
            model=model, tasks=tasks,
            api_base=args.api_base, api_key=api_key,
            searchapi_key=searchapi_key, output_dir=args.output_dir,
            workers=args.workers, max_turns=args.max_turns,
            max_searches=args.max_searches, max_browses=args.max_browses,
        )
        all_results[model] = results

    _log("\nDone.")


if __name__ == "__main__":
    main()
