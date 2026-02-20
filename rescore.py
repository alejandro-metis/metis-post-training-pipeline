"""Re-score existing eval results with enriched sources.

Walks eval result directories, re-enriches snippet-only sources via Jina
(using JINA_API_KEY for higher rate limits), re-runs grade_task(), and
updates 2_scraped_sources.json, 3_autograder_results.json, and the scoring
section of trace.json.

Does NOT re-run models or re-do web searches — only re-scores from saved artifacts.

Usage:
    # Re-score tasks with unenriched sources (default — only re-scores what needs it)
    python rescore.py --results-dir eval_results/gpt-5.2_fewshot

    # Force re-score ALL tasks (re-enriches and re-grades everything)
    python rescore.py --results-dir eval_results/gpt-5.2_fewshot --all

    # Dry run: show what would be re-scored without writing
    python rescore.py --results-dir eval_results --dry-run
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from ace_scoring.product import map_products_to_sources
from ace_scoring.scorer import grade_task
from ace_scoring.sources import enrich_snippet_sources

_print_lock = Lock()


def _log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def _has_unenriched_sources(sources: list[dict], min_chars: int = 500) -> bool:
    """Check if any sources have only snippet-level content."""
    for src in sources:
        wc = src.get("webpage_content", {})
        markdown = wc.get("markdown", "")
        if 0 < len(markdown) < min_chars and not wc.get("enrichment_error"):
            return True
    return False


def rescore_run(
    run_dir: Path, dry_run: bool = False, only_failed: bool = False
) -> dict:
    """Re-score a single run directory.

    Returns a result dict with task_id, run, old/new scores, etc.
    """
    trace_path = run_dir / "trace.json"
    sources_path = run_dir / "2_scraped_sources.json"
    results_path = run_dir / "3_autograder_results.json"

    if not trace_path.exists() or not sources_path.exists():
        return {"dir": str(run_dir), "status": "skip", "reason": "missing files"}

    with open(trace_path) as f:
        trace = json.load(f)
    with open(sources_path) as f:
        autograder_input = json.load(f)

    task_id = trace["task_id"]
    domain = trace["domain"]
    sources = autograder_input["sources"]

    # Check if re-enrichment is needed
    if only_failed and not _has_unenriched_sources(sources):
        return {
            "dir": str(run_dir),
            "task_id": task_id,
            "status": "skip",
            "reason": "sources already enriched",
        }

    old_scoring = trace.get("scoring") or {}
    old_score = old_scoring.get("total_score", "?")
    old_hurdle = old_scoring.get("total_hurdle_score", "?")

    if dry_run:
        unenriched = sum(
            1
            for s in sources
            if 0 < len(s.get("webpage_content", {}).get("markdown", "")) < 500
        )
        return {
            "dir": str(run_dir),
            "task_id": task_id,
            "status": "dry_run",
            "old_score": old_score,
            "unenriched_sources": unenriched,
        }

    # Step 1: Re-enrich snippet sources with Jina API key
    enrich_snippet_sources(sources)

    # Step 2: Re-map products to sources (source content may have changed)
    response_text = trace["response_text"]
    query = autograder_input.get("query", trace.get("prompt", ""))
    existing_products = [
        p["product_name"] for p in autograder_input.get("productSourceMap", [])
    ]
    product_source_map = map_products_to_sources(existing_products, sources)
    autograder_input["productSourceMap"] = product_source_map

    # Step 3: Re-run grade_task()
    criteria = autograder_input["criteria"]
    score_result = grade_task(
        task_id=task_id,
        response_text=response_text,
        criteria=criteria,
        sources=sources,
        product_source_map=product_source_map,
        products=existing_products,
        query=query,
        domain=domain,
        shop_vs_product=autograder_input.get("shop_vs_product", "Product"),
    )

    # Step 4: Write updated files
    # 4a: Update 2_scraped_sources.json (enriched sources + updated productSourceMap)
    with open(sources_path, "w") as f:
        json.dump(autograder_input, f, indent=2, ensure_ascii=False)

    # 4b: Overwrite 3_autograder_results.json
    with open(results_path, "w") as f:
        json.dump(score_result.to_dict(), f, indent=2, ensure_ascii=False)

    # 4c: Update scoring section in trace.json
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
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    new_score = score_result.total_score
    new_hurdle = score_result.total_hurdle_score
    changed = old_score != new_score or old_hurdle != new_hurdle

    return {
        "dir": str(run_dir),
        "task_id": task_id,
        "status": "rescored",
        "old_score": old_score,
        "new_score": new_score,
        "old_hurdle": old_hurdle,
        "new_hurdle": new_hurdle,
        "changed": changed,
    }


def find_run_dirs(results_dir: Path) -> list[Path]:
    """Find all run directories (containing trace.json) under results_dir."""
    run_dirs = []
    for trace_path in sorted(results_dir.rglob("trace.json")):
        run_dirs.append(trace_path.parent)
    return run_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Re-score eval results with enriched sources"
    )
    parser.add_argument(
        "--results-dir", required=True, help="Path to eval results directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be re-scored without writing",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-score all tasks, not just ones with unenriched sources",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel workers (default: 8)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    run_dirs = find_run_dirs(results_dir)
    if not run_dirs:
        print(f"No run directories found under {results_dir}")
        sys.exit(1)

    _log(f"Found {len(run_dirs)} run directories under {results_dir}")
    if args.dry_run:
        _log("DRY RUN — no files will be modified\n")

    start = time.time()
    results = []

    if args.workers <= 1:
        for i, run_dir in enumerate(run_dirs):
            result = rescore_run(
                run_dir, dry_run=args.dry_run, only_failed=not args.all
            )
            results.append(result)
            if result["status"] == "rescored":
                tag = " CHANGED" if result["changed"] else ""
                _log(
                    f"[{i + 1}/{len(run_dirs)}] {run_dir}: "
                    f"{result['old_score']} -> {result['new_score']} "
                    f"(hurdle {result['old_hurdle']} -> {result['new_hurdle']}){tag}"
                )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for run_dir in run_dirs:
                fut = executor.submit(
                    rescore_run,
                    run_dir,
                    dry_run=args.dry_run,
                    only_failed=not args.all,
                )
                futures[fut] = run_dir

            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                results.append(result)
                run_dir = futures[fut]
                if result["status"] == "rescored":
                    tag = " CHANGED" if result["changed"] else ""
                    _log(
                        f"[{i + 1}/{len(run_dirs)}] {run_dir}: "
                        f"{result['old_score']} -> {result['new_score']} "
                        f"(hurdle {result['old_hurdle']} -> {result['new_hurdle']}){tag}"
                    )

    elapsed = time.time() - start

    # Summary
    rescored = [r for r in results if r["status"] == "rescored"]
    changed = [r for r in rescored if r.get("changed")]
    skipped = [r for r in results if r["status"] == "skip"]

    _log(f"\n{'=' * 60}")
    _log(
        f"Total: {len(results)}, Re-scored: {len(rescored)}, "
        f"Changed: {len(changed)}, Skipped: {len(skipped)}"
    )
    _log(f"Time: {elapsed:.1f}s")

    if changed:
        _log("\nScore changes:")
        for r in sorted(changed, key=lambda x: x["dir"]):
            _log(
                f"  {r['dir']}: {r['old_score']} -> {r['new_score']} "
                f"(hurdle {r['old_hurdle']} -> {r['new_hurdle']})"
            )


if __name__ == "__main__":
    main()
