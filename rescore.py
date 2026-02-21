"""Re-score failed hurdle criteria in existing eval results.

Walks eval result directories, re-enriches snippet-only sources via Jina,
re-runs grade_criterion() ONLY on failed hurdle criteria (score <= 0,
hurdle_tag == "Hurdle"), and marks them as rescored so subsequent runs skip them.

Also updates summary.json files with new scores and recomputed aggregates.

Does NOT re-run models or re-do web searches — only re-scores from saved artifacts.

Usage:
    # Re-score failed hurdle criteria (skips already-rescored)
    python rescore.py --results-dir eval_results/gpt-5.2_fewshot

    # Force re-score even already-rescored hurdle criteria
    python rescore.py --results-dir eval_results/gpt-5.2_fewshot --force

    # Dry run: show what would be re-scored without writing
    python rescore.py --results-dir eval_results --dry-run
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from threading import Lock

from ace_scoring.product import map_products_to_sources
from ace_scoring.scorer import grade_criterion
from ace_scoring.sources import enrich_snippet_sources

_print_lock = Lock()


def _log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def _recalculate_task_scores(detailed: list[dict]) -> tuple[list, int, int, dict]:
    """Recalculate task-level aggregates from detailed criterion results.

    Returns (criteria_scores, total_score, total_hurdle_score, summary).
    """
    scores_with_types = []
    scores = []
    for r in detailed:
        s = r.get("score", 0)
        t = r.get("type", "")
        h = r.get("hurdle_tag", "Not")
        scores_with_types.append([s, t, h])
        scores.append(s)

    total_score = sum(scores)
    hurdle_scores = [swt[0] for swt in scores_with_types if swt[2] == "Hurdle"]
    if hurdle_scores and any(s <= 0 for s in hurdle_scores):
        total_hurdle_score = 0
    else:
        total_hurdle_score = total_score

    summary = {
        "pass_count": sum(1 for s in scores if s == 1),
        "fail_response_count": sum(1 for s in scores if s == 0),
        "fail_source_count": sum(1 for s in scores if s == -1),
        "total": len(scores),
        "hurdle_count": len(hurdle_scores),
        "hurdle_pass_count": sum(1 for s in hurdle_scores if s == 1),
    }

    return scores_with_types, total_score, total_hurdle_score, summary


def rescore_run(run_dir: Path, dry_run: bool = False, force: bool = False) -> dict:
    """Re-score failed hurdle criteria in a single run directory.

    Only re-grades criteria where hurdle_tag == "Hurdle" and score <= 0.
    Marks rescored criteria with "rescored": true so subsequent runs skip them.

    Args:
        force: If True, re-score even criteria already marked as rescored.
    """
    trace_path = run_dir / "trace.json"
    sources_path = run_dir / "2_scraped_sources.json"
    results_path = run_dir / "3_autograder_results.json"

    if not trace_path.exists() or not sources_path.exists():
        return {"dir": str(run_dir), "status": "skip", "reason": "missing files"}

    if not results_path.exists():
        return {"dir": str(run_dir), "status": "skip", "reason": "no existing results"}

    with open(trace_path) as f:
        trace = json.load(f)
    with open(sources_path) as f:
        autograder_input = json.load(f)
    with open(results_path) as f:
        existing_results = json.load(f)

    task_id = trace["task_id"]
    domain = trace["domain"]
    sources = autograder_input["sources"]
    criteria = autograder_input["criteria"]

    # Find failed hurdle criteria that need rescoring
    detailed = existing_results.get("detailed_results", [])
    failed_hurdle_indices = []
    for i, r in enumerate(detailed):
        if r.get("hurdle_tag") != "Hurdle":
            continue
        if r.get("score", 1) > 0:
            continue
        if not force and r.get("rescored", False):
            continue
        failed_hurdle_indices.append(i)

    if not failed_hurdle_indices:
        return {
            "dir": str(run_dir),
            "task_id": task_id,
            "status": "skip",
            "reason": "no failed hurdle criteria to rescore",
        }

    old_score = existing_results.get("total_score", "?")
    old_hurdle = existing_results.get("total_hurdle_score", "?")

    if dry_run:
        failed_ids = [
            detailed[i].get("criterion_id", "?") for i in failed_hurdle_indices
        ]
        return {
            "dir": str(run_dir),
            "task_id": task_id,
            "status": "dry_run",
            "old_score": old_score,
            "old_hurdle": old_hurdle,
            "failed_hurdle_criteria": failed_ids,
        }

    # Step 1: Re-enrich snippet sources
    _, had_transient_failures = enrich_snippet_sources(sources)

    # Step 2: Re-map products to sources (source content may have changed)
    response_text = trace["response_text"]
    existing_products = [
        p["product_name"] for p in autograder_input.get("productSourceMap", [])
    ]
    product_source_map = map_products_to_sources(existing_products, sources)
    autograder_input["productSourceMap"] = product_source_map

    # Step 3: Build criterion lookup from original criteria
    criteria_by_id = {}
    for c in criteria:
        cid = str(c.get("criterion_id", c.get("id", "")))
        criteria_by_id[cid] = c

    # Step 4: Re-grade only failed hurdle criteria
    for idx in failed_hurdle_indices:
        old_result = detailed[idx]
        cid = str(old_result.get("criterion_id", ""))
        criterion = criteria_by_id.get(cid)
        if criterion is None:
            _log(
                f"  Warning: criterion {cid} not found in criteria list, using old result"
            )
            criterion = old_result

        new_cr = grade_criterion(
            criterion,
            response_text,
            existing_products,
            sources,
            product_source_map,
            domain=domain,
            shop_vs_product=autograder_input.get("shop_vs_product", "Product"),
        )

        new_dict = asdict(new_cr)
        # Only mark as rescored if enrichment had no transient failures —
        # timeouts/429s leave unmarked so next run retries
        new_dict["rescored"] = not had_transient_failures
        detailed[idx] = new_dict

    # Step 5: Recalculate task-level scores
    scores_with_types, total_score, total_hurdle_score, summary = (
        _recalculate_task_scores(detailed)
    )

    existing_results["criteria_scores"] = scores_with_types
    existing_results["total_score"] = total_score
    existing_results["total_hurdle_score"] = total_hurdle_score
    existing_results["summary"] = summary
    existing_results["detailed_results"] = detailed

    # Step 6: Save updated files
    with open(sources_path, "w") as f:
        json.dump(autograder_input, f, indent=2, ensure_ascii=False)
    with open(results_path, "w") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)

    trace["scoring"] = {
        "total_score": total_score,
        "total_hurdle_score": total_hurdle_score,
        "num_criteria": existing_results.get("num_criteria", len(criteria)),
        "products": existing_results.get("products", existing_products),
        "summary": summary,
        "per_criterion": [
            {
                "criterion_id": r.get("criterion_id", ""),
                "description": r.get("description", ""),
                "type": r.get("type", ""),
                "score": r.get("score", 0),
                "hurdle_tag": r.get("hurdle_tag", "Not"),
                "stage_reached": r.get("stage_reached", ""),
                "reasoning": r.get("reasoning", ""),
                "rescored": r.get("rescored", False),
            }
            for r in detailed
        ],
    }
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    new_score = total_score
    new_hurdle = total_hurdle_score
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
        "rescored_criteria": len(failed_hurdle_indices),
    }


def update_summaries(results_dir: Path, dry_run: bool = False) -> list[Path]:
    """Find and update all summary.json files under results_dir.

    Re-reads each task's 3_autograder_results.json to pick up rescored values,
    then recomputes all aggregates (domain_stats, score_pct, criteria_type_scores)
    via build_eval_summary().

    Returns list of updated summary paths.
    """
    from ace_eval import build_eval_summary

    summary_paths = sorted(results_dir.rglob("summary.json"))
    updated = []

    for summary_path in summary_paths:
        with open(summary_path) as f:
            summary = json.load(f)

        model_dir = summary_path.parent
        results_list = summary.get("results", [])
        any_changed = False

        for r in results_list:
            if r.get("status") != "ok":
                continue

            task_id = r["task_id"]
            domain = r.get("domain", "unknown")
            run_id = r.get("run_id")

            # Find the autograder results file on disk
            task_dir = model_dir / domain / f"task_{task_id}"
            if run_id is not None:
                ag_path = task_dir / f"run_{run_id}" / "3_autograder_results.json"
            else:
                ag_path = task_dir / "3_autograder_results.json"

            if not ag_path.exists():
                continue

            with open(ag_path) as f:
                ag = json.load(f)

            new_score = ag.get("total_score", r.get("total_score", 0))
            new_hurdle = ag.get("total_hurdle_score", r.get("total_hurdle_score", 0))
            new_cs = ag.get("criteria_scores", r.get("criteria_scores", []))

            if new_score != r.get("total_score") or new_hurdle != r.get(
                "total_hurdle_score"
            ):
                any_changed = True

            r["total_score"] = new_score
            r["total_hurdle_score"] = new_hurdle
            r["criteria_scores"] = new_cs

        if not any_changed:
            _log(f"  summary {summary_path}: no score changes")
            continue

        if dry_run:
            _log(f"  summary {summary_path}: would update (dry run)")
            continue

        # Recompute all aggregates
        model = summary.get("model", "unknown")
        total_time = summary.get("total_time_seconds", 0)
        new_summary = build_eval_summary(model, results_list, total_time)

        with open(summary_path, "w") as f:
            json.dump(new_summary, f, indent=2, ensure_ascii=False)

        _log(f"  summary {summary_path}: updated")
        updated.append(summary_path)

    return updated


def find_run_dirs(results_dir: Path) -> list[Path]:
    """Find all run directories (containing trace.json) under results_dir."""
    run_dirs = []
    for trace_path in sorted(results_dir.rglob("trace.json")):
        run_dirs.append(trace_path.parent)
    return run_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Re-score failed hurdle criteria in eval results"
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
        "--force",
        action="store_true",
        help="Re-score even criteria already marked as rescored",
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
            result = rescore_run(run_dir, dry_run=args.dry_run, force=args.force)
            results.append(result)
            if result["status"] == "rescored" and result["changed"]:
                _log(
                    f"  {run_dir}: "
                    f"{result['old_score']} -> {result['new_score']} "
                    f"(hurdle {result['old_hurdle']} -> {result['new_hurdle']})"
                )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for run_dir in run_dirs:
                fut = executor.submit(
                    rescore_run,
                    run_dir,
                    dry_run=args.dry_run,
                    force=args.force,
                )
                futures[fut] = run_dir

            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                results.append(result)
                run_dir = futures[fut]
                if result["status"] == "rescored" and result["changed"]:
                    _log(
                        f"  {run_dir}: "
                        f"{result['old_score']} -> {result['new_score']} "
                        f"(hurdle {result['old_hurdle']} -> {result['new_hurdle']})"
                    )

    elapsed = time.time() - start

    retried = [r for r in results if r["status"] == "rescored"]
    changed = [r for r in retried if r.get("changed")]
    skipped = [r for r in results if r["status"] == "skip"]
    _log(f"\n{'=' * 60}")
    _log(
        f"Total: {len(results)}, Retried: {len(retried)} "
        f"({len(changed)} changed), Skipped: {len(skipped)}"
    )
    _log(f"Time: {elapsed:.1f}s")

    # Update summary.json files
    if retried:
        _log("\nUpdating summary.json files...")
        update_summaries(results_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
