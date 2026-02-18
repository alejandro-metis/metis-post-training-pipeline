"""
Convert Mercor ACE dataset CSVs to verl-compatible parquet format.

Each ACE CSV has one row per criterion. verl expects one row per task (prompt).
This script groups criteria by Task ID and bundles them into the ground_truth field.

Usage:
    python ace_to_verl.py --ace_dir /path/to/apex-evals/ace/dataset --output_dir /path/to/output
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import datasets
import pyarrow as pa
import pyarrow.parquet as pq


DOMAIN_FILES = {
    "diy": "ACE-DIY.csv",
    "food": "ACE-Food.csv",
    "gaming": "ACE-Gaming.csv",
    "shopping": "ACE-Shopping.csv",
}

HF_DOMAIN_NAMES = {
    "diy": "DIY",
    "food": "Food",
    "gaming": "Gaming",
    "shopping": "Shopping",
}


def load_ace_rows(domain, ace_dir=None, from_hf=False):
    """Load ACE data as list of dicts, either from CSV or HuggingFace."""
    if from_hf:
        hf_name = HF_DOMAIN_NAMES[domain]
        ds = datasets.load_dataset("mercor/ACE", hf_name, split="train")
        return [row for row in ds]
    else:
        filepath = os.path.join(ace_dir, DOMAIN_FILES[domain])
        with open(filepath, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


def group_by_task(rows):
    """Group criterion rows by Task ID."""
    tasks = defaultdict(lambda: {"prompt": None, "criteria": []})

    for row in rows:
        task_id = str(row["Task ID"])

        if tasks[task_id]["prompt"] is None:
            tasks[task_id]["prompt"] = row.get("Specified Prompt") or row["Prompt"]

        tasks[task_id]["criteria"].append(
            {
                "criterion_id": str(row["Criterion ID"]),
                "description": row["Description"],
                "hurdle_tag": row["Hurdle Tag"],
                "criteria_type": row["Criteria type"],
                "grounding_check": row["Criterion Grounding Check"],
            }
        )

    return tasks


def convert_domain(domain, ace_dir=None, from_hf=False, data_source="mercor/ACE"):
    """Convert a single domain into verl-format records."""
    rows = load_ace_rows(domain, ace_dir=ace_dir, from_hf=from_hf)
    tasks = group_by_task(rows)
    records = []

    for task_id, task_data in tasks.items():
        prompt_text = task_data["prompt"]
        criteria = task_data["criteria"]

        record = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": f"ace_{domain}",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(criteria),
            },
            "extra_info": {
                "domain": domain,
                "task_id": task_id,
                "num_criteria": len(criteria),
                "num_hurdles": sum(1 for c in criteria if c["hurdle_tag"] == "Hurdle"),
                "num_grounded": sum(1 for c in criteria if c["grounding_check"] == "Grounded"),
            },
        }
        records.append(record)

    return records


def records_to_parquet(records, output_path):
    """Write records to parquet using pyarrow directly (no HF datasets metadata)."""
    # Build Arrow arrays for each column
    data_sources = [r["data_source"] for r in records]
    abilities = [r["ability"] for r in records]

    # prompt: list<struct<role: string, content: string>>
    prompts = [
        [{"role": m["role"], "content": m["content"]} for m in r["prompt"]]
        for r in records
    ]

    # reward_model: struct<style: string, ground_truth: string>
    reward_styles = [r["reward_model"]["style"] for r in records]
    reward_gts = [r["reward_model"]["ground_truth"] for r in records]

    # extra_info: struct<domain: string, task_id: string, num_criteria: int64, ...>
    ei_domain = [r["extra_info"]["domain"] for r in records]
    ei_task_id = [r["extra_info"]["task_id"] for r in records]
    ei_num_criteria = [r["extra_info"]["num_criteria"] for r in records]
    ei_num_hurdles = [r["extra_info"]["num_hurdles"] for r in records]
    ei_num_grounded = [r["extra_info"]["num_grounded"] for r in records]

    # Define Arrow schema
    prompt_type = pa.list_(
        pa.struct([("role", pa.string()), ("content", pa.string())])
    )
    reward_model_type = pa.struct(
        [("style", pa.string()), ("ground_truth", pa.string())]
    )
    extra_info_type = pa.struct(
        [
            ("domain", pa.string()),
            ("task_id", pa.string()),
            ("num_criteria", pa.int64()),
            ("num_hurdles", pa.int64()),
            ("num_grounded", pa.int64()),
        ]
    )

    table = pa.table(
        {
            "data_source": pa.array(data_sources, type=pa.string()),
            "prompt": pa.array(prompts, type=prompt_type),
            "ability": pa.array(abilities, type=pa.string()),
            "reward_model": pa.StructArray.from_arrays(
                [pa.array(reward_styles), pa.array(reward_gts)],
                names=["style", "ground_truth"],
            ),
            "extra_info": pa.StructArray.from_arrays(
                [
                    pa.array(ei_domain),
                    pa.array(ei_task_id),
                    pa.array(ei_num_criteria, type=pa.int64()),
                    pa.array(ei_num_hurdles, type=pa.int64()),
                    pa.array(ei_num_grounded, type=pa.int64()),
                ],
                names=["domain", "task_id", "num_criteria", "num_hurdles", "num_grounded"],
            ),
        }
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pq.write_table(table, output_path)
    print(f"  Wrote {len(records)} tasks to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert ACE CSVs to verl parquet format")
    parser.add_argument(
        "--ace_dir",
        default="apex-evals/ace/dataset",
        help="Path to ACE dataset directory containing CSVs",
    )
    parser.add_argument(
        "--output_dir",
        default="ace_verl_data",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DOMAIN_FILES.keys()),
        choices=list(DOMAIN_FILES.keys()),
        help="Which domains to convert (default: all)",
    )
    parser.add_argument(
        "--from_hf",
        action="store_true",
        help="Load from HuggingFace (mercor/ACE) instead of local CSVs",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.75,
        help="Train/test split ratio (default: 0.75 = 15 train / 5 test per domain)",
    )
    args = parser.parse_args()

    all_train = []
    all_test = []

    for domain in args.domains:
        if not args.from_hf:
            csv_path = os.path.join(args.ace_dir, DOMAIN_FILES[domain])
            if not os.path.exists(csv_path):
                print(f"  Skipping {domain}: {csv_path} not found")
                continue

        records = convert_domain(domain, ace_dir=args.ace_dir, from_hf=args.from_hf)
        print(f"{domain}: {len(records)} tasks, "
              f"{sum(r['extra_info']['num_criteria'] for r in records)} total criteria")

        # Split
        split_idx = int(len(records) * args.split_ratio)
        all_train.extend(records[:split_idx])
        all_test.extend(records[split_idx:])

    if all_train:
        records_to_parquet(all_train, os.path.join(args.output_dir, "train.parquet"))
    if all_test:
        records_to_parquet(all_test, os.path.join(args.output_dir, "test.parquet"))

    print(f"\nDone: {len(all_train)} train, {len(all_test)} test")


if __name__ == "__main__":
    main()