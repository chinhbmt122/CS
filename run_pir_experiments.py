from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


SPLITS = [
    ("2025-09", "2025-09-01", "2025-10-01"),
    ("2025-10", "2025-10-01", "2025-11-01"),
    ("2025-11", "2025-11-01", "2025-12-01"),
    ("2025-12", "2025-12-01", "2026-01-01"),
]


CONFIGS = [
    {
        "name": "baseline_hl60",
        "half_life_days": 60,
        "copurchase_weight": 0,
        "category_weight": 0,
        "category_level": "category_l2",
        "global_weight": 0.05,
    },
    {
        "name": "copurchase_hl60_w015",
        "half_life_days": 60,
        "copurchase_weight": 0.15,
        "category_weight": 0,
        "category_level": "category_l2",
        "global_weight": 0.05,
    },
    {
        "name": "brand_l3_hl60_cw015_mw006",
        "half_life_days": 60,
        "copurchase_weight": 0.15,
        "category_weight": 0.06,
        "category_level": "brand,category_l3",
        "global_weight": 0.05,
    },
    {
        "name": "brand_l3_hl60_cw015_mw010",
        "half_life_days": 60,
        "copurchase_weight": 0.15,
        "category_weight": 0.10,
        "category_level": "brand,category_l3",
        "global_weight": 0.05,
    },
    {
        "name": "brand_l3_hl90_cw015_mw006",
        "half_life_days": 90,
        "copurchase_weight": 0.15,
        "category_weight": 0.06,
        "category_level": "brand,category_l3",
        "global_weight": 0.05,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PIR rolling validation experiments.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pipeline", default="pir_pipeline.py")
    parser.add_argument("--output-dir", default="pir_experiments")
    parser.add_argument("--summary", default="pir_experiments/summary.csv")
    parser.add_argument("--splits", default=",".join(split[0] for split in SPLITS))
    parser.add_argument("--configs", default=",".join(config["name"] for config in CONFIGS))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--keep-recommendations", action="store_true")
    return parser.parse_args()


def selected(items: list[tuple] | list[dict], names: set[str], name_getter) -> list:
    return [item for item in items if name_getter(item) in names]


def run_one(args: argparse.Namespace, split: tuple[str, str, str], config: dict, output_dir: Path) -> dict:
    split_name, cutoff, valid_end = split
    safe_name = f"{split_name}_{config['name']}"
    metrics_path = output_dir / f"{safe_name}_metrics.json"
    recommendations_path = output_dir / f"{safe_name}_recommendations.json"

    command = [
        args.python,
        args.pipeline,
        "--mode",
        "validate",
        "--cutoff",
        cutoff,
        "--valid-end",
        valid_end,
        "--k",
        str(args.k),
        "--half-life-days",
        str(config["half_life_days"]),
        "--copurchase-weight",
        str(config["copurchase_weight"]),
        "--category-weight",
        str(config["category_weight"]),
        "--category-level",
        str(config["category_level"]),
        "--global-weight",
        str(config["global_weight"]),
        "--copurchase-cache",
        str(output_dir / f"{split_name}_copurchase_cache.pkl"),
        "--metrics-output",
        str(metrics_path),
        "--output",
        str(recommendations_path),
    ]
    subprocess.run(command, check=True)
    if not args.keep_recommendations and recommendations_path.exists():
        recommendations_path.unlink()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "split": split_name,
        "config": config["name"],
        "cutoff": cutoff,
        "valid_end": valid_end,
        "total_correct": metrics["total_correct"],
        "precision_at_10_micro": metrics["precision_at_10_micro"],
        "mean_iou": metrics["mean_iou"],
        "global_iou": metrics["global_iou"],
        "mrr_first_hit": metrics["mean_reciprocal_rank_first_hit"],
        "map_at_k": metrics["map_at_k"],
        "seen_precision_at_10": metrics["seen_user_metrics"]["precision_at_10_micro"],
        "cold_precision_at_10": metrics["cold_user_metrics"]["precision_at_10_micro"],
    }


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "config",
        "cutoff",
        "valid_end",
        "total_correct",
        "precision_at_10_micro",
        "mean_iou",
        "global_iou",
        "mrr_first_hit",
        "map_at_k",
        "seen_precision_at_10",
        "cold_precision_at_10",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = set(args.splits.split(","))
    config_names = set(args.configs.split(","))
    splits = selected(SPLITS, split_names, lambda split: split[0])
    configs = selected(CONFIGS, config_names, lambda config: config["name"])

    rows: list[dict] = []
    for split in splits:
        for config in configs:
            print(f"RUN split={split[0]} config={config['name']}", flush=True)
            row = run_one(args, split, config, output_dir)
            rows.append(row)
            write_summary(Path(args.summary), rows)

    print(f"Wrote {len(rows)} rows to {args.summary}")


if __name__ == "__main__":
    main()
