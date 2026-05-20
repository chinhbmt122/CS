from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PIR validation ceilings.")
    parser.add_argument("--transactions", default="transaction_full_2025.parquet")
    parser.add_argument("--events", default="event_full_2025.parquet")
    parser.add_argument("--cutoff", default="2025-12-01")
    parser.add_argument("--valid-end", default="2026-01-01")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default="pir_ceiling_analysis.json")
    return parser.parse_args()


def metric_summary(lf: pl.LazyFrame, prefix: str) -> dict[str, float | int]:
    row = lf.select(
        pl.len().alias(f"{prefix}_users"),
        pl.col("target_n").sum().alias(f"{prefix}_target_items"),
        pl.col("target_n").mean().alias(f"{prefix}_avg_target_items"),
        pl.col("target_n").median().alias(f"{prefix}_median_target_items"),
        pl.col("target_n").quantile(0.75).alias(f"{prefix}_p75_target_items"),
        pl.col("target_n").quantile(0.90).alias(f"{prefix}_p90_target_items"),
        pl.col("target_n").quantile(0.99).alias(f"{prefix}_p99_target_items"),
        (pl.col("target_n") >= 2).sum().alias(f"{prefix}_users_with_2plus_items"),
        (pl.col("target_n") >= 3).sum().alias(f"{prefix}_users_with_3plus_items"),
    ).collect().to_dicts()[0]
    return row


def overlap_summary(joined: pl.LazyFrame, overlap_col: str, k: int, prefix: str) -> dict[str, float | int]:
    row = joined.select(
        pl.len().alias(f"{prefix}_users"),
        pl.col("target_n").sum().alias(f"{prefix}_target_items"),
        pl.col(overlap_col).sum().alias(f"{prefix}_recoverable_hits"),
        (pl.col(overlap_col) > 0).sum().alias(f"{prefix}_users_with_any_recoverable"),
        (pl.min_horizontal(pl.col("target_n"), pl.lit(k))).sum().alias(f"{prefix}_perfect_hits_at_k"),
    ).collect().to_dicts()[0]
    users = row[f"{prefix}_users"]
    target_items = row[f"{prefix}_target_items"]
    recoverable_hits = row[f"{prefix}_recoverable_hits"]
    perfect_hits = row[f"{prefix}_perfect_hits_at_k"]
    row[f"{prefix}_recoverable_precision_at_{k}"] = recoverable_hits / (users * k) if users else 0.0
    row[f"{prefix}_recoverable_recall"] = recoverable_hits / target_items if target_items else 0.0
    row[f"{prefix}_perfect_precision_at_{k}"] = perfect_hits / (users * k) if users else 0.0
    row[f"{prefix}_user_recoverable_rate"] = row[f"{prefix}_users_with_any_recoverable"] / users if users else 0.0
    return row


def main() -> None:
    args = parse_args()
    cutoff = datetime.fromisoformat(args.cutoff)
    valid_end = datetime.fromisoformat(args.valid_end)

    trx = pl.scan_parquet(args.transactions)
    evt = pl.scan_parquet(args.events)

    target = (
        trx.filter((pl.col("updated_date") >= cutoff) & (pl.col("updated_date") < valid_end))
        .group_by("customer_id")
        .agg(pl.col("item_id").cast(pl.Utf8).unique().alias("target_items"))
        .with_columns(pl.col("target_items").list.len().alias("target_n"))
    )
    prior_purchase = (
        trx.filter(pl.col("updated_date") < cutoff)
        .group_by("customer_id")
        .agg(pl.col("item_id").cast(pl.Utf8).unique().alias("prior_purchase_items"))
    )
    prior_event = (
        evt.filter((pl.col("event_date") >= datetime(2025, 1, 1)) & (pl.col("event_date") < cutoff))
        .group_by("customer_id")
        .agg(pl.col("item_id").cast(pl.Utf8).unique().alias("prior_event_items"))
    )
    prior_any = (
        pl.concat(
            [
                trx.filter(pl.col("updated_date") < cutoff).select("customer_id", pl.col("item_id").cast(pl.Utf8)),
                evt.filter((pl.col("event_date") >= datetime(2025, 1, 1)) & (pl.col("event_date") < cutoff)).select(
                    "customer_id", pl.col("item_id").cast(pl.Utf8)
                ),
            ],
            how="vertical",
        )
        .group_by("customer_id")
        .agg(pl.col("item_id").unique().alias("prior_any_items"))
    )

    joined = (
        target.join(prior_purchase, on="customer_id", how="left")
        .join(prior_event, on="customer_id", how="left")
        .join(prior_any, on="customer_id", how="left")
        .with_columns(
            pl.col("prior_purchase_items").fill_null([]),
            pl.col("prior_event_items").fill_null([]),
            pl.col("prior_any_items").fill_null([]),
        )
        .with_columns(
            pl.col("target_items").list.set_intersection("prior_purchase_items").list.len().alias("repeat_purchase_hits"),
            pl.col("target_items").list.set_intersection("prior_event_items").list.len().alias("prior_event_hits"),
            pl.col("target_items").list.set_intersection("prior_any_items").list.len().alias("prior_any_hits"),
            (pl.col("prior_any_items").list.len() > 0).alias("seen_before_cutoff"),
        )
    )

    seen = joined.filter(pl.col("seen_before_cutoff"))
    cold = joined.filter(~pl.col("seen_before_cutoff"))

    summary: dict[str, float | int | str] = {
        "cutoff": args.cutoff,
        "valid_end": args.valid_end,
        "k": args.k,
    }
    summary.update(metric_summary(joined, "all"))
    summary.update(metric_summary(seen, "seen"))
    summary.update(metric_summary(cold, "cold"))
    summary.update(overlap_summary(joined, "repeat_purchase_hits", args.k, "all_repeat_purchase"))
    summary.update(overlap_summary(seen, "repeat_purchase_hits", args.k, "seen_repeat_purchase"))
    summary.update(overlap_summary(joined, "prior_event_hits", args.k, "all_prior_event"))
    summary.update(overlap_summary(seen, "prior_event_hits", args.k, "seen_prior_event"))
    summary.update(overlap_summary(joined, "prior_any_hits", args.k, "all_prior_any"))
    summary.update(overlap_summary(seen, "prior_any_hits", args.k, "seen_prior_any"))

    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
