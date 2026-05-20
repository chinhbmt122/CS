from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import polars as pl

from pir_pipeline import build_copurchase_neighbors


FEATURES = [
    "user_txns",
    "user_items",
    "user_avg_price",
    "user_days_since_last",
    "user_txns_30d",
    "user_txns_90d",
    "user_velocity",
    "user_recency_x_freq",
    "item_txns",
    "item_buyers",
    "item_avg_price",
    "item_txns_7d",
    "item_txns_30d",
    "item_txns_90d",
    "item_trend_ratio",
    "item_penetration",
    "item_log_buyers",
    "ui_txns",
    "ui_qty",
    "ui_days_since_last",
    "purchase_share",
    "price_ratio",
    "ui_recency_x_freq",
    "event_views",
    "event_carts",
    "source_user_hist",
    "source_global",
    "source_group",
    "source_copurchase",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature-based LightGBM ranker experiment.")
    parser.add_argument("--transactions", default="transaction_full_2025.parquet")
    parser.add_argument("--events", default="event_full_2025.parquet")
    parser.add_argument("--items", default="items .parquet")
    parser.add_argument("--train-cutoff", default="2025-11-01")
    parser.add_argument("--train-end", default="2025-12-01")
    parser.add_argument("--eval-cutoff", default="2025-12-01")
    parser.add_argument("--eval-end", default="2026-01-01")
    parser.add_argument("--train-users", type=int, default=120_000)
    parser.add_argument("--eval-users", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--user-candidates", type=int, default=25)
    parser.add_argument("--global-candidates", type=int, default=20)
    parser.add_argument("--group-candidates", type=int, default=8)
    parser.add_argument("--copurchase-seeds", type=int, default=4)
    parser.add_argument("--copurchase-neighbors", type=int, default=10)
    parser.add_argument("--num-boost-round", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=40)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-fraction", type=float, default=0.85)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--tune-trials", type=int, default=0)
    parser.add_argument("--output", default="feature_ranker_experiment.json")
    return parser.parse_args()


def sample_target_users(transactions: str, start: datetime, end: datetime, n_users: int, seed: int) -> list[int]:
    users = (
        pl.scan_parquet(transactions)
        .filter((pl.col("updated_date") >= pl.lit(start)) & (pl.col("updated_date") < pl.lit(end)))
        .select(pl.col("customer_id").cast(pl.Int64).unique())
        .collect()
        .get_column("customer_id")
        .to_list()
    )
    rng = np.random.default_rng(seed)
    if len(users) > n_users:
        users = rng.choice(np.array(users, dtype=np.int64), size=n_users, replace=False).tolist()
    return [int(user_id) for user_id in users]


def recency_expr(date_col: str, cutoff: datetime) -> pl.Expr:
    return (pl.lit(cutoff) - pl.col(date_col)).dt.total_days().cast(pl.Float64).clip(lower_bound=0.0)


def build_truth(transactions: str, start: datetime, end: datetime, users: list[int]) -> dict[int, set[str]]:
    user_df = pl.DataFrame({"customer_id": users}, schema={"customer_id": pl.Int64}).lazy()
    df = (
        pl.scan_parquet(transactions)
        .filter((pl.col("updated_date") >= pl.lit(start)) & (pl.col("updated_date") < pl.lit(end)))
        .select(pl.col("customer_id").cast(pl.Int64), pl.col("item_id").cast(pl.Utf8))
        .join(user_df, on="customer_id", how="inner")
        .group_by("customer_id")
        .agg(pl.col("item_id").unique().alias("items"))
        .collect(engine="streaming")
    )
    return {int(row["customer_id"]): set(row["items"]) for row in df.iter_rows(named=True)}


def evaluate(recommendations: dict[int, list[str]], truth: dict[int, set[str]], k: int) -> dict[str, float | int]:
    total_correct = 0
    total_union = 0
    rr_sum = 0.0
    ap_sum = 0.0
    iou_sum = 0.0
    users = 0
    for user_id, true_items in truth.items():
        pred = recommendations.get(user_id, [])[:k]
        pred_set = set(pred)
        hits = pred_set & true_items
        users += 1
        total_correct += len(hits)
        total_union += len(pred_set | true_items)
        iou_sum += len(hits) / len(pred_set | true_items) if pred_set or true_items else 0.0
        hit_count = 0
        rr = 0.0
        ap = 0.0
        for rank, item in enumerate(pred, start=1):
            if item in true_items:
                hit_count += 1
                if rr == 0.0:
                    rr = 1.0 / rank
                ap += hit_count / rank
        rr_sum += rr
        ap_sum += ap / min(len(true_items), k) if true_items else 0.0
    return {
        "users_evaluated": users,
        "k": k,
        "total_correct": total_correct,
        "precision_at_10_micro": total_correct / (users * k) if users else 0.0,
        "mean_iou": iou_sum / users if users else 0.0,
        "global_iou": total_correct / total_union if total_union else 0.0,
        "mean_reciprocal_rank_first_hit": rr_sum / users if users else 0.0,
        "map_at_k": ap_sum / users if users else 0.0,
    }


def add_source(candidates: pl.DataFrame, name: str) -> pl.DataFrame:
    return candidates.with_columns(pl.lit(1.0).alias(name))


def build_candidates(args: argparse.Namespace, cutoff: datetime, users: list[int]) -> pl.DataFrame:
    user_df = pl.DataFrame({"customer_id": users}, schema={"customer_id": pl.Int64}).lazy()

    prior_user_items = (
        pl.scan_parquet(args.transactions)
        .filter(pl.col("updated_date") < pl.lit(cutoff))
        .select(
            pl.col("customer_id").cast(pl.Int64),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("quantity").clip(lower_bound=1).sum().over(["customer_id", "item_id"]).alias("qty_tmp"),
            pl.col("updated_date"),
        )
        .join(user_df, on="customer_id", how="inner")
        .group_by(["customer_id", "item_id"])
        .agg(
            pl.len().alias("n"),
            pl.col("qty_tmp").max().alias("qty"),
            pl.col("updated_date").max().alias("last_date"),
        )
        .with_columns((pl.col("n") * 2.0 + pl.col("qty").log()).alias("score"))
        .sort(["customer_id", "score", "last_date"], descending=[False, True, True])
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").head(args.user_candidates))
        .explode("item_id")
        .collect(engine="streaming")
    )
    frames = [add_source(prior_user_items, "source_user_hist")]

    global_items = (
        pl.scan_parquet(args.transactions)
        .filter(pl.col("updated_date") < pl.lit(cutoff))
        .group_by("item_id")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(args.global_candidates)
        .select(pl.col("item_id").cast(pl.Utf8))
        .collect()
    )
    global_candidates = pl.DataFrame({"customer_id": users}, schema={"customer_id": pl.Int64}).join(global_items, how="cross")
    frames.append(add_source(global_candidates, "source_global"))

    item_meta = pl.scan_parquet(args.items).select(
        pl.col("item_id").cast(pl.Utf8),
        pl.col("brand").cast(pl.Utf8).fill_null("unknown").alias("brand"),
        pl.col("category_l3").cast(pl.Utf8).fill_null("unknown").alias("category_l3"),
    )
    item_pop = (
        pl.scan_parquet(args.transactions)
        .filter(pl.col("updated_date") < pl.lit(cutoff))
        .group_by("item_id")
        .agg(pl.len().alias("item_n"))
        .join(item_meta, on="item_id", how="inner")
    )
    group_top = []
    for group_col in ["brand", "category_l3"]:
        group_top.append(
            item_pop.sort([group_col, "item_n"], descending=[False, True])
            .group_by(group_col, maintain_order=True)
            .agg(pl.col("item_id").head(args.group_candidates))
            .explode("item_id")
            .rename({group_col: "group_value"})
            .with_columns(pl.lit(group_col).alias("group_type"))
            .select("group_type", "group_value", "item_id")
        )
    group_top_df = pl.concat(group_top, how="vertical")
    user_groups = []
    prior_items_with_meta = prior_user_items.lazy().join(item_meta, on="item_id", how="inner")
    for group_col in ["brand", "category_l3"]:
        user_groups.append(
            prior_items_with_meta.group_by(["customer_id", group_col])
            .agg(pl.len().alias("n"))
            .sort(["customer_id", "n"], descending=[False, True])
            .group_by("customer_id", maintain_order=True)
            .agg(pl.col(group_col).head(2))
            .explode(group_col)
            .rename({group_col: "group_value"})
            .with_columns(pl.lit(group_col).alias("group_type"))
            .select("customer_id", "group_type", "group_value")
        )
    group_candidates = (
        pl.concat(user_groups, how="vertical")
        .join(group_top_df.lazy(), on=["group_type", "group_value"], how="inner")
        .select("customer_id", "item_id")
        .collect(engine="streaming")
    )
    frames.append(add_source(group_candidates, "source_group"))

    copurchase_args = SimpleNamespace(
        transactions=args.transactions,
        max_basket_items=12,
        max_copurchase_neighbors=args.copurchase_neighbors,
        copurchase_cache=f"feature_ranker_copurchase_{cutoff.date()}.pkl",
    )
    neighbors = build_copurchase_neighbors(copurchase_args, cutoff)
    seeds = (
        prior_user_items.group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").head(args.copurchase_seeds))
        .iter_rows(named=True)
    )
    rows = []
    for row in seeds:
        user_id = int(row["customer_id"])
        for seed_item in row["item_id"]:
            for item_id, _ in neighbors.get(seed_item, []):
                rows.append((user_id, item_id))
    if rows:
        copurchase_candidates = pl.DataFrame(rows, schema=["customer_id", "item_id"], orient="row")
        frames.append(add_source(copurchase_candidates, "source_copurchase"))

    candidates = pl.concat(frames, how="diagonal").fill_null(0.0)
    source_cols = ["source_user_hist", "source_global", "source_group", "source_copurchase"]
    candidates = (
        candidates.with_columns(pl.col("customer_id").cast(pl.Int64), pl.col("item_id").cast(pl.Utf8))
        .group_by(["customer_id", "item_id"])
        .agg([pl.col(c).max().alias(c) for c in source_cols])
        .with_columns([pl.col(c).fill_null(0.0) for c in source_cols])
    )
    return candidates


def build_feature_frame(
    args: argparse.Namespace,
    cutoff: datetime,
    end: datetime,
    users: list[int],
    candidates: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[int, set[str]]]:
    user_df = pl.DataFrame({"customer_id": users}, schema={"customer_id": pl.Int64}).lazy()
    truth = build_truth(args.transactions, cutoff, end, users)
    label_df = pl.DataFrame(
        [(user_id, item_id, 1) for user_id, items in truth.items() for item_id in items],
        schema=["customer_id", "item_id", "label"],
        orient="row",
    ).lazy()
    prior = (
        pl.scan_parquet(args.transactions)
        .filter(pl.col("updated_date") < pl.lit(cutoff))
        .select(
            pl.col("customer_id").cast(pl.Int64),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("price").cast(pl.Float64),
            pl.col("quantity").cast(pl.Float64),
            pl.col("updated_date"),
        )
    )
    prior_users = prior.join(user_df, on="customer_id", how="inner")

    user_stats = (
        prior_users.group_by("customer_id")
        .agg(
            pl.len().alias("user_txns"),
            pl.col("item_id").n_unique().alias("user_items"),
            pl.col("price").mean().alias("user_avg_price"),
            recency_expr("updated_date", cutoff).min().alias("user_days_since_last"),
            (recency_expr("updated_date", cutoff) <= 30).sum().alias("user_txns_30d"),
            (recency_expr("updated_date", cutoff) <= 90).sum().alias("user_txns_90d"),
        )
        .with_columns(
            (pl.col("user_txns_30d") / (pl.col("user_txns_90d") + 1.0)).alias("user_velocity"),
            (pl.col("user_txns") / (pl.col("user_days_since_last") + 1.0)).alias("user_recency_x_freq"),
        )
    )
    item_stats = (
        prior.group_by("item_id")
        .agg(
            pl.len().alias("item_txns"),
            pl.col("customer_id").n_unique().alias("item_buyers"),
            pl.col("price").mean().alias("item_avg_price"),
            (recency_expr("updated_date", cutoff) <= 7).sum().alias("item_txns_7d"),
            (recency_expr("updated_date", cutoff) <= 30).sum().alias("item_txns_30d"),
            (recency_expr("updated_date", cutoff) <= 90).sum().alias("item_txns_90d"),
        )
        .with_columns(
            (pl.col("item_txns_30d") / (pl.col("item_txns_90d") + 1.0)).alias("item_trend_ratio"),
            (pl.col("item_buyers") / pl.lit(max(len(users), 1))).alias("item_penetration"),
            (pl.col("item_buyers") + 1.0).log().alias("item_log_buyers"),
        )
    )
    ui_stats = (
        prior_users.group_by(["customer_id", "item_id"])
        .agg(
            pl.len().alias("ui_txns"),
            pl.col("quantity").sum().alias("ui_qty"),
            recency_expr("updated_date", cutoff).min().alias("ui_days_since_last"),
        )
    )
    event_stats = (
        pl.scan_parquet(args.events)
        .filter((pl.col("event_date") >= pl.lit(datetime(2025, 1, 1))) & (pl.col("event_date") < pl.lit(cutoff)))
        .select(pl.col("customer_id").cast(pl.Int64), pl.col("item_id").cast(pl.Utf8), "event_type")
        .join(user_df, on="customer_id", how="inner")
        .group_by(["customer_id", "item_id"])
        .agg(
            (pl.col("event_type") == "view_item").sum().alias("event_views"),
            (pl.col("event_type") == "add_to_cart").sum().alias("event_carts"),
        )
    )
    item_price = pl.scan_parquet(args.items).select(pl.col("item_id").cast(pl.Utf8), pl.col("price").cast(pl.Float64).alias("meta_price"))

    df = (
        candidates.lazy()
        .join(label_df, on=["customer_id", "item_id"], how="left")
        .join(user_stats, on="customer_id", how="left")
        .join(item_stats, on="item_id", how="left")
        .join(ui_stats, on=["customer_id", "item_id"], how="left")
        .join(event_stats, on=["customer_id", "item_id"], how="left")
        .join(item_price, on="item_id", how="left")
        .with_columns(
            pl.col("label").fill_null(0).cast(pl.Int8),
            pl.col("user_txns").fill_null(0.0),
            pl.col("user_items").fill_null(0.0),
            pl.col("user_avg_price").fill_null(pl.col("meta_price")).fill_null(0.0),
            pl.col("user_days_since_last").fill_null(9999.0),
            pl.col("user_txns_30d").fill_null(0.0),
            pl.col("user_txns_90d").fill_null(0.0),
            pl.col("user_velocity").fill_null(0.0),
            pl.col("user_recency_x_freq").fill_null(0.0),
            pl.col("item_txns").fill_null(0.0),
            pl.col("item_buyers").fill_null(0.0),
            pl.col("item_avg_price").fill_null(pl.col("meta_price")).fill_null(0.0),
            pl.col("item_txns_7d").fill_null(0.0),
            pl.col("item_txns_30d").fill_null(0.0),
            pl.col("item_txns_90d").fill_null(0.0),
            pl.col("item_trend_ratio").fill_null(0.0),
            pl.col("item_penetration").fill_null(0.0),
            pl.col("item_log_buyers").fill_null(0.0),
            pl.col("ui_txns").fill_null(0.0),
            pl.col("ui_qty").fill_null(0.0),
            pl.col("ui_days_since_last").fill_null(9999.0),
            pl.col("event_views").fill_null(0.0),
            pl.col("event_carts").fill_null(0.0),
        )
        .with_columns(
            (pl.col("ui_txns") / (pl.col("user_txns") + 1.0)).alias("purchase_share"),
            (pl.col("item_avg_price") / (pl.col("user_avg_price") + 1.0)).alias("price_ratio"),
            (pl.col("ui_txns") / (pl.col("ui_days_since_last") + 1.0)).alias("ui_recency_x_freq"),
        )
        .select("customer_id", "item_id", "label", *FEATURES)
        .collect(engine="streaming")
    )
    return df, truth


def train_ranker(train_df: pl.DataFrame, args: argparse.Namespace, params_override: dict | None = None) -> lgb.Booster:
    train_df = train_df.sort("customer_id")
    groups = train_df.group_by("customer_id", maintain_order=True).len().get_column("len").to_list()
    x = train_df.select(FEATURES).to_numpy().astype(np.float32)
    y = train_df.get_column("label").to_numpy().astype(np.int32)
    dataset = lgb.Dataset(x, label=y, group=groups, feature_name=FEATURES)
    params = {
        "objective": "lambdarank",
        "metric": "map",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": 1,
        "feature_fraction": args.feature_fraction,
        "lambda_l2": args.lambda_l2,
        "seed": args.seed,
        "num_threads": -1,
        "verbosity": -1,
    }
    if params_override:
        params.update(params_override)
    return lgb.train(
        params,
        dataset,
        num_boost_round=int(params.pop("num_boost_round", args.num_boost_round)),
    )


def predict_ranker(model: lgb.Booster, df: pl.DataFrame, k: int) -> dict[int, list[str]]:
    scores = model.predict(df.select(FEATURES).to_numpy().astype(np.float32))
    ranked = (
        df.select("customer_id", "item_id")
        .with_columns(pl.Series("score", scores))
        .sort(["customer_id", "score"], descending=[False, True])
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").head(k))
    )
    return {int(row["customer_id"]): row["item_id"] for row in ranked.iter_rows(named=True)}


def candidate_recall(df: pl.DataFrame, truth: dict[int, set[str]], k: int) -> dict[str, float | int]:
    by_user = df.group_by("customer_id").agg(pl.col("item_id")).iter_rows(named=True)
    hits = 0
    perfect_hits = 0
    total_targets = sum(len(v) for v in truth.values())
    for row in by_user:
        user_id = int(row["customer_id"])
        cand = set(row["item_id"])
        true_items = truth.get(user_id, set())
        hits += len(cand & true_items)
        perfect_hits += min(len(true_items), k)
    return {
        "candidate_hits": hits,
        "target_items": total_targets,
        "candidate_recall": hits / total_targets if total_targets else 0.0,
        "oracle_precision_at_10": min(hits, perfect_hits) / (len(truth) * k) if truth else 0.0,
    }


def random_param_trials(args: argparse.Namespace) -> list[dict]:
    rng = np.random.default_rng(args.seed + 999)
    trials = [
        {
            "name": "default",
            "params": {
                "num_boost_round": args.num_boost_round,
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "min_data_in_leaf": args.min_data_in_leaf,
                "feature_fraction": args.feature_fraction,
                "bagging_fraction": args.bagging_fraction,
                "lambda_l2": args.lambda_l2,
            },
        }
    ]
    for i in range(args.tune_trials):
        trials.append(
            {
                "name": f"trial_{i + 1}",
                "params": {
                    "num_boost_round": int(rng.choice([120, 180, 240, 320, 420])),
                    "learning_rate": float(rng.choice([0.025, 0.035, 0.05, 0.07, 0.10])),
                    "num_leaves": int(rng.choice([31, 47, 63, 95, 127])),
                    "min_data_in_leaf": int(rng.choice([20, 40, 80, 120, 200])),
                    "feature_fraction": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
                    "bagging_fraction": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
                    "lambda_l2": float(rng.choice([0.0, 0.1, 0.5, 1.0, 3.0])),
                },
            }
        )
    return trials


def run_baseline(args: argparse.Namespace, users: list[int], cutoff: str, end: str, truth: dict[int, set[str]]) -> dict[str, float | int]:
    target_path = Path("feature_ranker_eval_users.csv")
    output_path = Path("feature_ranker_baseline_recs.json")
    target_path.write_text("customer_id\n" + "\n".join(map(str, users)), encoding="utf-8")
    command = [
        sys.executable,
        "pir_pipeline.py",
        "--mode",
        "validate",
        "--cutoff",
        cutoff,
        "--valid-end",
        end,
        "--target-customers",
        str(target_path),
        "--k",
        str(args.k),
        "--half-life-days",
        "60",
        "--copurchase-weight",
        "0.15",
        "--category-weight",
        "0.06",
        "--category-level",
        "brand,category_l3",
        "--location-weight",
        "0",
        "--global-weight",
        "0.05",
        "--output",
        str(output_path),
        "--metrics-output",
        "feature_ranker_baseline_metrics.json",
    ]
    subprocess.run(command, check=True)
    recs = {int(k): v for k, v in json.loads(output_path.read_text(encoding="utf-8")).items()}
    return evaluate(recs, truth, args.k)


def main() -> None:
    args = parse_args()
    train_cutoff = datetime.fromisoformat(args.train_cutoff)
    train_end = datetime.fromisoformat(args.train_end)
    eval_cutoff = datetime.fromisoformat(args.eval_cutoff)
    eval_end = datetime.fromisoformat(args.eval_end)

    train_users = sample_target_users(args.transactions, train_cutoff, train_end, args.train_users, args.seed)
    eval_users = sample_target_users(args.transactions, eval_cutoff, eval_end, args.eval_users, args.seed + 1)

    print(f"building train candidates for {len(train_users)} users", flush=True)
    train_candidates = build_candidates(args, train_cutoff, train_users)
    print(f"train candidates rows={train_candidates.height}", flush=True)
    train_df, train_truth = build_feature_frame(args, train_cutoff, train_end, train_users, train_candidates)
    train_positive_users = (
        train_df.group_by("customer_id").agg(pl.col("label").sum().alias("positives")).filter(pl.col("positives") > 0)
    )
    train_df = train_df.join(train_positive_users.select("customer_id"), on="customer_id", how="inner")
    print(f"train feature rows={train_df.height} positives={train_df['label'].sum()}", flush=True)

    print(f"building eval candidates for {len(eval_users)} users", flush=True)
    eval_candidates = build_candidates(args, eval_cutoff, eval_users)
    print(f"eval candidates rows={eval_candidates.height}", flush=True)
    eval_df, eval_truth = build_feature_frame(args, eval_cutoff, eval_end, eval_users, eval_candidates)
    print(f"eval feature rows={eval_df.height} positives={eval_df['label'].sum()}", flush=True)

    baseline_metrics = run_baseline(args, eval_users, args.eval_cutoff, args.eval_end, eval_truth)
    recall = candidate_recall(eval_df, eval_truth, args.k)

    trial_results = []
    best = None
    best_model = None
    for trial in random_param_trials(args):
        print(f"training ranker {trial['name']} params={trial['params']}", flush=True)
        model = train_ranker(train_df, args, trial["params"])
        recs = predict_ranker(model, eval_df, args.k)
        metrics = evaluate(recs, eval_truth, args.k)
        row = {"name": trial["name"], "params": trial["params"], "metrics": metrics}
        trial_results.append(row)
        if best is None or metrics["map_at_k"] > best["metrics"]["map_at_k"]:
            best = row
            best_model = model

    assert best is not None and best_model is not None
    importances = sorted(zip(FEATURES, best_model.feature_importance(importance_type="gain").tolist()), key=lambda kv: kv[1], reverse=True)

    payload = {
        "train_cutoff": args.train_cutoff,
        "train_end": args.train_end,
        "eval_cutoff": args.eval_cutoff,
        "eval_end": args.eval_end,
        "train_users": len(train_users),
        "eval_users": len(eval_users),
        "train_rows": train_df.height,
        "eval_rows": eval_df.height,
        "candidate_recall": recall,
        "baseline_metrics": baseline_metrics,
        "ranker_metrics": best["metrics"],
        "best_trial": best,
        "trial_results": trial_results,
        "feature_importances": importances,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
