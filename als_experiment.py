from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


ACTION_WEIGHTS = {
    "purchase": 8.0,
    "add_to_cart": 3.0,
    "view_item": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALS candidate experiment for PIR.")
    parser.add_argument("--transactions", default="transaction_full_2025.parquet")
    parser.add_argument("--events", default="event_full_2025.parquet")
    parser.add_argument("--base-recommendations", default="pir_validation_recommendations_category_brand_l3_006.json")
    parser.add_argument("--cutoff", default="2025-12-01")
    parser.add_argument("--valid-end", default="2026-01-01")
    parser.add_argument("--event-start", default="2025-01-01")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--top-items", type=int, default=10000)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--regularization", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--als-candidates", type=int, default=50)
    parser.add_argument("--include-events", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="als_experiment_metrics.json")
    return parser.parse_args()


def recency_expr(date_col: str, anchor: datetime, half_life_days: float = 60.0) -> pl.Expr:
    age_days = (pl.lit(anchor) - pl.col(date_col)).dt.total_days().cast(pl.Float64)
    return (-(age_days.clip(lower_bound=0.0)) / half_life_days).exp()


def load_truth(path: str, cutoff: datetime, valid_end: datetime) -> dict[int, set[str]]:
    df = (
        pl.scan_parquet(path)
        .filter((pl.col("updated_date") >= pl.lit(cutoff)) & (pl.col("updated_date") < pl.lit(valid_end)))
        .group_by("customer_id")
        .agg(pl.col("item_id").cast(pl.Utf8).unique())
        .collect(engine="streaming")
    )
    return {int(row["customer_id"]): set(row["item_id"]) for row in df.iter_rows(named=True)}


def evaluate(recommendations: dict[int, list[str]], truth: dict[int, set[str]], k: int) -> dict[str, float | int]:
    total_correct = 0
    total_union = 0
    rr_sum = 0.0
    iou_sum = 0.0
    ap_sum = 0.0
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


def top_items(args: argparse.Namespace, cutoff: datetime) -> list[str]:
    return (
        pl.scan_parquet(args.transactions)
        .filter(pl.col("updated_date") < pl.lit(cutoff))
        .group_by("item_id")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(args.top_items)
        .select(pl.col("item_id").cast(pl.Utf8))
        .collect()
        .get_column("item_id")
        .to_list()
    )


def build_matrix(args: argparse.Namespace, cutoff: datetime, target_users: list[int], item_ids: list[str]):
    target_df = pl.DataFrame({"customer_id": target_users}, schema={"customer_id": pl.Int64}).lazy()
    item_df = pl.DataFrame({"item_id": item_ids}, schema={"item_id": pl.Utf8}).lazy()

    frames = [
        pl.scan_parquet(args.transactions)
        .filter(pl.col("updated_date") < pl.lit(cutoff))
        .select(
            pl.col("customer_id").cast(pl.Int64),
            pl.col("item_id").cast(pl.Utf8),
            (
                pl.lit(ACTION_WEIGHTS["purchase"])
                * (1.0 + pl.col("quantity").clip(lower_bound=1).log())
                * recency_expr("updated_date", cutoff)
            ).alias("score"),
        )
    ]
    if args.include_events:
        frames.append(
            pl.scan_parquet(args.events)
            .filter(
                (pl.col("event_date") >= pl.lit(datetime.fromisoformat(args.event_start)))
                & (pl.col("event_date") < pl.lit(cutoff))
                & (pl.col("event_type").is_in(["view_item", "add_to_cart"]))
            )
            .select(
                pl.col("customer_id").cast(pl.Int64),
                pl.col("item_id").cast(pl.Utf8),
                (
                    pl.when(pl.col("event_type") == "add_to_cart")
                    .then(pl.lit(ACTION_WEIGHTS["add_to_cart"]))
                    .otherwise(pl.lit(ACTION_WEIGHTS["view_item"]))
                    * (1.0 + pl.col("quantity").clip(lower_bound=1).log())
                    * recency_expr("event_date", cutoff)
                ).alias("score"),
            )
        )

    interactions = (
        pl.concat(frames, how="vertical")
        .join(target_df, on="customer_id", how="inner")
        .join(item_df, on="item_id", how="inner")
        .group_by(["customer_id", "item_id"])
        .agg(pl.col("score").sum().alias("score"))
        .collect(engine="streaming")
    )

    users = interactions.get_column("customer_id").unique().sort().to_list()
    user_to_idx = {int(user_id): idx for idx, user_id in enumerate(users)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    rows = np.array([user_to_idx[int(v)] for v in interactions.get_column("customer_id")], dtype=np.int32)
    cols = np.array([item_to_idx[v] for v in interactions.get_column("item_id")], dtype=np.int32)
    data = np.array(interactions.get_column("score"), dtype=np.float32) * np.float32(args.alpha)
    matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(item_ids)), dtype=np.float32)
    return matrix, users


def top_als_recommendations(
    target_users: list[int],
    users: list[int],
    item_ids: list[str],
    model: AlternatingLeastSquares,
    n_candidates: int,
) -> dict[int, list[str]]:
    user_to_idx = {int(user_id): idx for idx, user_id in enumerate(users)}
    item_t = model.item_factors.T
    out: dict[int, list[str]] = {}
    known = [user_id for user_id in target_users if user_id in user_to_idx]
    batch_size = 2048
    for start in range(0, len(known), batch_size):
        batch_users = known[start : start + batch_size]
        idx = np.array([user_to_idx[user_id] for user_id in batch_users], dtype=np.int32)
        scores = model.user_factors[idx] @ item_t
        top_idx = np.argpartition(-scores, kth=min(n_candidates, scores.shape[1] - 1), axis=1)[:, :n_candidates]
        row_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-row_scores, axis=1)
        sorted_idx = np.take_along_axis(top_idx, order, axis=1)
        for user_id, item_idx_row in zip(batch_users, sorted_idx):
            out[user_id] = [item_ids[int(i)] for i in item_idx_row[:n_candidates]]
    return out


def blend(base: dict[int, list[str]], als: dict[int, list[str]], truth: dict[int, set[str]], k: int) -> dict[str, dict]:
    weights = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
    results: dict[str, dict] = {}
    target_users = list(truth)
    for weight in weights:
        recs: dict[int, list[str]] = {}
        for user_id in target_users:
            scores: dict[str, float] = {}
            for rank, item in enumerate(base.get(user_id, []), start=1):
                scores[item] = scores.get(item, 0.0) + 1.0 / rank
            for rank, item in enumerate(als.get(user_id, []), start=1):
                scores[item] = scores.get(item, 0.0) + weight / rank
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            recs[user_id] = [item for item, _ in ranked[:k]]
        results[f"base_plus_als_{weight:g}"] = evaluate(recs, truth, k)
    for keep_base in [3, 5, 7, 9]:
        recs = {}
        for user_id in target_users:
            out = []
            seen = set()
            for item in base.get(user_id, [])[:keep_base]:
                if item not in seen:
                    out.append(item)
                    seen.add(item)
            for item in als.get(user_id, []):
                if item not in seen:
                    out.append(item)
                    seen.add(item)
                if len(out) == k:
                    break
            for item in base.get(user_id, []):
                if item not in seen:
                    out.append(item)
                    seen.add(item)
                if len(out) == k:
                    break
            recs[user_id] = out[:k]
        results[f"base_top{keep_base}_then_als"] = evaluate(recs, truth, k)
    oracle_recs = {}
    for user_id in target_users:
        true_items = truth[user_id]
        union = []
        seen = set()
        for item in base.get(user_id, []) + als.get(user_id, []):
            if item not in seen:
                union.append(item)
                seen.add(item)
        hits = [item for item in union if item in true_items]
        misses = [item for item in union if item not in true_items]
        oracle_recs[user_id] = (hits + misses)[:k]
    results["oracle_base_als_union"] = evaluate(oracle_recs, truth, k)
    results["als_only"] = evaluate({user_id: items[:k] for user_id, items in als.items()}, truth, k)
    results["base_only"] = evaluate(base, truth, k)
    return results


def main() -> None:
    args = parse_args()
    cutoff = datetime.fromisoformat(args.cutoff)
    valid_end = datetime.fromisoformat(args.valid_end)
    truth = load_truth(args.transactions, cutoff, valid_end)
    target_users = list(truth)
    base_raw = json.loads(Path(args.base_recommendations).read_text(encoding="utf-8"))
    base = {int(user_id): items for user_id, items in base_raw.items()}

    item_ids = top_items(args, cutoff)
    matrix, users = build_matrix(args, cutoff, target_users, item_ids)
    print(f"training ALS users={matrix.shape[0]} items={matrix.shape[1]} nnz={matrix.nnz}", flush=True)
    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        random_state=42,
    )
    model.fit(matrix, show_progress=True)
    als = top_als_recommendations(target_users, users, item_ids, model, args.als_candidates)
    results = blend(base, als, truth, args.k)
    payload = {
        "cutoff": args.cutoff,
        "valid_end": args.valid_end,
        "top_items": args.top_items,
        "factors": args.factors,
        "iterations": args.iterations,
        "regularization": args.regularization,
        "alpha": args.alpha,
        "include_events": args.include_events,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
