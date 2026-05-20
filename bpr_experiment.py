from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight BPR candidate experiment for PIR.")
    parser.add_argument("--transactions", default="transaction_full_2025.parquet")
    parser.add_argument("--base-recommendations", default="pir_validation_recommendations_category_brand_l3_006.json")
    parser.add_argument("--cutoff", default="2025-12-01")
    parser.add_argument("--valid-end", default="2026-01-01")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--top-items", type=int, default=8000)
    parser.add_argument("--factors", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--batches-per-epoch", type=int, default=700)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--reg", type=float, default=0.0005)
    parser.add_argument("--bpr-candidates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="bpr_experiment_metrics.json")
    return parser.parse_args()


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


def build_training_data(args: argparse.Namespace, cutoff: datetime, target_users: list[int]):
    target_df = pl.DataFrame({"customer_id": target_users}, schema={"customer_id": pl.Int64}).lazy()
    top_items = (
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
    top_item_df = pl.DataFrame({"item_id": top_items}, schema={"item_id": pl.Utf8}).lazy()
    pairs = (
        pl.scan_parquet(args.transactions)
        .filter(pl.col("updated_date") < pl.lit(cutoff))
        .select(pl.col("customer_id").cast(pl.Int64), pl.col("item_id").cast(pl.Utf8))
        .join(target_df, on="customer_id", how="inner")
        .join(top_item_df, on="item_id", how="inner")
        .unique()
        .collect(engine="streaming")
    )
    users = pairs.get_column("customer_id").unique().sort().to_list()
    items = top_items
    user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(items)}
    pos_lists: list[list[int]] = [[] for _ in users]
    for row in pairs.iter_rows(named=True):
        pos_lists[user_to_idx[int(row["customer_id"])]].append(item_to_idx[row["item_id"]])
    valid_user_idx = np.array([idx for idx, items_ in enumerate(pos_lists) if items_], dtype=np.int32)
    pos_arrays = [np.array(items_, dtype=np.int32) for items_ in pos_lists]
    return users, items, user_to_idx, pos_arrays, valid_user_idx


def train_bpr(args: argparse.Namespace, pos_arrays: list[np.ndarray], valid_user_idx: np.ndarray, n_items: int):
    rng = np.random.default_rng(args.seed)
    user_factors = rng.normal(0, 0.05, size=(len(pos_arrays), args.factors)).astype(np.float32)
    item_factors = rng.normal(0, 0.05, size=(n_items, args.factors)).astype(np.float32)

    for epoch in range(args.epochs):
        for _ in range(args.batches_per_epoch):
            users = rng.choice(valid_user_idx, size=args.batch_size, replace=True)
            pos = np.fromiter(
                (pos_arrays[u][rng.integers(len(pos_arrays[u]))] for u in users),
                dtype=np.int32,
                count=args.batch_size,
            )
            neg = rng.integers(0, n_items, size=args.batch_size, dtype=np.int32)

            u_vec = user_factors[users]
            p_vec = item_factors[pos]
            n_vec = item_factors[neg]
            diff = np.sum(u_vec * (p_vec - n_vec), axis=1)
            coeff = 1.0 / (1.0 + np.exp(diff))
            coeff = coeff.astype(np.float32)

            u_update = coeff[:, None] * (p_vec - n_vec) - args.reg * u_vec
            p_update = coeff[:, None] * u_vec - args.reg * p_vec
            n_update = -coeff[:, None] * u_vec - args.reg * n_vec

            np.add.at(user_factors, users, args.lr * u_update)
            np.add.at(item_factors, pos, args.lr * p_update)
            np.add.at(item_factors, neg, args.lr * n_update)
        print(f"epoch {epoch + 1}/{args.epochs} complete", flush=True)
    return user_factors, item_factors


def top_bpr_recommendations(
    target_users: list[int],
    user_to_idx: dict[int, int],
    users: list[int],
    items: list[str],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    n_candidates: int,
) -> dict[int, list[str]]:
    item_t = item_factors.T
    out: dict[int, list[str]] = {}
    batch_size = 2048
    known_targets = [user_id for user_id in target_users if user_id in user_to_idx]
    for start in range(0, len(known_targets), batch_size):
        batch_users = known_targets[start : start + batch_size]
        idx = np.array([user_to_idx[user_id] for user_id in batch_users], dtype=np.int32)
        scores = user_factors[idx] @ item_t
        top_idx = np.argpartition(-scores, kth=min(n_candidates, scores.shape[1] - 1), axis=1)[:, :n_candidates]
        row_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-row_scores, axis=1)
        sorted_idx = np.take_along_axis(top_idx, order, axis=1)
        for user_id, item_idx_row in zip(batch_users, sorted_idx):
            out[user_id] = [items[int(i)] for i in item_idx_row[:n_candidates]]
    return out


def blend(base: dict[int, list[str]], bpr: dict[int, list[str]], truth: dict[int, set[str]], k: int) -> dict[str, dict]:
    weights = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
    results: dict[str, dict] = {}
    target_users = list(truth)
    for weight in weights:
        recs: dict[int, list[str]] = {}
        for user_id in target_users:
            scores: dict[str, float] = {}
            for rank, item in enumerate(base.get(user_id, []), start=1):
                scores[item] = scores.get(item, 0.0) + 1.0 / rank
            for rank, item in enumerate(bpr.get(user_id, []), start=1):
                scores[item] = scores.get(item, 0.0) + weight / rank
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            recs[user_id] = [item for item, _ in ranked[:k]]
        results[f"base_plus_bpr_{weight:g}"] = evaluate(recs, truth, k)
    results["bpr_only"] = evaluate({user_id: items[:k] for user_id, items in bpr.items()}, truth, k)
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

    users, items, user_to_idx, pos_arrays, valid_user_idx = build_training_data(args, cutoff, target_users)
    print(
        f"training users={len(users)} items={len(items)} valid_users={len(valid_user_idx)} positives={sum(len(p) for p in pos_arrays)}",
        flush=True,
    )
    user_factors, item_factors = train_bpr(args, pos_arrays, valid_user_idx, len(items))
    bpr = top_bpr_recommendations(
        target_users,
        user_to_idx,
        users,
        items,
        user_factors,
        item_factors,
        args.bpr_candidates,
    )
    results = blend(base, bpr, truth, args.k)
    payload = {
        "cutoff": args.cutoff,
        "valid_end": args.valid_end,
        "top_items": args.top_items,
        "factors": args.factors,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "batches_per_epoch": args.batches_per_epoch,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
