from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import polars as pl


ACTION_WEIGHTS = {
    "purchase": 8.0,
    "add_to_cart": 3.0,
    "view_item": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Personalized Item Recommendation pipeline")
    parser.add_argument("--transactions", default="transaction_full_2025.parquet")
    parser.add_argument("--events", default="event_full_2025.parquet")
    parser.add_argument("--items", default="items .parquet")
    parser.add_argument("--mode", choices=["validate", "predict"], default="validate")
    parser.add_argument("--cutoff", default="2025-12-01", help="Train on rows before this date.")
    parser.add_argument("--valid-end", default="2026-01-01", help="Validation end date, exclusive.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--half-life-days", type=float, default=60.0)
    parser.add_argument("--max-user-items", type=int, default=80)
    parser.add_argument("--repeat-weight", type=float, default=1.0)
    parser.add_argument("--copurchase-weight", type=float, default=0.35)
    parser.add_argument("--category-weight", type=float, default=0.0)
    parser.add_argument("--location-weight", type=float, default=0.0)
    parser.add_argument("--global-weight", type=float, default=0.05)
    parser.add_argument("--max-basket-items", type=int, default=12)
    parser.add_argument("--max-copurchase-neighbors", type=int, default=40)
    parser.add_argument("--max-copurchase-seeds", type=int, default=20)
    parser.add_argument(
        "--category-level",
        default="category_l2",
        help="Metadata level(s) for candidate expansion, comma-separated. Allowed: category_l1, category_l2, category_l3, brand.",
    )
    parser.add_argument("--max-category-items", type=int, default=30)
    parser.add_argument("--max-location-items", type=int, default=40)
    parser.add_argument("--copurchase-cache", default="pir_copurchase_cache.pkl")
    parser.add_argument("--target-customers", default=None, help="Optional CSV/parquet with customer_id.")
    parser.add_argument("--output", default="pir_recommendations.json")
    parser.add_argument("--metrics-output", default="pir_validation_metrics.json")
    parser.add_argument(
        "--include-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use view_item and add_to_cart as auxiliary train signals.",
    )
    parser.add_argument(
        "--event-start",
        default="2025-01-01",
        help="Ignore event rows before this date to avoid old backfill noise.",
    )
    return parser.parse_args()


def recency_expr(date_col: str, anchor: datetime, half_life_days: float) -> pl.Expr:
    age_days = (pl.lit(anchor) - pl.col(date_col)).dt.total_days().cast(pl.Float64)
    return (-(age_days.clip(lower_bound=0.0)) / half_life_days).exp()


def purchase_signals(path: str, start: datetime | None, end: datetime, anchor: datetime, half_life_days: float) -> pl.LazyFrame:
    lf = pl.scan_parquet(path)
    filters = [pl.col("updated_date") < pl.lit(end)]
    if start is not None:
        filters.append(pl.col("updated_date") >= pl.lit(start))
    return (
        lf.filter(pl.all_horizontal(filters))
        .select(
            pl.col("customer_id").cast(pl.Int64),
            pl.col("item_id").cast(pl.Utf8),
            pl.lit("purchase").alias("signal"),
            pl.col("updated_date").alias("signal_date"),
            (
                pl.lit(ACTION_WEIGHTS["purchase"])
                * (1.0 + pl.col("quantity").clip(lower_bound=1).log())
                * recency_expr("updated_date", anchor, half_life_days)
            ).alias("score"),
        )
    )


def event_signals(path: str, start: datetime, end: datetime, anchor: datetime, half_life_days: float) -> pl.LazyFrame:
    lf = pl.scan_parquet(path)
    return (
        lf.filter(
            (pl.col("event_date") >= pl.lit(start))
            & (pl.col("event_date") < pl.lit(end))
            & (pl.col("event_type").is_in(["view_item", "add_to_cart"]))
        )
        .select(
            pl.col("customer_id").cast(pl.Int64),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("event_type").alias("signal"),
            pl.col("event_date").alias("signal_date"),
            (
                pl.when(pl.col("event_type") == "add_to_cart")
                .then(pl.lit(ACTION_WEIGHTS["add_to_cart"]))
                .otherwise(pl.lit(ACTION_WEIGHTS["view_item"]))
                * (1.0 + pl.col("quantity").clip(lower_bound=1).log())
                * recency_expr("event_date", anchor, half_life_days)
            ).alias("score"),
        )
    )


def build_train_signals(args: argparse.Namespace, cutoff: datetime) -> pl.LazyFrame:
    frames = [purchase_signals(args.transactions, None, cutoff, cutoff, args.half_life_days)]
    if args.include_events:
        frames.append(event_signals(args.events, datetime.fromisoformat(args.event_start), cutoff, cutoff, args.half_life_days))
    return pl.concat(frames, how="vertical")


def build_global_items(signals: pl.LazyFrame, k: int) -> list[str]:
    # Sale status 0 appears in the item metadata. Treat other values as less desirable, not invalid.
    return (
        signals.group_by("item_id")
        .agg(pl.col("score").sum().alias("score"), pl.len().alias("n"))
        .sort(["score", "n"], descending=[True, True])
        .head(max(k * 20, 200))
        .collect()
        .get_column("item_id")
        .to_list()
    )


def build_global_scores(signals: pl.LazyFrame, limit: int) -> dict[str, float]:
    scored = (
        signals.group_by("item_id")
        .agg(pl.col("score").sum().alias("score"), pl.len().alias("n"))
        .sort(["score", "n"], descending=[True, True])
        .head(limit)
        .collect()
    )
    max_score = scored.get_column("score").max()
    if not max_score:
        return {}
    return {row["item_id"]: float(row["score"]) / float(max_score) for row in scored.iter_rows(named=True)}


def build_user_item_scores(signals: pl.LazyFrame, max_user_items: int) -> dict[int, list[tuple[str, float]]]:
    scored = (
        signals.group_by(["customer_id", "item_id"])
        .agg(pl.col("score").sum().alias("score"), pl.col("signal_date").max().alias("last_date"))
        .sort(["customer_id", "score", "last_date"], descending=[False, True, True])
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").head(max_user_items), pl.col("score").head(max_user_items))
        .collect(engine="streaming")
    )
    out: dict[int, list[tuple[str, float]]] = {}
    for row in scored.iter_rows(named=True):
        scores = [float(v) for v in row["score"]]
        max_score = max(scores) if scores else 1.0
        out[int(row["customer_id"])] = [
            (item, score / max_score) for item, score in zip(row["item_id"], scores)
        ]
    return out


def build_copurchase_neighbors(args: argparse.Namespace, cutoff: datetime) -> dict[str, list[tuple[str, float]]]:
    cache_key = {
        "transactions": str(Path(args.transactions).resolve()),
        "cutoff": cutoff.isoformat(),
        "max_basket_items": args.max_basket_items,
        "max_copurchase_neighbors": args.max_copurchase_neighbors,
    }
    cache_path = Path(args.copurchase_cache) if args.copurchase_cache else None
    if cache_path and cache_path.exists():
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == cache_key:
            return cached["neighbors"]

    baskets = (
        pl.scan_parquet(args.transactions)
        .filter(pl.col("updated_date") < pl.lit(cutoff))
        .group_by("bill_id")
        .agg(
            pl.col("item_id").cast(pl.Utf8).unique().alias("items"),
            pl.col("item_id").cast(pl.Utf8).n_unique().alias("n_items"),
        )
        .filter((pl.col("n_items") > 1) & (pl.col("n_items") <= args.max_basket_items))
        .select("items")
        .collect(engine="streaming")
    )

    pair_scores: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (items,) in baskets.iter_rows():
        unique_items = sorted(items)
        basket_weight = 1.0 / (len(unique_items) - 1)
        for i, item_a in enumerate(unique_items):
            for item_b in unique_items[i + 1 :]:
                pair_scores[item_a][item_b] += basket_weight
                pair_scores[item_b][item_a] += basket_weight

    neighbors: dict[str, list[tuple[str, float]]] = {}
    for item, related in pair_scores.items():
        top = sorted(related.items(), key=lambda kv: kv[1], reverse=True)[: args.max_copurchase_neighbors]
        if not top:
            continue
        max_score = top[0][1]
        neighbors[item] = [(other, score / max_score) for other, score in top]
    if cache_path:
        with cache_path.open("wb") as f:
            pickle.dump({"key": cache_key, "neighbors": neighbors}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return neighbors


def parse_category_levels(category_level: str) -> list[str]:
    allowed = {"category_l1", "category_l2", "category_l3", "brand"}
    levels = [level.strip() for level in category_level.split(",") if level.strip()]
    unknown = [level for level in levels if level not in allowed]
    if unknown:
        raise ValueError(f"Unknown category level(s): {unknown}. Allowed: {sorted(allowed)}")
    return levels or ["category_l2"]


def build_item_groups(args: argparse.Namespace) -> dict[str, list[str]]:
    levels = parse_category_levels(args.category_level)
    df = (
        pl.scan_parquet(args.items)
        .select(pl.col("item_id").cast(pl.Utf8), *[pl.col(level).cast(pl.Utf8).fill_null("unknown") for level in levels])
        .collect()
    )
    out: dict[str, list[str]] = {}
    for row in df.iter_rows(named=True):
        out[row["item_id"]] = [f"{level}:{row[level]}" for level in levels]
    return out


def build_group_top_items(
    args: argparse.Namespace,
    signals: pl.LazyFrame,
    item_groups: dict[str, list[str]],
) -> dict[str, list[tuple[str, float]]]:
    pairs = [
        {"item_id": item_id, "group": group}
        for item_id, groups in item_groups.items()
        for group in groups
    ]
    groups = pl.DataFrame(
        pairs,
        schema={"item_id": pl.Utf8, "group": pl.Utf8},
    ).lazy()
    scored = (
        signals.group_by("item_id")
        .agg(pl.col("score").sum().alias("score"))
        .join(groups, on="item_id", how="inner")
        .sort(["group", "score"], descending=[False, True])
        .group_by("group", maintain_order=True)
        .agg(pl.col("item_id").head(args.max_category_items), pl.col("score").head(args.max_category_items))
        .collect(engine="streaming")
    )
    out: dict[str, list[tuple[str, float]]] = {}
    for row in scored.iter_rows(named=True):
        scores = [float(v) for v in row["score"]]
        max_score = max(scores) if scores else 1.0
        out[row["group"]] = [(item, score / max_score) for item, score in zip(row["item_id"], scores)]
    return out


def build_user_locations(args: argparse.Namespace, cutoff: datetime, target_users: list[int]) -> dict[int, int]:
    targets = pl.DataFrame({"customer_id": target_users}, schema={"customer_id": pl.Int64}).lazy()
    scored = (
        pl.scan_parquet(args.transactions)
        .filter(pl.col("updated_date") < pl.lit(cutoff))
        .select(pl.col("customer_id").cast(pl.Int64), "location", "updated_date")
        .join(targets, on="customer_id", how="inner")
        .group_by(["customer_id", "location"])
        .agg(pl.len().alias("rows"), pl.col("updated_date").max().alias("last_date"))
        .sort(["customer_id", "rows", "last_date"], descending=[False, True, True])
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("location").first())
        .collect(engine="streaming")
    )
    return {int(row["customer_id"]): int(row["location"]) for row in scored.iter_rows(named=True)}


def build_location_top_items(args: argparse.Namespace, cutoff: datetime) -> dict[int, list[tuple[str, float]]]:
    scored = (
        pl.scan_parquet(args.transactions)
        .filter(pl.col("updated_date") < pl.lit(cutoff))
        .select(
            pl.col("location").cast(pl.Int64),
            pl.col("item_id").cast(pl.Utf8),
            (
                pl.lit(ACTION_WEIGHTS["purchase"])
                * (1.0 + pl.col("quantity").clip(lower_bound=1).log())
                * recency_expr("updated_date", cutoff, args.half_life_days)
            ).alias("score"),
        )
        .group_by(["location", "item_id"])
        .agg(pl.col("score").sum().alias("score"))
        .sort(["location", "score"], descending=[False, True])
        .group_by("location", maintain_order=True)
        .agg(pl.col("item_id").head(args.max_location_items), pl.col("score").head(args.max_location_items))
        .collect(engine="streaming")
    )
    out: dict[int, list[tuple[str, float]]] = {}
    for row in scored.iter_rows(named=True):
        scores = [float(v) for v in row["score"]]
        max_score = max(scores) if scores else 1.0
        out[int(row["location"])] = [(item, score / max_score) for item, score in zip(row["item_id"], scores)]
    return out


def load_targets(args: argparse.Namespace, cutoff: datetime, valid_end: datetime) -> list[int]:
    if args.target_customers:
        path = Path(args.target_customers)
        if path.suffix.lower() == ".parquet":
            return pl.read_parquet(path).get_column("customer_id").cast(pl.Int64).unique().to_list()
        return pl.read_csv(path).get_column("customer_id").cast(pl.Int64).unique().to_list()

    if args.mode == "validate":
        return (
            pl.scan_parquet(args.transactions)
            .filter((pl.col("updated_date") >= pl.lit(cutoff)) & (pl.col("updated_date") < pl.lit(valid_end)))
            .select(pl.col("customer_id").cast(pl.Int64).unique())
            .collect()
            .get_column("customer_id")
            .to_list()
        )

    users = [pl.scan_parquet(args.transactions).select(pl.col("customer_id").cast(pl.Int64))]
    if args.include_events:
        users.append(pl.scan_parquet(args.events).select(pl.col("customer_id").cast(pl.Int64)))
    return pl.concat(users).select(pl.col("customer_id").unique()).collect().get_column("customer_id").to_list()


def recommend_for_user(
    user_id: int,
    user_item_scores: dict[int, list[tuple[str, float]]],
    copurchase_neighbors: dict[str, list[tuple[str, float]]],
    item_groups: dict[str, list[str]],
    group_top_items: dict[str, list[tuple[str, float]]],
    user_locations: dict[int, int],
    location_top_items: dict[int, list[tuple[str, float]]],
    global_scores: dict[str, float],
    global_items: list[str],
    args: argparse.Namespace,
) -> list[str]:
    candidates: defaultdict[str, float] = defaultdict(float)
    user_items = user_item_scores.get(user_id, [])

    for item, score in user_items:
        candidates[item] += args.repeat_weight * score

    for seed_item, seed_score in user_items[: args.max_copurchase_seeds]:
        for item, related_score in copurchase_neighbors.get(seed_item, []):
            candidates[item] += args.copurchase_weight * seed_score * related_score

    if args.category_weight:
        used_groups: set[str] = set()
        for seed_item, seed_score in user_items[: args.max_copurchase_seeds]:
            for group in item_groups.get(seed_item, []):
                if group in used_groups:
                    continue
                used_groups.add(group)
                for item, group_score in group_top_items.get(group, []):
                    candidates[item] += args.category_weight * seed_score * group_score

    if args.location_weight:
        location = user_locations.get(user_id)
        if location is not None:
            for item, location_score in location_top_items.get(location, []):
                candidates[item] += args.location_weight * location_score

    for item, score in global_scores.items():
        candidates[item] += args.global_weight * score

    ranked = sorted(candidates.items(), key=lambda kv: (kv[1], global_scores.get(kv[0], 0.0), kv[0]), reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for item, _ in ranked:
        if item not in seen:
            out.append(item)
            seen.add(item)
            if len(out) == args.k:
                return out
    for item in global_items:
        if item not in seen:
            out.append(item)
            seen.add(item)
            if len(out) == args.k:
                return out
    return out


def ground_truth(path: str, cutoff: datetime, valid_end: datetime) -> dict[int, set[str]]:
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
    reciprocal_rank_sum = 0.0
    avg_iou_sum = 0.0
    avg_precision_sum = 0.0
    users = 0

    for user_id, true_items in truth.items():
        pred = recommendations.get(user_id, [])[:k]
        pred_set = set(pred)
        hits = pred_set & true_items
        users += 1
        total_correct += len(hits)
        total_union += len(pred_set | true_items)
        avg_iou_sum += len(hits) / len(pred_set | true_items) if pred_set or true_items else 0.0

        first_hit_rr = 0.0
        precision_hits = 0
        ap = 0.0
        for rank, item in enumerate(pred, start=1):
            if item in true_items:
                precision_hits += 1
                if first_hit_rr == 0.0:
                    first_hit_rr = 1.0 / rank
                ap += precision_hits / rank
        reciprocal_rank_sum += first_hit_rr
        avg_precision_sum += ap / min(len(true_items), k) if true_items else 0.0

    return {
        "users_evaluated": users,
        "k": k,
        "total_correct": total_correct,
        "precision_at_10_micro": total_correct / (users * k) if users else 0.0,
        "mean_iou": avg_iou_sum / users if users else 0.0,
        "global_iou": total_correct / total_union if total_union else 0.0,
        "mean_reciprocal_rank_first_hit": reciprocal_rank_sum / users if users else 0.0,
        "map_at_k": avg_precision_sum / users if users else 0.0,
    }


def filter_truth(truth: dict[int, set[str]], users: set[int], include: bool) -> dict[int, set[str]]:
    if include:
        return {user_id: items for user_id, items in truth.items() if user_id in users}
    return {user_id: items for user_id, items in truth.items() if user_id not in users}


def write_json(path: str, payload: object) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_recommendations(path: str, recommendations: dict[str, list[str]]) -> None:
    output_path = Path(path)
    if output_path.suffix.lower() == ".pkl":
        with output_path.open("wb") as f:
            pickle.dump(recommendations, f, protocol=pickle.HIGHEST_PROTOCOL)
        return
    write_json(path, recommendations)


def main() -> None:
    args = parse_args()
    cutoff = datetime.fromisoformat(args.cutoff)
    valid_end = datetime.fromisoformat(args.valid_end)

    signals = build_train_signals(args, cutoff)
    global_items = build_global_items(signals, args.k)
    global_scores = build_global_scores(signals, max(args.k * 20, 200))
    user_item_scores = build_user_item_scores(signals, args.max_user_items)
    copurchase_neighbors = build_copurchase_neighbors(args, cutoff) if args.copurchase_weight else {}
    item_groups = build_item_groups(args) if args.category_weight else {}
    group_top_items = build_group_top_items(args, signals, item_groups) if args.category_weight else {}
    targets = load_targets(args, cutoff, valid_end)
    user_locations = build_user_locations(args, cutoff, targets) if args.location_weight else {}
    location_top_items = build_location_top_items(args, cutoff) if args.location_weight else {}

    recommendations = {
        str(user_id): recommend_for_user(
            user_id,
            user_item_scores,
            copurchase_neighbors,
            item_groups,
            group_top_items,
            user_locations,
            location_top_items,
            global_scores,
            global_items,
            args,
        )
        for user_id in targets
    }
    write_recommendations(args.output, recommendations)

    if args.mode == "validate":
        truth = ground_truth(args.transactions, cutoff, valid_end)
        int_recommendations = {int(k): v for k, v in recommendations.items()}
        seen_users = set(user_item_scores)
        metrics = evaluate(int_recommendations, truth, args.k)
        metrics.update(
            {
                "cutoff": args.cutoff,
                "valid_end": args.valid_end,
                "include_events": args.include_events,
                "half_life_days": args.half_life_days,
                "repeat_weight": args.repeat_weight,
                "copurchase_weight": args.copurchase_weight,
                "category_weight": args.category_weight,
                "category_level": args.category_level,
                "location_weight": args.location_weight,
                "global_weight": args.global_weight,
                "max_basket_items": args.max_basket_items,
                "max_copurchase_neighbors": args.max_copurchase_neighbors,
                "recommendation_rows": len(recommendations),
                "seen_user_metrics": evaluate(int_recommendations, filter_truth(truth, seen_users, True), args.k),
                "cold_user_metrics": evaluate(int_recommendations, filter_truth(truth, seen_users, False), args.k),
            }
        )
        write_json(args.metrics_output, metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print(f"Wrote {len(recommendations)} recommendation lists to {args.output}")


if __name__ == "__main__":
    main()
