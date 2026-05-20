# Personalized Item Recommendation Notes

## What matters most

This is an implicit-feedback ranking problem, not a rating-prediction problem. The target event is `Purchase`; `view_item` and `add_to_cart` should help only if validation proves that they improve future purchases.

Use a time split that mimics the blind month:

1. Train on history before a cutoff.
2. Validate on the next month of purchases only.
3. Tune only against validation metrics.
4. Retrain on all available 2025 data before producing the January 2026 submission.

For the current files, a useful local validation is:

- Train: `2025-01-01` to `2025-11-30`.
- Validation: `2025-12-01` to `2025-12-31`, purchases only.
- Final model: all 2025 interactions, then score January target customers if their IDs are provided.

## High-value baseline

Start with simple models because they expose data leakage and metric mistakes quickly:

- Recency-weighted global purchase popularity for cold-start users.
- Per-user repeat/history recommendations for returning users.
- Add auxiliary signals with lower weights: purchase > add-to-cart > view.
- Keep repeated purchase items unless the business explicitly forbids it. The metric evaluates future purchased items, and repeat purchase is often strong in retail.

The implemented baseline uses:

- `Purchase = 8.0`
- `add_to_cart = 3.0`
- `view_item = 1.0`
- exponential time decay with configurable half-life
- blended user repeat, item-to-item co-purchase, and global fallback scores

Run validation:

```powershell
python pir_pipeline.py --mode validate --cutoff 2025-12-01 --valid-end 2026-01-01 --k 10 --half-life-days 60 --copurchase-weight 0.15 --category-weight 0.06 --category-level brand,category_l3 --global-weight 0.05
```

Run final prediction when target customers are known:

```powershell
python pir_pipeline.py --mode predict --cutoff 2026-01-01 --target-customers january_customers.csv --output pir_submission.json
```

The target customer file must contain a `customer_id` column.

## What to avoid

- Random train/test splits. They leak future behavior into training and overestimate performance.
- Optimizing only offline loss. The required outputs are ranked lists, so use ranking metrics.
- Treating views, carts, and purchases as equally strong. Views are abundant but weak.
- Dropping cold-start users. The task explicitly includes January 2026 new users with purchases.
- Assuming January target users are all present in 2025. New accounts require a fallback list.
- Filtering all previously purchased items by default. That may destroy repeat-purchase signal.
- Tuning on the blind month after seeing labels. Use December 2025 as the decision gate.

## Improve after the baseline

Add improvements one at a time and keep an ablation table:

- Purchase-only versus purchase plus event signals.
- Half-life tuning: 30, 60, 90, 180 days.
- Item-to-item co-purchase candidates from baskets (`bill_id`).
- Category-aware fallback using item metadata.
- Matrix factorization or LightGCN only after the baseline is trusted.

Stop improving when a change fails to beat the current validation metrics, increases runtime too much for small gain, or improves one metric by damaging the metric the business cares about most.

## Current December 2025 validation

Evaluation target: December 2025 purchases from customers who purchased in December.

| Variant | Correct@10 | Precision@10 | Mean IoU | MRR first hit | MAP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global only, half-life 90 | 163,309 | 0.018924 | 0.012993 | 0.084701 | 0.027005 |
| Purchase only, half-life 60 | 691,138 | 0.080089 | 0.059766 | 0.343806 | 0.181966 |
| Purchase + events, half-life 60 | 698,748 | 0.080971 | 0.060528 | 0.346092 | 0.183709 |
| Purchase + events, half-life 75 | 700,916 | 0.081222 | 0.060693 | 0.344939 | 0.183198 |
| Purchase + events, half-life 90 | 701,337 | 0.081271 | 0.060723 | 0.343359 | 0.182375 |
| Blended repeat + global, half-life 90 | 704,713 | 0.081662 | 0.061009 | 0.344631 | 0.183082 |
| Blended repeat + co-purchase + global, half-life 60 | 720,447 | 0.083486 | 0.062609 | 0.345275 | 0.183997 |
| + category_l3, category weight 0.10 | 724,035 | 0.083902 | 0.063013 | 0.346915 | 0.185119 |
| + brand, category weight 0.06 | 724,554 | 0.083962 | 0.063041 | 0.346148 | 0.184793 |
| + brand + category_l3, category weight 0.06 | 725,940 | 0.084122 | 0.063198 | 0.347068 | 0.185355 |
| + brand + category_l3, category weight 0.10 | 725,317 | 0.084050 | 0.063152 | 0.347620 | 0.185518 |

The current best setting for total correct, precision@10, and IoU is `half-life-days=60`, `copurchase-weight=0.15`, `category-weight=0.06`, `category-level=brand,category_l3`, `global-weight=0.05`. If first-hit rank and MAP matter more, use `category-weight=0.10`.

Segment check for half-life 90:

| Segment | Users | Correct@10 | Precision@10 | MAP@10 |
| --- | ---: | ---: | ---: | ---: |
| Seen before cutoff | 660,769 | 681,422 | 0.103126 | 0.229420 |
| Cold in validation | 202,189 | 19,915 | 0.009850 | 0.028627 |

The main weakness is cold-start recommendation. The next best improvement is not a bigger model first; it is better cold-start and candidate generation: item metadata popularity, category-level popularity, recent local popularity if `location` matters, and item-to-item co-purchase expansion.

After adding item-to-item co-purchase, seen-user precision@10 improved from about `0.1031` to `0.1060`, but cold-user precision@10 stayed around `0.00985`. This confirms that co-purchase is useful for users with history, while true cold-start needs external context such as January events, location, signup metadata, or a better business-specific fallback.

## Rolling Validation

Rolling validation was run for September through December 2025:

```powershell
python run_pir_experiments.py --splits 2025-09,2025-10,2025-11,2025-12 --configs baseline_hl60,copurchase_hl60_w015,brand_l3_hl60_cw015_mw006,brand_l3_hl60_cw015_mw010
```

The merged output is `pir_experiments/summary_rolling.csv`.

Average across the four validation months:

| Config | Avg correct@10 | Avg precision@10 | Avg mean IoU | Avg MRR | Avg MAP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_hl60 | 687,586.5 | 0.085648 | 0.063949 | 0.364743 | 0.196011 |
| copurchase_hl60_w015 | 710,470.0 | 0.088491 | 0.066258 | 0.365065 | 0.196829 |
| brand_l3_hl60_cw015_mw006 | 715,789.5 | 0.089155 | 0.066881 | 0.366665 | 0.198104 |
| brand_l3_hl60_cw015_mw010 | 715,173.8 | 0.089080 | 0.066842 | 0.367001 | 0.198157 |

`brand_l3_hl60_cw015_mw006` is the robust production choice if total correct, precision@10, and IoU matter most. `brand_l3_hl60_cw015_mw010` is the rank-sensitive alternative if MRR/MAP matter more.

Location-aware popularity was tested on the December split using the robust config plus dominant-user-location item popularity:

| Location weight | Correct@10 | Precision@10 | Mean IoU | MRR | MAP@10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 725,935 | 0.084122 | 0.063197 | 0.347068 | 0.185357 |
| 0.01 | 725,581 | 0.084081 | 0.063162 | 0.346919 | 0.185186 |
| 0.03 | 724,793 | 0.083989 | 0.063082 | 0.346540 | 0.184805 |
| 0.06 | 723,314 | 0.083818 | 0.062934 | 0.345963 | 0.184209 |

Conclusion: do not include location in the current production blend. Historical store popularity displaces stronger repeat/co-purchase/category candidates and does not help true cold-start because prediction-time location is unavailable.

## Latent-Factor Experiments

A lightweight BPR-style latent factor candidate model was tested as the last model-based enhancement attempt:

```powershell
python bpr_experiment.py --cutoff 2025-12-01 --valid-end 2026-01-01 --base-recommendations pir_validation_recommendations_category_brand_l3_006.json --top-items 8000 --factors 64 --epochs 4 --batches-per-epoch 700 --batch-size 8192 --bpr-candidates 50 --output bpr_experiment_dec2025_stronger.json
```

December result:

| Model | Correct@10 | Precision@10 | Mean IoU | MRR | MAP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current best base | 725,940 | 0.084122 | 0.063198 | 0.347068 | 0.185355 |
| BPR only | 232,694 | 0.026965 | 0.019845 | 0.119341 | 0.057239 |
| Base + BPR weight 0.02 | 725,940 | 0.084122 | 0.063198 | 0.347066 | 0.185341 |
| Base + BPR weight 0.20 | 720,595 | 0.083503 | 0.062745 | 0.345229 | 0.182626 |

Conclusion: do not include BPR in the production blend. It does not add useful top-10 candidates beyond the current repeat/co-purchase/metadata model, and meaningful BPR weight degrades every metric.

ALS was also tested with the standard `implicit` package. Purchase-only ALS is much stronger than BPR, but still weaker than the current base model on its own. It can provide a small rank-sensitive reranking benefit.

December ALS result:

| Model | Correct@10 | Precision@10 | Mean IoU | MRR | MAP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current best base | 725,940 | 0.084122 | 0.063198 | 0.347068 | 0.185355 |
| ALS only, purchase-only | 600,457 | 0.069581 | 0.052182 | 0.245813 | 0.129205 |
| Base + ALS weight 0.10 | 725,940 | 0.084122 | 0.063198 | 0.347223 | 0.185443 |
| Base top 9 + ALS fill | 726,904 | 0.084234 | 0.063381 | 0.347333 | 0.185469 |

Rolling spot check:

| Split | Base correct@10 | Base top 9 + ALS correct@10 | Decision |
| --- | ---: | ---: | --- |
| 2025-10 | 704,138 | 703,942 | hurts total correct |
| 2025-11 | 783,171 | 783,412 | small gain |
| 2025-12 | 725,940 | 726,904 | small gain |

Conclusion: ALS is useful as an optional rank-sensitive reranker at low weight (`0.10`) because it improves MRR/MAP slightly while preserving total correct in tested months. It is not robust enough to replace or aggressively blend into the primary production config for total correct / precision@10.

With the current data only, stop enhancement here. The robust production config remains:

```powershell
python pir_pipeline.py --mode predict --cutoff 2026-01-01 --k 10 --half-life-days 60 --copurchase-weight 0.15 --category-weight 0.06 --category-level brand,category_l3 --location-weight 0 --global-weight 0.05 --output pir_submission_all_2025_users.json
```

## December Ceiling Analysis

Run:

```powershell
python analyze_pir_ceiling.py --cutoff 2025-12-01 --valid-end 2026-01-01 --output pir_ceiling_analysis_dec2025.json
```

December target size:

| Segment | Users | Target items | Avg items/user | Median | P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| All buyers | 862,958 | 3,432,162 | 3.98 | 2 | 9 |
| Seen before Dec | 660,769 | 2,914,732 | 4.41 | 3 | 10 |
| Cold before Dec | 202,189 | 517,430 | 2.56 | 1 | 5 |

Recoverable-history ceiling:

| Signal source | Recoverable hits | Recoverable precision@10 | Recoverable recall |
| --- | ---: | ---: | ---: |
| Prior purchases only, all users | 927,585 | 0.107489 | 0.270263 |
| Prior purchase/event, all users | 971,917 | 0.112626 | 0.283179 |
| Prior purchase/event, seen users only | 971,917 | 0.147089 | 0.333450 |

Current best December precision@10 is `0.084122` overall and `0.106849` for seen users. That is about 75% of the all-user prior-history recoverable precision ceiling and 73% of the seen-user prior-history ceiling. To move far beyond this, the model needs to predict genuinely new items, not just rank known repeat/history items better.
