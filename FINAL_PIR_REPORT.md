# Personalized Item Recommendation Final Report

## 1. Problem Understanding

The task is Personalized Item Recommendation (PIR). For each `customer_id`, the system must return a ranked list of items:

```python
{
  customer_id: [item_1, item_2, ..., item_k]
}
```

The training data is from 2025. The hidden test period is January 2026, and the official evaluation only considers the `purchased` event. The task also explicitly includes cold-start users: customers who are newly created or first appear during the test month.

Main metrics:

- total number of correct recommendations
- IoU between recommended and purchased item sets
- `1 / rank(first hit item)`
- `precision@10`
- MAP

Because the target is future purchases, this is an implicit-feedback ranking problem, not a rating-prediction problem.

## 2. Data Understanding

Available files:

| File | Role |
| --- | --- |
| `transaction_full_2025.parquet` | purchase transactions |
| `event_full_2025.parquet` | `view_item` and `add_to_cart` events |
| `items .parquet` | item metadata |

Transaction data:

- Rows: `41,470,317`
- Customers: `3,020,253`
- Items: `20,393`
- Bills/baskets: `19,456,172`
- Date range: `2025-01-01` to `2025-12-31`
- Has `location`, but only for historical purchases

Event data:

- Rows: `30,322,216`
- Customers: `729,097`
- Items: `19,833`
- Event types: `view_item`, `add_to_cart`
- No `location` column

Item metadata:

- Rows/items: `29,823`
- Useful fields: `category_l1`, `category_l2`, `category_l3`, `brand`, `price`, `sale_status`

Two extra files with `(1)` in their names were later added, but they were exact duplicates of the original transaction/event files. They did not add new information.

## 3. Validation Strategy

Random train/test split was avoided because it leaks future behavior into training. Instead, temporal validation was used:

```text
train before cutoff -> validate on next month purchases
```

Primary December validation:

```text
Train: before 2025-12-01
Validation: 2025-12-01 to 2025-12-31 purchases
```

Rolling validation was also run:

```text
Train before Sep -> validate Sep
Train before Oct -> validate Oct
Train before Nov -> validate Nov
Train before Dec -> validate Dec
```

This was used to avoid choosing a setting that only worked by luck on one month.

## 4. Main Pipeline

The final pipeline is a hybrid recommender, not a single neural model. It combines several ranking signals.

### 4.1 Interaction Weighting

Different user actions have different strengths:

| Signal | Weight |
| --- | ---: |
| purchase | `8.0` |
| add_to_cart | `3.0` |
| view_item | `1.0` |

This reflects the idea that purchases are much stronger intent signals than views.

### 4.2 Recency Decay

Recent behavior is more important than old behavior. The final model uses exponential decay with:

```text
half-life = 60 days
```

### 4.3 Repeat/User History

For each customer-item pair, historical interaction scores are aggregated. This captures repeat purchases and repeated interest.

### 4.4 Item-to-Item Co-Purchase

Using `bill_id`, baskets were used to learn item-to-item relationships:

```text
if item A and item B appear in the same bill, they are co-purchased
```

This helped recommend items that are related to what the customer previously bought.

### 4.5 Brand and Category Expansion

Item metadata was used to expand recommendations:

- same brand
- same `category_l3`

This improved recommendations when exact repeat items were too narrow.

### 4.6 Global Popularity Fallback

For users with little or no history, the model falls back to globally popular items.

This is necessary for cold-start users, but it is naturally weaker than personalized recommendations.

## 5. Feature Engineering Tried

| Feature / Strategy | Result |
| --- | --- |
| Action weighting | kept |
| Recency decay | kept |
| Repeat purchase score | kept |
| Co-purchase from baskets | kept |
| Brand/category expansion | kept |
| Global fallback | kept |
| Historical location popularity | rejected |
| BPR latent factor model | rejected |
| ALS latent factor model | optional only, not primary |
| LightGBM feature-based ranker | promising improvement |

## 6. Experiments and Results

### 6.1 December Validation Progress

| Model | Correct@10 | Precision@10 | Mean IoU | MRR | MAP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global only | 163,309 | 0.018924 | 0.012993 | 0.084701 | 0.027005 |
| Purchase only | 691,138 | 0.080089 | 0.059766 | 0.343806 | 0.181966 |
| Purchase + events | 698,748 | 0.080971 | 0.060528 | 0.346092 | 0.183709 |
| + co-purchase | 720,447 | 0.083486 | 0.062609 | 0.345275 | 0.183997 |
| + brand/category | 725,940 | 0.084122 | 0.063198 | 0.347068 | 0.185355 |

The largest useful improvement came from co-purchase. Brand/category expansion gave a smaller but consistent gain.

### 6.2 Rolling Validation

Average across September, October, November, and December:

| Config | Avg correct@10 | Avg precision@10 | Avg mean IoU | Avg MRR | Avg MAP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_hl60 | 687,586.5 | 0.085648 | 0.063949 | 0.364743 | 0.196011 |
| + co-purchase | 710,470.0 | 0.088491 | 0.066258 | 0.365065 | 0.196829 |
| + brand/category, weight 0.06 | 715,789.5 | 0.089155 | 0.066881 | 0.366665 | 0.198104 |
| + brand/category, weight 0.10 | 715,173.8 | 0.089080 | 0.066842 | 0.367001 | 0.198157 |

Decision:

- Use `category_weight = 0.06` for total correct, precision@10, and IoU.
- Use `category_weight = 0.10` only if MRR/MAP are more important.

The final selected setting uses `category_weight = 0.06`.

## 7. What Did Not Work

### 7.1 Location

Historical purchase data has `location`, but event data does not. For old users, we can infer dominant historical location. For new users, we cannot know location offline unless the test system provides it.

Location-aware popularity was tested:

| Location weight | Correct@10 | Precision@10 | Mean IoU | MRR | MAP@10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 725,935 | 0.084122 | 0.063197 | 0.347068 | 0.185357 |
| 0.01 | 725,581 | 0.084081 | 0.063162 | 0.346919 | 0.185186 |
| 0.03 | 724,793 | 0.083989 | 0.063082 | 0.346540 | 0.184805 |
| 0.06 | 723,314 | 0.083818 | 0.062934 | 0.345963 | 0.184209 |

Conclusion: location was rejected because it hurt validation metrics.

### 7.2 BPR

A lightweight BPR latent-factor model was tested. It was much weaker than the hybrid pipeline.

| Model | Correct@10 | Precision@10 | MAP@10 |
| --- | ---: | ---: | ---: |
| Current best base | 725,940 | 0.084122 | 0.185355 |
| BPR only | 232,694 | 0.026965 | 0.057239 |
| Base + BPR weight 0.20 | 720,595 | 0.083503 | 0.182626 |

Conclusion: BPR was rejected.

### 7.3 ALS

ALS was stronger than BPR, but still weaker than the base model alone. It gave small ranking improvements, but not robust enough for the primary configuration.

| Model | Correct@10 | Precision@10 | MRR | MAP@10 |
| --- | ---: | ---: | ---: | ---: |
| Current best base | 725,940 | 0.084122 | 0.347068 | 0.185355 |
| ALS only | 600,457 | 0.069581 | 0.245813 | 0.129205 |
| Base + ALS weight 0.10 | 725,940 | 0.084122 | 0.347223 | 0.185443 |
| Base top 9 + ALS fill | 726,904 | 0.084234 | 0.347333 | 0.185469 |

Rolling spot check for `base top 9 + ALS fill`:

| Split | Base | Base + ALS | Result |
| --- | ---: | ---: | --- |
| 2025-10 | 704,138 | 703,942 | worse |
| 2025-11 | 783,171 | 783,412 | small gain |
| 2025-12 | 725,940 | 726,904 | small gain |

Conclusion: ALS is optional for rank-sensitive reranking, but not used in the primary final output.

## 8. Feature-Based Ranker Upgrade

After the first hybrid model was finished, we identified one important missing part: a supervised second-stage ranker.

The hybrid recommender manually combines signals with fixed weights. A feature-based ranker instead learns how to combine many user, item, and user-item features from labeled next-month purchases.

Implemented model:

```text
Candidate generation -> feature table -> LightGBM LambdaRank -> rerank top candidates
```

Feature families used:

| Feature family | Examples |
| --- | --- |
| User behavior | `user_txns`, `user_items`, `user_days_since_last`, `user_velocity`, `user_recency_x_freq` |
| Item popularity/trend | `item_txns`, `item_buyers`, `item_txns_7d`, `item_txns_30d`, `item_trend_ratio` |
| User-item affinity | `ui_txns`, `ui_qty`, `ui_days_since_last`, `purchase_share`, `ui_recency_x_freq` |
| Event behavior | `event_views`, `event_carts` |
| Price affinity | `user_avg_price`, `item_avg_price`, `price_ratio` |
| Candidate source flags | `source_user_hist`, `source_global`, `source_group`, `source_copurchase` |

The ranker was trained leakage-safely:

```text
features before month M -> labels from purchases in month M
```

### Sample Validation Results

December sample:

| Model | Users | Correct@10 | Precision@10 | Mean IoU | MRR | MAP@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid baseline | 120,000 | 100,943 | 0.084119 | 0.063178 | 0.346841 | 0.185436 |
| LightGBM ranker | 120,000 | 102,389 | 0.085324 | 0.063765 | 0.364569 | 0.194145 |

November rolling sample:

| Model | Users | Correct@10 | Precision@10 | Mean IoU | MRR | MAP@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid baseline | 80,000 | 76,918 | 0.096148 | 0.071267 | 0.391731 | 0.211477 |
| LightGBM ranker | 80,000 | 79,614 | 0.099518 | 0.073473 | 0.413045 | 0.223845 |

Top feature importances were consistently:

- `ui_recency_x_freq`
- `ui_days_since_last`
- `purchase_share`
- item popularity and trend features
- `source_copurchase`
- price ratio and group-source signals

Conclusion: the feature-based ranker is the best next enhancement. It improves both precision and ranking metrics on sampled rolling validation. The current final submission file was produced by the hybrid model, but the ranker is now the recommended next production layer if we have time to scale it to all users.

### Hyperparameter Optimization

We also ran bounded random-search HPO for the LightGBM ranker. This is more valuable than exhaustive tuning of the hybrid pipeline because the ranker has many interacting tree parameters and already showed clear validation signal.

Search space included:

- `num_boost_round`
- `learning_rate`
- `num_leaves`
- `min_data_in_leaf`
- `feature_fraction`
- `bagging_fraction`
- `lambda_l2`

December 40k-user sample:

| Model | Correct@10 | Precision@10 | Mean IoU | MRR | MAP@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid baseline | 33,473 | 0.083683 | 0.062851 | 0.345176 | 0.184890 |
| Default LightGBM ranker | 33,594 | 0.083985 | 0.062720 | 0.361880 | 0.191742 |
| Tuned LightGBM ranker | 33,914 | 0.084785 | 0.063348 | 0.363221 | 0.193012 |

Best sampled parameters:

```text
num_boost_round = 240
learning_rate = 0.07
num_leaves = 63
min_data_in_leaf = 80
feature_fraction = 0.7
bagging_fraction = 0.9
lambda_l2 = 3.0
```

Conclusion: automated HPO helps the feature ranker. It gives a small precision gain and a clearer MAP/MRR improvement. A larger HPO run on more users would be the next best optimization step if compute time is available.

## 9. Cold-Start Analysis

December 2025 validation:

| Segment | Users | Target items | Avg items/user | Median | P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| All buyers | 862,958 | 3,432,162 | 3.98 | 2 | 9 |
| Seen before Dec | 660,769 | 2,914,732 | 4.41 | 3 | 10 |
| Cold before Dec | 202,189 | 517,430 | 2.56 | 1 | 5 |

About `23.4%` of December buyers were cold-start users for our model.

Current best December precision:

| Segment | Precision@10 |
| --- | ---: |
| Overall | 0.084122 |
| Seen users | about 0.1068 |
| Cold users | about 0.00985 |

Cold-start is the main limitation. Without January target users, January events, customer metadata, signup information, or prediction-time location/session context, true cold-start users can only receive generic fallback recommendations.

## 10. Is the Current Precision Good Enough?

The final December precision@10 is about `0.0841`, meaning the model finds about `0.84` correct purchased items inside every 10 recommended items per purchasing customer.

This number should not be judged alone. Recommendation precision depends on:

- catalog size: around `20,393` purchased items
- number of items each user buys in the target month
- how predictable repeat purchases are
- cold-start share
- whether test-time behavior/context is available

December buyers purchased a median of only `2` distinct items, and about `23.4%` of buyers were cold-start users. These two facts make precision@10 difficult: many users have only a few possible correct answers, and many new users have no historical behavior.

Recoverable-history ceiling:

| Signal source | Recoverable precision@10 | Recoverable recall |
| --- | ---: | ---: |
| Prior purchases only | 0.107489 | 0.270263 |
| Prior purchases + prior events | 0.112626 | 0.283179 |
| Prior purchases + events, seen users only | 0.147089 | 0.333450 |

The final model gets about `0.0841` overall precision@10. This is around 75% of the all-user prior-history recoverable precision ceiling.

Conclusion: the current precision is reasonable for the available data and task constraints, especially because cold-start users are included in evaluation. It is not perfect, but further large improvement likely requires extra context, such as January behavior, customer profile, target-time location/session data, or a provided target customer list.

## 11. Final Output

Final file:

```text
pir_submission_all_2025_users.pkl
```

Verification:

| Check | Value |
| --- | ---: |
| Customers | 3,122,302 |
| Recommendations per customer | 10 |
| Empty lists | 0 |
| Short lists | 0 |
| Lists with duplicate items | 0 |
| PKL file size | about 518 MB |

The output covers all customers observed in 2025 transaction/event data. It cannot include January-only unseen customer IDs unless a target customer list is provided.

## 12. Final Configuration

```powershell
python pir_pipeline.py --mode predict --cutoff 2026-01-01 --k 10 --half-life-days 60 --copurchase-weight 0.15 --category-weight 0.06 --category-level brand,category_l3 --location-weight 0 --global-weight 0.05 --output pir_submission_all_2025_users.pkl
```

## 13. Final Decision

With the current data, enhancement should stop here.

The final pipeline is robust because:

- validation is time-based, not random
- improvements were tested by ablation
- the selected setting won rolling validation for the main metrics
- failed features/models were rejected instead of kept
- cold-start limitations were measured and explained

Future improvement requires new data, especially:

- January target customer IDs
- January `view_item` / `add_to_cart`
- customer profile or signup date
- prediction-time store/session location
- item availability for January
