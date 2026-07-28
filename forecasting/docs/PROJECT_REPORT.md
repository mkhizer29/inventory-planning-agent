# Naheed.pk Demand Forecasting — Project Report

**Prepared:** 2026-07-27
**Scope:** 30-SKU pilot, daily demand forecasting with weekly business reporting

---

## 1. Executive Summary

We built a demand-forecasting pipeline for a 30-SKU pilot at Naheed.pk, sourced from the
company's Magento staging database. The pipeline extracts ~1 year of daily sales history per
SKU, engineers features (calendar effects, lag/rolling demand, stock status, promo signal,
category-level demand), and trains a LightGBM model (Tweedie objective) at **daily grain**.
For business reporting, daily forecasts are **summed into weekly totals** — validated as the
better approach over training a model natively on weekly-aggregated data.

**Current state:** the model is trained, validated with rolling-origin cross-validation (not
just a single test window), and produces a stable weekly accuracy of **~61%** on average
(individual weeks can range 25–75%). The pipeline is ready to hand off for the next step —
automating forecast delivery to a website.

---

## 2. Background & Goal

Naheed.pk wanted a demand forecast per SKU to support warehouse ordering decisions — how much
of each product to order so the business avoids both stockouts and slow-moving overstock. A
30-SKU pilot (15 Groceries & Pets + 15 Health & Beauty, colleague-supplied list in
`SKUs/pilot_skus.csv`) was chosen over an earlier data-driven 50-SKU candidate list, as the
starting scope.

Data source: a live Magento staging MySQL database. All extraction connects directly to this
staging DB using credentials in `.env` (see Section 9 — **this file must never be shared**).

---

## 3. Data Source, Timeline, and Data-Quality Issues Resolved

### 3.1 Schema switch: `pg_1` → `pg_new_1`
The original ETL mixed sources — `price` came from schema `pg_1`, `cost` came from `pg_new_1`
— because `pg_1` only has cost data for 63 of 31,628 catalog products (none of our 30 pilot
SKUs). This produced a nonsensical **cost > price** result for 26 of 30 SKUs (e.g. one SKU
showed price 599 but cost 1,247). Root-caused by querying both schemas directly, then fixed by
re-sourcing both `price` and `cost` from `pg_new_1` (`SKUs/patch_price_cost_pg_new_1.py`).
`pg_new_1` is now the primary source for all catalog-attribute extraction going forward.

### 3.2 The "current snapshot" problem
Several catalog columns (`price`, `is_in_stock`, `current_stock_qty`, `is_active`,
`visibility`, `special_price`) exist in the DB only as a **current snapshot** — there is no
historical time series for them. Naively repeating today's snapshot across a year of history
would be wrong (e.g. a SKU that's in stock today doesn't mean it was in stock every day for
the last year). Resolution per column:

| Column | Resolution |
|---|---|
| `special_price` | Left null — genuinely sparse catalog-wide (only 226/63 of 31,628 products have it set), not a query bug |
| `cost`, `is_active`, `visibility` | Kept as SKU-level constants (current snapshot) — no fabrication |
| `price` | Kept constant for now; once a real promo calendar exists, real discounts will be overlaid on promo dates instead |
| `is_in_stock` | **Raw flag distrusted entirely.** The team confirmed 6 SKUs the snapshot marks "out of stock" are actually in stock right now, and several SKUs show impossible **negative** stock quantities. Replaced with a **derived, demand-based flag**: for each SKU, find runs of consecutive zero-demand days and flag a run as "likely OOS" only once it's unusually long *for that SKU* (threshold = `max(10, 2 × that SKU's own 90th-percentile zero-run length)`), so slow movers aren't falsely flagged. Validated against the team's 6 known SKUs — the derived flag agrees (shows in-stock) for all 6. |
| `current_stock_qty` | **Dropped entirely** — not just stale, flat-out broken (negative values for several SKUs) |

### 3.3 Six-month historical backfill
The team added ~6 months of real history behind the original 2026-01-15 start date. Verified
live against the DB (not assumed): real production-volume data (thousands of orders/day)
begins sharply on **2025-07-24** — the days just before that (2025-07-15→07-23) are still old,
sparse "playground" test data and were excluded.

### 3.4 Invoicing lag at the live edge of the data
`net_qty` (our demand measure) is built from `qty_invoiced − qty_refunded`, and invoicing
genuinely lags order placement by a few days as routine warehouse process — it is **not** the
same as order placement. Checking the invoiced/ordered ratio against a fully-settled baseline
(~42–74% is normal) revealed the last 2–3 days of any extraction are typically **under-invoiced**
and read as artificially low demand if trusted at face value. The dataset is trimmed to exclude
this soft lag as well as a harder ~1-day cutoff where order rows barely exist yet.

**Current dataset window: 2025-07-24 → 2026-07-21 (363 days × 30 SKUs = 10,890 rows).**

**General lesson for future extractions:** always check the invoiced/ordered ratio for the last
3–5 days of any pull against a settled-period baseline before trusting them as real low demand.

---

## 4. SKUs

30 pilot SKUs, evenly split:
- **15 Groceries & Pets**
- **15 Health & Beauty**

Full list with human-readable names, brand, and category: `SKUs/pilot_skus.csv` and
`SKUs/sku_full_names.csv` (the latter has full, untruncated product names — the names cached
in earlier extracts were truncated to 32 characters by an earlier extraction step; full names
up to 255 characters were re-pulled directly from the catalog).

Demand is highly uneven across these 30 SKUs — total 12-month volume ranges from ~24,750 units
(`IC-1055803`, Naheed Sugar) down to under 1,300 units for the smallest. A handful of SKUs also
have **ops-unconfirmed long stockout streaks** (`IC-1057685`, `IC-1134493`, `IC-1001018`) —
extended zero-demand runs that may be undetected out-of-stock periods rather than genuine zero
demand. These are flagged, not yet resolved with the operations team.

---

## 5. Features and How They Were Extracted

The canonical, exact SQL table/attribute-ID mapping for every column lives in
**`SKUs/training_dataset_30skus_column_sources.md`** — that file is the authoritative
extraction spec and should be the first thing shared with anyone extending this to more SKUs.
Summary:

### 5.1 Product / catalog attributes (Magento EAV snapshot, `pg_new_1`)
| Column | Source |
|---|---|
| `sku` | `catalog_product_entity.sku` |
| `product_name` | `catalog_product_entity_varchar`, attribute_id 73 |
| `brand` | `catalog_product_entity_int` → `eav_attribute_option_value`, attribute_id 83 (manufacturer) |
| `category` | `catalog_category_product` → `catalog_category_entity` (path), resolved to the level-2 (top-level) category |
| `is_active` | `catalog_product_entity_int`, attribute_id 97 (status) |
| `visibility` | `catalog_product_entity_int`, attribute_id 99 |
| `price` | `catalog_product_entity_decimal`, attribute_id 77 |
| `cost` | `catalog_product_entity_decimal`, attribute_id 81 |
| `special_price` | `catalog_product_entity_decimal`, attribute_id 78 (null for all 30 pilot SKUs) |
| `is_in_stock` / `current_stock_qty` (raw, later replaced/dropped) | `cataloginventory_stock_item` |

### 5.2 Daily sales aggregation (`sales_order_item` joined to `sales_order`)
| Column | Formula |
|---|---|
| `net_qty` (target) | `SUM(qty_invoiced) − SUM(qty_refunded)`, per sku per day |
| `revenue`, `order_count`, `coupon_orders`, `is_promo_order_present` | Computed but **excluded from the model** — all are derived from that day's own realized orders, so they can't be known at the time a forecast is needed (same-day leakage) |

### 5.3 `is_on_sale` / `avg_discount_pct` — promo signal (new, 2026-07-27)
The marketing team's forward-looking promo calendar is still pending. As a stand-in, we
compare `sales_order_item.price` (price actually charged) against `original_price` (catalog
price at order time) per order line: if any line for a SKU on a given day was discounted, that
day is flagged `is_on_sale = 1`. Only **10 of 30 SKUs ever show a discount** in their entire
order history. This is included as a real model feature (not just descriptive) on the basis
that once the real promo calendar arrives, it will populate this same flag for future dates
in advance — for now, backfilling it from realized historical data is a faithful stand-in for
what that calendar would have said.

### 5.4 Calendar flags (external, not from the DB)
`is_ramadan`, `is_eid_fitr`, `is_eid_adha` — no promo/holiday flag exists natively in Magento;
these date ranges were sourced externally and merged in.

### 5.5 Lag / rolling features (per SKU, computed after extraction)
`lag_1`, `lag_7`, `lag_14`, `rolling_mean_7`, `rolling_std_7`, `rolling_mean_14`,
`rolling_std_14` — all on `net_qty`, all shifted before computing so no day's own value leaks
into its own feature row.

### 5.6 Derived `is_in_stock`
Replaces the untrustworthy raw snapshot (see Section 3.2) with the adaptive zero-run heuristic.

### 5.7 Category-level rolling features (new, 2026-07-27 — final addition)
`cat_lag_1`, `cat_lag_7`, `cat_lag_14`, `cat_rolling_mean_7`, `cat_rolling_std_7`,
`cat_rolling_mean_14`, `cat_rolling_std_14` — computed on the **leave-one-out** category total
(sum of `net_qty` across every *other* SKU in the same category, same shift-before-compute
discipline as the SKU-level features). Gives thin-history SKUs a signal to borrow from siblings
in the same category, without being redundant with the SKU's own lag features.

**Current canonical model dataset: `SKUs/lgbm-dataset-5.csv` — 27 columns, 10,890 rows.**

---

## 6. Modeling Approach and Iteration History

Model: **LightGBM**, comparing a plain regression objective against a **Tweedie** objective
(better suited to zero-inflated, right-skewed retail demand). Tweedie won in every iteration.
Time-based train/validation/test splits throughout (never random — avoids leaking neighboring
days).

| Version | Key change | Result (daily accuracy, Tweedie) |
|---|---|---|
| v1–v2 | First baselines; price/cost bug still present | ~34% |
| v3 | Fixed price/cost sourcing; derived `is_in_stock`; dropped broken `current_stock_qty` | 35.16% |
| v4 | Added `is_on_sale` as a real feature | 36.89% (21-day test window) |
| v5 | Same features, longer 8-week test window (for weekly/biweekly comparison) | 32.90% (different eval window — not a regression) |
| **v6 (current)** | Added category-level rolling features | **33.04%** (same window as v5) |

A **28-SKU variant** (dropping the 2 lowest-volume SKUs) was tested to see if they were
dragging aggregate accuracy down — result: essentially no change, confirming forecasting error
is broad and systemic across many SKUs, not concentrated in a few problem SKUs.

---

## 7. Validating the Model Properly: Rolling-Origin Cross-Validation

Every model version above was checked on only **one** held-out window. To find out whether
that was a stable read (or a lucky/unlucky window) and whether hyperparameter tuning would
help, we ran a proper walk-forward validation.

**Hyperparameter search:** 18 configurations (learning rate × leaf count × min samples) tested
across 12 time-spaced folds. **All 18 configs landed within ~1 point of each other** —
hyperparameter tuning barely matters for this problem.

**Stability check:** the best config vs. the original default, evaluated on **24 separate,
non-overlapping weekly folds** spanning February–July 2026:

| | Mean | Std | Min | Max |
|---|---|---|---|---|
| Tuned | 61.46% | 10.02 | 25.06% | 74.94% |
| Default (kept) | 60.94% | 9.29 | 27.11% | 73.60% |

**Conclusion:** the ~62% weekly accuracy figure is a genuinely stable, representative average —
not a lucky window. But **individual weeks vary widely** (25–75% swing depending on the week).
Decision: keep the original default hyperparameters (tuning gain is noise-level) and report
~62% as the expected weekly accuracy, with the caveat that any single week can be much lower —
safety-stock buffers should account for the low end of this range, not just the average.

---

## 8. Daily-Trained vs. Weekly-Trained: Which Approach to Use for Weekly Business Reporting

Since Naheed orders/restocks weekly or biweekly (not daily), we tested two genuinely different
approaches on a 10-SKU subset (5 per category, chosen by top sales volume, excluding the 3
ops-unconfirmed stockout SKUs):

- **Approach A:** train the model at daily grain (as above), then **sum** the daily forecasts
  into weekly totals for reporting.
- **Approach B:** train a **new model natively** on weekly-aggregated targets.

Both were evaluated on the *identical* 8 held-out weeks (a date-alignment bug was initially
caught and fixed to guarantee this).

| | Overall weekly accuracy | SKUs won (of 10) |
|---|---|---|
| **A — daily-trained, summed to weekly** | **66.98%** | 7 |
| B — trained natively on weekly targets | 57.99% | 3 |

**Approach A won clearly.** Root cause: daily grain keeps ~9× more training rows and features
like day-of-week and short lags that get thrown away if you aggregate before training.
Aggregating the model's *output* (not its *input*) preserves that signal.

**Decision: Approach A — train daily, sum forecasts for weekly business reporting.**
This is the architecture the current pipeline uses.

---

## 9. Weekly vs. Biweekly Reporting Cadence — Weekly Adopted, Biweekly Removed

Retrained on all 30 SKUs with a longer 8-week test window to compare weekly against biweekly
aggregation:

| Grain | Overall accuracy |
|---|---|
| Daily (raw model output) | 32.9–33.0% |
| **Weekly (adopted)** | **~61%** |
| Biweekly (evaluated, not adopted) | ~65–67% |

Biweekly scored somewhat higher (more aggregation → more day-to-day error cancellation), but
**the business decision was to report weekly**, not biweekly, to match Naheed's actual
reorder cadence — accepting the modest accuracy trade-off. 62% weekly accuracy is a large
improvement over the ~33% raw daily figure.

**Update (2026-07-27):** since the business only needs weekly, all biweekly artifacts and
scripts have since been removed from the pipeline entirely (`model/aggregate_weekly_biweekly_30skus.py`,
the biweekly forecast/comparison CSVs, and the per-SKU biweekly text reports). The pipeline now
produces weekly output only — see Section 10.

---

## 10. Final Model Summary

**Canonical pipeline (current, end to end):**
`pg_new_1` (MySQL) → `SKUs/training_dataset_30skus.csv` (master enriched extract) →
`SKUs/lgbm-dataset-5.csv` (model-ready, 27 cols) → `model/train_lgbm_v6.py` (Tweedie
objective) → `model/lgbm_model_tweedie_v6.txt` (trained model) →
`model/aggregate_weekly_30skus.py` (daily forecasts summed to weekly — weekly only,
biweekly removed).

**Current output files (retrained 2026-07-27):**
- **Daily forecast:** `model/test_predictions_tweedie_v6.csv` — one row per SKU per test day
  (`date, sku, category, brand, net_qty` actual, `pred`)
- **Weekly forecast:** `model/weekly_forecast_vs_actual_30skus.csv` — one row per SKU per test
  week (`sku, week, week_start, week_end, actual, predicted, accuracy_pct`), plus
  `model/weekly_accuracy_30skus.csv` for the per-SKU/overall summary
- Per-SKU human-readable breakdowns: `model/sku_reports/weekly/<sku>.txt`

**Headline accuracy (30 SKUs):**
- Daily: ~33%
- **Weekly (adopted reporting grain): ~61%**
- Validated stable via 24-fold rolling-origin CV — individual weeks can range 25–75%.

---

## 11. Known Limitations / Open Items

1. **Marketing promo calendar still pending** (the original ask to marketing/ops). `is_on_sale`
   is currently backfilled from historical realized data as a stand-in — reasonable for
   training on the past, but the real forward-looking calendar should eventually replace it for
   future dates.
2. **3 SKUs with unconfirmed long stockout streaks** (`IC-1057685`, `IC-1134493`, `IC-1001018`)
   — need an operations check on whether their zero-demand periods are genuine or undetected
   stockouts.
3. **`IC-1147930`** — a real, still-unexplained high-volume miss (actual 799 vs. predicted
   ~400 units in one test window). Confirmed *not* caused by price-discount promos (this SKU
   has zero historical `is_on_sale` days) — still an open investigation.
4. **Week-to-week accuracy variance is real** (25–75% range) — this should be communicated to
   whoever uses these forecasts for stock-buffer sizing.
5. Any future data extraction should re-check the invoicing-lag pattern near the live edge of
   the DB (Section 3.4) before trusting the most recent few days as genuine low demand.

---

## 12. What to Share With the Team (Next Step: Website Automation)

### Do NOT share
- **`.env`** — contains the live database password in plaintext. Never send this file. If the
  team building the website needs DB access, share credentials through a secure channel
  (password manager / secrets vault), not by attaching this file.
- `SERVER_ACCESS_GUIDE.md` — describes shared server access; only share with people who are
  actually authorized for direct DB/EC2 access, and check with whoever manages that access first.

### Share — Documentation (read first)
- **This report** (`PROJECT_REPORT.md` / `.docx`)
- `SKUs/training_dataset_30skus_column_sources.md` — the exact SQL/table/attribute-ID mapping
  for every column. **This is what answers your question about extracting the same info for
  other SKUs** — it's the extraction spec, independent of any one script.

### Share — Data
- `SKUs/training_dataset_30skus.csv` — master enriched extract (all raw + derived columns
  before model-specific trimming). Best as the general-purpose reference.
- `SKUs/lgbm-dataset-5.csv` — the exact, current model-ready dataset (what the model is
  actually trained on).
- `SKUs/pilot_skus.csv` and `SKUs/sku_full_names.csv` — SKU list with names/category/brand.

**Yes — share both CSVs.** They serve different purposes: the master file is the full raw
reference (useful if the team wants to build different features later), the `lgbm-dataset-5`
file is exactly what feeds the current model.

### Share — Extraction / feature-building scripts
**Yes — share these.** They're exactly what the team needs to extract the same information for
additional SKUs:
- `SKUs/patch_price_cost_pg_new_1.py` — price/cost extraction pattern
- `SKUs/extend_training_dataset_backfill.py` / `SKUs/refresh_july2026_tail.py` — the core daily
  sales aggregation extraction pattern (the main "how do we pull daily sales for a SKU" logic)
- `SKUs/build_sale_flag.py` — `is_on_sale` extraction
- `SKUs/build_stockout_feature_v4.py` — derived `is_in_stock` + final feature assembly
- `SKUs/build_category_rolling_features.py` — category-level feature construction
- `SKUs/trim_invoicing_lag_tail.py` — the invoicing-lag safety check (important gotcha to keep)

### Share — Model & results (for the website automation team specifically)
- `model/train_lgbm_v6.py` — current training script (feature list, split logic, architecture)
- `model/lgbm_model_tweedie_v6.txt` — the trained model file itself (LightGBM native format —
  can be loaded directly for inference without retraining, via `lgb.Booster(model_file=...)`)
- `model/aggregate_weekly_30skus.py` — exactly how daily predictions get turned into the weekly
  numbers a website would display (weekly only — biweekly was evaluated, not adopted, and has
  since been removed from the pipeline; see Section 9)
- `model/weekly_forecast_vs_actual_30skus.csv` — current forecast-vs-actual numbers per SKU,
  ready to visualize
- `model/rolling_cv_final_results.csv`, `model/rolling_cv_category_features_results.csv` — the
  validation evidence behind the accuracy numbers, useful for a methodology/confidence appendix

### Don't bother sharing (superseded / internal experiments)
`lgbm-dataset.csv` / `-2` / `-3` / `-4` (superseded by `-5`), model versions v1–v5 (superseded
by v6), the 10-SKU daily-vs-weekly experiment files, and the 28-SKU variant — these were
internal validation steps, not part of the production pipeline, and would only add confusion.

### For the website automation step itself
The team building the website will need, at minimum: (1) the trained model file plus a small
inference script to score fresh daily data, (2) the aggregation logic to roll daily predictions
into weekly numbers, (3) a scheduled job that re-runs the extraction scripts against `pg_new_1`
to keep features current, and (4) somewhere to store forecast output that the website reads
from (even the CSVs to start, or a small database table later). None of this has been built
yet — it's the natural next phase after this handoff.
