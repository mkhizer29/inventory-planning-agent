# Demand Forecasting (Layer 3) — technical reference

This is the demand-forecasting component of the **Inventory Planning Agent**. It trains
a per-SKU daily demand model (LightGBM, tweedie objective) on a 30-SKU pilot and rolls
the daily forecasts up into weekly numbers for reporting/reorder decisions.

**Read `docs/PROJECT_REPORT.md` first** — it's the full write-up (DB source, feature
decisions, validation, results, known gaps). This README is just an orientation map.

## Where things live

| Item | Location |
|---|---|
| Full project report | `docs/PROJECT_REPORT.md` |
| Column/SQL source spec (every feature's exact query) | `docs/training_dataset_30skus_column_sources.md` |
| Master enriched extract (raw + all derived columns) | `data/training_dataset_30skus.csv` |
| Current model-ready dataset (30 SKUs, 27 cols) | `data/lgbm-dataset-5.csv` |
| SKU reference lists | `data/pilot_skus.csv`, `data/sku_full_names.csv` |
| DB extraction / feature-building scripts | `scripts/` |
| Training script, trained model, predictions, results | `model/` |

## Pipeline (run in this order from a fresh `pg_new_1` pull)

```
training_dataset_30skus.csv   (raw extract — not built by a script here; see docs/ for the query spec)
        │
        ▼  scripts/build_stockout_feature_v4.py     → data/lgbm-dataset-4.csv   (+ derived is_in_stock, drops leakage cols)
        ▼  scripts/build_category_rolling_features.py → data/lgbm-dataset-5.csv (+ category leave-one-out rolling features)
        ▼  model/train_lgbm_v6.py                    → model/lgbm_model_tweedie_v6.txt
                                                        model/test_predictions_tweedie_v6.csv
        ▼  model/aggregate_weekly_30skus.py           → model/weekly_forecast_vs_actual_30skus.csv
                                                        model/weekly_accuracy_30skus.csv
```

`scripts/patch_price_cost_pg_new_1.py`, `extend_training_dataset_backfill.py`,
`refresh_july2026_tail.py`, `build_sale_flag.py`, and `trim_invoicing_lag_tail.py` are the
extraction/maintenance scripts used to build and keep `training_dataset_30skus.csv`
current (price/cost patching, historical backfill, tail refresh, sale-flag derivation,
invoicing-lag trim) — see each script's docstring and `docs/training_dataset_30skus_column_sources.md`
for what each does and why.

All scripts expect `.env` (DB credentials, copied from the repo-root `.env.example`) at
the repository root, and are meant to be run **from inside `forecasting/scripts/` or
`forecasting/model/`** (paths inside each script are relative to that).

## Current results (tweedie_v6, 2026-07-27)

- Daily test accuracy: **33.04%** (56-day test window)
- Weekly aggregate accuracy: **~61%** average — the number to report/plan against —
  but individual weeks can swing 25–75%, validated via 24-fold rolling-origin CV
  (`model/rolling_cv_final_results.csv`, `model/rolling_cv_category_features_results.csv`).
  Size any safety-stock buffer off the low end of that range, not just the mean.

## Known gaps / next steps (see `docs/PROJECT_REPORT.md` §11–12 for full detail)

- Promo/marketing calendar still pending from marketing/ops — the biggest likely accuracy
  lever left; `is_on_sale` (DB-derived discount flag) is a partial stand-in already in the model.
- A few SKUs have long ops-unconfirmed stockout streaks that still drag down per-SKU accuracy.
- Next natural step: a small inference script to score fresh daily data with
  `model/lgbm_model_tweedie_v6.txt` on a schedule, plus somewhere for a website/dashboard
  to read the weekly output from.
