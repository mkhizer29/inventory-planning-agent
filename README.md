# Inventory Planning Agent

An explainable, forward-looking inventory decision-support system for Naheed's buying
team. The full vision (see the Technical Specification) is a six-layer agent that
**forecasts demand**, **predicts stockouts**, **recommends reorder quantities**, and
**allocates stock across channels** (store / naheed.pk / foodpanda), with plain-English
explanations and a dashboard.

## Status

| Layer | Component | Status |
|---|---|---|
| **L2** | **ETL / data pipeline** | ✅ **Implemented** (`inventory_etl/`) |
| **L3** | **Forecasting** (30-SKU pilot, daily LightGBM, tweedie objective, weekly rollup) | ✅ **Implemented** (`forecasting/`, this branch) |
| L3 | Anomaly/spike detection, stockout risk, perishable classifier | ⏳ Future work |
| L4 | Reorder engine, multi-channel allocation, what-if simulation | ⏳ Future work |
| L5 | LLM explainability, alerts, feedback loop | ⏳ Future work |
| L6 | Dashboard, API | ⏳ Future work |

This branch (`aqib-lgbm`) adds the **Layer 3 demand-forecasting pipeline** on top of the
Layer 2 ETL. It turns the raw Magento `pg_new_1` database into clean, analysis-ready
tables (Layer 2), then trains a per-SKU daily demand model and rolls the forecasts up
to weekly numbers for reorder decisions (Layer 3).

## Repository structure

```
Inventory-Planning-Agent/
├─ .env.example            # template for DB credentials (copy to .env)
├─ .env                    # your real credentials (git-ignored, never committed)
├─ requirements.txt        # Python dependencies (Layer 2 / ETL)
├─ pyproject.toml          # packaging (package `etl` under inventory_etl/) + pytest config
├─ README.md               # this file
├─ inventory_etl/          # Layer 2 — ETL pipeline
│  ├─ config/config.yaml   # ETL business rules & assumptions
│  ├─ etl/                 # the `etl` Python package (extract/transform/load/CLI)
│  ├─ tests/                # unit + path regression tests
│  ├─ output/               # generated: inventory.db, csv/, data_quality_report.md (git-ignored)
│  ├─ run_etl.bat           # Windows launcher (double-click or CLI)
│  └─ README.md             # detailed ETL technical reference
└─ forecasting/            # Layer 3 — demand forecasting (this branch)
   ├─ requirements.txt      # extra deps on top of the root ones (lightgbm, scikit-learn)
   ├─ docs/                 # full project report + exact column/SQL source spec
   ├─ data/                 # master extract + current model-ready dataset + SKU lists
   ├─ scripts/              # DB extraction / feature-building scripts (run from pg_new_1)
   ├─ model/                # training script, trained model, predictions, CV results
   └─ README.md             # pipeline order, current results, how to run
```

**How the two layers connect:** `inventory_etl/` is the general-purpose ETL for the
canonical warehouse tables used across all future layers. `forecasting/` currently pulls
directly from the Magento DB with its own extraction scripts (built before the ETL
package existed) rather than reading `inventory_etl/`'s output tables — reconciling the
two into one shared extraction path is open follow-up work, not done yet.

## Setup (Windows PowerShell)

Run everything **from the repository root**:

```powershell
cd "path\to\Inventory-Planning-Agent"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r forecasting\requirements.txt   # only needed for the forecasting pipeline
python -m pip install -e .
Copy-Item .env.example .env
```

Then open `.env` and fill in the database credentials (`STAGING_PASSWORD` and/or
`LOCAL_PASSWORD`). **Never commit `.env`** — it is git-ignored and holds secrets.

(macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`, then the same `pip`
commands and `cp .env.example .env`.)

## Run the ETL

```powershell
python -m etl.run_etl --help
python -m etl.run_etl --source staging
python -m etl.run_etl --source local_backup --sales-since 2024-01-01
```

Or use the launcher `inventory_etl\run_etl.bat` (double-click, or
`inventory_etl\run_etl.bat --source staging`). It auto-uses `.venv` if present and
falls back to `python` on PATH.

## Run the tests

```powershell
python -m pytest -q
```

Tests are offline — they do not connect to the database or read secrets.

## Where outputs appear

After a run, in `inventory_etl/output/`:

- `inventory.db` — SQLite warehouse with all canonical + supporting tables
- `csv/*.csv` — one CSV per table (Excel-friendly)
- `data_quality_report.md` — row counts, coverage, cleansing flags, warnings

## Run the forecasting pipeline (Layer 3)

Scripts run from **inside** `forecasting/scripts/` or `forecasting/model/` (their paths
are relative to that), against a filled-in `.env` at the repo root:

```powershell
cd forecasting\scripts
python build_stockout_feature_v4.py           # data/training_dataset_30skus.csv -> data/lgbm-dataset-4.csv
python build_category_rolling_features.py     # data/lgbm-dataset-4.csv -> data/lgbm-dataset-5.csv (canonical dataset)
cd ..\model
python train_lgbm_v6.py                       # trains, saves model + daily predictions in this folder
python aggregate_weekly_30skus.py             # daily predictions -> weekly forecast-vs-actual
```

`data/training_dataset_30skus.csv` is a committed extract (not regenerated by a script
here) — see `forecasting/docs/training_dataset_30skus_column_sources.md` for the exact
DB query behind every column, and `forecasting/scripts/` for the maintenance scripts
that patch/extend/trim it as fresher DB data arrives.

Current results: ~33% daily accuracy, ~61% weekly aggregate accuracy (30-SKU pilot,
tweedie objective). Full detail, validation, and known gaps in
**`forecasting/docs/PROJECT_REPORT.md`** and **`forecasting/README.md`**.

## More detail

See **`inventory_etl/README.md`** for the ETL table catalogue, data-handling decisions,
configuration reference, and known data gaps. See **`forecasting/README.md`** for the
forecasting pipeline order, file map, and current results.
