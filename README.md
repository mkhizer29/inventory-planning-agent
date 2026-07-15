# Inventory Planning Agent

An explainable, forward-looking inventory decision-support system for Naheed's buying
team. The full vision (see the Technical Specification) is a six-layer agent that
**forecasts demand**, **predicts stockouts**, **recommends reorder quantities**, and
**allocates stock across channels** (store / naheed.pk / foodpanda), with plain-English
explanations and a dashboard.

## Status

| Layer | Component | Status |
|---|---|---|
| **L2** | **ETL / data pipeline** | ✅ **Implemented** (this repo) |
| L3 | Forecasting, anomaly/spike detection, stockout risk, perishable classifier | ⏳ Future work |
| L4 | Reorder engine, multi-channel allocation, what-if simulation | ⏳ Future work |
| L5 | LLM explainability, alerts, feedback loop | ⏳ Future work |
| L6 | Dashboard, API | ⏳ Future work |

The **only implemented component today is the Layer 2 ETL pipeline**, which turns the raw
Magento `pg_1` database into clean, analysis-ready tables for the (future) modelling layers.

## Repository structure

```
Inventory-Planning-Agent/
├─ .env.example            # template for DB credentials (copy to .env)
├─ .env                    # your real credentials (git-ignored, never committed)
├─ requirements.txt        # Python dependencies
├─ pyproject.toml          # packaging (package `etl` under inventory_etl/) + pytest config
├─ README.md               # this file
└─ inventory_etl/
   ├─ config/config.yaml   # ETL business rules & assumptions
   ├─ etl/                 # the `etl` Python package (extract/transform/load/CLI)
   ├─ tests/               # unit + path regression tests
   ├─ output/              # generated: inventory.db, csv/, data_quality_report.md (git-ignored)
   ├─ run_etl.bat          # Windows launcher (double-click or CLI)
   └─ README.md            # detailed ETL technical reference
```

## Setup (Windows PowerShell)

Run everything **from the repository root**:

```powershell
cd "path\to\Inventory-Planning-Agent"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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

## More detail

See **`inventory_etl/README.md`** for the table catalogue, data-handling decisions,
configuration reference, and known data gaps.
