# Inventory Planning Agent

Naheed AI Explorers Program 2026 - Team 3

An explainable, forward-looking inventory decision-support system for Naheed's ecommerce planning workflow. The project turns warehouse sales and inventory data into 7- and 14-day demand forecasts, forecast-driven stockout risk, prioritized reorder proposals, deadstock diagnostics, and an interactive Streamlit dashboard.

> **Final project scope:** the implemented pilot is ecommerce-focused. It supports human decision-making; it does **not** automatically create or submit purchase orders, perform live ERP write-back, or claim full store/web/Foodpanda allocation optimization.

## Project status

The final pilot is implemented end-to-end:

- ETL from Naheed Magento data into a local SQLite analytics warehouse.
- Dynamic category and Top-N SKU selection.
- Top-N ranking by historical units sold **or** a pre-forecast stockout-risk proxy.
- Run-specific, leakage-aware daily forecasting datasets.
- Baseline, Holt-Winters/ETS, and global LightGBM forecasting families.
- Shared chronological 7/14-day backtesting and deterministic model ranking.
- Automatic selection of an operational forecast for downstream decisions.
- Phase B forecast-driven stockout-risk analysis and daily risk trajectories.
- Phase C reorder recommendations with MOQ/pack-size handling and human-approval safeguards.
- Standalone ecommerce deadstock analysis using real warehouse inventory and sales.
- Run history, exports, explanations, data-quality views, and interactive dashboard workflows.

## What the system does

```text
Magento / pg_1
      |
      v
ETL -> SQLite warehouse (inventory.db)
      |
      +-------------------------------> Deadstock analysis
      |
      v
Category + Top-N selection
      |
      v
Run-specific pilot data preparation
      |
      v
Baselines | Holt-Winters | LightGBM
      |
      v
Locked-holdout model ranking
      |
      v
selected_forecasts.parquet
      |
      v
Phase B: Stockout Risk
      |
      v
Phase C: Reorder Recommendations
      |
      v
Streamlit Dashboard
```

Forecast runs are isolated under `runs/<run_id>/`. The dashboard reads the completed artifacts instead of recomputing forecasting or inventory-decision logic inside the UI.

## Repository structure

```text
Inventory-Planning-Agent/
|- inventory_etl/
|  |- etl/                     # extract, cleanse, transform, load, DQ reporting
|  |- config/config.yaml       # auditable business rules and pilot assumptions
|  |- tests/
|  `- output/                  # generated inventory.db / CSVs / DQ report
|- src/
|  |- dynamic_selection.py     # category discovery and deterministic Top-N selection
|  |- selection_risk.py        # pre-forecast risk proxy used only for Top-N ranking
|  |- prepare_pilot_data.py    # run/model-ready daily ecommerce datasets
|  |- evaluation.py            # shared chronological evaluation contract
|  |- baselines.py             # naive + moving-average baselines
|  |- holtwinters.py           # per-SKU ETS/Holt-Winters family
|  |- lgbm_global.py           # pooled/global LightGBM demand model
|  |- forecast_orchestrator.py # complete isolated forecast-run lifecycle
|  |- stockout_risk.py         # Phase B
|  |- reorder_recommendations.py # Phase C
|  `- deadstock_analysis.py    # standalone ecommerce inactivity/deadstock scan
|- dashboard/
|  |- app.py                   # Streamlit UI
|  |- run_service.py           # safe run discovery/launch/context layer
|  `- export_utils.py          # CSV / Excel / PDF exports
|- data/processed/             # generated fixed-pilot/model datasets
|- outputs/                    # generated fixed-pilot/model outputs
|- runs/                       # generated isolated dynamic runs
|- pilot_skus.csv              # legacy/fixed pilot SKU list
|- requirements.txt
|- pyproject.toml
`- README.md
```

Generated data, databases, credentials, and run outputs should not be treated as source code. Never commit `.env` or private server/database credentials.

## Data and ETL

The ETL can read the Naheed Magento `pg_1` database from a staging connection or locally loaded production backup and produces `inventory_etl/output/inventory.db` plus CSV exports and a data-quality report.

Core warehouse tables include:

- `sku_master`
- `sales_transactions`
- `inventory_snapshot`
- `channel_master`
- `external_signals`

Supporting tables cover available stock history, shipments, returns, promotions, geography, related products, product views, stock alerts, and search signals when those source tables are available.

See `inventory_etl/README.md` for the ETL table catalogue, data-handling rules, configuration, and known source-data gaps.

## Forecasting data contract

The final forecasting contract deliberately separates demand truth from inventory assumptions.

### Real demand

`units_observed` comes from real Naheed ecommerce sales. Demand is not synthetically generated, capped, or replaced for model training.

### Synthetic historical stock where required

Naheed does not provide a reliable daily historical inventory series covering the forecasting period. Missing historical `stock_on_hand` is therefore reconstructed deterministically for pilot context. This synthetic stock is **not** a demand target and is excluded from demand-model feature inputs and demand-evaluation eligibility.

### Real current inventory when valid

Inventory context uses an eligible real stock snapshot when one exists for the requested as-of date; otherwise the pilot contract records the fallback/assumption transparently.

### Forecast features

The prepared daily panel includes causal demand lags/rolling statistics plus known calendar/price/promotion signals. Examples include:

- lagged demand (1/7/14 days) and rolling demand statistics;
- effective price, discount and promotion indicators;
- day-of-week, weekend, week and month;
- Pakistan public-holiday and payday-window indicators;
- configured Ramadan indicators.

Inventory, unit cost, lead-time, MOQ, and pack-size fields are not allowed to leak into the demand forecast target/features outside the documented model contract.

## Dynamic category and Top-N selection

The primary run flow starts with an exact category, Top-N count, selection cutoff, and minimum history requirement.

Two ranking modes are implemented:

1. **Units sold (`units`)** - ranks forecast-eligible SKUs by historical real ecommerce units on or before the selection cutoff.
2. **Stockout risk (`stockout_risk`)** - ranks the eligible category population using a pre-forecast risk proxy built from trailing real ecommerce demand, real warehouse stock, and lead time.

The risk proxy exists to avoid a circular dependency: authoritative Phase B risk requires a forecast, but SKU selection happens before forecasting. The proxy only chooses which SKUs are forecast; Phase B later recomputes authoritative risk from the selected operational forecast.

Because the warehouse may not contain historical stock snapshots for an arbitrary cutoff, the default risk-selection policy can use the latest real inventory snapshot even when it postdates the selection cutoff. The run metadata records the snapshot date and whether post-cutoff stock influenced selection so the trade-off is auditable.

## Forecasting models

### Baselines

Four reference methods establish honest benchmarks:

- last-day naive;
- seasonal naive (7-day);
- 7-day moving average;
- 14-day moving average.

### Holt-Winters / ETS

The Holt-Winters implementation is a per-SKU univariate ETS family. Candidate additive structures are selected per SKU through leakage-free historical model selection, then evaluated on the shared locked holdout. Forecasts are non-negative and include uncertainty intervals/fallback behavior.

### Global LightGBM

LightGBM pools eligible SKU histories into one global gradient-boosted model using `sku` as a categorical identity feature plus the approved causal demand/calendar/price/promotion features. Multi-step future forecasts are recursive: future lag/rolling features are rebuilt using prior predictions rather than future actual demand.

## Evaluation and operational-model selection

All forecasting families use the same chronological evaluation contract for 7- and 14-day horizons. The test window is the most recent horizon-sized block; there is no random train/test split.

Reported demand metrics include:

- WAPE (primary ranking metric)
- MASE
- MAE
- RMSE
- bias

The orchestrator validates model outputs and dataset fingerprints before comparing models. It then deterministically ranks valid forecast candidates on the locked holdout (the individual baseline methods, Holt-Winters, and LightGBM) and selects the winner for the operational horizon (normally the largest requested horizon, 14 days). The selected candidate's future forecast is written to `selected_forecasts.parquet` and is the only forecast consumed by Phase B and Phase C.

Model accuracy is estimated from historical backtesting; it is not guaranteed future accuracy.

## Phase B - Forecast-driven stockout risk

Phase B combines the selected operational forecast with inventory context, lead time, service-level policy, and forecast/residual uncertainty.

For each selected SKU/channel it produces auditable fields such as:

- stockout probability and probability risk tier;
- forecast days of cover and cover tier;
- overall risk tier;
- projected stockout date;
- expected shortage units;
- safety stock and reorder point;
- estimated revenue at risk;
- uncertainty/confidence method;
- assumption flags and deterministic `reason_trace`.

Uncertainty follows an explicit fallback order: available forecast intervals, per-SKU backtest residuals, pooled same-model residuals, historical demand variability, then a deterministic zero-uncertainty fallback when no estimate exists. Cumulative uncertainty uses root-sum-of-squares under the documented independent-daily-error pilot assumption.

Outputs:

- `decisions/stockout_risk.parquet`
- `decisions/stockout_trajectory.parquet`

## Phase C - Reorder recommendations

Phase C consumes the validated Phase B results, operational forecast, inventory context, and replenishment policy. It does not recompute stockout risk.

Exactly one proposed action is produced per selected SKU/channel:

- `order_now`
- `monitor`
- `no_order`
- `vendor_follow_up`
- `manual_review`

Order quantities are planning proposals derived from the forecast-driven inventory need, target-cover policy, MOQ handling, and pack-size rounding. Missing/invalid critical inputs fall back to `manual_review` with no fabricated actionable quantity. Dropship/vendor-fulfilled products can be routed to `vendor_follow_up`.

**No purchase order is placed by this system.** Actionable recommendations require buyer review/approval.

Outputs:

- `decisions/reorder_recommendations.parquet`
- `decisions/reorder_summary.json`

## Deadstock analysis

Deadstock is a separate read-only ecommerce diagnostic and does not depend on forecast runs. It compares the latest real warehouse inventory with ecommerce sales activity over a configurable inactivity interval (90 days by default).

It can classify stock-carrying SKUs as:

- `Deadstock Candidate`
- `Never Sold`
- `Manual Review`
- `Not Deadstock` (when the full population is requested programmatically)

The analysis can estimate deadstock value when a usable unit cost is available. Ecommerce inactivity should not be interpreted as proof that a product had no physical-store sales.

## Dashboard

Run the Streamlit dashboard from the repository root:

```powershell
python -m streamlit run dashboard/app.py
```

The dashboard includes:

1. Executive Overview
2. Demand Analytics
3. Forecast Runs
4. Forecast Explorer
5. Inventory & Reorder
6. Deadstock
7. Stockout Risk
8. Data Quality & Assumptions

`Forecast Runs` can launch isolated forecast runs through a safe argument-list subprocess (`shell=False`), show progress/status, browse completed history, and activate a completed run for the remaining pages. Tables can be exported through the dashboard in CSV, Excel, and PDF formats where supported.

## Run artifacts and reproducibility

A dynamic forecast is self-contained under `runs/<run_id>/`:

```text
runs/<run_id>/
|- request.json
|- status.json
|- run_manifest.json
|- selected_forecasts.parquet
|- combined_scorecard.csv
|- model_ranking.csv
|- processed/
|  |- model_panel.parquet
|  |- forecast_frame.parquet
|  |- inventory_context.parquet
|  `- pilot_manifest.json
|- outputs/
|  |- baseline_*
|  |- holtwinters_*
|  `- lightgbm_*
`- decisions/
   |- stockout_risk.parquet
   |- stockout_trajectory.parquet
   |- reorder_recommendations.parquet
   `- reorder_summary.json
```

The final manifest records request metadata, selected SKUs, ranking mode, completed/failed models, dataset fingerprint, model metrics, operational model/horizon, decision summaries, software versions, errors/warnings, and an artifact inventory.

## Setup

Python 3.12 is the tested project target. From the repository root:

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
Copy-Item .env.example .env
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
cp .env.example .env
```

Populate `.env` only when ETL source-database access is required. Never commit credentials.

## Run the ETL

```powershell
python -m etl.run_etl --help
python -m etl.run_etl --source staging
python -m etl.run_etl --source local_backup --sales-since 2024-01-01
```

On Windows, `inventory_etl\run_etl.bat` is also available.

## Run an end-to-end forecast

The dashboard is the easiest operational entry point. The backend orchestrator can also be called directly:

```powershell
python src/forecast_orchestrator.py `
  --category "Groceries & Pets" `
  --top-n 10 `
  --as-of-date 2026-06-30 `
  --selection-cutoff 2026-06-30 `
  --min-history-days 28 `
  --horizons 7 14 `
  --ranking-metric units
```

To rank the eligible category population by the pre-forecast stockout-risk proxy instead:

```powershell
python src/forecast_orchestrator.py `
  --category "Groceries & Pets" `
  --top-n 10 `
  --as-of-date 2026-06-30 `
  --selection-cutoff 2026-06-30 `
  --min-history-days 28 `
  --horizons 7 14 `
  --ranking-metric stockout_risk
```

Use exact eligible category names from the dashboard/category-discovery flow. The dashboard also applies a trailing-extract completeness guard when suggesting the latest usable sales date so a partial raw extract tail is not silently treated as a normal forecasting cutoff.

## Tests

Run the complete automated suite from the project environment:

```powershell
python -m pytest -q
```

The suite covers ETL/data handling, pilot preparation, model/evaluation contracts, dynamic selection, run orchestration, stockout/reorder decisioning, deadstock logic, dashboard run services, and export/UI-support utilities. Keep the environment installed from `requirements.txt` before interpreting import failures as application failures.

## Important assumptions and limitations

This is a completed pilot, not an autonomous production procurement system.

- **Operational scope:** forecasting/decisioning is ecommerce-focused (`naheed_web` convention); this implementation does not optimize stock allocation across every physical-store/web/marketplace channel.
- **Historical inventory:** daily historical stock is incomplete, so pilot historical stock may be deterministically reconstructed and clearly flagged as synthetic.
- **Replenishment inputs:** supplier lead time, MOQ, pack size, on-order quantity, and cost may fall back to configured assumptions/imputation when reliable source values are unavailable.
- **Risk-based Top-N cutoff purity:** the current real inventory snapshot used by the selection proxy can postdate the historical selection cutoff; this is explicitly recorded in run metadata.
- **Uncertainty:** Phase B uses a documented independent-daily-error approximation and fallback hierarchy; the risk outputs are planning estimates, not observed historical stockout labels.
- **Deadstock scope:** inactivity is assessed from ecommerce sales, so it does not prove there were no offline/store sales.
- **Perishability:** relevant source fields are retained where available, but the final decision policy does not implement a complete differentiated perishable-goods optimization engine.
- **External signals:** calendar/payday/Ramadan signals are supported; live weather integration is not enabled.
- **Human in the loop:** reorder outputs are recommendations only. No PO is created, transmitted, or written back to an ERP.
- **Accuracy:** backtest metrics estimate historical forecasting performance and do not guarantee future accuracy.

## Project brief coverage

The final pilot directly addresses the core Team 3 brief by providing:

- short-term 7-14 day forecasting for more than the minimum five sample SKUs;
- clear forecast-driven stockout risk with reasoning;
- prioritized reorder quantities/actions with reasoning;
- an interactive decision-support dashboard;
- additional deadstock and dynamic Top-N capabilities beyond the minimum final deliverable.

The broader Technical Specification also describes longer-term capabilities such as full multi-channel allocation, live alerts, richer perishability optimization, what-if simulation, LLM-generated narratives, and production ERP integration. Those are roadmap concepts and should not be presented as implemented features of this final pilot.

## Final-project boundary

The forecasting, stockout-risk, reorder, deadstock, and dashboard logic in this repository represents the final submitted project implementation. Further changes should be treated as post-project maintenance or future roadmap work rather than silent changes to the evaluated final logic.
