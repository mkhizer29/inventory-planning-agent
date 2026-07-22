# Teammate Setup — Forecasting Pilot

This is the shared guide for the 3-person forecasting pilot. It covers cloning the
repo, setting up Python, (re)generating the data, and the branch-per-person workflow.
If you're not a git expert, just follow the steps top to bottom.

> **Golden rule:** `src/evaluation.py` is the **locked shared scorecard**. All three
> models `import` it so everyone is judged identically. **Import it — never edit it.**
> (Aiman owns it; any change to it is coordinated across the team.)

---

## 1. Get the code

```bash
git clone <the-repo-url> Inventory-Planning-Agent
cd Inventory-Planning-Agent
git checkout main
git pull origin main          # make sure you have the latest main
```

## 2. Set up the Python environment

Do this once. Run everything from the **repository root** (the folder with this file) —
the scorecard reads `data/processed/` and writes `outputs/` using paths relative to
wherever you launch Python.

**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Get the data — it's already provided ✅  (daily, ecommerce-only)

The shared inputs are **committed to the repo** as the locked pilot contract, so after
cloning you already have them (in `data/processed/`):

```
data/processed/
  model_panel.parquet        REAL daily demand (one row per sku × naheed_web × date; target: units_observed)
                             + reconstructed SYNTHETIC stock_on_hand (inventory context, NOT a demand feature)
  forecast_frame.parquet     next 14 future days per SKU (leakage-safe: no actuals, no cost) — 420 rows
  inventory_context.parquet  one current-inventory row per SKU: stock (real snapshot if <= as_of, else synthetic),
                             VALIDATED unit cost, lead time / MOQ / pack size (real or assumed), reorder recommendation
  pilot_manifest.json        real/synthetic contract, assumptions, counts, validation
```

- **Frequency is daily; horizons are 7 and 14 days.** Channel is ecommerce-only **`naheed_web`**
  (physical `store` excluded; `foodpanda` has no data yet).
- **Demand forecasting uses REAL sales only.** `units_observed` is real Naheed demand — never
  altered, capped, or replaced. There is **no synthetic demand, no synthetic sales, no scenarios**.
  Use the demand-feature whitelist (lags/rollings/price/discount/promo/calendar); score rows on
  `forecast_training_eligible`. **Never** use `stock_on_hand` or `unit_cost` as a demand feature.
- **Only missing daily `stock_on_hand` is synthetic** — a deterministic per-SKU reconstruction
  (`stock[t] = stock[t-1] + assumed_replenishment[t] − units_observed[t]`), flagged
  `stock_on_hand_is_synthetic`. Real stock snapshots are used in `inventory_context.parquet` only
  when `snapshot_date <= as_of_date` (so July snapshots never touch the 2026-06-30 run).
- **You do NOT need database access or the ETL to model.** Load + score via `src/evaluation.py`:
  `load_model_panel()` / `load_forecast_frame()`, then `evaluate(preds, horizon=7 or 14)` where
  `preds` has **`sku, channel, date, y_pred`** (never pass `y_true`). `evaluate()` asserts at runtime
  that synthetic stock cannot change your scored rows.

> Locked contract — everyone models the same 30 SKUs, days and chronological split.
> Don't regenerate or edit these on your model branch. Read `pilot_manifest.json` to see
> which values are real vs assumed (lead time, MOQ, pack size are assumptions where marked
> `assumed_default`; unit-cost currency/basis await Naheed confirmation).

## Model architecture — three separate stages

**Stage A — real demand forecast.** Train on real `units_observed` (calendar + leakage-safe
lag/rolling + real price/promo features). Baselines (seasonal-naive m=7, moving average), then
ETS/Holt-Winters, intermittent-demand and tree methods where justified. Rolling-origin backtest,
report WAPE / MAE / RMSE / MASE (not MAPE alone — real sales contain zeros), plus intervals.
Refit the selected model on **all** valid history through `as_of_date` before forecasting.

**Stage B — forecast-driven stockout risk (not a trained classifier).**
`P(stockout in lead time) = P(cumulative forecast demand over lead time > stock_on_hand + confirmed inbound)`,
via Monte-Carlo draws or forecast quantiles. Also days-of-cover, expected lead-time demand,
reorder-point breach, risk tier. Because stock is synthetic, results are **pilot estimates, not
validated against real Naheed stockouts**.

**Stage C — reorder recommendation.**
`target = forecast demand over (lead time + review) + safety stock`;
`raw = max(0, target − stock_on_hand − on_order)`; round up to pack size; enforce MOQ;
purchase value = qty × `unit_cost_effective`. State which inputs are real / assumed / synthetic /
imputed. It's a recommendation for human approval — never an actual placed order.
The multi-scenario what-if simulator is a **future roadmap** item, not part of `prepare_pilot_data.py`.

**Only the pipeline owner regenerates them** (when source data changes):
```bash
# needs DB credentials in .env (copied from .env.example); see the root README
python -m etl.run_etl --source staging --sales-since 2026-01-15   # -> inventory_etl/output/inventory.db
python src/prepare_pilot_data.py --as-of-date 2026-07-15          # -> data/processed/ (4 files)
```
Modelers can ignore this step — the outputs are already in the repo.

## 4. Branch-per-person workflow

We each work on **our own model file, on our own branch**, and only merge to `main`
when the model actually runs and beats the baseline. This keeps `main` always working.

1. Start from an up-to-date `main`:
   ```bash
   git checkout main
   git pull origin main
   ```
2. Create **your** branch (pick the one for you):
   ```bash
   git checkout -b model/baselines      # Aiman
   git checkout -b model/holtwinters    # Khizer
   git checkout -b model/lgbm           # Aqib
   ```
3. Work **only on your own model file.** Don't edit teammates' files or the scorecard.
4. Commit and push often so your work is backed up and visible:
   ```bash
   git add <your-model-file>
   git commit -m "clear message about what you changed"
   git push -u origin <your-branch>     # first push; later just: git push
   ```
5. **Merge into `main` only after** your model runs cleanly **and beats Aiman's
   baseline** on the shared scorecard. Open it for a quick look before merging.

## 5. Cheat sheet

| I want to… | Command |
|---|---|
| See which branch I'm on | `git branch --show-current` |
| Get the latest `main` | `git checkout main && git pull origin main` |
| Start my model branch | `git checkout -b model/<mine>` |
| Save + share my work | `git add . && git commit -m "…" && git push` |
| Score my model | `import src.evaluation as ev; ev.score_model(preds, sales, "mymodel")` |

Results land in `outputs/` as `preds_<model>.csv` and `metrics_<model>.csv` (git-ignored;
the `outputs/` folder itself stays in git via `.gitkeep`).
