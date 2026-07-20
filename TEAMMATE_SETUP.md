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
model_panel.parquet        daily history — one row per sku × channel × date (target: units_observed)
forecast_features.parquet  the next 14 future days per SKU (leakage-safe: no actuals/discounts)
inventory_context.parquet  as-of stock + replenishment context (assumptions flagged)
pilot_manifest.json         validation stats, coverage, and the explicit assumptions block
```

- **Frequency is daily; horizons are 7 and 14 days.** Channel is ecommerce-only:
  **`naheed_web`** (physical `store` excluded; `foodpanda` has no data yet).
- **You do NOT need database access or the ETL to model.** Load + score via `src/evaluation.py`:
  `load_model_panel()`, then `evaluate(preds, horizon=7 or 14)` where `preds` has
  **`sku, channel, date, y_pred`** (never pass `y_true`).

> Locked contract — everyone models the same 30 SKUs, days and chronological split.
> Don't regenerate or edit these on your model branch. Read `pilot_manifest.json` to see
> which values are real vs assumed (lead time, MOQ, stock-in-transit, perishability are assumptions).

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
