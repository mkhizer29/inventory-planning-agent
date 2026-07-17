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

## 3. Get the data — regenerate it from code, don't pass files around

The whole point: **data is rebuilt from the pipeline, never emailed/copied as files.**
Nobody commits data — it's git-ignored.

> ### ⚠️ Known blocker (read this before you start modelling)
> The models read `data/processed/weekly_sales.parquet` and
> `data/processed/weekly_signals.parquet`. **The data-prep step that produces those
> two parquet files does not exist yet — it is currently in progress.**
>
> **Until that prep step lands, you cannot run your model against the real data.**
> What you *can* do now:
> - build and **dry-run your model against a small hand-made sample** (create a tiny
>   `weekly_sales.parquet` / `weekly_signals.parquet` with a few SKUs and weeks so
>   your code path and `score_model(...)` run end-to-end), and
> - get your model file and its call into `src/evaluation.py` working.
>
> Watch `main` / ask Aiman for when the prep step is merged; after that, regenerate
> the real parquet files and re-run. **Do not treat step 3 as a working step yet.**

Once the prep step exists, the flow will be: run the ETL locally to build the
processed data (see the root `README.md` for the ETL — it needs your database
credentials in `.env`, copied from `.env.example`), which lands the parquet files in
`data/processed/`. You then run your model, which writes to `outputs/`.

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
