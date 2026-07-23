# Pilot Forecasting — Build Document 3 of 3
## Owner: Aqib · Global LightGBM Model (the machine-learning workhorse)

> Your mission: build **one** machine-learning model that forecasts **all 30
> products at once**, learning shared patterns across them and using the extra
> signals (holidays, payday, promotions) that the classical models can't. This is
> the most powerful method on the team — and the one with the most ways to go
> subtly wrong, so read the pitfalls (§7) carefully.

---

## Getting started

New to the repo? Follow **[TEAMMATE_SETUP.md](TEAMMATE_SETUP.md)** (repo root) for
cloning, the Python environment, regenerating the data, and the branch workflow.
**Work on branch `model/lgbm`.** Import the shared scorecard from
`src/evaluation.py` — don't edit it.

---

## ⚠️ Contract update (v2 — daily, ecommerce-only) — READ THIS FIRST

The pilot moved from **weekly** to **daily**, and to **ecommerce-only**. The weekly
sections below are background; the **binding contract is here**:

- **Inputs** (already generated in `data/processed/`):
  - `model_panel.parquet` — REAL daily demand (one row per `sku × naheed_web × date`; target = **`units_observed`**) plus a reconstructed **synthetic `stock_on_hand`** (inventory context only, never a demand feature).
  - `forecast_frame.parquet` — the next **14 future days** per SKU (leakage-safe: no actuals, no cost). *(Renamed from `forecast_features.parquet`.)*

> **🔄 v4 update — REAL demand, SYNTHETIC daily stock only (supersedes anything above that conflicts):**
> - **Demand forecasting uses REAL sales.** `units_observed` is real Naheed naheed_web demand — never altered, capped, or replaced. There is **no synthetic demand, no synthetic sales, no scenarios, no lost sales**.
> - **Only missing daily `stock_on_hand` is synthetic** — one deterministic per-SKU balance `stock[t] = stock[t-1] + assumed_replenishment[t] − units_observed[t]`, flagged `stock_on_hand_is_synthetic=True`. It is **downstream inventory context, not a demand feature**.
> - **Feature whitelist:** `units_lag_1/7/14`, `units_roll_mean_7/28`, `units_roll_std_7`, price/discount/promo, calendar (incl. Ramazan: `is_ramadan`/`ramadan_day`/`ramadan_week`). **Never** feed `stock_on_hand` or `unit_cost` to the model — this matters most for LightGBM, which will greedily use any column you hand it.
> - **Row eligibility = `forecast_training_eligible`** (real-data quality + ≥14 days history) — independent of synthetic stock; `evaluate()` asserts this at runtime.
> - Lead time / MOQ / pack size are **pilot assumptions** (real per-SKU values override defaults). Unit cost is validated in `inventory_context.parquet`; currency PKR and cost unit/pack **basis await Naheed confirmation**.
> - **Three separate stages** (see TEAMMATE_SETUP → "Model architecture"): **A** real demand forecast → **B** forecast-driven stockout risk → **C** reorder recommendation. B and C use synthetic stock, so they are **pilot estimates, not validated against real Naheed stockouts**.
> - **Removed:** the old `data/synthetic/` scenario outputs (stockout_scenarios / replenishment_events / simulation_parameters) — do not read them.
- **Channel**: ecommerce only → **`naheed_web`** (physical `store` excluded).
- **Horizons**: forecast **7 and 14 daily** steps, **chronological** split — `TEST_WEEKS` is gone.
- **Scoring**: `from evaluation import evaluate` → `evaluate(preds, horizon=7 or 14)`.
  Submit **`sku, channel, date, y_pred`** only — **never pass `y_true`**. Optional `lower_bound`/`upper_bound`.
  Reported: **WAPE, MAE, MASE, RMSE, bias** + interval coverage.
- Supplier **lead time & MOQ are downstream configurable assumptions**, **not forecasting features**.

**Your model (global LightGBM, daily).** Leakage-safe daily features:
`lag_1, lag_2, lag_3, lag_7, lag_14, lag_28, rolling_mean_7, rolling_mean_14, rolling_std_7,
day_of_week, is_public_holiday, is_payday_window, is_ramadan, ramadan_day, ramadan_week,
on_promo (historical), price, category, brand, channel`.
(`is_ramadan` / `ramadan_day` / `ramadan_week` are known-in-advance Karachi calendar signals,
configured in `config.external_signals.ramadan_periods`; only the 2026 period exists so far, so
any Ramazan effect is exploratory.)
Train only on rows with `date <= cutoff`. For **future** rows use only what's in
`forecast_frame.parquet` — planned/calendar signals — **never** realized future discounts
(`planned_promo` is currently `unavailable`, so treat promo as off for the future window).
Forecast 7 & 14 days recursively; submit `preds[sku, channel, date, y_pred]` (no `y_true`).

---

## 0. How your piece fits the team

| Person | Method | Document |
|---|---|---|
| Aiman | Baselines + scorecard | Doc 1 |
| Khizer | Exponential smoothing | Doc 2 |
| **Aqib (you)** | Global LightGBM | This one |

You import Aiman's `src/evaluation.py` for the split and the metrics — same as
everyone. Your target: **MASE below the best baseline**, and ideally below
Khizer's Holt model too. But note: on small data, ML does **not** automatically
win. You must *prove* it beats the simpler methods; if it doesn't, that's a valid
finding, not a failure.

---

## 1. What you're building, in plain words

Khizer builds a separate model per product. You do the opposite: **one model
trained on all 30 products stacked together.** Why? Because most products have
thin history, and a single product's 21 training weeks aren't much to learn from.
But *30 products × 21 weeks* pooled together, plus features telling the model which
category/brand/price/week each row is, lets it **borrow patterns across similar
products.** A slow-selling beverage benefits from what the model learned on the
fast-selling ones in the same category.

The model is **LightGBM** — a gradient-boosting method that builds many small
decision trees, each fixing the errors of the last. It's the go-to for exactly
this kind of tabular retail data (it's what wins real forecasting competitions on
Walmart-style sales data).

The core trick that turns forecasting into something LightGBM can do: we hand it,
for each product-week, a row of **features** describing that week, and ask it to
predict `units`. The features are what make or break this — that's §4.

---

## 2. Setup (do this once)

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install pandas numpy pyarrow lightgbm scikit-learn
python -c "import lightgbm; print('ready')"
```

Project layout:
```
pilot/
├─ data/processed/          <- shared input files (see §3)
├─ src/
│  ├─ evaluation.py         <- Aiman's shared file — import, don't edit
│  └─ lgbm_global.py        <- YOU write this (full code in §5)
└─ outputs/
```

---

## 3. The data you will receive (the shared contract)

You do **not** extract data. The ETL pipeline gives all three of us the same two
files, weekly, zero-filled. **You use BOTH files** — the signals are your edge.

**File 1 — `data/processed/weekly_sales.parquet`** (one row per SKU per week):

| column | type | meaning |
|---|---|---|
| `sku` | text | product code |
| `category` | text | one of the 2 pilot categories |
| `brand` | text | brand name |
| `price` | number | unit price |
| `week_start` | date | Monday of the week |
| `units` | integer | units sold that week (0 if none) — this is the TARGET |
| `on_promo` | 0/1 | on promotion that week |

**File 2 — `data/processed/weekly_signals.parquet`** (one row per week):

| column | type | meaning |
|---|---|---|
| `week_start` | date | Monday of the week |
| `holiday_days` | integer | holiday days that week |
| `payday_days` | integer | payday days that week |

### The split (same for everyone)
Per SKU: train = all weeks except the last 5; test = last 5 weeks. Chronological.
`TEST_WEEKS = 5` from `evaluation.py`. **Critical for you specifically:** because
your model uses lag features (last week's sales, etc.), you must make sure a test
week's features never contain information from the future — see §7.

### The metric (same for everyone)
MASE primary (< 1 beats naive), plus MAE and RMSE, from Aiman's `score_model()`.

---

## 4. The method explained — features are everything

For each product-week, you build a feature row. Categories of feature:

**Lag features (the most important).** Last week's units, 2 weeks ago, 3, 4. These
tell the model the recent trajectory. `lag_1` is usually the single strongest
predictor.

**Rolling feature.** The average of the last 4 weeks — a smoothed "recent level."

**Calendar features.** Week-of-year and month — lets the model place the week in time.

**Signal features.** `holiday_days`, `payday_days` (from File 2), and `on_promo`
(from File 1). This is your advantage over the classical models: the ML model can
learn "sales jump when payday_days > 0" directly.

**Product features.** `category`, `brand` (handled as **categorical** — LightGBM
supports this natively, no manual encoding), and `price`. These are how the model
tells products apart and borrows across similar ones.

### The multi-week horizon problem (read this twice)
You must forecast **5 weeks** ahead. But `lag_1` for week +2 is the units in week
+1 — which hasn't happened yet at forecast time. The standard solution is
**recursive forecasting**: predict week +1, then **feed that prediction back in**
as the `lag_1` for week +2, predict +2, feed it in for +3, and so on. The code in
§5 does exactly this, per SKU, week by week. It's fiddly but it's the correct way
to make a lag-based model produce a multi-week forecast comparable to the other
two methods.

### Small-data discipline
30 SKUs × ~21 usable training weeks ≈ a few hundred rows. That is **small** for a
tree model, which can memorize (overfit) tiny datasets. So the code sets
conservative settings: few leaves per tree, a floor on samples per leaf, and
regularization. Do not crank these up chasing a better training score — you'll
just overfit and do worse on the test weeks.

---

## 5. Build it — `src/lgbm_global.py`

Complete, tested file. The comments walk through each stage.

```python
"""lgbm_global.py — Aqib. ONE gradient-boosted model across all 30 SKUs.
Features: lags, rolling mean, calendar, holiday/payday/promo, category/brand/price.
Multi-week horizon via RECURSIVE forecasting. Small data -> heavy regularization."""
from __future__ import annotations
import numpy as np, pandas as pd, lightgbm as lgb
import sys; sys.path.append("src")
from evaluation import load_weekly_sales, load_signals, TEST_WEEKS, score_model

LAGS = [1, 2, 3, 4]
CAT_FEATURES = ["category", "brand"]   # LightGBM handles these natively


def build_features(df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw weekly table into a feature table (one row per SKU-week)."""
    df = df.sort_values(["sku", "week_start"]).copy()
    g = df.groupby("sku")["units"]
    for L in LAGS:                                  # last-1..4-week sales
        df[f"lag_{L}"] = g.shift(L)
    # rolling mean of the 4 weeks BEFORE this one (shift(1) avoids using today)
    df["roll_mean_4"] = g.shift(1).rolling(4).mean().reset_index(level=0, drop=True)
    df["woy"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    df = df.merge(signals, on="week_start", how="left")   # add holiday/payday
    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")
    return df


FEATURES = ([f"lag_{L}" for L in LAGS] + ["roll_mean_4", "woy", "month",
            "holiday_days", "payday_days", "on_promo", "price"] + CAT_FEATURES)


def run():
    sales = load_weekly_sales()
    signals = load_signals()
    feat = build_features(sales, signals)

    weeks = np.sort(sales.week_start.unique())
    cutoff = weeks[-TEST_WEEKS]                     # first test week
    # train only on weeks before the cutoff, and only rows that HAVE their lags
    train = feat[feat.week_start < cutoff].dropna(subset=[f"lag_{L}" for L in LAGS])

    # conservative settings — small data, don't overfit
    model = lgb.LGBMRegressor(
        objective="regression_l1",   # optimise absolute error (robust to spikes)
        n_estimators=300, learning_rate=0.05,
        num_leaves=15,               # small trees
        min_child_samples=10,        # each leaf must cover >=10 rows
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, verbose=-1)
    model.fit(train[FEATURES], train["units"], categorical_feature=CAT_FEATURES)

    # ---- RECURSIVE multi-week forecast, per SKU ----
    sig_idx = signals.set_index("week_start")
    static = sales.groupby("sku").agg(category=("category", "first"),
                                      brand=("brand", "first"),
                                      price=("price", "first"))
    rows = []
    for sku, g in sales.groupby("sku"):
        s = g.sort_values("week_start")
        hist = list(s[s.week_start < cutoff]["units"].astype(float))  # known past
        promo_map = dict(zip(s.week_start, s.on_promo))
        test = s[s.week_start >= cutoff]
        for wk, yt in zip(test.week_start, test.units):
            feat_row = {
                **{f"lag_{L}": hist[-L] for L in LAGS},   # lags from history
                "roll_mean_4": np.mean(hist[-4:]),
                "woy": int(pd.Timestamp(wk).isocalendar().week),
                "month": pd.Timestamp(wk).month,
                "holiday_days": int(sig_idx.loc[wk, "holiday_days"]),
                "payday_days": int(sig_idx.loc[wk, "payday_days"]),
                "on_promo": int(promo_map.get(wk, 0)),
                "price": float(static.loc[sku, "price"]),
                "category": static.loc[sku, "category"],
                "brand": static.loc[sku, "brand"],
            }
            X = pd.DataFrame([feat_row])
            for c in CAT_FEATURES:
                X[c] = X[c].astype("category")
            yp = max(0.0, round(float(model.predict(X[FEATURES])[0])))
            rows.append({"sku": sku, "week_start": wk, "y_true": yt, "y_pred": yp})
            hist.append(yp)          # <-- feed the prediction back in for next week

    preds = pd.DataFrame(rows)
    m = score_model(preds, sales, "lgbm_global")
    print(m[m.sku == "ALL"][["mae", "rmse", "mase"]].round(3).to_string(index=False))
    # which features mattered — good for the report
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\ntop features:\n", imp.head(6).to_string())


if __name__ == "__main__":
    run()
```

Run it:
```bash
python src/lgbm_global.py
```

Expected shape (numbers vary with real data):
```
 mae  rmse  mase
9.14 10.36 0.888

top features:
 roll_mean_4    740
lag_4          593
lag_1          573
...
```

The feature-importance list is a bonus deliverable: it tells the buying team *what
actually drives demand* in the model (recent sales? promos? payday?), which is
great for the explainability story.

---

## 6. How to evaluate and read your results

`score_model()` writes `outputs/metrics_lgbm_global.csv` (per-SKU + `ALL`) and
`outputs/preds_lgbm_global.csv`. What to check:
1. **`ALL` MASE** vs Aiman's baseline bar and Khizer's Holt number.
2. **Feature importances** — do they make sense? (Lags and rolling mean on top is
   healthy. If `price` or `woy` dominates on 6 months of data, be suspicious.)
3. **Per-SKU spread** — where does the global model shine vs struggle?
4. **Overfitting check** — if training error is tiny but test MASE is bad, reduce
   `n_estimators` / `num_leaves` further. Small data punishes greedy models.

### Stretch goal — prediction intervals
The other docs output a single number. You can output a **range** cheaply: train
two extra LightGBM models with `objective="quantile"` and `alpha=0.1` / `alpha=0.9`
to get a lower and upper bound. That gives the "likely 40–60" range the buying
team actually wants. Do this only after the point forecast beats the baselines.

---

## 7. Pitfalls to avoid (the ML-specific traps)

- **Leakage via lags — the #1 danger.** A feature must only use information
  available *before* the week being predicted. `groupby.shift(L)` and feeding
  predictions back recursively both exist to prevent this. Never build `lag_1` from
  the same week's units, and never let a test week's row see a real future value.
- **Leakage via the split.** Train **only** on weeks before the cutoff. If you
  train on all weeks and test on the last 5, you've trained on the answers.
- **Categoricals as text.** Pass `category`/`brand` as pandas `category` dtype and
  via `categorical_feature=...`. If you leave them as plain strings LightGBM errors;
  if you one-hot them by hand you lose the native handling. Use the dtype.
- **Overfitting tiny data.** Resist raising `num_leaves`/`n_estimators` to chase a
  better fit. Test MASE is the only score that matters.
- **Assuming ML wins.** On small or near-random data, simple methods often beat
  LightGBM (you saw this in the pilot dry-run). Report the honest comparison. If
  Holt or `ma4` wins, say so — that's a real result about your data.
- **Forgetting to clip/round.** `max(0.0, round(...))` — no negative or fractional units.

---

## 8. Definition of done (your checklist)

- [ ] `src/lgbm_global.py` runs, prints `ALL` metrics + feature importances.
- [ ] `outputs/metrics_lgbm_global.csv` and `preds_lgbm_global.csv` exist.
- [ ] You can state your `ALL` MASE vs the baseline bar AND vs Khizer's Holt.
- [ ] You've confirmed no leakage (lags shifted, trained only on pre-cutoff weeks).
- [ ] Feature importances make intuitive sense.
- [ ] (Stretch) prediction-interval version produces sensible lower/upper bounds.

---

## 9. Handoff

Report your `ALL` MASE, your per-SKU metrics, and your feature-importance list.
Your model is the team's "how far can ML push it" answer. The three-way comparison
— baseline vs Holt vs LightGBM, all scored identically — is the pilot's headline
result. Whichever wins, you'll be able to say *why*, backed by numbers everyone
computed the same way.
