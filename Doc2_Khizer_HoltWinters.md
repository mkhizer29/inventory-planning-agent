# Pilot Forecasting — Build Document 2 of 3
## Owner: Khizer · Exponential Smoothing (Holt / Holt-Winters family)

> Your mission: build a proper **classical time-series** forecaster for each of
> the 30 products — one that learns the level and trend of each product's demand
> and projects it 5 weeks forward. It must beat Aiman's best baseline.

---

## Getting started

New to the repo? Follow **[TEAMMATE_SETUP.md](TEAMMATE_SETUP.md)** (repo root) for
cloning, the Python environment, regenerating the data, and the branch workflow.
**Work on branch `model/holtwinters`.** Import the shared scorecard from
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
> - **Feature whitelist:** `units_lag_1/7/14`, `units_roll_mean_7/28`, `units_roll_std_7`, price/discount/promo, calendar (incl. Ramazan: `is_ramadan`/`ramadan_day`/`ramadan_week`). **Never** use `stock_on_hand` or `unit_cost` as a demand feature.
> - **Row eligibility = `forecast_training_eligible`** (real-data quality + ≥14 days history) — independent of synthetic stock; `evaluate()` asserts this at runtime.
> - Lead time / MOQ / pack size are **pilot assumptions** (real per-SKU values override defaults). Unit cost is validated in `inventory_context.parquet`; currency PKR and cost unit/pack **basis await Naheed confirmation**.
> - **Three separate stages** (see TEAMMATE_SETUP → "Model architecture"): **A** real demand forecast → **B** forecast-driven stockout risk → **C** reorder recommendation. B and C use synthetic stock, so they are **pilot estimates, not validated against real Naheed stockouts**.
> - **Removed:** the old `data/synthetic/` scenario outputs (stockout_scenarios / replenishment_events / simulation_parameters) — do not read them.
- **Channel**: ecommerce only → **`naheed_web`** (physical `store` excluded).
- **Horizons**: forecast **7 and 14 daily** steps. Splits are **chronological** — `TEST_WEEKS` is gone.
- **Scoring**: `from evaluation import evaluate` → `evaluate(preds, horizon=7 or 14)`.
  Submit **`sku, channel, date, y_pred`** only — **never pass `y_true`**. Optional `lower_bound`/`upper_bound`.
  Reported: **WAPE, MAE, MASE, RMSE, bias** + interval coverage.
- Supplier **lead time & MOQ are downstream configurable assumptions**, **not forecasting features**.

**Your model (daily exponential smoothing).** Fit per SKU on the chronological training
slice; forecast 7 and 14 days ahead. Evaluate **SES → Holt (trend) → damped Holt**, and
**Holt-Winters with weekly seasonality (m=7) only if there are enough repeated 7-day cycles**
(≈25 weeks of daily data here is borderline — justify if you use it). **Additive only**
(zeros are present). Submit `preds[sku, channel, date, y_pred]` (no `y_true`).

---

## 0. How your piece fits the team

Three of us build different methods on the **same 30 products** and the **same
data**, then compare:

| Person | Method | Document |
|---|---|---|
| Aiman | Baselines + scorecard | Doc 1 |
| **Khizer (you)** | Exponential smoothing | This one |
| Aqib | Global LightGBM | Doc 3 |

You import Aiman's `src/evaluation.py` for splitting and scoring. Do **not** write
your own metric — comparability across the three of us depends on everyone using
that one file. Your target is simple: **MASE below the best baseline MASE** Aiman
gives you.

---

## 1. What you're building, in plain words

Exponential smoothing forecasts the future by taking a **weighted average of the
past, where recent weeks count more than old ones.** The "how much more" is
controlled by a smoothing number the model learns automatically. There are three
levels of the method, adding one idea at a time:

1. **SES (Simple Exponential Smoothing)** — tracks the **level** (roughly "what's
   the current typical sales rate"). No trend, no seasonality.
2. **Holt's method** — adds a **trend** (is demand drifting up or down?).
3. **Holt-Winters** — adds **seasonality** (a repeating pattern, e.g. every
   December).

The name of your task is "Holt-Winters," but read §4 carefully — with only 6
months of data you actually **cannot** use the seasonal version honestly, and
knowing *why* is a big part of doing this right.

---

## 2. Setup (do this once)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pandas numpy pyarrow statsmodels
python -c "import statsmodels; print('ready')"
```

`statsmodels` is the library that contains the exponential smoothing models.

Project layout:
```
pilot/
├─ data/processed/          <- shared input files (see §3)
├─ src/
│  ├─ evaluation.py         <- Aiman's shared file — you import it, don't edit it
│  └─ holtwinters.py        <- YOU write this (full code in §5)
└─ outputs/
```

---

## 3. The data you will receive (the shared contract)

You do **not** extract data. The team ETL pipeline produces two files that are
identical for all three of us. Sales are already aggregated to **weekly** buckets
and weeks with no sale are filled with **zero** (keep the zeros — a zero means
"nobody bought it," which the model needs to see).

**File 1 — `data/processed/weekly_sales.parquet`** (one row per SKU per week):

| column | type | meaning |
|---|---|---|
| `sku` | text | product code |
| `category` | text | one of the 2 pilot categories |
| `brand` | text | brand name |
| `price` | number | unit price |
| `week_start` | date | Monday of the week |
| `units` | integer | units sold that week (0 if none) |
| `on_promo` | 0/1 | on promotion that week |

**File 2 — `data/processed/weekly_signals.parquet`** — holiday/payday counts per
week. Classical exponential smoothing can't easily use these extra signals, so
for your model you mainly need File 1 (the `units` series per SKU). That's fine —
Aqib's model in Doc 3 is the one that uses the signals.

### The split (same for everyone)
Per SKU: **train = all weeks except the last 5; test = the last 5 weeks.**
Chronological only. This lives in `evaluation.py` as `TEST_WEEKS = 5`.

### The metric (same for everyone)
**MASE is primary.** MASE < 1 = you beat the naive forecast; > 1 = worse than
doing nothing. You also report MAE (average error in units) and RMSE. All three
come from Aiman's `score_model()`.

---

## 4. The method explained — and the honest seasonality caveat

### The three smoothing parameters
- **alpha (α)** controls the **level** — how fast the model forgets old sales.
  High α = very reactive to the latest week; low α = smooth and slow.
- **beta (β)** controls the **trend** — how fast it updates the up/down drift.
- **gamma (γ)** controls **seasonality** — only exists in full Holt-Winters.

`statsmodels` estimates α, β, (γ) for you by fitting to each product's history.
You don't set them by hand.

### Why you must NOT use seasonal Holt-Winters here (important)
Seasonality means a **repeating cycle**. To *learn* a cycle, the model needs to
see it happen **at least twice**. Our data is ~26 weekly points over 6 months.
- A **yearly** cycle (52 weeks) — we haven't seen even one full year. Impossible.
- Even a **monthly-ish** cycle — we'd have only a handful of repeats, too few to
  trust.

So the correct, defensible choice is **Holt's method (level + trend, no
seasonality).** If you force the seasonal version, `statsmodels` will either error
out or fit noise and produce nonsense. Writing in your report "we used Holt's
trend method because 6 months is too short for seasonal Holt-Winters, which needs
≥2 full cycles" is exactly the kind of judgment that reads as senior-level.

*(Alternative if the team later switches to DAILY series: with daily data you get
a 7-day weekly cycle repeated ~26 times, so seasonal Holt-Winters with a period
of 7 becomes valid. For the weekly pilot, stick with Holt's trend method.)*

### The zero problem
Exponential smoothing comes in **additive** and **multiplicative** flavours.
Multiplicative **breaks on zeros** (it multiplies, and zeros/negatives are
undefined). Our data is full of zero-sales weeks. So always use **additive**
trend, never multiplicative. The code in §5 does this and also falls back to SES
if a fit fails on a very sparse SKU.

---

## 5. Build it — `src/holtwinters.py`

Complete, tested file:

```python
"""holtwinters.py — Khizer. Exponential smoothing (Holt's trend method).
Seasonal Holt-Winters is NOT used: <2 seasonal cycles in 6 months. Falls back
to Simple Exponential Smoothing if a fit fails on a sparse SKU."""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
import sys; sys.path.append("src")
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from evaluation import load_weekly_sales, TEST_WEEKS, score_model
warnings.filterwarnings("ignore")   # statsmodels is chatty on short series


def forecast_es(train: pd.Series, h: int) -> np.ndarray:
    """Fit Holt's additive trend (damped) and forecast h weeks.
    'damped' stops the trend line running away to silly numbers far out."""
    y = train.astype(float).reset_index(drop=True)
    try:
        fit = ExponentialSmoothing(
            y,
            trend="add",            # additive trend (safe with zeros)
            damped_trend=True,      # gently flatten the trend over the horizon
            seasonal=None,          # NO seasonality — see Doc §4
            initialization_method="estimated",
        ).fit()
        fc = fit.forecast(h)
    except Exception:
        # very sparse/short SKU: drop the trend, just smooth the level
        fit = SimpleExpSmoothing(y, initialization_method="estimated").fit()
        fc = fit.forecast(h)
    # clean up: no negative sales, whole units only
    return np.clip(np.round(np.asarray(fc)), 0, None)


def run():
    sales = load_weekly_sales()
    rows = []
    for sku, g in sales.groupby("sku"):          # fit ONE model per product
        s = g.sort_values("week_start")
        train = s["units"].iloc[:-TEST_WEEKS]     # all but last 5 weeks
        test = s.iloc[-TEST_WEEKS:]               # last 5 weeks = the truth
        yhat = forecast_es(train, TEST_WEEKS)
        for (wk, yt), yp in zip(zip(test.week_start, test.units), yhat):
            rows.append({"sku": sku, "week_start": wk,
                         "y_true": yt, "y_pred": yp})
    preds = pd.DataFrame(rows)
    m = score_model(preds, sales, "holtwinters")   # writes outputs, returns metrics
    print(m[m.sku == "ALL"][["mae", "rmse", "mase"]].round(3).to_string(index=False))


if __name__ == "__main__":
    run()
```

Run it:
```bash
python src/holtwinters.py
```

Expected shape (numbers vary with real data):
```
  mae  rmse  mase
8.55  9.76  0.849
```

MASE 0.849 < 1 means it beats naive. Now compare that number to Aiman's best
baseline — if yours is lower, your model earns its place; if not, that's a real
(and reportable) finding that classical smoothing doesn't add value on these SKUs.

---

## 6. How to evaluate and read your results

`score_model()` writes two files:
- `outputs/metrics_holtwinters.csv` — MAE/RMSE/MASE per SKU, plus an `ALL` average row.
- `outputs/preds_holtwinters.csv` — every prediction vs the actual, so you can plot.

What to actually look at:
1. **The `ALL` MASE** — your headline number vs the baseline bar.
2. **Per-SKU MASE spread** — sort the per-SKU file. Which products does the model
   do well on (steady sellers) vs badly on (lumpy ones)? This tells the story of
   *where* classical smoothing works, which is great material for the writeup.
3. **A plot for 2–3 SKUs** — the last-5-weeks actuals vs your forecast. Seeing it
   is more convincing than any table.

---

## 7. Pitfalls to avoid

- **Never use `trend="mul"` or `seasonal="mul"`.** Multiplicative dies on zeros,
  and our data is full of them. Additive only.
- **Never turn seasonality on** for the weekly pilot — you don't have 2 cycles.
  If you're curious, try it on one SKU and watch it fail; then leave it off.
- **Don't panic at convergence warnings** on short/sparse series — that's why the
  code wraps the fit in try/except and falls back to SES. As long as it produces a
  forecast, you're fine.
- **Don't fit one model for all products.** Exponential smoothing is per-SKU: each
  product gets its own fitted α/β. (Aqib's LightGBM is the shared-across-all one.)
- **Clip and round** every forecast — no negative or fractional units.

---

## 8. Definition of done (your checklist)

- [ ] `src/holtwinters.py` runs and prints the `ALL` metrics row.
- [ ] `outputs/metrics_holtwinters.csv` and `preds_holtwinters.csv` exist.
- [ ] You can say your **`ALL` MASE** and whether it beats Aiman's baseline bar.
- [ ] You've plotted forecast vs actual for at least 2 SKUs.
- [ ] Your writeup states clearly **why seasonal Holt-Winters was not used** (short history).

---

## 9. Handoff

Report your `ALL` MASE and your per-SKU metrics file to the team. In the joint
comparison, your model represents "the best a well-understood classical method can
do." Whether it wins or loses against Aqib's LightGBM, that contrast is one of the
most interesting results the pilot will produce.
