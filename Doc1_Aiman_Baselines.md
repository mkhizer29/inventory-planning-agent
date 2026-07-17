# Pilot Forecasting — Build Document 1 of 3
## Owner: Aiman · Baseline Models + the Shared Scorecard

> Your mission: build the **measuring stick**. Simple forecasts that every other
> model on the team must beat to justify its existence — plus the shared scoring
> code that Khizer and Aqib will both import. You are the keeper of "what counts
> as good." Nothing ships unless it beats your baselines.

---

## Getting started

New to the repo? Follow **[TEAMMATE_SETUP.md](TEAMMATE_SETUP.md)** (repo root) for
cloning, the Python environment, regenerating the data, and the branch workflow.
**Work on branch `model/baselines`.** You own `src/evaluation.py`, the locked shared
scorecard that Khizer and Aqib import — coordinate any change to it with the team.

---

## 0. How your piece fits the team

Three of us are each building a different forecasting method on the **same 30
products** and the **same data**, then comparing:

| Person | Method | Document |
|---|---|---|
| **Aiman (you)** | Baselines + scorecard | This one |
| Khizer | Exponential smoothing (Holt) | Doc 2 |
| Aqib | Global LightGBM | Doc 3 |

The only way three people's results are comparable is if we all: read the **same
input files**, use the **same train/test split**, and score with the **same
metric**. You own that last part. The file `src/evaluation.py` (in this document)
is the single source of truth for splitting and scoring. Khizer and Aqib import
it. If it changes, everyone's numbers change — so treat it as a locked contract
once we agree on it.

---

## 1. What you're building, in plain words

A "baseline" is a deliberately dumb forecast. Examples:
- **Naive:** next week = last week's sales.
- **Mean:** next week = the average of all past weeks.
- **Moving average:** next week = the average of the last 4 weeks.

These feel too simple to matter. They are the most important thing in the whole
project. Here's why: a fancy model that can't beat "same as last week" is
**worse than useless** — it's burning effort to be wrong. In our earlier attempt,
a model that always predicted zero looked like it "won" on a badly-chosen metric.
Baselines + the right metric are what stop that from happening again. You are
building the honesty layer.

---

## 2. Setup (do this once)

You need Python 3.11+ installed. Then, from the project folder:

```bash
# 1. Make an isolated environment so versions don't clash with your other work
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 2. Install exactly what the pilot needs
pip install pandas numpy pyarrow

# 3. Confirm it worked
python -c "import pandas, numpy; print('ready')"
```

Project layout you'll be working in:

```
pilot/
├─ data/processed/          <- the shared input files land here (see §3)
├─ src/
│  ├─ evaluation.py         <- YOU own this (full code in §6)
│  └─ baselines.py          <- YOU write this (full code in §5)
└─ outputs/                 <- your results are written here
```

---

## 3. The data you will receive (the shared contract)

Everyone gets the **same two files**, produced by the team's ETL pipeline. You do
NOT extract data yourself — you read these. The pipeline has already: pulled the
30 pilot SKUs, aggregated sales to **weekly** buckets, filled weeks with no sale
as **zero** (a zero is real information — "nobody bought it that week" — never
drop it), and attached the signal flags.

**File 1 — `data/processed/weekly_sales.parquet`** (one row per SKU per week):

| column | type | meaning |
|---|---|---|
| `sku` | text | product code, e.g. `IC-1196653` |
| `category` | text | one of the 2 pilot categories |
| `brand` | text | brand name |
| `price` | number | unit price |
| `week_start` | date | Monday of that week |
| `units` | integer | units sold that week (0 if none) |
| `on_promo` | 0/1 | was the SKU on promotion that week |

With 30 SKUs × ~26 weeks that's ~780 rows. Small enough to open and eyeball.

**File 2 — `data/processed/weekly_signals.parquet`** (one row per week):

| column | type | meaning |
|---|---|---|
| `week_start` | date | Monday of that week |
| `holiday_days` | integer | how many holiday days fell in that week |
| `payday_days` | integer | how many payday days fell in that week |

You (Aiman) mostly need File 1. Khizer and Aqib use both.

### The split everyone uses
For **each** SKU, sort weeks oldest→newest, then:
- **Train** = all weeks except the last 5.
- **Test** = the **last 5 weeks**.

Always split by **time**, never randomly. A random split would let a model "see"
future weeks while training — that's cheating and it inflates scores.
`TEST_WEEKS = 5` lives in `evaluation.py` so all three of us hold out the same weeks.

### The metrics everyone uses
- **MAE** — average size of the error, in units. Easy to explain to the buying team.
- **RMSE** — like MAE but punishes big misses harder.
- **MASE** — *the primary metric.* It divides your error by the naive forecast's
  error. So: **MASE < 1 means you beat naive; MASE > 1 means you're worse than
  doing nothing.** This is the number that cannot be fooled by predicting flat
  zero — which is precisely why we use it instead of raw percentage error.

---

## 4. The methods explained

Each baseline takes the training weeks and produces a forecast for the 5 test
weeks.

**Naive (last value).** Forecast every future week as the last observed week.
Reasoning: for many products, "recent" is the best guess. It's the reference
point MASE is built on.

**Historical mean.** Forecast every future week as the average of all training
weeks. Good when a product sells at a roughly stable rate with no trend.

**Moving average (4-week).** Forecast as the average of the last 4 weeks. A middle
ground: responds to recent changes but smooths out one-off spikes.

There is no "training" in the machine-learning sense here — you're just computing
simple summaries. That simplicity is the point.

---

## 5. Build it — `src/baselines.py`

This is the complete, tested file. Read the comments; they explain each move.

```python
"""baselines.py — Aiman. Naive, moving averages, mean. The yardstick."""
from __future__ import annotations
import numpy as np, pandas as pd
import sys; sys.path.append("src")
from evaluation import load_weekly_sales, TEST_WEEKS, score_model


# --- the three baseline forecasters ---------------------------------------
# Each takes the training series `train` and a horizon `h` (=5 weeks) and
# returns h numbers. They repeat one value across all 5 weeks — baselines
# don't try to be clever about *which* future week it is.
def forecast_naive(train, h):   return np.repeat(train.iloc[-1], h)   # last week
def forecast_mean(train, h):    return np.repeat(train.mean(), h)     # all-history avg
def forecast_ma(train, h, w=4): return np.repeat(train.iloc[-w:].mean(), h)  # last-4 avg


BASELINES = {
    "naive_last": lambda tr, h: forecast_naive(tr, h),
    "mean":       lambda tr, h: forecast_mean(tr, h),
    "ma4":        lambda tr, h: forecast_ma(tr, h, 4),
}


def run():
    sales = load_weekly_sales()          # shared loader from evaluation.py
    results = {}
    for name, fn in BASELINES.items():
        rows = []
        for sku, g in sales.groupby("sku"):         # one SKU at a time
            s = g.sort_values("week_start")
            train = s["units"].iloc[:-TEST_WEEKS]    # all but last 5 weeks
            test = s.iloc[-TEST_WEEKS:]              # last 5 weeks (the truth)
            # forecast, then clean it: no negative sales, whole units only
            yhat = np.clip(np.round(fn(train, TEST_WEEKS)), 0, None)
            for (wk, yt), yp in zip(zip(test.week_start, test.units), yhat):
                rows.append({"sku": sku, "week_start": wk,
                             "y_true": yt, "y_pred": yp})
        preds = pd.DataFrame(rows)
        m = score_model(preds, sales, name)          # writes outputs, returns metrics
        results[name] = m[m.sku == "ALL"].iloc[0]    # the averaged row
    # print a tidy comparison of all baselines
    print(pd.DataFrame(results).T[["mae", "rmse", "mase"]].round(3))


if __name__ == "__main__":
    run()
```

Run it:
```bash
python src/baselines.py
```

Expected shape of output (numbers vary with real data):
```
              mae     rmse    mase
naive_last  10.45   12.59   1.036
mean         8.88   10.11   0.868
ma4          8.68   10.01   0.855
```

Read that as: on this data, `ma4` beats naive (MASE 0.855 < 1) and is the
baseline to beat. Whichever baseline has the **lowest MASE** becomes the team's
official bar — Khizer and Aqib must come in under it.

---

## 6. The shared scorecard — `src/evaluation.py` (you own this)

This is the file Khizer and Aqib import. It is identical for all three of us. Put
it in `src/evaluation.py` and don't let it drift.

```python
"""evaluation.py — the SHARED scorecard. All three models import from here so
every result is measured identically. Owned by Aiman."""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

TEST_WEEKS = 5                       # hold out the last 5 weeks of every SKU
PROC = Path("data/processed")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)


def load_weekly_sales() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "weekly_sales.parquet")
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df.sort_values(["sku", "week_start"]).reset_index(drop=True)


def load_signals() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "weekly_signals.parquet")
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df.sort_values("week_start").reset_index(drop=True)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))

def mase(y_true, y_pred, y_train) -> float:
    """Error scaled by the naive one-step error on the TRAINING series.
    < 1 beats naive, > 1 is worse than naive. Cannot be gamed by flat zero."""
    y_train = np.asarray(y_train, dtype=float)
    denom = np.mean(np.abs(np.diff(y_train))) if len(y_train) > 1 else np.nan
    if not denom or np.isnan(denom):     # a flat/degenerate training series
        return np.nan
    return mae(y_true, y_pred) / denom


def score_model(preds: pd.DataFrame, sales: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """preds columns: [sku, week_start, y_true, y_pred]. Returns per-SKU + an
    'ALL' average row; writes outputs/metrics_<model>.csv and preds_<model>.csv."""
    rows = []
    for sku, g in preds.groupby("sku"):
        train = (sales.loc[sales.sku == sku]
                 .sort_values("week_start")["units"].iloc[:-TEST_WEEKS])
        rows.append({"sku": sku,
                     "mae": mae(g.y_true, g.y_pred),
                     "rmse": rmse(g.y_true, g.y_pred),
                     "mase": mase(g.y_true, g.y_pred, train)})
    m = pd.DataFrame(rows)
    overall = {"sku": "ALL", "mae": m.mae.mean(), "rmse": m.rmse.mean(),
               "mase": m.mase.dropna().mean()}
    m = pd.concat([m, pd.DataFrame([overall])], ignore_index=True)
    preds.to_csv(OUT / f"preds_{model_name}.csv", index=False)
    m.to_csv(OUT / f"metrics_{model_name}.csv", index=False)
    return m
```

**Why MASE's denominator is the naive error:** MASE literally measures "how much
better than naive are you." A model predicting flat zero gets a *huge* MASE on any
SKU that actually sells — the metric refuses to reward it. That's the safeguard.

---

## 7. Pitfalls to avoid

- **Don't drop zero-sales weeks.** A zero is a real demand observation. Dropping
  them silently turns "sold nothing for 3 weeks" into "no data," which lies to
  every model. The pipeline already zero-fills; don't undo it.
- **Don't let forecasts go negative or fractional.** `np.clip(..., 0, None)` and
  `np.round(...)` handle this. You can't sell −3 or 2.4 units.
- **Don't change `TEST_WEEKS` quietly.** If you change the split, tell Khizer and
  Aqib — their numbers become incomparable to yours otherwise.
- **A SKU with a totally flat training series** gives MASE = NaN (division by
  zero). That's expected; `score_model` drops NaNs from the average. Note how many
  SKUs this affects — if it's many, the pilot picks were too sparse.

---

## 8. Definition of done (your checklist)

- [ ] `src/evaluation.py` written and agreed with Khizer + Aqib as the locked contract.
- [ ] `src/baselines.py` runs and prints the comparison table.
- [ ] `outputs/metrics_naive_last.csv`, `metrics_mean.csv`, `metrics_ma4.csv` exist.
- [ ] You can state, in one sentence, **which baseline is the bar to beat** and its MASE.
- [ ] You've eyeballed a few SKUs' predictions in `outputs/preds_*.csv` and they look sane.

---

## 9. Handoff

Give the team two things: (1) the locked `evaluation.py`, and (2) the **best
baseline MASE**. Everyone else's job is now defined as "get MASE below that
number." You've turned a vague goal ("forecast well") into a hard target.
