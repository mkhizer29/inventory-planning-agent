# Pilot Forecasting — Build Document 2 of 3
## Owner: Khizer · Exponential Smoothing (ETS / Holt-Winters family)

> Your mission: build a proper **univariate classical time-series** forecaster for each of
> the 30 pilot products — one that learns each product's level (and, where justified, trend
> and weekly seasonality) from its **real daily sales** and projects it **7 and 14 days**
> forward with prediction intervals. Compare it honestly to the seasonal-naive benchmark.

---

## Getting started

New to the repo? Follow **[TEAMMATE_SETUP.md](TEAMMATE_SETUP.md)** (repo root) for
cloning, the Python environment, and the branch workflow. **Work on branch
`model/holtwinters`.** Import the shared scorecard from `src/evaluation.py` — don't edit it.

---

## ⚠️ Contract (v4 — daily, ecommerce-only, real demand) — READ THIS FIRST

- **Inputs** (already generated in `data/processed/`, built by `src/prepare_pilot_data.py`):
  - `model_panel.parquet` — REAL daily demand (one row per `sku × naheed_web × date`; target = **`units_observed`**) plus a reconstructed **synthetic `stock_on_hand`** (inventory context only, never a demand feature).
  - `forecast_frame.parquet` — the next **14 future days** per SKU (leakage-safe: no actuals, no cost).

> **REAL demand, SYNTHETIC daily stock only:**
> - **Demand forecasting uses REAL sales.** `units_observed` is real Naheed naheed_web demand — never altered, capped, or replaced. There is **no synthetic demand, no synthetic sales, no scenarios, no lost sales**.
> - **ETS is UNIVARIATE.** It consumes only each SKU's own `units_observed` history. **No exogenous inputs** — no `stock_on_hand`, unit cost, lead time, MOQ, pack size, price, promo, holiday or payday columns enter a fit. (Those belong to Aqib's LightGBM.)
> - **Additive only.** Real zero-demand days exist (~22% of rows), so no multiplicative error/trend/seasonality and no log/Box-Cox — those need strictly positive data.
> - **Horizons 7 and 14; splits chronological.** `from evaluation import evaluate` → `evaluate(preds, horizon=7 or 14)`. Submit **`sku, channel, date, y_pred`** only — **never pass `y_true`** (the evaluator holds the truth and runs a synthetic-stock independence check).
> - Lead time / MOQ / pack size are downstream (Stage C) assumptions, **not forecasting features**.
> - **Three stages** (see TEAMMATE_SETUP → "Model architecture"): **A** real demand forecast (this doc) → **B** forecast-driven stockout risk → **C** reorder recommendation. Accuracy here is **backtest-estimated, not guaranteed**; the forecasts feed B and C. No supplier order is created.

---

## 0. How your piece fits the team

| Person | Method | Document |
|---|---|---|
| Aiman | Baselines + scorecard | Doc 1 |
| **Khizer (you)** | Exponential smoothing (ETS) | This one |
| Aqib | Global LightGBM | Doc 3 |

You import Aiman's `src/evaluation.py` for splitting and scoring so all three methods are
comparable. Your target is **MASE below the seasonal-naive benchmark** on the locked holdout.

---

## 1. What you're building, in plain words

Exponential smoothing forecasts the future as a **weighted average of the past, where
recent days count more than old ones.** `statsmodels` learns the weights per product. We
build a small family and add one idea at a time:

1. **SES** — level only ("current typical daily sales rate"). No trend, no seasonality.
2. **Holt** — adds a **trend** (drifting up/down); optionally **damped** so the trend flattens.
3. **Holt-Winters (m=7)** — adds a **weekly seasonal** shape (Mon-vs-Sun pattern).

Because we now have **daily** data (~22–24 weeks per SKU, i.e. 22+ repeats of the 7-day
cycle), weekly seasonality (period 7) **is** learnable — unlike the retired weekly pilot.
We still let the backtest decide per SKU whether seasonality actually helps.

---

## 2. The candidate family (`src/holtwinters.py`)

Six **additive** ETS structures (`statsmodels.tsa.exponential_smoothing.ets.ETSModel`,
`initialization_method="estimated"`; the library estimates α/β/γ/φ):

| model_id | error | trend | damped | seasonal | m |
|---|---|---|---|---|---|
| `ets_A_N_N`  | add | – | – | – | – |
| `ets_A_A_N`  | add | add | no | – | – |
| `ets_A_Ad_N` | add | add | yes | – | – |
| `ets_A_N_A7` | add | – | – | add | 7 |
| `ets_A_A_A7` | add | add | no | add | 7 |
| `ets_A_Ad_A7`| add | add | yes | add | 7 |

Point forecasts are floats with negatives clipped to 0 — **never rounded** here (rounding is
the Stage C reorder/pack-size concern).

---

## 3. Leakage-free model selection (backtest)

Expanding-window **rolling-origin** evaluation, per SKU:

- The evaluator's locked holdout (`max_date − h`) is **never** used for selection.
- For each horizon `h ∈ {7,14}`: selection origins at `locked_cutoff − {1,2,3}×h`.
- At each origin: train on `date ≤ origin`, validate `origin+1 … origin+h`. Every validation
  date is `≤ locked_cutoff`, so selection never touches the holdout.
- Metrics use `evaluation.py`'s exact definitions (MAE, RMSE, WAPE, bias, MASE). **MASE is
  primary** — MAPE is not used because demand contains zeros.

**Per-SKU rule:** lowest **mean finite MASE** across all selection folds/horizons; candidates
within **2% relative MASE** are tied and broken by (1) lower complexity, (2) lower WAPE,
(3) lower |bias|, (4) alphabetical id. A candidate must produce complete valid forecasts for
every required fold to be selected normally; relaxations (incomplete folds, or MAE-based when
MASE is undefined for a constant series) are recorded per SKU.

**Fallback (truthful, at fit time):** selected ETS → SES(A,N,N) → seasonal-naive(7) →
trailing-7-day mean. A fallback is used only if the model genuinely fails to fit/converge/
simulate, and the output records `model_actually_used`, `fit_status`, `fallback_used`,
`fallback_reason`, `converged`, and `warnings`. A fallback is **never** mislabelled as ETS.

---

## 4. Locked backtest, intervals, production forecast

- **Locked backtest:** after selection is frozen, each SKU is refit through the evaluator's
  `max_date − h` cutoff, the final `h` days are predicted, and `evaluate(preds, horizon=h)`
  scores them. A **same-day-last-week seasonal-naive (m=7)** benchmark is scored on the same
  windows (comparison only — it does not override a selected model).
- **Prediction intervals:** 80% and 95% from `ETSResults.simulate(anchor="end",
  repetitions=5000, random_state=<stable seed>)` (additive), negatives clipped, quantiles
  per step. Seeds are SHA-256 of `base_seed(2026)|sku|channel|origin|model_id` (never Python
  `hash()`). If simulation fails, a deterministic residual bootstrap is used and labelled
  `residual_bootstrap_fallback`. Order `0 ≤ lower_95 ≤ lower_80 ≤ point ≤ upper_80 ≤ upper_95`
  is enforced. Call these **prediction intervals, not guarantees**.
- **Production forecast:** each frozen model is refit on **all** valid history through
  `as_of_date` and forecasts exactly the `forecast_frame.parquet` keys (14 days × 30 SKUs).

---

## 5. Run it

```bash
python src/holtwinters.py             # writes outputs/
python src/holtwinters.py --selfcheck # also recompute in-process and assert identical
```

Outputs (generated, git-ignored):
- `outputs/holtwinters_backtest_metrics.csv` — selection-fold + locked + seasonal-naive metrics
- `outputs/holtwinters_backtest_predictions.parquet` — locked predictions + intervals + benchmark
- `outputs/demand_forecasts_holtwinters.parquet` — final 14-day production forecast
- `outputs/holtwinters_model_selection.json` — full audit: fingerprints, candidates, folds,
  selection per SKU, locked scorecards, coverage, fallbacks

The script fails loudly on any contract/leakage/output violation; it does not serialize models
(30 SKUs refit fast, and pickles are version-sensitive).

---

## 6. Results from the current run (backtest-ESTIMATED, not guaranteed)

Selected structure counts across 30 SKUs: **`ets_A_N_N` 20, `ets_A_A_N` 7, `ets_A_N_A7` 2,
`ets_A_A_A7` 1** — SES dominates because much of this demand is low/intermittent.

Locked scorecard (overall, via `evaluate`):

| horizon | ETS WAPE | ETS MASE | ETS MAE | sNaive WAPE | sNaive MASE | 80% cov | 95% cov |
|---|---|---|---|---|---|---|---|
| 7  | 0.677 | 0.787 | 8.89 | 1.035 | 1.074 | 0.857 | 0.929 |
| 14 | 0.657 | 0.898 | 9.56 | 0.821 | 1.158 | 0.843 | 0.938 |

ETS beats the seasonal-naive benchmark on **MASE and WAPE at both horizons**, and MASE < 1
(better than the in-sample naive). Interval coverage is close to nominal — 80% intervals
slightly conservative, 95% intervals slightly under (0.929 / 0.938 vs 0.95). Intervals are
estimates, not guarantees. Numbers regenerate deterministically (identical across runs).

---

## 7. Honest caveats

- Demand is **real** `units_observed`; the model is **univariate** per SKU; **no synthetic
  inventory** was used; **no future actual** entered an earlier fit.
- Accuracy is **estimated by backtesting**, not guaranteed. ~6 months of history means
  seasonal fits see ~22 weekly cycles (fine) but no annual pattern (out of scope).
- These forecasts feed **Stage B (stockout risk)** and **Stage C (reorder)** — which also use
  the *synthetic* inventory context and are therefore pilot estimates, **not** validated
  against real Naheed stockouts. No supplier order is created here.
