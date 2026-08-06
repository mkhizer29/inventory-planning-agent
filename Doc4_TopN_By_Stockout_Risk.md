# Top-N by Stockout Risk

Adds a second way to choose which products a forecast run covers. Until now a run
always took the **Top N by units sold**; it can now take the **Top N by stockout
risk** instead — the products most likely to run out, rather than the ones that
sell the most.

Nothing about the existing behaviour changes. `units` remains the default, every
run created before this branch reads back as units-ranked, and the forecasting
models, evaluation and decision layers are untouched.

---

## 1. Why this needed a proxy

The obvious implementation — "rank by stockout risk, then forecast the top N" —
does not work, because of a circular dependency.

The authoritative stockout risk is **Phase B** (`src/stockout_risk.py`). It is
*forecast-driven*: it reads `runs/<run_id>/selected_forecasts.parquet`, so it only
exists for SKUs that have **already been selected and forecast**. Selection happens
first, so it cannot consume Phase B's output.

```
selection  ->  prepare  ->  train + forecast  ->  Phase B risk  ->  Phase C reorder
    ^                                                  |
    |__________ cannot read this ______________________|
```

So selection uses a **pre-forecast proxy** (`src/selection_risk.py`) that reproduces
Phase B's arithmetic with a naive flat forecast substituted for the model forecast:

```
mean_daily   = mean daily ecommerce units over the trailing window (default 28d)
sigma_daily  = std  daily ecommerce units over the same window
lt_mean      = mean_daily  * lead_time_days
lt_sigma     = sigma_daily * sqrt(lead_time_days)     # independent daily errors (RSS)
inventory    = stock_on_hand from the warehouse snapshot (on-order excluded)
P(stockout)  = 1 - Phi((inventory - lt_mean) / lt_sigma)
```

This is not an invented heuristic. `historical_demand_std` is already Phase B's own
documented **Method-3** uncertainty fallback (see `stockout_risk._daily_sigma`), so
the proxy walks a path the codebase already trusts.

**The proxy only ORDERS candidates.** Phase B still computes the authoritative
per-SKU risk after the models run, and the two can legitimately disagree — the
dashboard says so explicitly before you launch.

### Measured agreement

On a real Groceries & Pets Top-10 run (`as_of 2026-07-15`), proxy tier vs the Phase B
tier computed afterwards:

| | proxy P | real P | proxy tier | real tier |
|---|---|---|---|---|
| IC-1015335 | 1.000 | 1.000 | critical | critical |
| IC-1200458 | 1.000 | 0.856 | critical | critical |
| IC-1119334 | 1.000 | 0.804 | critical | critical |

**10/10 tier agreement.** The proxy is optimistic on absolute probability — a flat
forecast overstates depletion versus a fitted model — but it picked the right
products.

---

## 2. The ranking key

```
round(stockout_probability, 6) DESC, expected_shortage_units DESC, sku ASC
```

**The rounding is the important part**, and it was not obvious up front.

Out-of-stock SKUs dominate the top of this ranking — 945 of 2,289 eligible
Groceries & Pets SKUs hold zero stock — and their probabilities *saturate*. They are
not exactly 1.0 (that only happens when demand is perfectly flat and `lt_sigma == 0`),
but they agree to roughly 13 decimal places:

```
IC-1015335   P = 0.9999999999859499   exposure  48.25
IC-1144527   P = 0.9999999998923177   exposure  47.75
IC-1186715   P = 0.9999999981593075   exposure 139.25
```

Sorting the raw float made those digits decide the order. They are numerical noise
from the far tail of the normal CDF, not a business distinction — every one means
"certain to stock out". Worse, the noise is driven by the **coefficient of variation**
(`z = −√L · mean/sigma`), so a *small steady* seller outranks a *large erratic* one.
On real data IC-1186715, exposed for **139 units**, sat fourth behind SKUs exposed for
47–63.

Rounding to 6 decimals collapses the saturated block into a genuine tie, so
`expected_shortage_units` orders it by the size of the exposure — out of stock **and**
selling fastest first:

```
IC-1186715  139.25    IC-1177119   90.50    IC-1030208   89.75    IC-1200458  63.75
```

Probabilities that differ meaningfully still decide the order before exposure is
consulted. Only the sort key is rounded; the reported `stockout_probability` keeps
full precision.

Unscored SKUs sort last and can never occupy a Top-N slot.

> **Testing note.** The original test used flat demand, which produces exact 1.0 ties,
> so it passed while the realistic case was broken. `test_saturated_probabilities_rank_by_exposure`
> now covers varying demand — the case that actually occurs — and
> `test_meaningful_probability_gap_still_beats_exposure` guards against over-rounding.

---

## 3. Eligibility is unchanged — and shared

Both metrics rank the **same candidate set**. Eligibility is still decided once, by
`dynamic_selection.list_eligible_skus`: ecommerce channels only, on/before the
cutoff, exact category match, `Free*`/`PACK*` excluded, and at least
`min_history_days` distinct active dates.

The risk metric scores the **full eligible pool**, never a units-ranked shortlist.
That distinction matters: trimming to the top 100 by units first would silently hide
every at-risk product that is not also a bestseller. `test_risk_metric_scores_full_pool_not_a_units_shortlist`
locks this in.

Ranking the two ways gives genuinely different runs — on Groceries & Pets Top-10 the
two sets had **zero overlap**, with the risk picks spanning 398–1,988 historical
units against the bestsellers' 3,520–11,851.

---

## 4. Known limitation: post-cutoff stock

**Read this before using risk-ranked runs for backtesting.**

The warehouse holds **one** inventory snapshot and no stock history, so a SKU's stock
on an arbitrary `selection_cutoff` cannot be reconstructed. Under the default
`stock_snapshot_policy: latest` the newest snapshot is used **even when it postdates
the cutoff**, so post-cutoff information influences which SKUs get selected. That
breaks the as-of purity `dynamic_selection` guarantees for the `units` metric.

This is a deliberate, **recorded** trade-off rather than a silent one:

- `pipeline.log` logs it at **WARNING** level
- `request.json` / `run_manifest.json` carry `stock_snapshot_date` and `stock_is_post_cutoff`
- the launch form warns before you click Generate
- `select_top_skus_detailed` returns it in warnings

Set `stock_snapshot_policy: on_or_before_cutoff` for strict purity — the scan then
yields no rows when every snapshot is newer, which is the honest outcome.

Related: stock in this pilot is partly synthetic (`schema 4.0-real-demand-synthetic-stock`),
but the **ranking** reads the real `inventory_snapshot`, the same source Deadstock
uses — never the synthetic reconstruction.

---

## 5. Configuration

```yaml
selection_risk:
  enabled: true
  demand_window_days: 28          # trailing days ending at selection_cutoff (>= 2).
                                  # Days with no sale count as zero demand, so the
                                  # mean divides by the window, not by active days.
  stock_snapshot_policy: latest   # latest | on_or_before_cutoff
  include_zero_stock: true        # already-out SKUs lead the ranking when true
  exclude_dropship: false
```

Risk tiers reuse `decisioning.probability_thresholds`, so proxy and Phase B tier
vocabulary stay consistent.

---

## 6. Usage

**Dashboard** — Forecast Runs → Generate Forecast → **Rank Top N by**:
*Units sold* / *Stockout risk*. Risk-ranked runs are marked `⚡` in run labels and
have a "Ranked by" column in Run History.

**Orchestrator**

```bash
python src/forecast_orchestrator.py \
  --category "Groceries & Pets" --top-n 10 \
  --as-of-date 2026-07-15 --selection-cutoff 2026-07-15 \
  --horizons 7 14 --ranking-metric stockout_risk
```

**Score without running a forecast** (read-only, useful for inspection)

```bash
python src/selection_risk.py --category "Groceries & Pets" \
  --selection-cutoff 2026-07-15 --top-n 20
```

---

## 7. Also in this branch: the as-of extract-tail guard

Separate fix, same branch, because it blocks any run — not just risk-ranked ones.

`get_latest_sales_date()` used to return plain `MAX(transaction_date)`. The warehouse
extract can stop part-way through a day, leaving trailing dates holding a handful of
stray rows — on the current DB, sales run to `2026-07-31` but the last real trading
day is **`2026-07-23`**:

```
2026-07-23    14,613 units   5,373 SKUs   <- last real day
2026-07-24       162 units     135 SKUs
2026-07-25/26/27   (no rows)
2026-07-28         3 units
2026-07-30         2 units
2026-07-31         2 units
```

The dashboard defaulted As-of to `07-31`, which puts the horizon-7 locked holdout
(`07-25 → 07-31`) in the empty tail. Zero actual units means WAPE's denominator is
zero, so it is `NaN` for every model and the run dies at
`no finite-WAPE ranking candidate for horizon 7`. This already caused a real
**units-ranked** run to fail before this branch existed.

`get_latest_sales_date()` now discounts trailing days below
`sales_calendar.min_share_of_median_daily_units` (default 10%) of the median daily
volume, with a 1-unit floor so genuinely low-volume warehouses are untouched. It
returns `2026-07-23`, and the page explains what was discounted rather than shifting
the default silently. `sales_date_diagnostics()` exposes `raw_max`, `usable_max`,
`ignored_dates`, `median_daily_units`, `threshold`.

---

## 8. Files

| File | Change |
|---|---|
| `src/selection_risk.py` | **New.** Read-only scorer, ranking, CLI |
| `src/dynamic_selection.py` | `stockout_risk` metric; `list_eligible_skus`, `select_top_skus_detailed`; contract docstring rewritten |
| `src/forecast_orchestrator.py` | `--ranking-metric` → `request.json` → `run_manifest.json` |
| `dashboard/run_service.py` | Metric plumbing, run labels, extract-tail guard |
| `dashboard/app.py` | Rank-by control, disclosure banners, "Ranked by" column |
| `inventory_etl/config/config.yaml` | `selection_risk:` and `sales_calendar:` blocks |
| `inventory_etl/tests/test_selection_risk.py` | **New**, 34 tests |
| `inventory_etl/tests/test_dashboard_runs.py` | +14 tests |

### Tests

```bash
python -m pytest inventory_etl/tests/test_selection_risk.py -q     # 34 passed
python -m pytest inventory_etl/tests/test_dynamic_selection.py -q  # 40 passed
python -m pytest inventory_etl/tests/test_dashboard_runs.py -q     # 133 passed
python -m pytest inventory_etl/tests/test_forecast_orchestrator.py -q
```

`test_paths.py::test_repo_and_etl_roots_resolve` fails on any clone whose directory
is not named `Inventory-Planning-Agent`. Pre-existing and unrelated to this branch.
