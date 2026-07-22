"""synthetic_inventory.py — CAUSAL, seeded synthetic inventory & stockout simulation.

Naheed keeps no real daily stock history and no historical stockout labels, so for the
pilot we simulate inventory on a SEPARATE latent demand series. Real `units_observed`
are never touched or used to drive same-day inventory. Everything here is synthetic and
is flagged as such by the caller.

Causality guarantees (see project brief §3–§7):
  * Demand parameters are estimated ONLY from real sales on/before a fixed calibration
    cut-off, with category then global fallbacks (recorded). No future/holdout data.
  * Latent demand for day t uses only calibration params + calendar/scenario info known
    for t (never units_observed[t] or later).
  * Inventory is processed strictly in ascending date order with an end-of-day snapshot:
        opening[t]   = ending[t-1] + arrivals[t]
        available[t] = max(0, opening[t])
        sales[t]     = min(latent_demand[t], available[t])
        lost[t]      = max(0, latent_demand[t] - available[t])
        ending[t]    = max(0, available[t] - sales[t])
    Orders are placed at end of day t and can only arrive at t + lead + supplier_delay.
  * Same (seed, config) -> identical output. Seed = global_seed ^ crc32(sku) ^ crc32(scenario).

Nothing here forces a stockout label; outcomes differ only through exogenous scenario
parameters (coverage, demand multiplier, lead time, supplier delay, MOQ, pack size).
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd


# ── demand calibration (past-only) ───────────────────────────────────────────────
def _series_stats(units: np.ndarray) -> tuple[float, float, int]:
    """Return (mean, variance, n) for a daily-units array."""
    u = np.asarray(units, dtype=float)
    if u.size == 0:
        return 0.0, 0.0, 0
    m = float(u.mean())
    v = float(u.var(ddof=1)) if u.size > 1 else m
    return m, v, int(u.size)


def _nb_size(mean: float, var: float, dispersion: float) -> float:
    """NB 'size' r (Gamma shape). Smaller r = more overdispersed. Falls back to the
    configured dispersion when the series is not overdispersed (var <= mean)."""
    if mean <= 0:
        return 1.0
    if var > mean:
        return float(np.clip(mean * mean / (var - mean), 0.1, 50.0))
    return float(np.clip(1.0 / max(dispersion, 1e-3), 0.1, 50.0))


def _weekday_shape(cal: pd.DataFrame) -> dict[int, float]:
    """Multiplicative weekday effect (mean-by-weekday / overall mean), known-at-time."""
    if cal.empty or cal["units"].mean() <= 0:
        return {d: 1.0 for d in range(7)}
    overall = cal["units"].mean()
    by = cal.groupby(cal["date"].dt.weekday)["units"].mean()
    return {d: float(np.clip(by.get(d, overall) / overall, 0.3, 3.0)) for d in range(7)}


def calibrate(real_daily: pd.DataFrame, sku_cat: dict[str, str],
              calibration_end: pd.Timestamp, cfg: dict) -> dict:
    """Estimate per-SKU demand params from real sales ON/BEFORE calibration_end only.
    SKU -> category -> global fallback, each recorded via `source`."""
    s = cfg["synthetic"]
    min_cal = int(s["min_calibration_days"])
    disp = float(s["negbin_dispersion"])
    cal = real_daily[real_daily["date"] <= calibration_end].copy()

    g_m, g_v, g_n = _series_stats(cal["units"].to_numpy())
    weekday = _weekday_shape(cal) if s.get("weekday_effects", True) else {d: 1.0 for d in range(7)}
    cat_stats: dict[str, tuple[float, float, int]] = {}
    if not cal.empty:
        cal_cat = cal.assign(category=cal["sku"].map(sku_cat))
        for cat, gc in cal_cat.groupby("category"):
            cat_stats[cat] = _series_stats(gc["units"].to_numpy())

    params: dict[str, dict] = {}
    for sku, cat in sku_cat.items():
        g = cal[cal["sku"] == sku]["units"].to_numpy()
        m, v, n = _series_stats(g)
        if n >= min_cal and m > 0:
            source = "sku"
        elif cat in cat_stats and cat_stats[cat][2] >= min_cal and cat_stats[cat][0] > 0:
            m, v, n = cat_stats[cat]
            source = "category_fallback"
        else:
            m, v, n = g_m, g_v, g_n
            source = "global_fallback"
        m = max(m, 0.05)
        params[sku] = {"lam": m, "std": math.sqrt(max(v, m)),
                       "nb_size": _nb_size(m, max(v, m), disp), "calib_source": source}
    return params


# ── one scenario for one SKU ──────────────────────────────────────────────────────
def _latent_demand(dates: pd.DatetimeIndex, p: dict, scenario: dict,
                   weekday: dict[int, float], cal: pd.DataFrame, cfg: dict,
                   rng: np.random.Generator) -> np.ndarray:
    """Causal latent demand via a Gamma-Poisson (Negative Binomial) mixture. Uses only
    calibration params + calendar/scenario info known for each date."""
    s = cfg["synthetic"]
    hol_mult, pay_mult = float(s["holiday_demand_mult"]), float(s["payday_demand_mult"])
    dmult = float(scenario["demand_multiplier"])
    r = p["nb_size"]
    cal_dates = cal  # DataFrame with date, is_public_holiday, is_payday_window (known-at-time)
    hol = set(cal_dates.loc[cal_dates["is_public_holiday"] == 1, "date"])
    pay = set(cal_dates.loc[cal_dates["is_payday_window"] == 1, "date"])
    out = np.zeros(len(dates), dtype=int)
    for i, d in enumerate(dates):
        m_t = p["lam"] * weekday.get(d.weekday(), 1.0) * dmult
        if d in hol:
            m_t *= hol_mult
        if d in pay:
            m_t *= pay_mult
        if m_t <= 0:
            continue
        lam = rng.gamma(shape=r, scale=m_t / r)   # Gamma-Poisson == Negative Binomial
        out[i] = int(rng.poisson(lam))
    return out


def _simulate_one(sku: str, product_id, channel: str, dates: pd.DatetimeIndex,
                  p: dict, scenario: dict, cfg: dict) -> tuple[list[dict], list[dict]]:
    """Chronological inventory simulation for one SKU x scenario. Returns (rows, events)."""
    s = cfg["synthetic"]
    seed = (int(s["seed"]) ^ zlib.crc32(str(sku).encode()) ^ zlib.crc32(str(scenario["id"]).encode())) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    # known-at-time calendar for these dates (recomputed by the caller-provided cal frame)
    weekday = _weekday_shape_cache
    latent = _latent_demand(dates, p, scenario, _weekday_shape_cache, _cal_cache, cfg, rng)

    lead = int(scenario["lead_time_days"])
    delay = int(scenario["supplier_delay_days"])
    review = int(scenario["review_period_days"])
    moq = int(scenario["moq"])
    pack = int(scenario["pack_size"])
    z = float(s["service_level_z"])

    exp_daily = p["lam"]                                   # causal expected daily demand
    std_daily = p["std"]
    safety = max(0.0, z * std_daily * math.sqrt(lead))
    reorder_point = exp_daily * lead + safety
    target_stock = reorder_point + exp_daily * review
    initial_stock = max(0, round(exp_daily * int(scenario["initial_coverage_days"])))

    n = len(dates)
    arrivals = np.zeros(n + lead + delay + 5)
    order_of_arrival = [[] for _ in range(len(arrivals))]  # track order dates arriving at index
    ending = np.zeros(n)
    rows: list[dict] = []
    events: list[dict] = []
    prev_ending = float(initial_stock)
    on_order = 0.0
    warmup = int(s["warmup_days"])

    for i, d in enumerate(dates):
        arr = float(arrivals[i])
        opening = prev_ending + arr                       # opening = prior ending + arrivals
        on_order = max(0.0, on_order - arr)               # received reduces on-order
        available = max(0.0, opening)
        demand = float(latent[i])
        sales = min(demand, available)
        lost = max(0.0, demand - available)
        end = max(0.0, available - sales)
        inv_position = end + on_order

        order_placed, order_qty = 0, 0.0
        if inv_position <= reorder_point:                 # continuous review at end of day
            raw = max(0.0, target_stock - inv_position)
            if raw > 0:
                rounded = math.ceil(raw / pack) * pack
                order_qty = float(max(moq, rounded))
                arr_i = i + lead + delay
                if arr_i < len(arrivals):
                    arrivals[arr_i] += order_qty
                    order_of_arrival[arr_i].append(i)
                on_order += order_qty
                order_placed = 1
                events.append({
                    "sku": sku, "product_id": product_id, "channel": channel,
                    "scenario_id": scenario["id"], "order_date": d,
                    "order_quantity": order_qty, "assumed_lead_time_days": lead,
                    "supplier_delay_days": delay,
                    "expected_arrival_date": d + pd.Timedelta(days=lead),
                    "actual_arrival_date": (dates[arr_i] if arr_i < n else d + pd.Timedelta(days=lead + delay)),
                    "received_quantity": order_qty, "moq": moq, "pack_size": pack,
                    "moq_is_assumed": True, "pack_size_is_assumed": True,
                    "lead_time_is_assumed": True, "is_synthetic": True,
                })
        ending[i] = end
        rows.append({
            "date": d, "sku": sku, "product_id": product_id, "channel": channel,
            "scenario_id": scenario["id"], "scenario_type": scenario["type"],
            "opening_stock": round(opening), "replenishment_received": round(arr),
            "latent_synthetic_demand": int(demand), "synthetic_sales": round(sales),
            "lost_sales": round(lost), "ending_stock": round(end),
            "stock_on_hand": round(end),                  # compatibility alias = ending stock
            "inventory_position": round(inv_position), "on_order_quantity": round(on_order),
            "order_placed": order_placed, "reorder_point": round(reorder_point, 2),
            "safety_stock": round(safety, 2), "days_of_cover": round(end / exp_daily, 2) if exp_daily > 0 else np.nan,
            "expected_daily_demand": round(exp_daily, 4), "demand_std": round(std_daily, 4),
            "is_stockout": bool(end == 0), "is_demand_censored_synthetic": bool(lost > 0),
            "assumed_lead_time_days": lead, "assumed_supplier_delay_days": delay,
            "assumed_moq": moq, "assumed_pack_size": pack,
            "stock_snapshot_timing": "end_of_day",
        })
        prev_ending = end

    # ── future stockout TARGETS (allowed to look ahead; features never do) ──────────
    horizons = list(s["stockout_target_horizons"])
    next_arrival = _next_arrival_index(arrivals, n)
    for i in range(n):
        for h in horizons:
            hi = ending[i + 1:i + 1 + h]
            rows[i][f"stockout_within_{h}d"] = bool((hi == 0).any()) if len(hi) else False
        fut = ending[i + 1:]
        zero = np.where(fut == 0)[0]
        rows[i]["days_until_stockout"] = int(zero[0] + 1) if zero.size else np.nan
        na = next_arrival[i]
        window = ending[i + 1:na] if na > i + 1 else np.array([])
        rows[i]["stockout_before_next_replenishment"] = bool((window == 0).any()) if len(window) else False
        rows[i]["stockout_training_eligible"] = bool(i >= warmup)   # skip init-artefact warmup
    return rows, events


def _next_arrival_index(arrivals: np.ndarray, n: int) -> list[int]:
    """For each day i, the index of the next arrival strictly after i (else n)."""
    nxt = [n] * n
    arr_days = [j for j in range(len(arrivals)) if arrivals[j] > 0]
    for i in range(n):
        nxt[i] = next((j for j in arr_days if j > i), n)
    return nxt


# module-level caches set per SKU by run() (kept simple; single-threaded pipeline)
_weekday_shape_cache: dict[int, float] = {}
_cal_cache: pd.DataFrame = pd.DataFrame()


# ── driver ─────────────────────────────────────────────────────────────────────────
def run(real_daily: pd.DataFrame, sku_meta: pd.DataFrame, calendar: pd.DataFrame,
        calibration_end: pd.Timestamp, as_of: pd.Timestamp, cfg: dict
        ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run every scenario for every SKU.

    real_daily : columns [sku, date, units]  (REAL, already filtered date <= as_of)
    sku_meta   : columns [sku, product_id, category, channel]
    calendar   : columns [date, is_public_holiday, is_payday_window]  (known-at-time)
    Returns (stockout_scenarios_df, replenishment_events_df, sim_params_dict).
    """
    global _weekday_shape_cache, _cal_cache
    s = cfg["synthetic"]
    sku_cat = dict(zip(sku_meta["sku"], sku_meta["category"]))
    meta = sku_meta.set_index("sku")
    params = calibrate(real_daily, sku_cat, calibration_end, cfg)
    _weekday_shape_cache = _weekday_shape(real_daily[real_daily["date"] <= calibration_end])
    _cal_cache = calendar[["date", "is_public_holiday", "is_payday_window"]].copy()

    all_rows: list[dict] = []
    all_events: list[dict] = []
    for sku, g in real_daily.groupby("sku"):
        # simulate strictly AFTER the calibration window (params are past-only)
        sim_dates = pd.DatetimeIndex(sorted(
            d for d in g["date"].unique() if pd.Timestamp(d) > calibration_end and pd.Timestamp(d) <= as_of))
        if len(sim_dates) == 0:
            continue
        p = params[sku]
        pid = meta.loc[sku, "product_id"]
        ch = meta.loc[sku, "channel"]
        for scenario in s["scenarios"]:
            rows, events = _simulate_one(sku, pid, ch, sim_dates, p, scenario, cfg)
            for r in rows:
                r.update({"calib_source": p["calib_source"], "is_synthetic": True,
                          "stock_is_synthetic": True, "stockout_label_is_synthetic": True,
                          "simulation_version": s["simulation_version"], "simulation_seed": int(s["seed"])})
            for e in events:
                e.update({"simulation_version": s["simulation_version"], "simulation_seed": int(s["seed"])})
            all_rows.extend(rows)
            all_events.extend(events)

    scenarios = pd.DataFrame(all_rows).sort_values(
        ["sku", "scenario_id", "date"]).reset_index(drop=True) if all_rows else pd.DataFrame()
    events = pd.DataFrame(all_events).sort_values(
        ["sku", "scenario_id", "order_date"]).reset_index(drop=True) if all_events else pd.DataFrame()

    sim_params = {
        "simulation_version": s["simulation_version"],
        "random_seed": int(s["seed"]),
        "historical_inventory_available": False,
        "historical_stockout_labels": "synthetic (simulation-based, not observed)",
        "demand_forecast_sales": "real (units_observed untouched)",
        "synthetic_demand_method": "Negative Binomial (Gamma-Poisson mixture), causal",
        "calibration_window": {"end": calibration_end.date().isoformat(),
                               "length_days": int(s["calibration_days"]),
                               "min_days_before_fallback": int(s["min_calibration_days"])},
        "fallback_order": ["sku", "category", "global"],
        "scenarios": s["scenarios"],
        "replenishment_assumptions": {"service_level_z": s["service_level_z"],
                                      "note": "lead time / MOQ / pack size are pilot assumptions"},
        "snapshot_timing": "end_of_day",
        "feature_cutoff_convention": "features at day t use info <= t; targets look at t+1 onward",
        "calibration_sources_used": pd.Series(
            {k: v["calib_source"] for k, v in params.items()}).value_counts().to_dict(),
    }
    return scenarios, events, sim_params
