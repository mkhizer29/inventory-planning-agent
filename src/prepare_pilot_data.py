"""prepare_pilot_data.py — build the daily, ecommerce-only, leakage-free pilot dataset.

Reads the ETL warehouse (inventory_etl/output/inventory.db) and writes:

  data/processed/model_panel.parquet        REAL demand history for forecast training/backtesting
  data/processed/forecast_features.parquet  next-14-day inputs, known at as_of only (no leakage, no cost)
  data/processed/inventory_context.parquet  as-of stock + validated cost + replenishment context
  data/processed/pilot_manifest.json        validation stats, reproducibility, assumptions

  data/synthetic/stockout_scenarios.parquet   causal synthetic inventory/stockout trajectories
  data/synthetic/replenishment_events.parquet  synthetic replenishment orders
  data/synthetic/simulation_parameters.json    what is real vs synthetic, method, assumptions

REAL vs SYNTHETIC (this is the whole point of the redesign):
  * `units_observed` in model_panel is REAL e-commerce sales and is NEVER overwritten by
    synthetic sales. Demand forecasting is trained/evaluated only on this.
  * Inventory levels, stockouts, lost sales and replenishment are a SEPARATE causal,
    seeded SIMULATION on a separate latent demand series (src/synthetic_inventory.py).
    They live under data/synthetic/ and are flagged is_synthetic / stock_is_synthetic /
    stockout_label_is_synthetic. They do NOT affect forecast training eligibility or
    forecast evaluation.
  * Validated unit cost is a FINANCIAL field only (inventory_context) — never a demand feature.

Scope: ECOMMERCE ONLY. `online_delivery` is normalised to `naheed_web`; `store`/`storepickup`
are excluded (and counted). `foodpanda` has no rows in this warehouse (documented).

Run from the repo root:
    python src/prepare_pilot_data.py --as-of-date 2026-06-30
    python src/prepare_pilot_data.py --reselect-pilot-skus --selection-cutoff 2026-05-31
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

import synthetic_inventory  # sibling module in src/

try:
    import holidays as _holidays
except ImportError:  # calendar features degrade gracefully if the lib is absent
    _holidays = None

SCHEMA_VERSION = "3.0-real-synthetic-split"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "inventory_etl" / "output" / "inventory.db"
DEFAULT_OUT = REPO_ROOT / "data" / "processed"
DEFAULT_SYNTH = REPO_ROOT / "data" / "synthetic"
CONFIG_PATH = REPO_ROOT / "inventory_etl" / "config" / "config.yaml"
PILOT_LIST = REPO_ROOT / "pilot_skus.csv"
CANDIDATE_LIST = REPO_ROOT / "pilot_skus_candidate.csv"

# model_panel is REAL-ONLY. Synthetic-inventory columns are deliberately ABSENT here;
# the three "*_synthetic / historical_*" flags below are constant markers documenting
# that no synthetic stock is attached to the demand panel.
MODEL_PANEL_COLS = [
    "sku", "product_id", "channel", "date", "category", "sub_category", "brand",
    "units_observed", "effective_unit_price", "discount_amount", "discount_pct",
    "on_promo", "promo_known_in_advance", "is_public_holiday", "holiday_name",
    "is_payday_window", "day_of_week", "is_weekend", "week_of_year", "month",
    "units_lag_1", "units_lag_7", "units_lag_14",
    "units_roll_mean_7", "units_roll_mean_28", "units_roll_std_7",
    "product_active", "historical_stockout_observed", "observed_demand_censored",
    "stock_is_synthetic", "stockout_label_is_synthetic",
    "forecast_training_eligible", "data_quality_flag",
]
# The ONLY columns a demand model may use as features. Everything inventory/cost/synthetic
# is excluded by construction (they are not in model_panel) and enumerated here for tests/docs.
DEMAND_FEATURE_WHITELIST = [
    "units_lag_1", "units_lag_7", "units_lag_14",
    "units_roll_mean_7", "units_roll_mean_28", "units_roll_std_7",
    "effective_unit_price", "discount_pct", "on_promo",
    "is_public_holiday", "is_payday_window", "day_of_week", "is_weekend",
    "week_of_year", "month",
]
DEMAND_FEATURE_FORBIDDEN = [
    "stock_on_hand", "opening_stock", "ending_stock", "is_stockout", "is_available",
    "lost_sales", "replenishment_received", "inventory_position", "on_order_quantity",
    "stockout_within_2d", "stockout_within_7d", "days_until_stockout",
    "unit_cost", "unit_cost_observed", "unit_cost_effective",
]
FORECAST_COLS = [
    "sku", "product_id", "channel", "date", "forecast_horizon_day", "category",
    "sub_category", "brand", "latest_known_price", "trailing_units_mean_7",
    "trailing_units_mean_28", "planned_promo", "planned_discount_pct",
    "is_public_holiday", "holiday_name", "is_payday_window", "day_of_week",
    "is_weekend", "week_of_year", "month", "feature_availability_flag",
]
INVENTORY_COLS = [
    "as_of_date", "sku", "product_id", "location_id",
    "stock_on_hand", "stock_is_synthetic", "scenario_id", "simulation_version",
    "stock_snapshot_timing", "inventory_position", "on_order_quantity",
    "reorder_point", "safety_stock", "days_of_cover", "is_stockout",
    "stock_in_transit", "supplier_lead_time_days", "moq", "pack_size",
    "is_perishable", "shelf_life_days", "price",
    "unit_cost_observed", "unit_cost_effective", "cost_source", "cost_is_valid",
    "cost_is_imputed", "cost_quality_flag", "cost_currency", "cost_basis",
    "recommended_order_quantity", "recommended_purchase_value", "inventory_value",
    "is_dropship", "lead_time_is_assumed", "moq_is_assumed", "pack_size_is_assumed",
    "stock_in_transit_is_assumed", "perishability_is_assumed", "assumption_notes",
]

COST_PRECEDENCE = ["magento_eav", "staging_margin", "product_flat"]


# ── config / db ───────────────────────────────────────────────────────────────
def load_config() -> dict:
    if not CONFIG_PATH.exists():
        sys.exit(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key in ("pilot", "replenishment", "external_signals", "synthetic", "cost"):
        if key not in cfg:
            sys.exit(f"Config section '{key}' missing in {CONFIG_PATH}")
    return cfg


@contextlib.contextmanager
def open_db(db_path: Path):
    if not db_path.exists():
        sys.exit(f"ETL warehouse not found: {db_path}\nRun the ETL first (see TEAMMATE_SETUP.md).")
    con = sqlite3.connect(db_path)
    try:
        yield con
    finally:
        con.close()


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}


# ── calendar (known in advance → safe as future features) ───────────────────────
def calendar_features(dates: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Holiday/payday/day-of-week features for any date range, computed deterministically
    (holidays known in advance). Reuses the payday rule from config.external_signals."""
    es = cfg["external_signals"]
    starts = set(es.get("payday_days_month_start", []))
    ends = set(es.get("payday_days_month_end", []))
    country = es.get("country", "PK")
    hol = {}
    if _holidays is not None and len(dates):
        yrs = range(dates.min().year, dates.max().year + 1)
        hol = dict(_holidays.country_holidays(country, years=yrs))

    def is_payday(d: dt.date) -> bool:
        last = (dt.date(d.year + d.month // 12, d.month % 12 + 1, 1) - dt.timedelta(days=1)).day
        return d.day in starts or d.day in ends or d.day == last

    df = pd.DataFrame({"date": pd.DatetimeIndex(dates)})
    dd = df["date"].dt.date
    df["is_public_holiday"] = dd.map(lambda d: d in hol).astype(int)
    df["holiday_name"] = dd.map(lambda d: hol.get(d))
    df["is_payday_window"] = dd.map(is_payday).astype(int)
    df["day_of_week"] = df["date"].dt.weekday          # Monday=0
    df["is_weekend"] = (df["date"].dt.weekday >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    return df


# ── channel mapping ─────────────────────────────────────────────────────────────
def map_channels(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, int, dict]:
    """Keep ecommerce rows only. Returns (ecommerce_df, physical_excluded, unknown_counts)."""
    emap = cfg["pilot"]["ecommerce_channel_map"]
    physical = set(cfg["pilot"]["physical_channels"])
    src = df["channel"].astype(str)
    physical_excluded = int(src.isin(physical).sum())
    known = set(emap) | physical
    unknown_mask = ~src.isin(known)
    unknown_counts = df.loc[unknown_mask, "channel"].value_counts().to_dict()
    keep = df[src.isin(emap)].copy()
    keep["channel"] = keep["channel"].map(emap)     # explicit map, never silent to naheed_web
    return keep, physical_excluded, unknown_counts


# ── sku selection / validation ──────────────────────────────────────────────────
def _ecommerce_sales_sql(cfg: dict) -> str:
    ecom = list(cfg["pilot"]["ecommerce_channel_map"])
    inlist = ",".join(f"'{c}'" for c in ecom)
    return (f"SELECT s.sku_id AS sku, st.channel, st.quantity_sold, st.transaction_date "
            f"FROM sales_transactions st JOIN sku_master s ON s.sku_id=st.sku_id "
            f"WHERE st.channel IN ({inlist})")


def reselect_candidates(con: sqlite3.Connection, cfg: dict, cutoff: str) -> pd.DataFrame:
    """Deterministic top-N-per-category selection using ONLY data on/before the cutoff.
    Never uses test-period demand; writes a candidate file (does not overwrite the approved list)."""
    p = cfg["pilot"]
    q = (f"SELECT s.sku_id AS sku, s.category, s.brand, s.sku_name AS name, "
         f"SUM(st.quantity_sold) AS units, COUNT(DISTINCT st.transaction_date) AS active_days "
         f"FROM sales_transactions st JOIN sku_master s ON s.sku_id=st.sku_id "
         f"WHERE st.channel IN ({','.join(repr(c) for c in p['ecommerce_channel_map'])}) "
         f"  AND st.transaction_date <= ? "
         f"  AND s.category IS NOT NULL AND TRIM(s.category)<>'' "
         f"  AND s.sku_id NOT LIKE 'Free%' AND s.sku_id NOT LIKE 'PACK%' "
         f"GROUP BY s.sku_id")
    df = pd.read_sql(q, con, params=[cutoff])
    df = df[df["active_days"] >= p["min_history_days"]]
    top_cats = (df.groupby("category")["units"].sum().sort_values(ascending=False)
                  .head(p["n_categories"]).index.tolist())
    parts = [df[df.category == c].sort_values("units", ascending=False).head(p["top_n_per_category"])
             for c in top_cats]
    out = pd.concat(parts, ignore_index=True) if parts else df.head(0)
    return out[["sku", "category", "brand", "name", "units"]]


def load_pilot_skus(con: sqlite3.Connection, cfg: dict, strict: bool) -> tuple[pd.DataFrame, list[str]]:
    """Load the frozen approved list and validate it against the warehouse."""
    if not PILOT_LIST.exists():
        sys.exit(f"Approved pilot list missing: {PILOT_LIST.name}. "
                 f"Run with --reselect-pilot-skus --selection-cutoff YYYY-MM-DD to propose one.")
    pilot = pd.read_csv(PILOT_LIST, encoding="utf-8-sig")
    if "sku" not in pilot.columns:
        sys.exit(f"{PILOT_LIST.name} must have a 'sku' column.")
    skus = pilot["sku"].astype(str).tolist()
    warnings: list[str] = []
    if len(set(skus)) != len(skus):
        warnings.append("pilot_skus.csv contains duplicate SKUs")
    existing = set(pd.read_sql("SELECT sku_id FROM sku_master", con)["sku_id"])
    missing = [s for s in skus if s not in existing]
    if missing:
        warnings.append(f"{len(missing)} pilot SKUs not found in sku_master: {missing[:5]}")
    cov = pd.read_sql(
        f"SELECT sku, COUNT(DISTINCT transaction_date) d FROM ({_ecommerce_sales_sql(cfg)}) "
        f"GROUP BY sku", con)
    cov = dict(zip(cov["sku"], cov["d"]))
    thin = [s for s in skus if cov.get(s, 0) < cfg["pilot"]["min_history_days"]]
    if thin:
        warnings.append(f"{len(thin)} pilot SKUs have < {cfg['pilot']['min_history_days']} "
                        f"days of ecommerce history: {thin[:5]}")
    if strict and (missing or len(set(skus)) != len(skus)):
        sys.exit(f"[strict] pilot list invalid: {warnings}")
    return pilot, warnings


# ── model panel (REAL demand only) ───────────────────────────────────────────────
def build_model_panel(con: sqlite3.Connection, pilot: pd.DataFrame, cfg: dict,
                      start: str | None, as_of: str | None
                      ) -> tuple[pd.DataFrame, dict]:
    """Daily SKU x channel panel of REAL sales over each SKU's active period.

    `as_of` is a HARD boundary: no sales row after as_of affects anything downstream.
    Contains NO synthetic inventory columns — those live in data/synthetic/.
    """
    skus = pilot["sku"].astype(str).tolist()
    ph = ",".join("?" * len(skus))
    attrs = pd.read_sql(
        f"SELECT sku_id AS sku, product_id, category, sub_category, brand, price "
        f"FROM sku_master WHERE sku_id IN ({ph})", con, params=skus)

    raw = pd.read_sql(
        f"SELECT sku_id AS sku, channel, transaction_date, quantity_sold, "
        f"       discount_amount, row_total FROM sales_transactions WHERE sku_id IN ({ph})",
        con, params=skus)
    raw["date"] = pd.to_datetime(raw["transaction_date"])
    ecom, physical_excluded, unknown = map_channels(raw, cfg)
    if start:
        ecom = ecom[ecom["date"] >= pd.Timestamp(start)]
    # HARD as_of boundary — drop every record after as_of BEFORE any aggregation/features
    as_of_ts = pd.Timestamp(as_of) if as_of else (ecom["date"].max() if not ecom.empty else pd.NaT)
    rows_after_as_of = int((ecom["date"] > as_of_ts).sum()) if as_of else 0
    ecom = ecom[ecom["date"] <= as_of_ts]
    if ecom.empty:
        sys.exit("No ecommerce sales found for the pilot SKUs in the given window (<= as_of).")
    window_end = as_of_ts

    agg = (ecom.groupby(["sku", "channel", "date"])
               .agg(units_observed=("quantity_sold", "sum"),
                    discount_amount=("discount_amount", "sum"),
                    net_rev=("row_total", "sum"))
               .reset_index())
    agg["units_observed"] = agg["units_observed"].clip(lower=0).round().astype(int)

    starts = agg.groupby(["sku", "channel"])["date"].min().reset_index(name="active_start")
    frames = []
    for (sku, ch), grp in starts.groupby(["sku", "channel"]):
        a0 = grp["active_start"].iloc[0]
        days = pd.date_range(a0, window_end, freq="D")
        frames.append(pd.DataFrame({"sku": sku, "channel": ch, "date": days}))
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.merge(agg, on=["sku", "channel", "date"], how="left")

    panel["units_observed"] = panel["units_observed"].fillna(0).astype(int)   # genuine zeros in-window
    panel["discount_amount"] = panel["discount_amount"].fillna(0.0)
    panel["net_rev"] = panel["net_rev"].fillna(0.0)
    panel["product_active"] = True
    panel = panel.sort_values(["sku", "channel", "date"]).reset_index(drop=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(panel["units_observed"] > 0,
                       panel["net_rev"] / panel["units_observed"].replace(0, np.nan), np.nan)
    panel["effective_unit_price"] = eff
    panel["effective_unit_price"] = panel.groupby(["sku", "channel"])["effective_unit_price"].ffill()
    gross = panel["net_rev"] + panel["discount_amount"]
    panel["discount_pct"] = np.where(gross > 0, panel["discount_amount"] / gross, 0.0)
    panel["on_promo"] = (panel["discount_amount"] > 0).astype(int)
    panel["promo_known_in_advance"] = 0        # historical realised promo, not a planned calendar

    cal = calendar_features(pd.DatetimeIndex(panel["date"].unique()), cfg)
    panel = panel.merge(cal, on="date", how="left")
    panel = panel.merge(attrs.drop(columns=["price"]), on="sku", how="left")

    # CAUSAL demand features: grouped shift BEFORE rolling so date t uses only data <= t-1.
    grp = panel.groupby(["sku", "channel"])["units_observed"]
    panel["units_lag_1"] = grp.shift(1)
    panel["units_lag_7"] = grp.shift(7)
    panel["units_lag_14"] = grp.shift(14)
    base = grp.shift(1)                                   # exclude the current day from rolling
    keyed = base.groupby([panel["sku"], panel["channel"]])
    panel["units_roll_mean_7"] = keyed.transform(lambda s: s.rolling(7, min_periods=1).mean())
    panel["units_roll_mean_28"] = keyed.transform(lambda s: s.rolling(28, min_periods=1).mean())
    panel["units_roll_std_7"] = keyed.transform(lambda s: s.rolling(7, min_periods=1).std())

    # Explicit REAL-vs-SYNTHETIC markers. Real daily stock history is unavailable, so
    # historical stockout / censoring are UNKNOWN here — never assume zero sales == stockout.
    panel["historical_stockout_observed"] = False
    panel["observed_demand_censored"] = pd.NA          # unknown (no real stock history)
    panel["stock_is_synthetic"] = False                # no synthetic stock attached to the demand panel
    panel["stockout_label_is_synthetic"] = False

    # forecast eligibility: depends ONLY on real-data validity + sufficient history.
    didx = panel.groupby(["sku", "channel"]).cumcount()
    min_hist = 14
    panel["forecast_training_eligible"] = panel["product_active"] & (didx >= min_hist)

    def flag(row_didx, price_bad) -> str:
        flags = ["activation_inferred_from_first_sale"]
        if row_didx < min_hist:
            flags.append("insufficient_history")
        if price_bad:
            flags.append("price_anomaly")
        return ";".join(flags)

    price_bad = (panel["units_observed"] > 0) & (
        panel["effective_unit_price"].isna() | (panel["effective_unit_price"] <= 0))
    panel["data_quality_flag"] = [flag(d, pb) for d, pb in zip(didx.to_numpy(), price_bad.to_numpy())]

    panel["holiday_name"] = panel["holiday_name"].where(panel["holiday_name"].notna(), None)
    stats = {"physical_store_rows_excluded": physical_excluded, "unknown_channel_rows": unknown,
             "window_end": window_end, "as_of": window_end, "rows_after_as_of_dropped": rows_after_as_of}
    return panel[MODEL_PANEL_COLS], stats


# ── forecast features (real, known-at-as_of, NO cost, NO synthetic) ───────────────
def build_forecast_features(panel: pd.DataFrame, con: sqlite3.Connection, pilot: pd.DataFrame,
                            cfg: dict, as_of: pd.Timestamp) -> pd.DataFrame:
    """Exactly N future days per SKU/channel, using only info known on as_of (no leakage)."""
    n_days = int(cfg["pilot"]["forecast_feature_days"])
    future = pd.date_range(as_of + pd.Timedelta(days=1), periods=n_days, freq="D")
    keys = panel[["sku", "product_id", "channel", "category", "sub_category", "brand"]].drop_duplicates()

    hist = panel[panel["date"] <= as_of]
    last_price = (hist.dropna(subset=["effective_unit_price"])
                      .sort_values("date").groupby(["sku", "channel"])["effective_unit_price"].last())
    # trailing real-demand means KNOWN at as_of (static per SKU/channel; safe future features)
    tail = hist.sort_values("date").groupby(["sku", "channel"])["units_observed"]
    trail7 = tail.apply(lambda s: float(s.tail(7).mean()) if len(s) else np.nan)
    trail28 = tail.apply(lambda s: float(s.tail(28).mean()) if len(s) else np.nan)
    skus = pilot["sku"].astype(str).tolist()
    cat_price = pd.read_sql(
        f"SELECT sku_id AS sku, price FROM sku_master WHERE sku_id IN ({','.join('?'*len(skus))})",
        con, params=skus).set_index("sku")["price"]

    rows = []
    for _, k in keys.iterrows():
        key = (k["sku"], k["channel"])
        lp = last_price.get(key)
        if pd.isna(lp):
            lp = cat_price.get(k["sku"], np.nan)
        for i, d in enumerate(future, start=1):
            rows.append({**k.to_dict(), "date": d, "forecast_horizon_day": i,
                         "latest_known_price": lp,
                         "trailing_units_mean_7": trail7.get(key, np.nan),
                         "trailing_units_mean_28": trail28.get(key, np.nan)})
    ff = pd.DataFrame(rows)
    cal = calendar_features(future, cfg)
    ff = ff.merge(cal, on="date", how="left")
    ff["planned_promo"] = 0
    ff["planned_discount_pct"] = np.nan
    ff["feature_availability_flag"] = np.where(
        ff["latest_known_price"].notna(), "price_ok;planned_promo_unavailable",
        "price_missing;planned_promo_unavailable")
    ff["holiday_name"] = ff["holiday_name"].where(ff["holiday_name"].notna(), None)
    return ff.sort_values(["sku", "channel", "date"]).reset_index(drop=True)[FORECAST_COLS]


# ── unit-cost validation ──────────────────────────────────────────────────────────
def classify_cost(candidates: list[tuple[str, object]], price, tol: float) -> dict:
    """Resolve one SKU's cost from precedence-ordered (source, value) candidates.

    A cost is valid iff numeric, finite and > 0. Zero/negative are INVALID (not just present).
    Returns observed value + source + validity + quality flags. No imputation here — the
    caller applies category/global median fallback for unit_cost_effective.
    """
    flags: list[str] = []
    numeric = [(s, pd.to_numeric(v, errors="coerce")) for s, v in candidates]
    numeric = [(s, float(v)) for s, v in numeric if pd.notna(v)]
    if not numeric:
        flags.append("MISSING_COST")
    if any(np.isfinite(v) and v <= 0 for _, v in numeric):
        flags.append("NON_POSITIVE_COST")
    if any(not np.isfinite(v) for _, v in numeric):
        flags.append("NON_FINITE_COST")
    valids = [(s, v) for s, v in numeric if np.isfinite(v) and v > 0]
    observed, source = (valids[0][1], valids[0][0]) if valids else (None, "missing")
    if len(valids) >= 2:
        lo, hi = min(v for _, v in valids), max(v for _, v in valids)
        if lo > 0 and (hi - lo) / lo > tol:
            flags.append("COST_SOURCE_CONFLICT")
    if observed is not None and pd.notna(price) and observed > float(price):
        flags.append("COST_ABOVE_PRICE")
    return {"observed": observed, "source": source, "valid": observed is not None, "flags": flags}


def resolve_costs(sm: pd.DataFrame, con: sqlite3.Connection, cfg: dict) -> pd.DataFrame:
    """Add validated cost columns to the SKU master frame.

    Uses per-source columns (eav_cost/margin_cost/flat_cost) if the warehouse carries them;
    otherwise treats sku_master.unit_cost as the already-precedence-resolved observed cost
    (its source recorded in cost_source if present). Applies category then global median
    fallback for unit_cost_effective, always retaining that the observed value was invalid.
    """
    ccfg = cfg["cost"]
    tol = float(ccfg.get("conflict_tolerance_pct", 0.25))
    cols = _table_columns(con, "sku_master")
    per_source = {"eav_cost", "margin_cost", "flat_cost"} <= cols
    has_source_col = "cost_source" in cols

    recs = []
    for _, r in sm.iterrows():
        if per_source:
            cands = [("magento_eav", r.get("eav_cost")),
                     ("staging_margin", r.get("margin_cost")),
                     ("product_flat", r.get("flat_cost"))]
        else:
            src = r.get("cost_source") if has_source_col and pd.notna(r.get("cost_source")) \
                else "warehouse_precedence"
            cands = [(src, r.get("unit_cost"))]
        recs.append(classify_cost(cands, r.get("price"), tol))

    sm = sm.copy()
    sm["unit_cost_observed"] = [x["observed"] for x in recs]
    sm["cost_source"] = [x["source"] for x in recs]
    sm["cost_is_valid"] = [x["valid"] for x in recs]
    sm["_flags"] = [x["flags"] for x in recs]

    # category -> global median fallback (from VALID observed costs only)
    valid_obs = sm.loc[sm["cost_is_valid"], ["category", "unit_cost_observed"]]
    cat_median = valid_obs.groupby("category")["unit_cost_observed"].median().to_dict()
    global_median = float(valid_obs["unit_cost_observed"].median()) if len(valid_obs) else np.nan

    eff, imputed, source, flags = [], [], [], []
    for i, r in sm.reset_index(drop=True).iterrows():
        fl = list(recs[i]["flags"])
        if r["cost_is_valid"]:
            eff.append(float(r["unit_cost_observed"]))
            imputed.append(False)
            source.append(r["cost_source"])
        else:
            cm = cat_median.get(r["category"], np.nan)
            if pd.notna(cm):
                eff.append(float(cm)); source.append("category_median_fallback"); fl.append("IMPUTED_CATEGORY_MEDIAN")
            elif pd.notna(global_median):
                eff.append(float(global_median)); source.append("global_median_fallback"); fl.append("IMPUTED_GLOBAL_MEDIAN")
            else:
                eff.append(np.nan); source.append("missing")
            imputed.append(True)
        fl.append("PACK_UNIT_BASIS_UNCONFIRMED")     # DB cost unit (SKU/case/pack) not confirmed
        flags.append(";".join(dict.fromkeys(fl)))     # de-dup, keep order
    sm["unit_cost_effective"] = eff
    sm["cost_is_imputed"] = imputed
    sm["cost_source"] = source
    sm["cost_quality_flag"] = flags
    sm["cost_currency"] = ccfg.get("currency", "PKR")
    sm["cost_basis"] = ccfg.get("basis", "sellable_sku_unit_unconfirmed")
    return sm.drop(columns=["_flags"])


# ── inventory context (as-of; validated cost; synthetic stock snapshot) ────────────
def build_inventory_context(con: sqlite3.Connection, pilot: pd.DataFrame, cfg: dict,
                            as_of: pd.Timestamp, scenarios: pd.DataFrame) -> pd.DataFrame:
    """One row per SKU x location at as_of. Stock-on-hand is the SYNTHETIC primary-scenario
    end-of-day snapshot (flagged). Cost is validated (observed vs effective). Replenishment
    inputs are flagged assumptions."""
    rep = cfg["replenishment"]
    syn = cfg["synthetic"]
    primary = syn["primary_scenario"]
    prm = next(s for s in syn["scenarios"] if s["id"] == primary)
    skus = pilot["sku"].astype(str).tolist()
    ph = ",".join("?" * len(skus))
    base_cols = "sku_id AS sku, product_id, category, is_perishable, shelf_life_days, unit_cost, price, pack_size, is_dropship"
    extra = [c for c in ("eav_cost", "margin_cost", "flat_cost", "cost_source")
             if c in _table_columns(con, "sku_master")]
    sel = base_cols + ("".join(f", {c}" for c in extra))
    sm = pd.read_sql(f"SELECT {sel} FROM sku_master WHERE sku_id IN ({ph})", con, params=skus)
    sm = resolve_costs(sm, con, cfg)

    # synthetic snapshot from the PRIMARY scenario, last simulated day <= as_of
    snap = {}
    if not scenarios.empty:
        prim = scenarios[(scenarios["scenario_id"] == primary) & (scenarios["date"] <= as_of)]
        snap = {sku: g.sort_values("date").iloc[-1] for sku, g in prim.groupby("sku")}

    lead = int(prm["lead_time_days"])
    moq = int(prm["moq"])
    pack = int(prm["pack_size"])
    review = int(prm["review_period_days"])
    assume_transit0 = bool(cfg["pilot"]["assume_stock_in_transit_zero"])
    note = (f"stock_on_hand is SYNTHETIC (end-of-day snapshot from the '{primary}' simulation "
            f"scenario — NOT real Naheed stock). Supplier lead time ({lead}d), MOQ ({moq}), "
            f"pack size ({pack}) and review period ({review}d) are pilot assumptions (no supplier "
            f"data). Stock-in-transit " + ("assumed 0. " if assume_transit0 else "unknown. ")
            + "Perishability unknown where shelf-life absent. Unit cost validated; effective "
            "cost may be an imputed fallback (see cost_quality_flag).")

    rows = []
    for _, r in sm.iterrows():
        s = snap.get(r["sku"])
        if s is not None:
            soh = float(s["ending_stock"]); inv_pos = float(s["inventory_position"])
            on_order = float(s["on_order_quantity"]); rop = float(s["reorder_point"])
            safety = float(s["safety_stock"]); doc = float(s["days_of_cover"]) if pd.notna(s["days_of_cover"]) else np.nan
            is_so = bool(s["is_stockout"]); exp_daily = float(s["expected_daily_demand"])
        else:
            soh = inv_pos = on_order = rop = safety = np.nan
            doc = np.nan; is_so = False; exp_daily = np.nan

        # causal reorder recommendation at as_of (same math as the simulation)
        if pd.notna(inv_pos) and pd.notna(exp_daily):
            target = rop + exp_daily * review
            raw = max(0.0, target - inv_pos)
            if inv_pos <= rop and raw > 0:
                rec_qty = float(max(moq, math.ceil(raw / pack) * pack))
            else:
                rec_qty = 0.0
        else:
            rec_qty = np.nan
        eff = r["unit_cost_effective"]
        rec_val = rec_qty * eff if (pd.notna(rec_qty) and pd.notna(eff)) else np.nan
        inv_val = soh * eff if (pd.notna(soh) and pd.notna(eff)) else np.nan

        rows.append({
            "as_of_date": as_of.date().isoformat(), "sku": r["sku"], "product_id": r["product_id"],
            "location_id": "ALL",
            "stock_on_hand": soh, "stock_is_synthetic": True, "scenario_id": primary,
            "simulation_version": syn["simulation_version"], "stock_snapshot_timing": "end_of_day",
            "inventory_position": inv_pos, "on_order_quantity": on_order,
            "reorder_point": round(rop, 2) if pd.notna(rop) else np.nan,
            "safety_stock": round(safety, 2) if pd.notna(safety) else np.nan,
            "days_of_cover": doc, "is_stockout": is_so,
            "stock_in_transit": 0.0 if assume_transit0 else np.nan,
            "supplier_lead_time_days": lead, "moq": moq, "pack_size": pack,
            "is_perishable": bool(r["is_perishable"]) if pd.notna(r["is_perishable"]) else None,
            "shelf_life_days": r["shelf_life_days"] if pd.notna(r["shelf_life_days"]) else None,
            "price": r["price"],
            "unit_cost_observed": r["unit_cost_observed"], "unit_cost_effective": eff,
            "cost_source": r["cost_source"], "cost_is_valid": bool(r["cost_is_valid"]),
            "cost_is_imputed": bool(r["cost_is_imputed"]), "cost_quality_flag": r["cost_quality_flag"],
            "cost_currency": r["cost_currency"], "cost_basis": r["cost_basis"],
            "recommended_order_quantity": rec_qty, "recommended_purchase_value": rec_val,
            "inventory_value": inv_val,
            "is_dropship": bool(r["is_dropship"]) if pd.notna(r["is_dropship"]) else None,
            "lead_time_is_assumed": True, "moq_is_assumed": True, "pack_size_is_assumed": True,
            "stock_in_transit_is_assumed": True, "perishability_is_assumed": True,
            "assumption_notes": note,
        })
    return pd.DataFrame(rows)[INVENTORY_COLS]


# ── validation ────────────────────────────────────────────────────────────────────
def validate_outputs(panel: pd.DataFrame, ff: pd.DataFrame, inv: pd.DataFrame,
                     scenarios: pd.DataFrame, cfg: dict) -> list[str]:
    problems: list[str] = []
    if list(panel.columns) != MODEL_PANEL_COLS:
        problems.append("model_panel columns mismatch")
    if list(ff.columns) != FORECAST_COLS:
        problems.append("forecast_features columns mismatch")
    if list(inv.columns) != INVENTORY_COLS:
        problems.append("inventory_context columns mismatch")
    # real panel must not carry any synthetic inventory feature
    leaked = [c for c in DEMAND_FEATURE_FORBIDDEN if c in panel.columns]
    if leaked:
        problems.append(f"LEAKAGE: synthetic/cost fields present in model_panel: {leaked}")
    if panel.duplicated(["sku", "channel", "date"]).any():
        problems.append("duplicate sku+channel+date in model_panel")
    if ff.duplicated(["sku", "channel", "date"]).any():
        problems.append("duplicate sku+channel+date in forecast_features")
    if inv.duplicated(["sku", "location_id"]).any():
        problems.append("duplicate sku+location in inventory_context")
    if (panel["units_observed"] < 0).any():
        problems.append("negative units_observed")
    if "units_observed" in ff.columns:
        problems.append("LEAKAGE: forecast_features must not contain units_observed")
    if any(c in ff.columns for c in ("unit_cost", "unit_cost_effective", "unit_cost_observed")):
        problems.append("LEAKAGE: forecast_features must not contain unit cost")
    n_days = int(cfg["pilot"]["forecast_feature_days"])
    per = ff.groupby(["sku", "channel"])["date"].nunique()
    if not (per == n_days).all():
        problems.append(f"forecast_features must have exactly {n_days} future days per sku/channel")
    allowed = set(cfg["pilot"]["ecommerce_channel_map"].values())
    if not set(panel["channel"]).issubset(allowed):
        problems.append(f"unexpected channels in model_panel: {set(panel['channel']) - allowed}")
    if (inv["pack_size"] < 1).any():
        problems.append("pack_size must be a positive integer")
    eff = pd.to_numeric(inv["unit_cost_effective"], errors="coerce")
    if ((eff <= 0) & eff.notna()).any():
        problems.append("unit_cost_effective must be positive when present")
    # synthetic invariants
    if not scenarios.empty:
        if (scenarios["ending_stock"] < 0).any():
            problems.append("synthetic ending_stock went negative")
        if (scenarios["synthetic_sales"] > scenarios["opening_stock"].clip(lower=0)).any():
            problems.append("synthetic_sales exceeded available stock")
        if not scenarios["is_synthetic"].all():
            problems.append("stockout_scenarios rows missing is_synthetic=True")
    return problems


# ── manifest ──────────────────────────────────────────────────────────────────────
def build_manifest(panel, ff, inv, scenarios, events, pilot, cfg, args, stats,
                   as_of, warnings, problems) -> dict:
    def rate(mask) -> float:
        return round(float(mask.mean()), 4) if len(mask) else 0.0

    by_scn = {}
    if not scenarios.empty:
        for scn, g in scenarios.groupby("scenario_id"):
            by_scn[scn] = {
                "rows": int(len(g)),
                "stockout_rate": rate(g["is_stockout"]),
                "stockout_within_2d_rate": rate(g.get("stockout_within_2d", pd.Series(dtype=bool))),
                "stockout_within_7d_rate": rate(g.get("stockout_within_7d", pd.Series(dtype=bool))),
                "lost_sales_days": int((g["lost_sales"] > 0).sum()),
                "lost_sales_units": int(g["lost_sales"].sum()),
            }

    eff = pd.to_numeric(inv["unit_cost_effective"], errors="coerce")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": args.generated_at,
        "as_of_date": as_of.date().isoformat(),
        "sku_selection_cutoff": args.selection_cutoff,
        "rows_after_as_of_dropped": stats["rows_after_as_of_dropped"],
        "source_warehouse_path": str(Path(args.db_path)),
        "data_frequency": "daily",
        "forecast_sales_source": "REAL Naheed e-commerce sales (units_observed, untouched)",
        "inventory_stockout_source": "SYNTHETIC causal simulation (NOT real historical stock)",
        "forecast_eligibility_independent_of_synthetic_stock": True,
        "forecast_horizons": cfg["pilot"]["forecast_horizons"],
        "history_start": panel["date"].min().date().isoformat(),
        "history_end": panel["date"].max().date().isoformat(),
        "sku_count": int(panel["sku"].nunique()),
        "ecommerce_channels": sorted(panel["channel"].unique().tolist()),
        "selected_skus": pilot["sku"].astype(str).tolist(),
        "category_distribution": panel.drop_duplicates("sku")["category"].value_counts().to_dict(),
        "physical_store_rows_excluded": stats["physical_store_rows_excluded"],
        "unknown_channel_rows": stats["unknown_channel_rows"],
        # real demand panel
        "real_sales_row_count": int(len(panel)),
        "forecast_eligible_row_count": int(panel["forecast_training_eligible"].sum()),
        "zero_sales_rate": rate(panel["units_observed"] == 0),
        "promotion_coverage": rate(panel["on_promo"] == 1),
        # synthetic simulation
        "synthetic_scenario_row_count": int(len(scenarios)),
        "n_scenarios": int(scenarios["scenario_id"].nunique()) if not scenarios.empty else 0,
        "replenishment_event_count": int(len(events)),
        "stockout_by_scenario": by_scn,
        "simulation_version": cfg["synthetic"]["simulation_version"],
        "simulation_seed": cfg["synthetic"]["seed"],
        # cost coverage / quality
        "cost_coverage_by_source": inv["cost_source"].value_counts().to_dict(),
        "cost_valid_count": int(inv["cost_is_valid"].sum()),
        "cost_invalid_or_missing_count": int((~inv["cost_is_valid"]).sum()),
        "cost_imputed_count": int(inv["cost_is_imputed"].sum()),
        "cost_non_positive_count": int(inv["cost_quality_flag"].str.contains("NON_POSITIVE_COST").sum()),
        "cost_missing_count": int(inv["cost_quality_flag"].str.contains("MISSING_COST").sum()),
        "cost_above_price_count": int(inv["cost_quality_flag"].str.contains("COST_ABOVE_PRICE").sum()),
        "cost_source_conflict_count": int(inv["cost_quality_flag"].str.contains("COST_SOURCE_CONFLICT").sum()),
        "cost_currency": cfg["cost"].get("currency"),
        "cost_basis": cfg["cost"].get("basis"),
        "row_counts": {
            "model_panel": int(len(panel)), "forecast_features": int(len(ff)),
            "inventory_context": int(len(inv)), "stockout_scenarios": int(len(scenarios)),
            "replenishment_events": int(len(events)),
        },
        "demand_feature_whitelist": DEMAND_FEATURE_WHITELIST,
        "assumptions": [
            "Daily stock, stockouts, lost sales and replenishment are a SYNTHETIC causal "
            "simulation (Naheed keeps no daily stock history). Not real, flagged everywhere.",
            "Demand forecasting uses REAL units_observed only; synthetic stock never affects "
            "forecast training/evaluation eligibility.",
            "Supplier lead time, MOQ, pack size and review period are pilot assumptions.",
            "Unit cost is validated (>0, finite); invalid costs fall back to category/global "
            "median and are flagged; cost unit/pack basis requires Naheed confirmation.",
            "Product activation inferred from first ecommerce sale (listing dates unknown).",
            "planned_promo unavailable: no promo calendar with valid dates.",
            "foodpanda has no rows in this warehouse; only naheed_web is modelled.",
        ],
        "warnings": warnings,
        "validation_status": "passed" if not problems else "failed",
        "problems": problems,
    }


# ── io ─────────────────────────────────────────────────────────────────────────────
def atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


# ── cli ──────────────────────────────────────────────────────────────────────────
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the daily ecommerce-only pilot dataset.")
    ap.add_argument("--db-path", default=str(DEFAULT_DB))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--synthetic-dir", default=str(DEFAULT_SYNTH))
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--as-of-date", default=None, help="HARD boundary; records after this are dropped")
    ap.add_argument("--selection-cutoff", default=None, help="YYYY-MM-DD; recorded; used with --reselect-pilot-skus")
    ap.add_argument("--reselect-pilot-skus", action="store_true")
    ap.add_argument("--strict", action="store_true")
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.generated_at = dt.datetime.now().isoformat(timespec="seconds")
    cfg = load_config()
    out_dir = Path(args.output_dir)
    synth_dir = Path(args.synthetic_dir)

    with open_db(Path(args.db_path)) as con:
        if args.reselect_pilot_skus:
            if not args.selection_cutoff:
                sys.exit("--reselect-pilot-skus requires --selection-cutoff YYYY-MM-DD")
            cand = reselect_candidates(con, cfg, args.selection_cutoff)
            cand.to_csv(CANDIDATE_LIST, index=False, encoding="utf-8-sig")
            print(f"Wrote {len(cand)} candidate SKUs -> {CANDIDATE_LIST.name} "
                  f"(review, then copy to {PILOT_LIST.name}; approved list NOT overwritten).")
            return 0

        pilot, warnings = load_pilot_skus(con, cfg, args.strict)
        panel, stats = build_model_panel(con, pilot, cfg, args.start_date, args.as_of_date)
        as_of = stats["as_of"]

        # REAL daily series for the synthetic simulation (never feeds same-day inventory)
        real_daily = (panel.groupby(["sku", "date"], as_index=False)["units_observed"]
                           .sum().rename(columns={"units_observed": "units"}))
        sku_meta = panel[["sku", "product_id", "category", "channel"]].drop_duplicates("sku")
        cal_full = calendar_features(
            pd.date_range(panel["date"].min(), as_of, freq="D"), cfg)
        calibration_end = panel["date"].min() + pd.Timedelta(days=int(cfg["synthetic"]["calibration_days"]))
        scenarios, events, sim_params = synthetic_inventory.run(
            real_daily, sku_meta, cal_full, calibration_end, as_of, cfg)
        sim_params["as_of_date"] = as_of.date().isoformat()
        sim_params["sku_selection_cutoff"] = args.selection_cutoff

        ff = build_forecast_features(panel, con, pilot, cfg, as_of)
        inv = build_inventory_context(con, pilot, cfg, as_of, scenarios)

    problems = validate_outputs(panel, ff, inv, scenarios, cfg)
    manifest = build_manifest(panel, ff, inv, scenarios, events, pilot, cfg, args, stats,
                              as_of, warnings, problems)

    if problems and args.strict:
        print("VALIDATION FAILED:", *problems, sep="\n  ")
        write_json(manifest, out_dir / "pilot_manifest.json")
        return 1

    atomic_write(panel, out_dir / "model_panel.parquet")
    atomic_write(ff, out_dir / "forecast_features.parquet")
    atomic_write(inv, out_dir / "inventory_context.parquet")
    write_json(manifest, out_dir / "pilot_manifest.json")
    atomic_write(scenarios, synth_dir / "stockout_scenarios.parquet")
    atomic_write(events if not events.empty else pd.DataFrame(columns=["sku"]),
                 synth_dir / "replenishment_events.parquet")
    write_json(sim_params, synth_dir / "simulation_parameters.json")

    print("============= pilot data built (real demand + synthetic inventory) =============")
    print(f"as_of_date          : {as_of.date()}   history {manifest['history_start']}..{manifest['history_end']}")
    print(f"SKUs / channels     : {manifest['sku_count']} / {manifest['ecommerce_channels']}")
    print(f"model_panel rows    : {manifest['real_sales_row_count']} (forecast-eligible {manifest['forecast_eligible_row_count']})")
    print(f"forecast rows       : {manifest['row_counts']['forecast_features']} (14 days x SKU x channel)")
    print(f"inventory rows      : {manifest['row_counts']['inventory_context']}")
    print(f"synthetic scenarios : {manifest['synthetic_scenario_row_count']} rows across {manifest['n_scenarios']} scenarios; {manifest['replenishment_event_count']} orders")
    print(f"rows dropped > as_of: {manifest['rows_after_as_of_dropped']}")
    print(f"cost: valid {manifest['cost_valid_count']} / imputed {manifest['cost_imputed_count']} / missing {manifest['cost_missing_count']} / nonpos {manifest['cost_non_positive_count']}")
    print("stockout rate by scenario:")
    for scn, d in manifest["stockout_by_scenario"].items():
        print(f"  {scn:<18} stockout {d['stockout_rate']:.0%}  2d {d['stockout_within_2d_rate']:.0%}  7d {d['stockout_within_7d_rate']:.0%}  lost-days {d['lost_sales_days']}")
    print(f"validation          : {manifest['validation_status']}")
    if warnings:
        print("warnings:", *warnings, sep="\n  ")
    if problems:
        print("PROBLEMS (non-strict, written anyway):", *problems, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
