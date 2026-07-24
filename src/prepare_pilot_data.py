"""prepare_pilot_data.py — build the daily, ecommerce-only pilot dataset (naheed_web).

Reads the ETL warehouse (inventory_etl/output/inventory.db) and writes:

  data/processed/model_panel.parquet      REAL daily demand + reconstructed synthetic stock_on_hand
  data/processed/forecast_frame.parquet    next-14-day date x SKU x channel frame awaiting predictions
  data/processed/inventory_context.parquet one current-inventory row per pilot SKU (real snap or synthetic)
  data/processed/pilot_manifest.json       real/synthetic contract, assumptions, counts, validation

WHAT IS REAL vs SYNTHETIC (authoritative contract):
  * Demand is REAL. `units_observed` is real Naheed naheed_web sales and is NEVER modified,
    capped, or replaced. There is NO synthetic demand, NO synthetic sales, NO lost sales,
    NO scenarios and NO exported purchase orders.
  * Only the MISSING daily historical `stock_on_hand` is synthetic — reconstructed with one
    simple deterministic per-SKU balance driven by real sales:
        stock_on_hand[t] = stock_on_hand[t-1] + assumed_replenishment[t] - units_observed[t]
    (assumed replenishment is internal only, to keep the balance plausible; never exported).
  * Real stock snapshots are used for inventory_context ONLY when snapshot_date <= as_of_date.
    July 2026 snapshots therefore never touch a 2026-06-30 run.
  * Unit cost is REAL where valid, transparently imputed otherwise. Cost is a financial field,
    never a demand feature.

The optional multi-scenario what-if simulator is a LATER phase and is intentionally not here.

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

try:
    import holidays as _holidays
except ImportError:  # calendar features degrade gracefully if the lib is absent
    _holidays = None

SCHEMA_VERSION = "4.0-real-demand-synthetic-stock"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "inventory_etl" / "output" / "inventory.db"
DEFAULT_OUT = REPO_ROOT / "data" / "processed"
CONFIG_PATH = REPO_ROOT / "inventory_etl" / "config" / "config.yaml"
PILOT_LIST = REPO_ROOT / "pilot_skus.csv"
CANDIDATE_LIST = REPO_ROOT / "pilot_skus_candidate.csv"

MODEL_PANEL_COLS = [
    "sku", "product_id", "sku_name", "channel", "date", "category", "sub_category", "brand",
    "units_observed", "effective_unit_price", "net_price_paid", "discount_amount", "discount_pct",
    "on_promo", "promo_known_in_advance", "is_public_holiday", "holiday_name",
    "is_payday_window", "day_of_week", "is_weekend", "week_of_year", "month",
    "is_ramadan", "ramadan_day", "ramadan_week",
    "units_lag_1", "units_lag_7", "units_lag_14",
    "units_roll_mean_7", "units_roll_mean_28", "units_roll_std_7",
    "stock_on_hand", "stock_on_hand_is_synthetic", "stock_source", "stock_generation_version",
    "product_active", "forecast_training_eligible", "data_quality_flag",
]
# The ONLY columns a demand model may use as features. Synthetic stock and cost are excluded.
DEMAND_FEATURE_WHITELIST = [
    "units_lag_1", "units_lag_7", "units_lag_14",
    "units_roll_mean_7", "units_roll_mean_28", "units_roll_std_7",
    "effective_unit_price", "discount_pct", "on_promo",
    "is_public_holiday", "is_payday_window", "day_of_week", "is_weekend",
    "week_of_year", "month",
    "is_ramadan", "ramadan_day", "ramadan_week",
]
# Columns that must NEVER be a demand feature (and the synthetic-demand fields that must not exist).
DEMAND_FEATURE_FORBIDDEN = [
    "stock_on_hand", "unit_cost", "unit_cost_observed", "unit_cost_effective",
]
BANNED_COLS = [                       # relics of the removed simulator — must never reappear
    "latent_synthetic_demand", "synthetic_sales", "lost_sales", "is_stockout",
    "scenario_id", "scenario_type", "opening_stock", "ending_stock",
    "replenishment_received", "stockout_within_2d", "stockout_within_7d",
]
FORECAST_FRAME_COLS = [
    "sku", "product_id", "channel", "date", "forecast_horizon_day", "category",
    "sub_category", "brand", "sku_name", "latest_known_price", "trailing_units_mean_7",
    "trailing_units_mean_28", "planned_promo", "planned_discount_pct",
    "is_public_holiday", "holiday_name", "is_payday_window", "day_of_week",
    "is_weekend", "week_of_year", "month", "is_ramadan", "ramadan_day", "ramadan_week",
    "feature_availability_flag",
]
INVENTORY_COLS = [
    "as_of_date", "sku", "product_id", "location_id",
    "stock_on_hand", "stock_on_hand_is_synthetic", "stock_source", "stock_snapshot_date",
    "stock_generation_method", "stock_generation_version",
    "on_order_quantity", "on_order_is_available",
    "expected_daily_demand", "lead_time_days", "lead_time_source",
    "moq", "moq_source", "pack_size", "pack_size_source",
    "safety_stock", "reorder_point", "target_stock", "days_of_cover",
    "is_perishable", "shelf_life_days", "price",
    "unit_cost_observed", "unit_cost_effective", "cost_source", "cost_is_valid",
    "cost_is_imputed", "cost_quality_flag", "cost_currency", "cost_basis",
    "recommended_order_quantity", "recommended_purchase_value", "inventory_value",
    "is_dropship", "assumption_notes",
]

COST_PRECEDENCE = ["magento_eav", "staging_margin", "product_flat"]


# ── config / db ───────────────────────────────────────────────────────────────
def load_config() -> dict:
    if not CONFIG_PATH.exists():
        sys.exit(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key in ("pilot", "replenishment", "external_signals", "synthetic_stock", "cost"):
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


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return bool(con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchall())


# ── calendar (known in advance → safe as future features) ───────────────────────
def calendar_features(dates: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Holiday/payday/day-of-week + Ramazan features for any date range, computed
    deterministically. All values are known in advance, so they are safe future features."""
    es = cfg["external_signals"]
    starts = set(es.get("payday_days_month_start", []))
    ends = set(es.get("payday_days_month_end", []))
    country = es.get("country", "PK")
    hol = {}
    if _holidays is not None and len(dates):
        yrs = range(dates.min().year, dates.max().year + 1)
        hol = dict(_holidays.country_holidays(country, years=yrs))
    # Manually configured public holidays (config wins on a date conflict) — for locally-observed
    # dates the library may miss or date differently, e.g. Eid al-Fitr (lunar calendar).
    for h in (es.get("extra_public_holidays") or []):
        try:
            hd = pd.Timestamp(h["date"]).date()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid extra_public_holidays entry {h!r}: {exc}")
        hol[hd] = h.get("name", "Public Holiday")

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

    # ── Ramazan (configured, known-in-advance; supports multiple years) ──────────────
    # is_ramadan: 1 on [start_date, end_date] inclusive, else 0.
    # ramadan_day: 1-based day count since start_date within the period, else 0.
    # ramadan_week: ((ramadan_day - 1) // 7) + 1 within the period, else 0.
    df["is_ramadan"] = 0
    df["ramadan_day"] = 0
    df["ramadan_week"] = 0
    for period in (es.get("ramadan_periods") or []):
        try:
            p_start = pd.Timestamp(period["start_date"])
            p_end = pd.Timestamp(period["end_date"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid ramadan_periods entry {period!r}: {exc}")
        if p_end < p_start:
            raise ValueError(
                f"Invalid Ramazan period: end_date {period.get('end_date')!r} is earlier than "
                f"start_date {period.get('start_date')!r}")
        mask = (df["date"] >= p_start) & (df["date"] <= p_end)
        if not mask.any():
            continue
        day = ((df["date"] - p_start).dt.days + 1)          # 1-based within the period
        df.loc[mask, "is_ramadan"] = 1
        df.loc[mask, "ramadan_day"] = day[mask].astype(int)
        df.loc[mask, "ramadan_week"] = (((day[mask] - 1) // 7) + 1).astype(int)
    df["is_ramadan"] = df["is_ramadan"].astype(int)
    df["ramadan_day"] = df["ramadan_day"].astype(int)
    df["ramadan_week"] = df["ramadan_week"].astype(int)
    return df


# ── channel mapping ─────────────────────────────────────────────────────────────
def map_channels(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, int, dict]:
    """Keep ecommerce rows only. Returns (ecommerce_df, physical_excluded, unknown_counts)."""
    emap = cfg["pilot"]["ecommerce_channel_map"]
    physical = set(cfg["pilot"]["physical_channels"])
    src = df["channel"].astype(str)
    physical_excluded = int(src.isin(physical).sum())
    known = set(emap) | physical
    unknown_counts = df.loc[~src.isin(known), "channel"].value_counts().to_dict()
    keep = df[src.isin(emap)].copy()
    keep["channel"] = keep["channel"].map(emap)     # explicit map, never silent
    return keep, physical_excluded, unknown_counts


# ── sku selection / validation ──────────────────────────────────────────────────
def _ecommerce_sales_sql(cfg: dict) -> str:
    ecom = list(cfg["pilot"]["ecommerce_channel_map"])
    inlist = ",".join(f"'{c}'" for c in ecom)
    return (f"SELECT s.sku_id AS sku, st.channel, st.quantity_sold, st.transaction_date "
            f"FROM sales_transactions st JOIN sku_master s ON s.sku_id=st.sku_id "
            f"WHERE st.channel IN ({inlist})")


def reselect_candidates(con: sqlite3.Connection, cfg: dict, cutoff: str) -> pd.DataFrame:
    """Deterministic top-N-per-category selection using ONLY data on/before the cutoff."""
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


# ── per-SKU replenishment assumptions (real overrides default) ────────────────────
def sku_replenishment_params(con: sqlite3.Connection, skus: list[str], cfg: dict) -> dict:
    """lead time / MOQ / pack size per SKU. Uses real sku_master values when they look
    real (>0 / >1), otherwise the documented pilot default. Records the source of each."""
    s = cfg["synthetic_stock"]
    ph = ",".join("?" * len(skus))
    sm = pd.read_sql(
        f"SELECT sku_id AS sku, moq, pack_size, supplier_lead_time_days "
        f"FROM sku_master WHERE sku_id IN ({ph})", con, params=skus).set_index("sku")
    out = {}
    for sku in skus:
        r = sm.loc[sku] if sku in sm.index else None
        lt = pd.to_numeric(r["supplier_lead_time_days"], errors="coerce") if r is not None else np.nan
        moq = pd.to_numeric(r["moq"], errors="coerce") if r is not None else np.nan
        pack = pd.to_numeric(r["pack_size"], errors="coerce") if r is not None else np.nan
        lead_ok = pd.notna(lt) and lt > 0
        moq_ok = pd.notna(moq) and moq > 1          # default is 1; >1 means a real MOQ was set
        pack_ok = pd.notna(pack) and pack > 1        # default is 1; >1 means a real case-pack
        out[sku] = {
            "lead_time_days": int(lt) if lead_ok else int(s["default_lead_time_days"]),
            "lead_time_source": "sku_master_picking_mode" if lead_ok else "assumed_default",
            "moq": int(moq) if moq_ok else int(s["default_moq"]),
            "moq_source": "sku_master" if moq_ok else "assumed_default",
            "pack_size": int(pack) if pack_ok else int(s["default_pack_size"]),
            "pack_size_source": "sku_master_case_pack" if pack_ok else "assumed_default",
        }
    return out


# ── synthetic daily stock reconstruction (deterministic; real sales untouched) ─────
def reconstruct_stock(panel: pd.DataFrame, repl: dict, cfg: dict) -> tuple[pd.Series, dict]:
    """Rebuild end-of-day daily stock_on_hand per SKU x channel with one simple balance:

        available   = prior_ending + assumed_replenishment_arrivals
        (if available < units_observed -> add an assumed replenishment so real sales are covered)
        stock_on_hand = available - units_observed      (>= 0, integral)

    Real `units_observed` are subtracted but never modified or capped. Assumed replenishment
    is internal only. Deterministic (no RNG): identical config+data -> identical output.
    Returns (stock Series aligned to panel.index, per-SKU expected daily demand).
    """
    s = cfg["synthetic_stock"]
    init_cd, tgt_cd, safe_d = int(s["initial_cover_days"]), int(s["target_cover_days"]), int(s["safety_days"])
    out = pd.Series(index=panel.index, dtype="int64")
    exp_demand: dict = {}
    for (sku, ch), g in panel.groupby(["sku", "channel"], sort=False):
        u = g["units_observed"].to_numpy(dtype=float)
        n = len(u)
        avg = max(float(u.mean()), 0.1)
        exp_demand[sku] = avg
        p = repl[sku]
        lead, moq, pack = p["lead_time_days"], p["moq"], p["pack_size"]
        safety = round(avg * safe_d)
        rop = round(avg * lead + safety)
        order_up_to = round(avg * tgt_cd + safety)
        initial = max(int(round(avg * init_cd)), int(math.ceil(u[0])) if n else 0)

        arrivals = np.zeros(n + lead + 2)
        soh = np.zeros(n, dtype=np.int64)
        prev = float(initial)
        pending = -1
        for t in range(n):
            available = prev + arrivals[t]
            if pending == t:
                pending = -1
            if available < u[t]:                     # keep the balance able to cover REAL sales
                deficit = u[t] - available
                available += max(moq, math.ceil(deficit / pack) * pack)
            end = max(0.0, available - u[t])         # subtract real sales; never negative
            soh[t] = int(round(end))
            if end <= rop and pending < 0:           # simple reorder so stock doesn't stay at 0
                qty = max(moq, math.ceil(max(0.0, order_up_to - end) / pack) * pack)
                j = t + lead
                if j < len(arrivals):
                    arrivals[j] += qty
                    pending = j
            prev = end
        out.loc[g.index] = soh
    return out, exp_demand


# ── model panel (REAL demand + synthetic stock) ───────────────────────────────────
def build_model_panel(con: sqlite3.Connection, pilot: pd.DataFrame, cfg: dict,
                      start: str | None, as_of: str | None
                      ) -> tuple[pd.DataFrame, dict]:
    """Daily SKU x naheed_web panel of REAL sales over each SKU's active period, plus a
    reconstructed synthetic `stock_on_hand`. `as_of` is a HARD boundary applied first."""
    skus = pilot["sku"].astype(str).tolist()
    ph = ",".join("?" * len(skus))
    attrs = pd.read_sql(
        f"SELECT sku_id AS sku, product_id, sku_name, category, sub_category, brand, price "
        f"FROM sku_master WHERE sku_id IN ({ph})", con, params=skus)

    raw = pd.read_sql(
        f"SELECT sku_id AS sku, channel, transaction_date, quantity_sold, qty_ordered, "
        f"       discount_amount, row_total FROM sales_transactions WHERE sku_id IN ({ph})",
        con, params=skus)
    raw["date"] = pd.to_datetime(raw["transaction_date"])
    ecom, physical_excluded, unknown = map_channels(raw, cfg)
    if start:
        ecom = ecom[ecom["date"] >= pd.Timestamp(start)]
    as_of_ts = pd.Timestamp(as_of) if as_of else (ecom["date"].max() if not ecom.empty else pd.NaT)
    rows_after_as_of = int((ecom["date"] > as_of_ts).sum()) if as_of else 0
    ecom = ecom[ecom["date"] <= as_of_ts]        # HARD as_of boundary — before any features
    if ecom.empty:
        sys.exit("No ecommerce sales found for the pilot SKUs in the given window (<= as_of).")
    window_end = as_of_ts

    agg = (ecom.groupby(["sku", "channel", "date"])
               .agg(units_observed=("quantity_sold", "sum"),
                    ordered_qty=("qty_ordered", "sum"),
                    discount_amount=("discount_amount", "sum"),
                    net_rev=("row_total", "sum"))
               .reset_index())
    agg["units_observed"] = agg["units_observed"].clip(lower=0).round().astype(int)

    starts = agg.groupby(["sku", "channel"])["date"].min().reset_index(name="active_start")
    frames = []
    for (sku, ch), grp in starts.groupby(["sku", "channel"]):
        days = pd.date_range(grp["active_start"].iloc[0], window_end, freq="D")
        frames.append(pd.DataFrame({"sku": sku, "channel": ch, "date": days}))
    panel = pd.concat(frames, ignore_index=True).merge(agg, on=["sku", "channel", "date"], how="left")

    panel["units_observed"] = panel["units_observed"].fillna(0).astype(int)   # genuine in-window zeros
    panel["ordered_qty"] = panel["ordered_qty"].fillna(0.0)
    panel["discount_amount"] = panel["discount_amount"].fillna(0.0)
    panel["net_rev"] = panel["net_rev"].fillna(0.0)
    panel["product_active"] = True
    panel = panel.sort_values(["sku", "channel", "date"]).reset_index(drop=True)

    # Per-unit price on the ORDERED-quantity basis. row_total is the gross line total
    # (unit_price * qty_ordered, before discount), so divide by qty_ordered — NOT by
    # units_observed (which nets out cancellations/refunds and previously inflated the price
    # up to ~50x when an order was mostly cancelled). effective_unit_price = list/selling price;
    # net_price_paid = after-discount price actually paid. Both carried forward on no-sale days.
    ordered = panel["ordered_qty"].replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        list_price = panel["net_rev"] / ordered
        net_paid = (panel["net_rev"] - panel["discount_amount"]) / ordered
    panel["effective_unit_price"] = list_price
    panel["effective_unit_price"] = panel.groupby(["sku", "channel"])["effective_unit_price"].ffill()
    panel["net_price_paid"] = net_paid
    panel["net_price_paid"] = panel.groupby(["sku", "channel"])["net_price_paid"].ffill()
    gross = panel["net_rev"] + panel["discount_amount"]
    panel["discount_pct"] = np.where(gross > 0, panel["discount_amount"] / gross, 0.0)
    panel["on_promo"] = (panel["discount_amount"] > 0).astype(int)
    panel["promo_known_in_advance"] = 0

    cal = calendar_features(pd.DatetimeIndex(panel["date"].unique()), cfg)
    panel = panel.merge(cal, on="date", how="left")
    panel = panel.merge(attrs.drop(columns=["price"]), on="sku", how="left")

    # CAUSAL demand features: grouped shift BEFORE rolling so date t uses only data <= t-1.
    grp = panel.groupby(["sku", "channel"])["units_observed"]
    panel["units_lag_1"] = grp.shift(1)
    panel["units_lag_7"] = grp.shift(7)
    panel["units_lag_14"] = grp.shift(14)
    base = grp.shift(1)
    keyed = base.groupby([panel["sku"], panel["channel"]])
    panel["units_roll_mean_7"] = keyed.transform(lambda x: x.rolling(7, min_periods=1).mean())
    panel["units_roll_mean_28"] = keyed.transform(lambda x: x.rolling(28, min_periods=1).mean())
    panel["units_roll_std_7"] = keyed.transform(lambda x: x.rolling(7, min_periods=1).std())

    # synthetic daily stock (missing history filled deterministically; real sales untouched)
    repl = sku_replenishment_params(con, skus, cfg)
    stock, _ = reconstruct_stock(panel, repl, cfg)
    panel["stock_on_hand"] = stock.astype(int)
    panel["stock_on_hand_is_synthetic"] = True
    panel["stock_source"] = "synthetic_reconstruction"
    panel["stock_generation_version"] = cfg["synthetic_stock"]["version"]

    # forecast eligibility: real-data validity + sufficient history ONLY (never synthetic-derived).
    didx = panel.groupby(["sku", "channel"]).cumcount()
    min_hist = 14
    panel["forecast_training_eligible"] = panel["product_active"] & (didx >= min_hist)

    price_bad = (panel["units_observed"] > 0) & (
        panel["effective_unit_price"].isna() | (panel["effective_unit_price"] <= 0))

    def flag(d, pb):
        f = ["activation_inferred_from_first_sale"]
        if d < min_hist:
            f.append("insufficient_history")
        if pb:
            f.append("price_anomaly")
        return ";".join(f)
    panel["data_quality_flag"] = [flag(d, pb) for d, pb in zip(didx.to_numpy(), price_bad.to_numpy())]

    panel["holiday_name"] = panel["holiday_name"].where(panel["holiday_name"].notna(), None)
    stats = {"physical_store_rows_excluded": physical_excluded, "unknown_channel_rows": unknown,
             "as_of": window_end, "rows_after_as_of_dropped": rows_after_as_of,
             "repl_params": repl}
    return panel[MODEL_PANEL_COLS], stats


# ── forecast frame (real, known-at-as_of; no target, no cost) ──────────────────────
def build_forecast_frame(panel: pd.DataFrame, con: sqlite3.Connection, pilot: pd.DataFrame,
                         cfg: dict, as_of: pd.Timestamp) -> pd.DataFrame:
    """Exactly N future days per SKU/channel awaiting predictions (no leakage, no actuals)."""
    n_days = int(cfg["pilot"]["forecast_feature_days"])
    future = pd.date_range(as_of + pd.Timedelta(days=1), periods=n_days, freq="D")
    keys = panel[["sku", "product_id", "sku_name", "channel", "category", "sub_category", "brand"]].drop_duplicates()

    hist = panel[panel["date"] <= as_of]
    last_price = (hist.dropna(subset=["effective_unit_price"])
                      .sort_values("date").groupby(["sku", "channel"])["effective_unit_price"].last())
    tail = hist.sort_values("date").groupby(["sku", "channel"])["units_observed"]
    trail7 = tail.apply(lambda x: float(x.tail(7).mean()) if len(x) else np.nan)
    trail28 = tail.apply(lambda x: float(x.tail(28).mean()) if len(x) else np.nan)
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
    return ff.sort_values(["sku", "channel", "date"]).reset_index(drop=True)[FORECAST_FRAME_COLS]


# ── real stock snapshot selection (only snapshot_date <= as_of) ────────────────────
def select_real_snapshot(con: sqlite3.Connection, prod_ids: list[int], as_of: pd.Timestamp
                         ) -> tuple[dict, int, int]:
    """Latest ELIGIBLE real stock per product (snapshot_date <= as_of), preferring the ALL
    location (company-wide total). Never sums ALL with warehouse rows (double count).
    Returns (product_id -> {stock, snapshot_date, location, table}, n_eligible, n_future_excluded)."""
    parts = []
    for t in ("inventory_snapshot", "inventory_snapshot_history"):
        if _table_exists(con, t) and prod_ids:
            ph = ",".join("?" * len(prod_ids))
            df = pd.read_sql(
                f"SELECT product_id, snapshot_date, location_id, stock_on_hand FROM {t} "
                f"WHERE product_id IN ({ph})", con, params=prod_ids)
            if not df.empty:
                df["table"] = t
                parts.append(df)
    if not parts:
        return {}, 0, 0
    snaps = pd.concat(parts, ignore_index=True)
    snaps["snapshot_date"] = pd.to_datetime(snaps["snapshot_date"])
    n_future_excluded = int(snaps.loc[snaps["snapshot_date"] > as_of, "snapshot_date"].nunique())
    snaps = snaps[snaps["snapshot_date"] <= as_of]              # HARD cutoff: no future snapshot
    chosen: dict = {}
    for pid, g in snaps.groupby("product_id"):
        latest_date = g["snapshot_date"].max()
        day = g[g["snapshot_date"] == latest_date]
        allrow = day[day["location_id"] == "ALL"]
        if len(allrow):                                        # prefer the company-wide ALL total
            r = allrow.iloc[0]
            stock, loc = float(r["stock_on_hand"]), "ALL"
        else:                                                  # else sum deduped warehouse rows
            wh = day.drop_duplicates(["location_id"])
            stock, loc = float(wh["stock_on_hand"].sum()), "+".join(sorted(wh["location_id"]))
        chosen[int(pid)] = {"stock": max(0, int(round(stock))),
                            "snapshot_date": latest_date.date().isoformat(),
                            "location": loc, "table": str(day.iloc[0]["table"])}
    return chosen, len(chosen), n_future_excluded


# ── unit-cost validation ──────────────────────────────────────────────────────────
def classify_cost(candidates: list[tuple[str, object]], price, tol: float) -> dict:
    """Resolve one SKU's cost from precedence-ordered (source, value) candidates.
    Valid iff numeric, finite and > 0. Zero/negative are INVALID (not just present)."""
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
    """Add validated cost columns: observed (valid DB value or null) vs effective (observed or
    documented category->global median fallback). Retains that invalid costs were invalid."""
    ccfg = cfg["cost"]
    tol = float(ccfg.get("conflict_tolerance_pct", 0.25))
    cols = _table_columns(con, "sku_master")
    per_source = {"eav_cost", "margin_cost", "flat_cost"} <= cols
    has_source_col = "cost_source" in cols

    recs = []
    for _, r in sm.iterrows():
        if per_source:
            cands = [("magento_eav", r.get("eav_cost")), ("staging_margin", r.get("margin_cost")),
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

    valid_obs = sm.loc[sm["cost_is_valid"], ["category", "unit_cost_observed"]]
    cat_median = valid_obs.groupby("category")["unit_cost_observed"].median().to_dict()
    global_median = float(valid_obs["unit_cost_observed"].median()) if len(valid_obs) else np.nan

    eff, imputed, source, flags = [], [], [], []
    for i, r in sm.reset_index(drop=True).iterrows():
        fl = list(recs[i]["flags"])
        if r["cost_is_valid"]:
            eff.append(float(r["unit_cost_observed"])); imputed.append(False); source.append(r["cost_source"])
        else:
            cm = cat_median.get(r["category"], np.nan)
            if pd.notna(cm):
                eff.append(float(cm)); source.append("category_median_fallback"); fl.append("IMPUTED_CATEGORY_MEDIAN")
            elif pd.notna(global_median):
                eff.append(float(global_median)); source.append("global_median_fallback"); fl.append("IMPUTED_GLOBAL_MEDIAN")
            else:
                eff.append(np.nan); source.append("missing")
            imputed.append(True)
        fl.append("PACK_UNIT_BASIS_UNCONFIRMED")
        flags.append(";".join(dict.fromkeys(fl)))
    sm["unit_cost_effective"] = eff
    sm["cost_is_imputed"] = imputed
    sm["cost_source"] = source
    sm["cost_quality_flag"] = flags
    sm["cost_currency"] = ccfg.get("currency", "PKR")
    sm["cost_basis"] = ccfg.get("basis", "sellable_sku_unit_unconfirmed")
    return sm.drop(columns=[], errors="ignore")


# ── inventory context (one current row per SKU) ────────────────────────────────────
def build_inventory_context(con: sqlite3.Connection, pilot: pd.DataFrame, cfg: dict,
                            as_of: pd.Timestamp, panel: pd.DataFrame, repl: dict,
                            snapshot: dict) -> pd.DataFrame:
    """Current inventory per SKU. Uses the eligible REAL snapshot (snapshot_date <= as_of) if
    one exists, otherwise the final synthetic reconstructed stock at as_of. Adds validated cost
    and a transparent reorder recommendation from a simple demand estimate."""
    s = cfg["synthetic_stock"]
    skus = pilot["sku"].astype(str).tolist()
    ph = ",".join("?" * len(skus))
    base_cols = "sku_id AS sku, product_id, category, is_perishable, shelf_life_days, unit_cost, price, is_dropship"
    extra = [c for c in ("eav_cost", "margin_cost", "flat_cost", "cost_source")
             if c in _table_columns(con, "sku_master")]
    sel = base_cols + "".join(f", {c}" for c in extra)
    sm = pd.read_sql(f"SELECT {sel} FROM sku_master WHERE sku_id IN ({ph})", con, params=skus)
    sm = resolve_costs(sm, con, cfg)

    # final synthetic stock at as_of + expected daily demand from the reconstruction
    hist = panel[panel["date"] <= as_of].sort_values("date")
    synth_last = hist.groupby("sku")["stock_on_hand"].last().to_dict()
    exp_daily = hist.groupby("sku")["units_observed"].mean().to_dict()
    tgt_cd, safe_d = int(s["target_cover_days"]), int(s["safety_days"])

    rows = []
    for _, r in sm.iterrows():
        sku, pid = r["sku"], int(r["product_id"])
        rp = repl[sku]
        lead, moq, pack = rp["lead_time_days"], rp["moq"], rp["pack_size"]
        snap = snapshot.get(pid)
        if snap is not None:                       # eligible REAL snapshot on/before as_of
            soh = int(snap["stock"]); is_syn = False
            src = f"real_snapshot:{snap['table']}:{snap['location']}"
            snap_date = snap["snapshot_date"]; method = "real_snapshot"
        else:                                      # no eligible real snapshot -> synthetic final
            soh = int(synth_last.get(sku, 0)); is_syn = True
            src = "synthetic_reconstruction"
            snap_date = as_of.date().isoformat(); method = s["method"]

        avg = max(float(exp_daily.get(sku, 0.0)), 0.0)
        safety = round(avg * safe_d)
        reorder_point = round(avg * lead + safety)
        target_stock = round(avg * tgt_cd + safety)
        on_order = 0                               # no confirmed inbound data
        raw = max(0, target_stock - soh - on_order)
        rec_qty = 0 if raw == 0 else int(max(moq, math.ceil(raw / pack) * pack))
        eff = r["unit_cost_effective"]
        rec_val = rec_qty * eff if pd.notna(eff) else np.nan
        inv_val = soh * eff if pd.notna(eff) else np.nan
        doc = round(soh / avg, 2) if avg > 0 else np.nan

        note = (f"stock_on_hand is {'a REAL snapshot' if not is_syn else 'SYNTHETIC (deterministic '\
                f'reconstruction — not real Naheed stock)'}. Lead time ({lead}d, {rp['lead_time_source']}), "
                f"MOQ ({moq}, {rp['moq_source']}), pack size ({pack}, {rp['pack_size_source']}) — "
                f"assumptions where sourced as 'assumed_default'. on_order unavailable (0). "
                f"Reorder qty is a recommendation from a simple demand estimate, not a placed order. "
                f"Unit cost validated; effective may be an imputed fallback (see cost_quality_flag).")
        rows.append({
            "as_of_date": as_of.date().isoformat(), "sku": sku, "product_id": pid,
            "location_id": (snap["location"] if snap is not None else "ALL"),
            "stock_on_hand": soh, "stock_on_hand_is_synthetic": is_syn, "stock_source": src,
            "stock_snapshot_date": snap_date, "stock_generation_method": method,
            "stock_generation_version": s["version"],
            "on_order_quantity": on_order, "on_order_is_available": False,
            "expected_daily_demand": round(avg, 4), "lead_time_days": lead,
            "lead_time_source": rp["lead_time_source"], "moq": moq, "moq_source": rp["moq_source"],
            "pack_size": pack, "pack_size_source": rp["pack_size_source"],
            "safety_stock": safety, "reorder_point": reorder_point, "target_stock": target_stock,
            "days_of_cover": doc,
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
            "assumption_notes": note,
        })
    return pd.DataFrame(rows)[INVENTORY_COLS]


# ── validation (fails loudly) ───────────────────────────────────────────────────────
def validate_outputs(panel, ff, inv, pilot, cfg, as_of, snapshot_used_real: bool) -> list[str]:
    problems: list[str] = []
    n_pilot = pilot["sku"].nunique()
    allowed = set(cfg["pilot"]["ecommerce_channel_map"].values())

    if list(panel.columns) != MODEL_PANEL_COLS:
        problems.append("model_panel columns mismatch")
    if list(ff.columns) != FORECAST_FRAME_COLS:
        problems.append("forecast_frame columns mismatch")
    if list(inv.columns) != INVENTORY_COLS:
        problems.append("inventory_context columns mismatch")
    for name, df in (("model_panel", panel), ("forecast_frame", ff), ("inventory_context", inv)):
        banned = [c for c in BANNED_COLS if c in df.columns]
        if banned:
            problems.append(f"LEAKAGE: banned synthetic-scenario columns in {name}: {banned}")
    if any(c in ff.columns for c in ("units_observed", "unit_cost", "unit_cost_effective")):
        problems.append("LEAKAGE: forecast_frame must not contain actuals or cost")
    # as_of hard boundary
    if panel["date"].max() > as_of:
        problems.append("model_panel contains dates after as_of")
    if pd.to_datetime(ff["date"]).min() <= as_of:
        problems.append("forecast_frame must contain only future dates (> as_of)")
    # channel / sku scope
    if not set(panel["channel"]).issubset(allowed):
        problems.append(f"unexpected channels in model_panel: {set(panel['channel']) - allowed}")
    if panel["sku"].nunique() != n_pilot:
        problems.append(f"model_panel has {panel['sku'].nunique()} SKUs, expected {n_pilot}")
    # unique daily keys
    if panel.duplicated(["sku", "channel", "date"]).any():
        problems.append("duplicate sku+channel+date in model_panel")
    if inv.duplicated(["sku"]).any():
        problems.append("duplicate sku in inventory_context")
    # real demand integrity
    if (panel["units_observed"] < 0).any():
        problems.append("negative units_observed")
    # stock sanity
    soh = pd.to_numeric(panel["stock_on_hand"], errors="coerce")
    if soh.isna().any() or (soh < 0).any() or not np.isfinite(soh).all():
        problems.append("stock_on_hand must be non-negative and finite")
    if (soh != soh.round()).any():
        problems.append("stock_on_hand must be integral")
    # current synthetic inventory == final synthetic history (only when synthetic used)
    if not snapshot_used_real:
        last = panel.sort_values("date").groupby("sku")["stock_on_hand"].last()
        inv_soh = inv.set_index("sku")["stock_on_hand"]
        if not last.reindex(inv_soh.index).astype(int).equals(inv_soh.astype(int)):
            problems.append("inventory_context synthetic stock != final model_panel stock")
    # snapshot cutoff (June run must be fully synthetic)
    if snapshot_used_real and (pd.to_datetime(inv["stock_snapshot_date"]) > as_of).any():
        problems.append("a real snapshot with snapshot_date > as_of was used")
    # cost
    eff = pd.to_numeric(inv["unit_cost_effective"], errors="coerce")
    if ((eff <= 0) & eff.notna()).any() or (~np.isfinite(eff.dropna())).any():
        problems.append("unit_cost_effective must be positive and finite when present")
    # forecast frame width
    n_days = int(cfg["pilot"]["forecast_feature_days"])
    per = ff.groupby(["sku", "channel"])["date"].nunique()
    if not (per == n_days).all():
        problems.append(f"forecast_frame must have exactly {n_days} future days per sku/channel")
    if len(ff) != n_pilot * n_days * len(allowed & set(panel["channel"])):
        pass  # informational; per-key check above is the binding one
    return problems


# ── manifest ──────────────────────────────────────────────────────────────────────
def build_manifest(panel, ff, inv, pilot, cfg, args, stats, as_of,
                   n_real_snap, n_future_excluded, warnings, problems) -> dict:
    def rate(mask):
        return round(float(mask.mean()), 4) if len(mask) else 0.0
    real_stock_rows = int((~inv["stock_on_hand_is_synthetic"]).sum())
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": args.generated_at,
        "as_of_date": as_of.date().isoformat(),
        "historical_window": [panel["date"].min().date().isoformat(), as_of.date().isoformat()],
        "sku_selection_cutoff": args.selection_cutoff,
        "channel_scope": sorted(panel["channel"].unique().tolist()),
        "forecast_horizon_days": cfg["pilot"]["forecast_horizons"],
        "future_frame_days": int(cfg["pilot"]["forecast_feature_days"]),
        "selected_skus": pilot["sku"].astype(str).tolist(),
        "sku_count": int(panel["sku"].nunique()),
        "rows_after_as_of_dropped": stats["rows_after_as_of_dropped"],
        "future_real_snapshots_excluded": n_future_excluded,
        "real_snapshots_eligible": n_real_snap,
        "inventory_context_real_stock_rows": real_stock_rows,
        "inventory_context_synthetic_stock_rows": int(inv["stock_on_hand_is_synthetic"].sum()),
        "real_fields": [
            "date", "sku", "product_id", "channel(naheed_web)", "units_observed",
            "effective_unit_price", "discount_amount", "discount_pct", "on_promo",
            "calendar features", "unit_cost_observed (where valid)",
            "inventory_context stock_on_hand (only when a real snapshot <= as_of exists)",
        ],
        "synthetic_or_assumed_fields": [
            "stock_on_hand (daily historical reconstruction)", "assumed replenishment (internal)",
            "lead_time_days / moq / pack_size where marked assumed_default",
            "unit_cost_effective where cost_is_imputed", "on_order_quantity (0, unavailable)",
        ],
        "synthetic_stock_method": cfg["synthetic_stock"]["method"],
        "synthetic_stock_version": cfg["synthetic_stock"]["version"],
        "stock_reconstruction_balance": "stock[t] = stock[t-1] + assumed_replenishment[t] - units_observed[t]",
        "assumptions": {
            "initial_cover_days": cfg["synthetic_stock"]["initial_cover_days"],
            "target_cover_days": cfg["synthetic_stock"]["target_cover_days"],
            "safety_days": cfg["synthetic_stock"]["safety_days"],
            "default_lead_time_days": cfg["synthetic_stock"]["default_lead_time_days"],
            "default_moq": cfg["synthetic_stock"]["default_moq"],
            "default_pack_size": cfg["synthetic_stock"]["default_pack_size"],
        },
        "real_snapshot_rule": "use latest real snapshot with snapshot_date <= as_of_date; "
                              "prefer ALL location; never sum ALL with warehouse rows; "
                              "never reverse-fill history from future snapshots",
        "unit_cost_imputation": "precedence magento_eav->staging_margin->product_flat (valid=finite>0); "
                                "invalid/missing -> category median -> global median (flagged imputed)",
        "cost_currency": cfg["cost"].get("currency"),
        "cost_basis": cfg["cost"].get("basis"),
        "cost_valid_count": int(inv["cost_is_valid"].sum()),
        "cost_imputed_count": int(inv["cost_is_imputed"].sum()),
        "cost_non_positive_count": int(inv["cost_quality_flag"].str.contains("NON_POSITIVE_COST").sum()),
        "cost_missing_count": int(inv["cost_quality_flag"].str.contains("MISSING_COST").sum()),
        "cost_above_price_count": int(inv["cost_quality_flag"].str.contains("COST_ABOVE_PRICE").sum()),
        "demand_feature_whitelist": DEMAND_FEATURE_WHITELIST,
        "demand_feature_excluded": DEMAND_FEATURE_FORBIDDEN,
        "ramadan_periods_configured": cfg["external_signals"].get("ramadan_periods", []),
        "extra_public_holidays_configured": cfg["external_signals"].get("extra_public_holidays", []),
        "ramadan_signal_note": (
            "Ramazan dates are configured known-in-advance Karachi calendar signals "
            "(manually configured business inputs, NOT auto-verified by an external calendar "
            "service). The current pilot contains only one Ramazan period (2026), so any learned "
            "Ramazan effect is exploratory."),
        "output_paths": {
            "model_panel": "data/processed/model_panel.parquet",
            "forecast_frame": "data/processed/forecast_frame.parquet",
            "inventory_context": "data/processed/inventory_context.parquet",
            "manifest": "data/processed/pilot_manifest.json",
        },
        "row_counts": {
            "model_panel": int(len(panel)),
            "forecast_frame": int(len(ff)),
            "inventory_context": int(len(inv)),
            "forecast_eligible": int(panel["forecast_training_eligible"].sum()),
        },
        "zero_sales_rate": rate(panel["units_observed"] == 0),
        "stock_source_counts": inv["stock_source"].value_counts().to_dict(),
        "physical_store_rows_excluded": stats["physical_store_rows_excluded"],
        "unknown_channel_rows": stats["unknown_channel_rows"],
        "note": "Demand is REAL and untouched. Only missing daily stock_on_hand is synthetic. "
                "Stockout risk / reorder recommendations are computed downstream from forecasts + "
                "this inventory context; they are pilot estimates, not validated against real "
                "Naheed stockouts.",
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


def cleanup_obsolete(out_dir: Path) -> list[str]:
    """Remove obsolete generated artifacts from the retired multi-scenario simulator.
    Only touches known GENERATED files under data/synthetic/ — never raw/source data."""
    removed = []
    synth = out_dir.parent / "synthetic"
    for name in ("stockout_scenarios.parquet", "replenishment_events.parquet",
                 "simulation_parameters.json"):
        f = synth / name
        if f.exists():
            f.unlink()
            removed.append(str(f.relative_to(REPO_ROOT)))
    # also drop the retired v3 output name if present
    old_ff = out_dir / "forecast_features.parquet"
    if old_ff.exists():
        old_ff.unlink()
        removed.append(str(old_ff.relative_to(REPO_ROOT)))
    if synth.exists() and not any(synth.iterdir()):
        synth.rmdir()
    return removed


# ── cli ──────────────────────────────────────────────────────────────────────────
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the daily naheed_web pilot dataset.")
    ap.add_argument("--db-path", default=str(DEFAULT_DB))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--as-of-date", default=None, help="HARD boundary; records after this are dropped")
    ap.add_argument("--selection-cutoff", default=None, help="recorded; used with --reselect-pilot-skus")
    ap.add_argument("--reselect-pilot-skus", action="store_true")
    ap.add_argument("--strict", action="store_true")
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.generated_at = dt.datetime.now().isoformat(timespec="seconds")
    cfg = load_config()
    out_dir = Path(args.output_dir)

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
        repl = stats["repl_params"]

        prod_ids = pd.read_sql(
            f"SELECT product_id FROM sku_master WHERE sku_id IN ({','.join('?'*len(pilot))})",
            con, params=pilot["sku"].astype(str).tolist())["product_id"].astype(int).tolist()
        snapshot, n_real_snap, n_future_excluded = select_real_snapshot(con, prod_ids, as_of)

        ff = build_forecast_frame(panel, con, pilot, cfg, as_of)
        inv = build_inventory_context(con, pilot, cfg, as_of, panel, repl, snapshot)

    snapshot_used_real = bool((~inv["stock_on_hand_is_synthetic"]).any())
    problems = validate_outputs(panel, ff, inv, pilot, cfg, as_of, snapshot_used_real)
    manifest = build_manifest(panel, ff, inv, pilot, cfg, args, stats, as_of,
                              n_real_snap, n_future_excluded, warnings, problems)

    if problems:                                   # fail loudly; do not write partial outputs
        print("VALIDATION FAILED — outputs NOT written:", *problems, sep="\n  ")
        write_json(manifest, out_dir / "pilot_manifest.json")
        return 1

    removed = cleanup_obsolete(out_dir)
    atomic_write(panel, out_dir / "model_panel.parquet")
    atomic_write(ff, out_dir / "forecast_frame.parquet")
    atomic_write(inv, out_dir / "inventory_context.parquet")
    write_json(manifest, out_dir / "pilot_manifest.json")

    print("============= pilot data built (REAL demand + SYNTHETIC daily stock) =============")
    print(f"as_of_date          : {as_of.date()}   history {manifest['historical_window'][0]}..{manifest['historical_window'][1]}")
    print(f"pilot SKUs / channel: {manifest['sku_count']} / {manifest['channel_scope']}")
    print(f"model_panel rows    : {manifest['row_counts']['model_panel']} (forecast-eligible {manifest['row_counts']['forecast_eligible']})")
    print(f"forecast_frame rows : {manifest['row_counts']['forecast_frame']} ({manifest['future_frame_days']} future days x SKU x channel)")
    print(f"inventory rows      : {manifest['row_counts']['inventory_context']}   synthetic stock rows in panel: {manifest['row_counts']['model_panel']}")
    print(f"rows dropped > as_of: {manifest['rows_after_as_of_dropped']}")
    print(f"real snapshots      : eligible (<= as_of) {n_real_snap}; future excluded (> as_of) {n_future_excluded}")
    print(f"stock source counts : {manifest['stock_source_counts']}")
    print(f"cost                : valid {manifest['cost_valid_count']} / imputed {manifest['cost_imputed_count']} / missing {manifest['cost_missing_count']} / nonpos {manifest['cost_non_positive_count']}")
    if removed:
        print(f"removed obsolete    : {removed}")
    print(f"validation          : {manifest['validation_status']}")
    if warnings:
        print("warnings:", *warnings, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
