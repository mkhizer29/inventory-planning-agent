"""prepare_pilot_data.py — build the daily, ecommerce-only, leakage-free pilot dataset.

Reads the ETL warehouse (inventory_etl/output/inventory.db) and writes four files to
data/processed/:

  model_panel.parquet        history for training/backtesting (SKU x channel x active day)
  forecast_features.parquet  next-14-day prediction inputs, known-at-as_of only (no leakage)
  inventory_context.parquet  as-of stock + replenishment context (SKU x location), assumptions flagged
  pilot_manifest.json        validation stats, reproducibility and an explicit assumptions block

Scope: ECOMMERCE ONLY. `online_delivery` is normalised to `naheed_web`; `store`/`storepickup`
are excluded (and counted). `foodpanda` has no rows in this warehouse — documented as a
limitation; the `channel` column is kept so it can be added later with no schema change.

Design rules (see the project brief): daily frequency; 7 & 14 day horizons; chronological
splits only; missing stock is never treated as zero; stockout days are marked demand-censored
(not proven zero demand); pre-activation dates are not zero-filled; supplier lead time / MOQ /
stock-in-transit / perishability are configurable assumptions, always flagged.

Run from the repo root:
    python src/prepare_pilot_data.py --as-of-date 2026-07-15
    python src/prepare_pilot_data.py --reselect-pilot-skus --selection-cutoff 2026-05-31
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
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

SCHEMA_VERSION = "2.0-daily-ecommerce"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "inventory_etl" / "output" / "inventory.db"
DEFAULT_OUT = REPO_ROOT / "data" / "processed"
CONFIG_PATH = REPO_ROOT / "inventory_etl" / "config" / "config.yaml"
PILOT_LIST = REPO_ROOT / "pilot_skus.csv"
CANDIDATE_LIST = REPO_ROOT / "pilot_skus_candidate.csv"

MODEL_PANEL_COLS = [
    "sku", "product_id", "channel", "date", "category", "sub_category", "brand",
    "units_observed", "effective_unit_price", "discount_amount", "discount_pct",
    "on_promo", "promo_known_in_advance", "is_public_holiday", "holiday_name",
    "is_payday_window", "day_of_week", "is_weekend", "week_of_year", "month",
    "stock_on_hand", "stock_observation_available", "stock_snapshot_stale",
    "is_available", "is_stockout", "demand_censored", "product_active",
    "training_eligible", "data_quality_flag",
]
FORECAST_COLS = [
    "sku", "product_id", "channel", "date", "forecast_horizon_day", "category",
    "sub_category", "brand", "latest_known_price", "planned_promo",
    "planned_discount_pct", "is_public_holiday", "holiday_name", "is_payday_window",
    "day_of_week", "is_weekend", "week_of_year", "month", "feature_availability_flag",
]
INVENTORY_COLS = [
    "as_of_date", "sku", "product_id", "location_id", "stock_on_hand",
    "stock_in_transit", "supplier_lead_time_days", "moq", "pack_size",
    "is_perishable", "shelf_life_days", "unit_cost", "price", "is_dropship",
    "stock_flag", "lead_time_is_assumed", "moq_is_assumed", "pack_size_is_assumed",
    "stock_in_transit_is_assumed", "perishability_is_assumed", "assumption_notes",
]


# ── config / db ───────────────────────────────────────────────────────────────
def load_config() -> dict:
    if not CONFIG_PATH.exists():
        sys.exit(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key in ("pilot", "replenishment", "external_signals"):
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


# ── calendar (known in advance → safe as future features) ───────────────────────
def calendar_features(dates: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Holiday/payday/day-of-week features for any date range, computed deterministically
    (holidays known in advance). Reuses the payday rule from config.external_signals."""
    es = cfg["external_signals"]
    starts = set(es.get("payday_days_month_start", []))
    ends = set(es.get("payday_days_month_end", []))
    country = es.get("country", "PK")
    hol = {}
    if _holidays is not None:
        yrs = range(dates.min().year, dates.max().year + 1)
        hol = dict(_holidays.country_holidays(country, years=yrs))

    def is_payday(d: dt.date) -> bool:
        last = (dt.date(d.year + d.month // 12, d.month % 12 + 1, 1) - dt.timedelta(days=1)).day
        return d.day in starts or d.day in ends or d.day == last

    df = pd.DataFrame({"date": dates})
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


# ── stock as-of join ────────────────────────────────────────────────────────────
def asof_stock(keys: pd.DataFrame, stock: pd.DataFrame, max_staleness: int) -> pd.DataFrame:
    """As-of join stock snapshots onto (product_id, date) keys with a staleness cap.
    Returns keys + stock_on_hand, stock_flag, snapshot_age_days (NaN where no snapshot)."""
    if stock.empty:
        out = keys.copy()
        out["stock_on_hand"] = np.nan
        out["stock_flag"] = None
        out["snapshot_age_days"] = np.nan
        return out
    left = keys.sort_values("date")
    right = stock.sort_values("snapshot_date")
    merged = pd.merge_asof(
        left, right, left_on="date", right_on="snapshot_date",
        by="product_id", direction="backward",
        tolerance=pd.Timedelta(days=max_staleness))
    merged["snapshot_age_days"] = (merged["date"] - merged["snapshot_date"]).dt.days
    return merged


# ── builders ─────────────────────────────────────────────────────────────────────
def build_model_panel(con: sqlite3.Connection, pilot: pd.DataFrame, cfg: dict,
                      start: str | None, end: str | None
                      ) -> tuple[pd.DataFrame, dict]:
    """Daily SKU x channel panel over each SKU's active period (no pre-activation zero-fill)."""
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
    if end:
        ecom = ecom[ecom["date"] <= pd.Timestamp(end)]
    if ecom.empty:
        sys.exit("No ecommerce sales found for the pilot SKUs in the given window.")

    window_end = pd.Timestamp(end) if end else ecom["date"].max()
    # daily aggregation per sku/channel/date
    agg = (ecom.groupby(["sku", "channel", "date"])
               .agg(units_observed=("quantity_sold", "sum"),
                    discount_amount=("discount_amount", "sum"),
                    net_rev=("row_total", "sum"))
               .reset_index())
    agg["units_observed"] = agg["units_observed"].clip(lower=0).round().astype(int)

    # per-SKU/channel active start = first ecommerce sale (listing date unknown -> documented fallback)
    starts = agg.groupby(["sku", "channel"])["date"].min().reset_index(name="active_start")

    # rectangular active grid: each sku/channel from its active_start..window_end
    frames = []
    for (sku, ch), grp in starts.groupby(["sku", "channel"]):
        a0 = grp["active_start"].iloc[0]
        days = pd.date_range(a0, window_end, freq="D")
        frames.append(pd.DataFrame({"sku": sku, "channel": ch, "date": days}))
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.merge(agg, on=["sku", "channel", "date"], how="left")

    # genuine zeros only within the active window
    panel["units_observed"] = panel["units_observed"].fillna(0).astype(int)
    panel["discount_amount"] = panel["discount_amount"].fillna(0.0)
    panel["net_rev"] = panel["net_rev"].fillna(0.0)
    panel["product_active"] = True

    # effective price: revenue/units on sale days, else carry last known price forward
    panel = panel.sort_values(["sku", "channel", "date"])
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(panel["units_observed"] > 0,
                       panel["net_rev"] / panel["units_observed"].replace(0, np.nan), np.nan)
    panel["effective_unit_price"] = eff
    panel["effective_unit_price"] = (panel.groupby(["sku", "channel"])["effective_unit_price"]
                                          .ffill())
    gross = panel["net_rev"] + panel["discount_amount"]
    panel["discount_pct"] = np.where(gross > 0, panel["discount_amount"] / gross, 0.0)
    panel["on_promo"] = (panel["discount_amount"] > 0).astype(int)
    panel["promo_known_in_advance"] = 0        # historical realised promo, not a planned calendar

    # calendar features
    cal = calendar_features(pd.DatetimeIndex(panel["date"].unique()), cfg)
    panel = panel.merge(cal, on="date", how="left")

    # attributes
    panel = panel.merge(attrs.drop(columns=["price"]), on="sku", how="left")

    # stock as-of join (product_id keyed history)
    stock_hist = pd.read_sql(
        "SELECT product_id, snapshot_date, stock_on_hand, stock_flag "
        "FROM inventory_snapshot_history", con)
    stock_hist["snapshot_date"] = pd.to_datetime(stock_hist["snapshot_date"])
    keys = panel[["product_id", "date"]].copy()
    st = asof_stock(keys.assign(_i=range(len(keys))), stock_hist,
                    cfg["pilot"]["max_stock_staleness_days"]).sort_values("_i")
    panel["stock_on_hand"] = st["stock_on_hand"].to_numpy()
    panel["stock_flag"] = st["stock_flag"].to_numpy()
    age = st["snapshot_age_days"].to_numpy()
    panel["stock_observation_available"] = ~pd.isna(panel["stock_on_hand"])
    panel["stock_snapshot_stale"] = panel["stock_observation_available"] & (np.nan_to_num(age, nan=0) > 0)

    soh = pd.to_numeric(panel["stock_on_hand"], errors="coerce")
    panel["is_stockout"] = (panel["stock_observation_available"] & (soh <= 0)).fillna(False)
    panel["is_available"] = (panel["stock_observation_available"] & (soh > 0)).fillna(False)
    panel["demand_censored"] = panel["is_stockout"]
    panel["training_eligible"] = panel["product_active"] & (~panel["demand_censored"])

    # data quality flag (compact, honest)
    def flag(row) -> str:
        flags = ["activation_inferred_from_first_sale"]
        if not row["stock_observation_available"]:
            flags.append("stock_unobserved")
        elif row["stock_snapshot_stale"]:
            flags.append("stock_stale")
        if row["units_observed"] > 0 and (pd.isna(row["effective_unit_price"]) or row["effective_unit_price"] <= 0):
            flags.append("price_anomaly")
        return ";".join(flags)
    panel["data_quality_flag"] = panel.apply(flag, axis=1)

    panel["holiday_name"] = panel["holiday_name"].where(panel["holiday_name"].notna(), None)
    panel = panel.sort_values(["sku", "channel", "date"]).reset_index(drop=True)
    stats = {"physical_store_rows_excluded": physical_excluded, "unknown_channel_rows": unknown,
             "window_end": window_end}
    return panel[MODEL_PANEL_COLS], stats


def build_forecast_features(panel: pd.DataFrame, con: sqlite3.Connection, pilot: pd.DataFrame,
                            cfg: dict, as_of: pd.Timestamp) -> pd.DataFrame:
    """Exactly N future days per SKU/channel, using only info known on as_of (no leakage)."""
    n_days = int(cfg["pilot"]["forecast_feature_days"])
    future = pd.date_range(as_of + pd.Timedelta(days=1), periods=n_days, freq="D")
    keys = panel[["sku", "product_id", "channel", "category", "sub_category", "brand"]].drop_duplicates()

    # latest price known on/before as_of (from history), fallback to catalog price
    hist = panel[panel["date"] <= as_of]
    last_price = (hist.dropna(subset=["effective_unit_price"])
                      .sort_values("date").groupby(["sku", "channel"])["effective_unit_price"].last())
    skus = pilot["sku"].astype(str).tolist()
    cat_price = pd.read_sql(
        f"SELECT sku_id AS sku, price FROM sku_master WHERE sku_id IN ({','.join('?'*len(skus))})",
        con, params=skus).set_index("sku")["price"]

    rows = []
    for _, k in keys.iterrows():
        lp = last_price.get((k["sku"], k["channel"]))
        if pd.isna(lp):
            lp = cat_price.get(k["sku"], np.nan)
        for i, d in enumerate(future, start=1):
            rows.append({**k.to_dict(), "date": d, "forecast_horizon_day": i,
                         "latest_known_price": lp})
    ff = pd.DataFrame(rows)
    cal = calendar_features(future, cfg)
    ff = ff.merge(cal, on="date", how="left")

    # planned promotions cannot be reliably derived (no calendar with valid dates) -> unavailable,
    # NEVER copied from future realised transaction discounts.
    ff["planned_promo"] = 0
    ff["planned_discount_pct"] = np.nan
    ff["feature_availability_flag"] = np.where(
        ff["latest_known_price"].notna(), "price_ok;planned_promo_unavailable",
        "price_missing;planned_promo_unavailable")
    ff["holiday_name"] = ff["holiday_name"].where(ff["holiday_name"].notna(), None)
    return ff.sort_values(["sku", "channel", "date"]).reset_index(drop=True)[FORECAST_COLS]


def build_inventory_context(con: sqlite3.Connection, pilot: pd.DataFrame, cfg: dict,
                            as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per SKU x location at as_of. Every replenishment assumption is flagged."""
    rep = cfg["replenishment"]
    skus = pilot["sku"].astype(str).tolist()
    ph = ",".join("?" * len(skus))
    sm = pd.read_sql(
        f"SELECT sku_id AS sku, product_id, is_perishable, shelf_life_days, unit_cost, "
        f"       price, pack_size, is_dropship FROM sku_master WHERE sku_id IN ({ph})",
        con, params=skus)

    stock_hist = pd.read_sql(
        "SELECT product_id, snapshot_date, stock_on_hand, stock_flag, location_id "
        "FROM inventory_snapshot_history", con)
    stock_hist["snapshot_date"] = pd.to_datetime(stock_hist["snapshot_date"])
    stock_hist = stock_hist[stock_hist["snapshot_date"] <= as_of]
    latest = (stock_hist.sort_values("snapshot_date").groupby("product_id").tail(1)
              if not stock_hist.empty else stock_hist)

    lead = int(rep["default_supplier_lead_time_days"])
    moq = int(rep["default_moq"])
    assume_transit0 = bool(cfg["pilot"]["assume_stock_in_transit_zero"])
    note = (f"Supplier lead time uses the configured {lead}-day default because supplier-level "
            f"lead-time data is unavailable. MOQ uses the configured default of {moq} because "
            f"supplier MOQ data is unavailable. Stock-in-transit "
            + ("assumed 0 (no source in warehouse). " if assume_transit0 else "left unknown. ")
            + "Perishability unknown (shelf-life data absent) — not confirmed durable.")

    rows = []
    for _, r in sm.iterrows():
        m = latest[latest["product_id"] == r["product_id"]] if not latest.empty else latest
        soh = float(m["stock_on_hand"].iloc[0]) if len(m) else np.nan
        sflag = m["stock_flag"].iloc[0] if len(m) else None
        loc = m["location_id"].iloc[0] if len(m) else "ALL"
        ps = r["pack_size"]
        ps_ok = pd.notna(ps) and int(ps) >= 1
        pack = int(ps) if ps_ok else 1
        pack_assumed = not (ps_ok and int(ps) > 1)     # only >1 counts as real box-pack data
        rows.append({
            "as_of_date": as_of.date().isoformat(), "sku": r["sku"], "product_id": r["product_id"],
            "location_id": loc or "ALL", "stock_on_hand": soh,
            "stock_in_transit": 0.0 if assume_transit0 else np.nan,
            "supplier_lead_time_days": lead, "moq": moq, "pack_size": pack,
            "is_perishable": bool(r["is_perishable"]) if pd.notna(r["is_perishable"]) else None,
            "shelf_life_days": r["shelf_life_days"] if pd.notna(r["shelf_life_days"]) else None,
            "unit_cost": r["unit_cost"] if pd.notna(r["unit_cost"]) else None,
            "price": r["price"], "is_dropship": bool(r["is_dropship"]) if pd.notna(r["is_dropship"]) else None,
            "stock_flag": sflag,
            "lead_time_is_assumed": True, "moq_is_assumed": True,
            "pack_size_is_assumed": pack_assumed,
            "stock_in_transit_is_assumed": True,
            "perishability_is_assumed": True, "assumption_notes": note,
        })
    return pd.DataFrame(rows)[INVENTORY_COLS]


# ── validation ────────────────────────────────────────────────────────────────────
def validate_outputs(panel: pd.DataFrame, ff: pd.DataFrame, inv: pd.DataFrame,
                     cfg: dict) -> list[str]:
    problems: list[str] = []
    if list(panel.columns) != MODEL_PANEL_COLS:
        problems.append("model_panel columns mismatch")
    if list(ff.columns) != FORECAST_COLS:
        problems.append("forecast_features columns mismatch")
    if list(inv.columns) != INVENTORY_COLS:
        problems.append("inventory_context columns mismatch")
    if panel.duplicated(["sku", "channel", "date"]).any():
        problems.append("duplicate sku+channel+date in model_panel")
    if ff.duplicated(["sku", "channel", "date"]).any():
        problems.append("duplicate sku+channel+date in forecast_features")
    if inv.duplicated(["sku", "location_id"]).any():
        problems.append("duplicate sku+location in inventory_context")
    if (panel["units_observed"] < 0).any():
        problems.append("negative units_observed")
    n_days = int(cfg["pilot"]["forecast_feature_days"])
    per = ff.groupby(["sku", "channel"])["date"].nunique()
    if not (per == n_days).all():
        problems.append(f"forecast_features must have exactly {n_days} future days per sku/channel")
    allowed = set(cfg["pilot"]["ecommerce_channel_map"].values())
    if not set(panel["channel"]).issubset(allowed):
        problems.append(f"unexpected channels in model_panel: {set(panel['channel']) - allowed}")
    if "units_observed" in ff.columns:
        problems.append("LEAKAGE: forecast_features must not contain units_observed")
    if (inv["pack_size"] < 1).any():
        problems.append("pack_size must be a positive integer")
    return problems


# ── manifest ──────────────────────────────────────────────────────────────────────
def build_manifest(panel, ff, inv, pilot, cfg, args, stats, as_of, warnings, problems) -> dict:
    def rate(mask) -> float:
        return round(float(mask.mean()), 4) if len(mask) else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": args.generated_at,
        "as_of_date": as_of.date().isoformat(),
        "source_database_profile": "sqlite:inventory.db",
        "source_warehouse_path": str(Path(args.db_path)),
        "data_frequency": "daily",
        "forecast_horizons": cfg["pilot"]["forecast_horizons"],
        "history_start": panel["date"].min().date().isoformat(),
        "history_end": panel["date"].max().date().isoformat(),
        "selection_cutoff": args.selection_cutoff,
        "sku_count": int(panel["sku"].nunique()),
        "ecommerce_channels": sorted(panel["channel"].unique().tolist()),
        "model_panel_row_count": int(len(panel)),
        "forecast_feature_row_count": int(len(ff)),
        "inventory_context_row_count": int(len(inv)),
        "selected_skus": pilot["sku"].astype(str).tolist(),
        "category_distribution": panel.drop_duplicates("sku")["category"].value_counts().to_dict(),
        "channel_distribution": panel["channel"].value_counts().to_dict(),
        "physical_store_rows_excluded": stats["physical_store_rows_excluded"],
        "unknown_channel_rows": stats["unknown_channel_rows"],
        "zero_sales_rate": rate(panel["units_observed"] == 0),
        "stockout_rate": rate(panel["is_stockout"]),
        "censored_demand_rate": rate(panel["demand_censored"]),
        "stock_observation_coverage": rate(panel["stock_observation_available"]),
        "promotion_coverage": rate(panel["on_promo"] == 1),
        "perishability_coverage": rate(inv["is_perishable"].fillna(False).astype(bool)),
        "assumed_lead_time_count": int(inv["lead_time_is_assumed"].sum()),
        "assumed_moq_count": int(inv["moq_is_assumed"].sum()),
        "assumed_pack_size_count": int(inv["pack_size_is_assumed"].sum()),
        "missing_value_counts": {
            "model_panel_stock_on_hand": int(panel["stock_on_hand"].isna().sum()),
            "inventory_unit_cost": int(inv["unit_cost"].isna().sum()),
            "inventory_shelf_life_days": int(inv["shelf_life_days"].isna().sum()),
        },
        "duplicate_key_counts": {
            "model_panel": int(panel.duplicated(["sku", "channel", "date"]).sum()),
            "forecast_features": int(ff.duplicated(["sku", "channel", "date"]).sum()),
            "inventory_context": int(inv.duplicated(["sku", "location_id"]).sum()),
        },
        "configuration_used": {
            "replenishment": cfg["replenishment"], "pilot": cfg["pilot"]},
        "assumptions": [
            "Supplier lead time is a configured assumption (no supplier data).",
            "MOQ is a configured assumption (no supplier data).",
            "Stock-in-transit assumed 0 (no source in warehouse).",
            "Pack size defaults to 1 (nhd_box_products empty) unless real box data exists.",
            "Perishability unknown (shelf-life absent) — not classified as durable.",
            "Product activation inferred from first ecommerce sale (listing dates unknown).",
            "planned_promo unavailable: no promo calendar with valid start/end dates.",
            "foodpanda has no rows in this warehouse; only naheed_web is modelled.",
        ],
        "warnings": warnings,
        "validation_status": "passed" if not problems else "failed",
    }


# ── io ─────────────────────────────────────────────────────────────────────────────
def atomic_write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


# ── cli ──────────────────────────────────────────────────────────────────────────
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the daily ecommerce-only pilot dataset.")
    ap.add_argument("--db-path", default=str(DEFAULT_DB))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--as-of-date", default=None, help="last known day; features cover the next 14 days")
    ap.add_argument("--selection-cutoff", default=None, help="YYYY-MM-DD; only used with --reselect-pilot-skus")
    ap.add_argument("--reselect-pilot-skus", action="store_true")
    ap.add_argument("--strict", action="store_true")
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.generated_at = dt.datetime.now().isoformat(timespec="seconds")
    cfg = load_config()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open_db(Path(args.db_path)) as con:
        if args.reselect_pilot_skus:
            if not args.selection_cutoff:
                sys.exit("--reselect-pilot-skus requires --selection-cutoff YYYY-MM-DD")
            cand = reselect_candidates(con, cfg, args.selection_cutoff)
            cand.to_csv(CANDIDATE_LIST, index=False, encoding="utf-8-sig")
            print(f"Wrote {len(cand)} candidate SKUs -> {CANDIDATE_LIST.name} "
                  f"(review, then copy to {PILOT_LIST.name}; the approved list was NOT overwritten).")
            return 0

        pilot, warnings = load_pilot_skus(con, cfg, args.strict)
        panel, stats = build_model_panel(con, pilot, cfg, args.start_date, args.end_date)
        as_of = pd.Timestamp(args.as_of_date) if args.as_of_date else stats["window_end"]
        ff = build_forecast_features(panel, con, pilot, cfg, as_of)
        inv = build_inventory_context(con, pilot, cfg, as_of)

    problems = validate_outputs(panel, ff, inv, cfg)
    manifest = build_manifest(panel, ff, inv, pilot, cfg, args, stats, as_of, warnings, problems)

    if problems and args.strict:
        print("VALIDATION FAILED:", *problems, sep="\n  ")
        write_json(manifest, out_dir / "pilot_manifest.json")
        return 1

    atomic_write(panel, out_dir / "model_panel.parquet")
    atomic_write(ff, out_dir / "forecast_features.parquet")
    atomic_write(inv, out_dir / "inventory_context.parquet")
    write_json(manifest, out_dir / "pilot_manifest.json")

    print("================ pilot data built (daily, ecommerce-only) ================")
    print(f"as_of_date          : {as_of.date()}   history {manifest['history_start']}..{manifest['history_end']}")
    print(f"SKUs / channels     : {manifest['sku_count']} / {manifest['ecommerce_channels']}")
    print(f"model_panel rows    : {manifest['model_panel_row_count']}")
    print(f"forecast rows       : {manifest['forecast_feature_row_count']} (14 days x SKU x channel)")
    print(f"inventory rows      : {manifest['inventory_context_row_count']}")
    print(f"physical excluded   : {manifest['physical_store_rows_excluded']}   unknown channels: {manifest['unknown_channel_rows']}")
    print(f"zero-sales rate     : {manifest['zero_sales_rate']:.0%}   stock coverage: {manifest['stock_observation_coverage']:.0%}   censored: {manifest['censored_demand_rate']:.0%}")
    print(f"validation          : {manifest['validation_status']}")
    if warnings:
        print("warnings:", *warnings, sep="\n  ")
    if problems:
        print("PROBLEMS (non-strict, written anyway):", *problems, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
