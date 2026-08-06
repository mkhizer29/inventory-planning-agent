"""selection_risk.py — pre-forecast stockout-risk proxy for Top-N SKU selection.

Scores forecast-eligible SKUs by stockout risk BEFORE any model has run, so
``src/dynamic_selection.py`` can rank Top-N by risk instead of by units sold.

Why a proxy is needed
---------------------
The real stockout risk (``src/stockout_risk.py``, Phase B) is FORECAST-DRIVEN: it reads
``runs/<run_id>/selected_forecasts.parquet``, so it only exists for SKUs that have already
been selected and forecast. Selection happens first, so it cannot consume that output
without a circular dependency. This module therefore reproduces Phase B's risk arithmetic
with a naive flat demand forecast in place of the model forecast:

    mean_daily   = mean daily ecommerce units over the trailing demand window
    sigma_daily  = std  daily ecommerce units over the same window (zero-filled calendar days)
    lt_mean      = mean_daily  * lead_time_days
    lt_sigma     = sigma_daily * sqrt(lead_time_days)      # decisioning.independent_daily_errors_rss
    inventory    = stock_on_hand from the warehouse snapshot (on-order excluded, as in Phase B)
    P(stockout)  = 1 - Phi((inventory - lt_mean) / lt_sigma)

This is not an invented heuristic: ``historical_demand_std`` is Phase B's own documented
Method-3 uncertainty fallback (see ``stockout_risk._daily_sigma``). The proxy ranks; Phase B
still computes the authoritative per-SKU risk after the forecast, and the two can disagree.

Post-cutoff stock — READ THIS
-----------------------------
The warehouse holds ONE inventory snapshot and no stock history, so a SKU's stock on an
arbitrary ``selection_cutoff`` cannot be reconstructed. Under the default
``stock_snapshot_policy: latest`` the newest snapshot is used even when it postdates the
cutoff, which means POST-CUTOFF information influences which SKUs get selected. That breaks
the as-of purity ``dynamic_selection`` guarantees for the ``units`` metric. It is a deliberate,
recorded trade-off: ``stock_snapshot_date`` and ``stock_is_post_cutoff`` are returned in the
metadata and persisted into the run request/manifest so any backtest reviewer can see it.
Set ``stock_snapshot_policy: on_or_before_cutoff`` to forbid it (the scan then yields no
rows when every snapshot postdates the cutoff).

Reads ONLY the real warehouse (``inventory_snapshot`` + ``sales_transactions`` + ``sku_master``)
— never the synthetic reconstructed daily stock, never a forecast. Never writes anything.

Public API:
    score_stockout_risk(db_path, eligible_skus, *, selection_cutoff, category, ...)
        -> (DataFrame, dict)
    rank_by_stockout_risk(scored) -> DataFrame

CLI::

    python src/selection_risk.py --category "Groceries & Pets" \
        --selection-cutoff 2026-07-31 --top-n 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "inventory_etl" / "config" / "config.yaml"
DEFAULT_DB_PATH = REPO_ROOT / "inventory_etl" / "output" / "inventory.db"

_ND = NormalDist()

# Output contract (column order is part of the contract).
SELECTION_RISK_COLUMNS = [
    "sku", "product_id",
    "stock_on_hand", "stock_snapshot_date", "stock_is_post_cutoff", "stock_source",
    "demand_window_days", "demand_window_start", "demand_mean_daily", "demand_sigma_daily",
    "demand_active_days",
    "lead_time_days", "lead_time_source",
    "lead_time_demand_mean", "lead_time_demand_sigma",
    "stockout_probability", "expected_shortage_units", "proxy_days_of_cover",
    "proxy_risk_tier", "risk_scored", "risk_exclusion_reason", "risk_assumption_flags",
]

# Ranking key for "highest stockout risk" (deterministic; see rank_by_stockout_risk).
RANK_COLUMNS = ["stockout_probability", "expected_shortage_units", "sku"]
RANK_ASCENDING = [False, False, True]

_TIER_ORDER = ("critical", "high", "medium", "low", "unknown")


class SelectionRiskError(Exception):
    """Warehouse missing/unreadable, config invalid, or the risk scan could not be built."""


# ── configuration ─────────────────────────────────────────────────────────────────────────
def load_selection_risk_config(config_path: "str | os.PathLike | None" = None) -> dict:
    """Read the ``selection_risk`` block plus the shared rules it must not duplicate
    (ecommerce channels, stock cleansing thresholds, lead-time fallback, risk tiers)."""
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise SelectionRiskError(f"Project config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    pilot = cfg.get("pilot") or {}
    channels = list((pilot.get("ecommerce_channel_map") or {}).keys())
    if not channels:
        raise SelectionRiskError("config.yaml is missing a valid 'pilot.ecommerce_channel_map'")

    sr = cfg.get("selection_risk") or {}
    cleansing = cfg.get("cleansing") or {}
    repl = cfg.get("replenishment") or {}
    prob = ((cfg.get("decisioning") or {}).get("probability_thresholds")
            or {"critical": 0.80, "high": 0.50, "medium": 0.20})

    policy = str(sr.get("stock_snapshot_policy", "latest")).strip().lower()
    if policy not in ("latest", "on_or_before_cutoff"):
        raise SelectionRiskError(
            f"selection_risk.stock_snapshot_policy must be 'latest' or 'on_or_before_cutoff', "
            f"got {policy!r}")

    window = int(sr.get("demand_window_days", 28))
    if window < 2:
        # variance needs >= 2 days; a 1-day window can never produce a sigma
        raise SelectionRiskError(
            f"selection_risk.demand_window_days must be >= 2, got {window}")

    return {
        "enabled": bool(sr.get("enabled", True)),
        "demand_window_days": window,
        "stock_snapshot_policy": policy,
        "include_zero_stock": bool(sr.get("include_zero_stock", True)),
        "exclude_dropship": bool(sr.get("exclude_dropship", False)),
        "ecommerce_channels": channels,
        "stock_sentinel_threshold": float(cleansing.get("stock_sentinel_threshold", 10000)),
        "stock_negative_floor": float(cleansing.get("stock_negative_floor", 0)),
        "default_lead_time_days": int(repl.get("default_supplier_lead_time_days", 7)),
        "probability_thresholds": {k: float(v) for k, v in prob.items()},
    }


# ── read-only warehouse access (mirrors dynamic_selection / deadstock_analysis) ────────────
def _connect_readonly(db_path: "str | os.PathLike") -> sqlite3.Connection:
    p = Path(db_path)
    if not p.exists() or not p.is_file():
        raise SelectionRiskError(f"Warehouse not found: {p}")
    uri = f"{p.resolve().as_uri()}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
        con.execute("PRAGMA query_only = TRUE")
        con.execute("PRAGMA temp_store = MEMORY")
        return con
    except sqlite3.Error as exc:                     # pragma: no cover - environment specific
        raise SelectionRiskError(f"Could not open warehouse read-only: {exc}") from exc


def _read_sql(con: sqlite3.Connection, sql: str, params) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, con, params=list(params))
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise SelectionRiskError(f"Warehouse query failed: {exc}") from exc


# ── demand statistics (aggregated in SQLite; the transaction table is never loaded) ────────
# Daily units are summed per (sku, date) first, then reduced to sum / sum-of-squares so the
# per-SKU mean and std can be derived without materialising a per-day matrix. Calendar days
# with no sale contribute 0 to both sums, so zero-filling is implicit and the mean divides by
# the full window (NOT by active days) — matching a forecast that emits a value every day.
_DEMAND_SQL = """
WITH daily AS (
    SELECT st.sku_id                AS sku,
           st.transaction_date      AS d,
           SUM(st.quantity_sold)    AS units
    FROM sales_transactions st
    WHERE st.channel IN ({channel_ph})
      AND st.transaction_date <= ?
      AND st.transaction_date >= ?
    GROUP BY st.sku_id, st.transaction_date
)
SELECT sku,
       SUM(units)        AS s1,
       SUM(units * units) AS s2,
       COUNT(*)          AS active_days
FROM daily
GROUP BY sku
"""


def _demand_stats(con: sqlite3.Connection, channels, cutoff: str, window_days: int) -> pd.DataFrame:
    """Per-SKU mean/std of daily ecommerce units over the trailing window ending at cutoff."""
    start = (pd.Timestamp(cutoff) - pd.Timedelta(days=window_days - 1)).strftime("%Y-%m-%d")
    sql = _DEMAND_SQL.format(channel_ph=",".join("?" for _ in channels))
    df = _read_sql(con, sql, [*channels, cutoff, start])

    w = float(window_days)
    s1 = pd.to_numeric(df["s1"], errors="coerce").fillna(0.0)
    s2 = pd.to_numeric(df["s2"], errors="coerce").fillna(0.0)
    mean = s1 / w
    # Population sum of squared deviations over the zero-filled window, then sample variance.
    var = (s2 - (s1 * s1) / w) / (w - 1.0)
    df["demand_mean_daily"] = mean
    df["demand_sigma_daily"] = np.sqrt(var.clip(lower=0.0))
    df["demand_active_days"] = pd.to_numeric(df["active_days"], errors="coerce").fillna(0).astype(int)
    df["demand_window_start"] = start
    return df[["sku", "demand_mean_daily", "demand_sigma_daily",
               "demand_active_days", "demand_window_start"]]


# ── inventory + lead time ─────────────────────────────────────────────────────────────────
def _resolve_snapshot_date(con: sqlite3.Connection, cutoff: str, policy: str) -> "str | None":
    if policy == "on_or_before_cutoff":
        row = con.execute(
            "SELECT MAX(snapshot_date) FROM inventory_snapshot WHERE snapshot_date <= ?",
            [cutoff]).fetchone()
    else:
        row = con.execute("SELECT MAX(snapshot_date) FROM inventory_snapshot").fetchone()
    return str(row[0]) if row and row[0] is not None else None


_STOCK_SQL = """
SELECT m.sku_id                  AS sku,
       m.product_id              AS product_id,
       inv.stock_on_hand         AS stock_on_hand,
       m.supplier_lead_time_days AS supplier_lead_time_days,
       m.is_dropship             AS is_dropship
FROM sku_master m
LEFT JOIN (
    SELECT product_id, SUM(stock_on_hand) AS stock_on_hand
    FROM inventory_snapshot
    WHERE snapshot_date = ?
    GROUP BY product_id
) inv ON inv.product_id = m.product_id
"""


def _stock_and_lead_time(con: sqlite3.Connection, snapshot_date: "str | None") -> pd.DataFrame:
    if snapshot_date is None:
        return pd.DataFrame(columns=["sku", "product_id", "stock_on_hand",
                                     "supplier_lead_time_days", "is_dropship"])
    return _read_sql(con, _STOCK_SQL, [snapshot_date])


# ── risk arithmetic (mirrors src/stockout_risk.py, with a flat demand forecast) ────────────
def _probability_tier(p, thresholds: dict) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "unknown"
    if p >= thresholds["critical"]:
        return "critical"
    if p >= thresholds["high"]:
        return "high"
    if p >= thresholds["medium"]:
        return "medium"
    return "low"


def _risk_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Vectorised P(stockout) and expected shortage over the lead-time window."""
    lt = pd.to_numeric(df["lead_time_days"], errors="coerce").fillna(0).to_numpy(float)
    mean_d = pd.to_numeric(df["demand_mean_daily"], errors="coerce").fillna(0.0).to_numpy(float)
    sig_d = pd.to_numeric(df["demand_sigma_daily"], errors="coerce").fillna(0.0).to_numpy(float)
    inv = pd.to_numeric(df["stock_on_hand"], errors="coerce").to_numpy(float)

    lt_mean = mean_d * lt
    lt_sigma = sig_d * np.sqrt(lt)          # independent daily errors -> RSS over the window

    prob = np.full(len(df), np.nan)
    short = np.full(len(df), np.nan)
    scored = np.isfinite(inv)

    # sigma == 0 is degenerate: the flat forecast is treated as certain.
    degenerate = scored & (lt_sigma <= 0)
    prob[degenerate] = np.where(lt_mean[degenerate] > inv[degenerate], 1.0, 0.0)
    short[degenerate] = np.maximum(0.0, lt_mean[degenerate] - inv[degenerate])

    normal = scored & (lt_sigma > 0)
    if normal.any():
        z = (inv[normal] - lt_mean[normal]) / lt_sigma[normal]
        cdf = np.array([_ND.cdf(v) for v in z])
        pdf = np.array([_ND.pdf(v) for v in z])
        prob[normal] = np.clip(1.0 - cdf, 0.0, 1.0)
        short[normal] = np.maximum(0.0, lt_sigma[normal] * pdf
                                   + (lt_mean[normal] - inv[normal]) * (1.0 - cdf))

    with np.errstate(divide="ignore", invalid="ignore"):
        cover = np.where(mean_d > 0, inv / np.where(mean_d > 0, mean_d, np.nan), np.nan)

    df = df.copy()
    df["lead_time_demand_mean"] = lt_mean
    df["lead_time_demand_sigma"] = lt_sigma
    df["stockout_probability"] = prob
    df["expected_shortage_units"] = short
    df["proxy_days_of_cover"] = cover
    df["proxy_risk_tier"] = [_probability_tier(p, cfg["probability_thresholds"]) for p in prob]
    return df


# ── public API ────────────────────────────────────────────────────────────────────────────
def score_stockout_risk(
    db_path: "str | os.PathLike",
    eligible_skus,
    *,
    selection_cutoff: str,
    config_path: "str | os.PathLike | None" = None,
) -> tuple[pd.DataFrame, dict]:
    """Score the given eligible SKUs by pre-forecast stockout risk.

    ``eligible_skus`` is the SKU id iterable already deemed forecast-eligible by
    ``dynamic_selection`` — eligibility rules are NOT re-implemented here, so the two can
    never drift. Returns ``(dataframe, meta)``; the dataframe has one row per input SKU in
    ``SELECTION_RISK_COLUMNS`` order, including rows that could not be scored
    (``risk_scored == False`` with a ``risk_exclusion_reason``). Missing stock is never
    treated as zero.
    """
    cfg = load_selection_risk_config(config_path)
    skus = [str(s) for s in (eligible_skus if eligible_skus is not None else [])]
    cutoff = pd.Timestamp(selection_cutoff).strftime("%Y-%m-%d")

    if not skus:
        return pd.DataFrame(columns=SELECTION_RISK_COLUMNS), {
            "scored": 0, "excluded": 0, "candidates": 0,
            "stock_snapshot_date": None, "stock_is_post_cutoff": False,
            "demand_window_days": cfg["demand_window_days"], "selection_cutoff": cutoff,
            "stock_snapshot_policy": cfg["stock_snapshot_policy"],
            "exclusion_reasons": {}, "tier_counts": {}, "warnings": [],
        }

    with closing(_connect_readonly(db_path)) as con:
        snapshot_date = _resolve_snapshot_date(con, cutoff, cfg["stock_snapshot_policy"])
        demand = _demand_stats(con, cfg["ecommerce_channels"], cutoff, cfg["demand_window_days"])
        stock = _stock_and_lead_time(con, snapshot_date)

    warnings: list[str] = []
    post_cutoff = bool(snapshot_date is not None and str(snapshot_date) > cutoff)
    if snapshot_date is None:
        warnings.append(
            f"No inventory snapshot available under policy "
            f"'{cfg['stock_snapshot_policy']}' at cutoff {cutoff}; no SKU can be risk-scored.")
    elif post_cutoff:
        warnings.append(
            f"Stock snapshot {snapshot_date} POSTDATES the selection cutoff {cutoff} — "
            f"selection used post-cutoff inventory information.")

    base = pd.DataFrame({"sku": skus})
    df = base.merge(demand, on="sku", how="left").merge(stock, on="sku", how="left")

    # A SKU with no rows in the trailing window has zero demand across it, not unknown demand.
    df["demand_mean_daily"] = pd.to_numeric(df["demand_mean_daily"], errors="coerce").fillna(0.0)
    df["demand_sigma_daily"] = pd.to_numeric(df["demand_sigma_daily"], errors="coerce").fillna(0.0)
    df["demand_active_days"] = pd.to_numeric(df["demand_active_days"], errors="coerce").fillna(0).astype(int)
    df["demand_window_start"] = df["demand_window_start"].fillna(
        (pd.Timestamp(cutoff) - pd.Timedelta(days=cfg["demand_window_days"] - 1)).strftime("%Y-%m-%d"))
    df["demand_window_days"] = int(cfg["demand_window_days"])

    # ── stock cleansing: sentinel -> unknown, negative -> floor. Never silently zeroed. ────
    raw_stock = pd.to_numeric(df["stock_on_hand"], errors="coerce")
    sentinel = raw_stock >= cfg["stock_sentinel_threshold"]
    negative = raw_stock < cfg["stock_negative_floor"]
    clean = raw_stock.mask(sentinel, np.nan).clip(lower=cfg["stock_negative_floor"])
    df["stock_on_hand"] = clean
    df["stock_snapshot_date"] = snapshot_date
    df["stock_is_post_cutoff"] = post_cutoff
    df["stock_source"] = "inventory_snapshot" if snapshot_date is not None else None

    # ── lead time: real per-SKU value, else the configured replenishment fallback ──────────
    lt_raw = pd.to_numeric(df["supplier_lead_time_days"], errors="coerce")
    lt_valid = lt_raw.notna() & (lt_raw >= 1)
    df["lead_time_days"] = lt_raw.where(lt_valid, cfg["default_lead_time_days"]).astype(int)
    df["lead_time_source"] = np.where(lt_valid, "sku_master.supplier_lead_time_days",
                                      "config.replenishment.default_supplier_lead_time_days")

    # ── exclusions (evaluated before scoring; first matching reason wins) ──────────────────
    dropship = pd.to_numeric(df["is_dropship"], errors="coerce").fillna(0.0) > 0
    zero_stock = df["stock_on_hand"].notna() & (df["stock_on_hand"] <= 0)
    reason = pd.Series([None] * len(df), dtype=object, index=df.index)
    if snapshot_date is None:
        reason = reason.fillna("no_inventory_snapshot")
    reason = reason.where(~sentinel.fillna(False), "stock_sentinel_value")
    reason = reason.where(df["stock_on_hand"].notna() | reason.notna(), "no_stock_row")
    if cfg["exclude_dropship"]:
        reason = reason.where(~dropship | reason.notna(), "dropship_excluded")
    if not cfg["include_zero_stock"]:
        reason = reason.where(~zero_stock | reason.notna(), "zero_stock_excluded")

    df = _risk_columns(df, cfg)

    # Anything without a usable probability is unscored regardless of the reason chain.
    unscored = df["stockout_probability"].isna()
    reason = reason.where(~(unscored & reason.isna()), "risk_not_computable")
    df["risk_exclusion_reason"] = reason
    df["risk_scored"] = reason.isna() & df["stockout_probability"].notna()

    # Excluded rows must never rank; blank their scores so a stale value cannot leak in.
    for col in ("stockout_probability", "expected_shortage_units", "proxy_days_of_cover"):
        df[col] = df[col].where(df["risk_scored"], np.nan)
    df["proxy_risk_tier"] = df["proxy_risk_tier"].where(df["risk_scored"], "unknown")

    flags = np.where(post_cutoff, "stock_snapshot_post_cutoff;", "")
    flags = flags + np.where(df["lead_time_source"].str.startswith("config."), "assumed_lead_time;", "")
    flags = flags + np.where(df["demand_sigma_daily"] <= 0, "zero_demand_sigma;", "")
    flags = flags + np.where(zero_stock.fillna(False), "already_out_of_stock;", "")
    df["risk_assumption_flags"] = [f.rstrip(";") for f in flags]

    out = df.reindex(columns=SELECTION_RISK_COLUMNS)

    excl_counts = (out.loc[~out["risk_scored"], "risk_exclusion_reason"]
                   .value_counts().to_dict())
    tier_counts = (out.loc[out["risk_scored"], "proxy_risk_tier"]
                   .value_counts().to_dict())
    n_scored = int(out["risk_scored"].sum())
    if n_scored == 0 and snapshot_date is not None:
        warnings.append("No eligible SKU could be risk-scored (see exclusion reasons).")

    meta = {
        "candidates": int(len(out)),
        "scored": n_scored,
        "excluded": int(len(out) - n_scored),
        "stock_snapshot_date": snapshot_date,
        "stock_is_post_cutoff": post_cutoff,
        "stock_snapshot_policy": cfg["stock_snapshot_policy"],
        "selection_cutoff": cutoff,
        "demand_window_days": int(cfg["demand_window_days"]),
        "include_zero_stock": cfg["include_zero_stock"],
        "exclude_dropship": cfg["exclude_dropship"],
        "already_out_of_stock": int((out["risk_scored"] & (out["stock_on_hand"] <= 0)).sum()),
        "exclusion_reasons": {str(k): int(v) for k, v in excl_counts.items()},
        "tier_counts": {t: int(tier_counts.get(t, 0)) for t in _TIER_ORDER if tier_counts.get(t)},
        "warnings": warnings,
    }
    return out, meta


def rank_by_stockout_risk(scored: pd.DataFrame) -> pd.DataFrame:
    """Deterministic highest-risk-first order.

    ``stockout_probability`` desc, then ``expected_shortage_units`` desc, then ``sku`` asc.

    The shortage tie-break is load-bearing rather than cosmetic. Out-of-stock SKUs dominate
    the top of this ranking (on the real warehouse, 945 of 2,289 eligible Groceries & Pets
    SKUs hold zero stock). Where demand varies at all their probabilities are merely very
    close, not equal — 0.9997 vs 0.9994 — so probability alone still discriminates. But a SKU
    with a perfectly flat demand history has ``lt_sigma == 0`` and scores an EXACT 1.0, and
    those tie with each other. Breaking on expected shortage orders that block by the size of
    the exposure (out of stock AND selling fastest first) instead of by SKU id.

    Unscored rows sort last and are never promoted into a Top-N.
    """
    if scored is None or scored.empty:
        return scored.copy() if scored is not None else scored
    d = scored.copy()
    d["_scored"] = ~d["risk_scored"].astype(bool)           # False sorts first
    d["_p"] = pd.to_numeric(d["stockout_probability"], errors="coerce")
    d["_s"] = pd.to_numeric(d["expected_shortage_units"], errors="coerce")
    d["_k"] = d["sku"].astype(str)
    d = d.sort_values(["_scored", "_p", "_s", "_k"],
                      ascending=[True, False, False, True],
                      na_position="last", kind="mergesort")
    return d.drop(columns=["_scored", "_p", "_s", "_k"]).reset_index(drop=True)


# ── CLI ───────────────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="selection_risk",
        description="Pre-forecast stockout-risk proxy scoring for Top-N selection (read-only).")
    ap.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--category", required=True, help="Exact category to score.")
    ap.add_argument("--selection-cutoff", required=True, help="As-of cutoff (YYYY-MM-DD).")
    ap.add_argument("--min-history-days", type=int, default=None,
                    help="Eligibility threshold (default: pilot.min_history_days).")
    ap.add_argument("--top-n", type=int, default=10, help="Rows to display (default 10).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", action="store_true", help="Emit the metadata dict as JSON.")
    return ap


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import dynamic_selection as dsel                       # noqa: PLC0415 - CLI-only import

    mhd = (args.min_history_days if args.min_history_days is not None
           else dsel.default_min_history_days())
    # The FULL eligible set — never a units-ranked shortlist, which would silently hide any
    # at-risk SKU that is not also a top seller.
    eligible = dsel.list_eligible_skus(args.db_path, args.category, args.selection_cutoff, mhd)
    scored, meta = score_stockout_risk(args.db_path, eligible["sku"].tolist(),
                                       selection_cutoff=args.selection_cutoff,
                                       config_path=args.config)
    ranked = rank_by_stockout_risk(scored)

    if args.json:
        print(json.dumps(meta, indent=2, default=str))
        return 0

    print("Pre-forecast stockout-risk proxy")
    print(f"  category        : {args.category}")
    print(f"  cutoff          : {meta['selection_cutoff']}")
    print(f"  demand window   : {meta['demand_window_days']}d")
    print(f"  stock snapshot  : {meta['stock_snapshot_date']} "
          f"(policy {meta['stock_snapshot_policy']})")
    print(f"  candidates      : {meta['candidates']}  scored: {meta['scored']}  "
          f"excluded: {meta['excluded']}")
    print(f"  already out     : {meta['already_out_of_stock']}")
    print(f"  tiers           : {meta['tier_counts']}")
    if meta["exclusion_reasons"]:
        print(f"  exclusions      : {meta['exclusion_reasons']}")
    for w in meta["warnings"]:
        print(f"  WARNING         : {w}")
    print()
    cols = ["sku", "stock_on_hand", "demand_mean_daily", "demand_sigma_daily", "lead_time_days",
            "lead_time_demand_mean", "stockout_probability", "expected_shortage_units",
            "proxy_days_of_cover", "proxy_risk_tier"]
    print(ranked.head(args.top_n)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
