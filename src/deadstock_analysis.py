"""deadstock_analysis.py — standalone ecommerce inventory-inactivity (deadstock) scan.

A SEPARATE diagnostic feature: which stock-carrying SKUs have positive current warehouse
stock but no recorded ECOMMERCE sale within a configurable inactivity window. It is fully
independent of demand forecasting, Top-N selection, stockout risk, reorder recommendations
and forecast run folders. It reads ONLY the real warehouse — the latest ``inventory_snapshot``
plus ``sales_transactions`` and ``sku_master`` — never synthetic/forecast stock, and it never
writes anything.

Public functions (function-based, no framework):
    list_deadstock_categories(db_path) -> list[str]
    analyse_deadstock(*, db_path, inactivity_days, category=None, config_path=None,
                      include_not_deadstock=False) -> tuple[pandas.DataFrame, dict]

The warehouse is opened strictly read-only (``mode=ro`` URI + ``PRAGMA query_only`` +
``PRAGMA temp_store = MEMORY``). Sales aggregation and latest-snapshot selection happen in
SQLite; the full transaction table is never loaded into pandas.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "inventory_etl" / "config" / "config.yaml"
DEFAULT_DB_PATH = REPO_ROOT / "inventory_etl" / "output" / "inventory.db"

# Allowed deadstock status vocabulary.
STATUS_CANDIDATE = "Deadstock Candidate"
STATUS_NEVER_SOLD = "Never Sold"
STATUS_NOT = "Not Deadstock"
STATUS_REVIEW = "Manual Review"
DEADSTOCK_STATUSES = (STATUS_CANDIDATE, STATUS_NEVER_SOLD)          # confirmed deadstock
RETURNED_STATUSES = (STATUS_CANDIDATE, STATUS_NEVER_SOLD, STATUS_REVIEW)

# Output dataframe contract (column order is part of the contract).
OUTPUT_COLUMNS = [
    "sku", "product_id", "sku_name", "category", "brand", "stock_on_hand", "snapshot_date",
    "last_sale_date", "days_since_last_sale", "product_created_date", "product_age_days",
    "inactivity_interval_days", "deadstock_status", "unit_cost", "cost_source",
    "estimated_deadstock_value", "is_dropship",
]


class DeadstockError(Exception):
    """Warehouse missing / unreadable, or the deadstock scan could not be built."""


# ── configuration (reuse the project's ecommerce channel rules; never duplicate them) ────
def _load_config(config_path: "str | os.PathLike | None" = None) -> dict:
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise DeadstockError(f"Project config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    d = cfg.get("deadstock") or {}
    pilot = cfg.get("pilot") or {}
    ecom = list((pilot.get("ecommerce_channel_map") or {}).keys())
    if not ecom:
        raise DeadstockError("config.yaml is missing a valid 'pilot.ecommerce_channel_map'")
    return {
        "enabled": bool(d.get("enabled", True)),
        "default_inactivity_days": int(d.get("default_inactivity_days", 90)),
        "minimum_interval_days": int(d.get("minimum_interval_days", 1)),
        "maximum_interval_days": int(d.get("maximum_interval_days", 365)),
        "minimum_stock_on_hand": float(d.get("minimum_stock_on_hand", 1)),
        "exclude_dropship": bool(d.get("exclude_dropship", True)),
        "active_products_only": bool(d.get("active_products_only", True)),
        "sales_scope": str(d.get("sales_scope", "ecommerce")),
        "ecommerce_channels": ecom,
    }


# ── read-only warehouse access (mirrors src/dynamic_selection conventions) ────────────────
def _connect_readonly(db_path: "str | os.PathLike") -> sqlite3.Connection:
    p = Path(db_path)
    if not p.exists() or not p.is_file():
        raise DeadstockError(f"Warehouse not found: {p}")
    uri = f"{p.resolve().as_uri()}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
        con.execute("PRAGMA query_only = TRUE")     # hard-guard against any write
        con.execute("PRAGMA temp_store = MEMORY")   # keep GROUP BY sorters in RAM (read-only safe)
        return con
    except sqlite3.Error as exc:                     # pragma: no cover - environment specific
        raise DeadstockError(f"Could not open warehouse read-only: {exc}") from exc


# ── public: category discovery (NOT forecast eligibility — every catalog category) ────────
def list_deadstock_categories(db_path: "str | os.PathLike" = DEFAULT_DB_PATH) -> list[str]:
    """Distinct, non-blank warehouse categories for active catalog products (sorted).

    Unlike forecasting category discovery this uses no cutoff / min-history / Top-N — it is
    the full set of categories a deadstock scan can target.
    """
    cfg = _load_config()
    active_clause = "AND status = 1" if cfg["active_products_only"] else ""
    sql = (f"SELECT DISTINCT TRIM(category) AS category FROM sku_master "
           f"WHERE category IS NOT NULL AND TRIM(category) <> '' {active_clause} "
           f"ORDER BY 1")
    with closing(_connect_readonly(db_path)) as con:
        try:
            rows = con.execute(sql).fetchall()
        except sqlite3.Error as exc:
            raise DeadstockError(f"Category query failed: {exc}") from exc
    return [str(r[0]).strip() for r in rows if r[0] is not None and str(r[0]).strip()]


# ── SQL: one row per scanned SKU (sales aggregated + latest snapshot summed in SQLite) ────
def _scan_sql(cfg: dict, category: "str | None") -> tuple[str, list]:
    ch = cfg["ecommerce_channels"]
    ch_ph = ",".join("?" for _ in ch)
    active_clause = "AND m.status = 1" if cfg["active_products_only"] else ""
    dropship_clause = ("AND (m.is_dropship = 0 OR m.is_dropship IS NULL)"
                       if cfg["exclude_dropship"] else "")
    category_clause = "AND TRIM(m.category) = ?" if category is not None else ""
    sql = f"""
    WITH latest AS (SELECT MAX(snapshot_date) AS d FROM inventory_snapshot),
    inv AS (
        SELECT product_id, SUM(stock_on_hand) AS stock_on_hand
        FROM inventory_snapshot
        WHERE snapshot_date = (SELECT d FROM latest)
        GROUP BY product_id
    ),
    lastsale AS (
        SELECT sku_id, MAX(transaction_date) AS last_sale_date
        FROM sales_transactions
        WHERE quantity_sold > 0
          AND channel IN ({ch_ph})
          AND transaction_date <= (SELECT d FROM latest)
        GROUP BY sku_id
    )
    SELECT m.sku_id                 AS sku,
           m.product_id             AS product_id,
           m.sku_name               AS sku_name,
           TRIM(m.category)         AS category,
           m.brand                  AS brand,
           inv.stock_on_hand        AS stock_on_hand,
           (SELECT d FROM latest)   AS snapshot_date,
           ls.last_sale_date        AS last_sale_date,
           m.created_at             AS product_created_at,
           m.unit_cost              AS unit_cost,
           m.cost_source            AS cost_source,
           m.is_dropship            AS is_dropship
    FROM sku_master m
    JOIN inv ON inv.product_id = m.product_id
    LEFT JOIN lastsale ls ON ls.sku_id = m.sku_id
    WHERE inv.stock_on_hand >= ?
      AND m.category IS NOT NULL AND TRIM(m.category) <> ''
      {active_clause}
      {dropship_clause}
      {category_clause}
    """
    params: list = [*ch, cfg["minimum_stock_on_hand"]]
    if category is not None:
        params.append(str(category).strip())
    return sql, params


# ── public: the scan ──────────────────────────────────────────────────────────────────────
def analyse_deadstock(*, db_path: "str | os.PathLike" = DEFAULT_DB_PATH, inactivity_days: int,
                      category: "str | None" = None, config_path: "str | os.PathLike | None" = None,
                      include_not_deadstock: bool = False) -> tuple[pd.DataFrame, dict]:
    """Scan active, non-dropship, stock-carrying SKUs against the latest warehouse snapshot.

    Returns ``(dataframe, summary)``. By default the dataframe holds Deadstock Candidate,
    Never Sold and Manual Review rows; pass ``include_not_deadstock=True`` to get every
    scanned SKU (the internal "all scanned" view for the dashboard). The summary always
    reflects the full scan universe. Missing cost / value stays null (never zeroed).
    """
    cfg = _load_config(config_path)
    interval = max(cfg["minimum_interval_days"],
                   min(int(inactivity_days), cfg["maximum_interval_days"]))
    cat = None if (category is None or str(category).strip().lower() in ("", "all categories")) else str(category).strip()
    sales_scope = cfg["sales_scope"]

    sql, params = _scan_sql(cfg, cat)
    with closing(_connect_readonly(db_path)) as con:
        try:
            scan = pd.read_sql_query(sql, con, params=params)
        except (sqlite3.Error, pd.errors.DatabaseError) as exc:
            raise DeadstockError(f"Deadstock scan query failed: {exc}") from exc

    snapshot_date = None
    if not scan.empty:
        snapshot_date = str(scan["snapshot_date"].iloc[0])
    else:                                   # empty snapshot or no SKU in scope
        with closing(_connect_readonly(db_path)) as con:
            row = con.execute("SELECT MAX(snapshot_date) FROM inventory_snapshot").fetchone()
        snapshot_date = str(row[0]) if row and row[0] is not None else None

    if scan.empty:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty, _summary(empty, snapshot_date, interval, category, sales_scope, scanned=0)

    snap = pd.to_datetime(snapshot_date)
    last_sale = pd.to_datetime(scan["last_sale_date"], errors="coerce")
    created = pd.to_datetime(scan["product_created_at"], errors="coerce")
    days_since = (snap - last_sale).dt.days
    age = (snap - created).dt.days

    has_sale = last_sale.notna()
    has_created = created.notna()
    status = np.select(
        [has_sale & (days_since >= interval),                 # old ecommerce sale -> candidate
         has_sale & (days_since < interval),                  # recent sale -> not deadstock
         (~has_sale) & has_created & (age >= interval),       # never sold, old enough -> never sold
         (~has_sale) & has_created & (age < interval),        # never sold but too new -> not deadstock
         (~has_sale) & (~has_created)],                       # never sold, unknown age -> manual review
        [STATUS_CANDIDATE, STATUS_NOT, STATUS_NEVER_SOLD, STATUS_NOT, STATUS_REVIEW],
        default=STATUS_NOT)

    cost = pd.to_numeric(scan["unit_cost"], errors="coerce")
    stock = pd.to_numeric(scan["stock_on_hand"], errors="coerce")
    valid_cost = cost.notna() & (cost > 0)
    est_value = pd.Series(np.where(valid_cost, stock * cost, np.nan), index=scan.index)

    out = pd.DataFrame({
        "sku": scan["sku"].astype(str),
        "product_id": scan["product_id"],
        "sku_name": scan["sku_name"],
        "category": scan["category"],
        "brand": scan["brand"],
        "stock_on_hand": stock,
        "snapshot_date": snapshot_date,
        "last_sale_date": scan["last_sale_date"].where(scan["last_sale_date"].notna(), None),
        "days_since_last_sale": days_since.astype("Int64"),
        "product_created_date": created.dt.strftime("%Y-%m-%d").where(has_created, None),
        "product_age_days": age.astype("Int64"),
        "inactivity_interval_days": int(interval),
        "deadstock_status": status,
        "unit_cost": cost.where(valid_cost, np.nan),
        "cost_source": scan["cost_source"],
        "estimated_deadstock_value": est_value,
        "is_dropship": scan["is_dropship"].fillna(0).astype(int).astype(bool),
    })[OUTPUT_COLUMNS]

    summary = _summary(out, snapshot_date, interval, category, sales_scope, scanned=len(out))

    if include_not_deadstock:
        result = out
    else:
        result = out[out["deadstock_status"].isin(RETURNED_STATUSES)]
    return result.reset_index(drop=True), summary


def _summary(out: pd.DataFrame, snapshot_date, interval, category, sales_scope, *, scanned) -> dict:
    if out.empty:
        dead = out
    else:
        dead = out[out["deadstock_status"].isin(DEADSTOCK_STATUSES)]
    est = pd.to_numeric(dead["estimated_deadstock_value"], errors="coerce") if not dead.empty \
        else pd.Series(dtype=float)
    status_col = out["deadstock_status"] if not out.empty else pd.Series(dtype=object)
    return {
        "snapshot_date": snapshot_date,
        "inactivity_interval_days": int(interval),
        "category": (str(category).strip() if category and str(category).strip()
                     and str(category).strip().lower() != "all categories" else "All Categories"),
        "products_scanned": int(scanned),
        "deadstock_candidate_count": int((status_col == STATUS_CANDIDATE).sum()),
        "never_sold_count": int((status_col == STATUS_NEVER_SOLD).sum()),
        "manual_review_count": int((status_col == STATUS_REVIEW).sum()),
        "deadstock_units": float(pd.to_numeric(dead["stock_on_hand"], errors="coerce").sum()) if not dead.empty else 0.0,
        "estimated_deadstock_value": float(est.dropna().sum()),
        "missing_cost_count": int(est.isna().sum()) if not dead.empty else 0,
        "sales_scope": sales_scope,
    }


# ══════════════════════════════════════════════════════════════════════════════════════════
# Pure DISPLAY helpers for the dashboard (sort / filter / aging buckets / export frame).
# They NEVER reclassify — they only order, filter, bucket, or reshape rows the backend already
# produced. Kept here (not in app.py) so they are unit-testable without a Streamlit runtime.
# ══════════════════════════════════════════════════════════════════════════════════════════
# Deterministic status priority for the queue (candidate first, not-deadstock last).
STATUS_ORDER = {STATUS_CANDIDATE: 0, STATUS_NEVER_SOLD: 1, STATUS_REVIEW: 2, STATUS_NOT: 3}
QUEUE_SORT_OPTIONS = ("Highest Value", "Highest Stock", "Longest Inactive", "Product Name")


def status_rank(status) -> int:
    """Queue ordering rank; unknown/other statuses sort last."""
    return STATUS_ORDER.get(str(status), 4)


def sort_deadstock_queue(df: pd.DataFrame, sort_by: str = "Highest Value") -> pd.DataFrame:
    """Deterministic queue order: status priority first, then the chosen within-status key with
    fixed tie-breakers (value desc, days-inactive desc, stock desc, product name asc, sku asc).
    Null values always sort last. Never mutates the input."""
    if df is None or df.empty:
        return df.copy() if df is not None else df
    d = df.copy()
    d["_rank"] = d["deadstock_status"].map(status_rank)
    d["_val"] = pd.to_numeric(d.get("estimated_deadstock_value"), errors="coerce")
    d["_days"] = pd.to_numeric(d.get("days_since_last_sale"), errors="coerce")
    d["_stock"] = pd.to_numeric(d.get("stock_on_hand"), errors="coerce")
    d["_name"] = d["sku_name"].astype("string").fillna(d["sku"].astype(str)).str.lower()
    d["_sku"] = d["sku"].astype(str)
    if sort_by == "Highest Stock":
        cols, asc = ["_rank", "_stock", "_val", "_days", "_name", "_sku"], [True, False, False, False, True, True]
    elif sort_by == "Longest Inactive":
        cols, asc = ["_rank", "_days", "_val", "_stock", "_name", "_sku"], [True, False, False, False, True, True]
    elif sort_by == "Product Name":
        cols, asc = ["_rank", "_name", "_sku", "_val", "_days", "_stock"], [True, True, True, False, False, False]
    else:  # "Highest Value" (default)
        cols, asc = ["_rank", "_val", "_days", "_stock", "_name", "_sku"], [True, False, False, False, True, True]
    d = d.sort_values(cols, ascending=asc, na_position="last", kind="mergesort")
    return d.drop(columns=["_rank", "_val", "_days", "_stock", "_name", "_sku"]).reset_index(drop=True)


def filter_deadstock(df: pd.DataFrame, *, query: "str | None" = None,
                     statuses: "list | tuple | None" = None) -> pd.DataFrame:
    """Filter by status set and a case-insensitive product-name/SKU substring. Never mutates."""
    if df is None or df.empty:
        return df.copy() if df is not None else df
    d = df
    if statuses is not None:
        d = d[d["deadstock_status"].astype(str).isin([str(s) for s in statuses])]
    if query and str(query).strip():
        q = str(query).strip().lower()
        mask = d["sku"].astype(str).str.lower().str.contains(q, regex=False)
        mask = mask | d["sku_name"].astype(str).str.lower().str.contains(q, regex=False, na=False)
        d = d[mask]
    return d.reset_index(drop=True)


def aging_bucket_labels(interval: int) -> list:
    """Ordered, mutually exclusive inactivity buckets that adapt to the configured interval,
    with Never Sold kept separate (it has no last-sale date)."""
    i = int(interval)
    return [f"{i}–{i + 29}d", f"{i + 30}–{i + 89}d",
            f"{i + 90}–{i + 274}d", f"{i + 275}d+", "Never Sold"]


def aging_bucket(days_since_last_sale, deadstock_status, interval: int) -> "str | None":
    """Return the single aging bucket for a row, or None for statuses that are not aged
    (Manual Review / Not Deadstock). Never Sold is its own bucket regardless of days."""
    i = int(interval)
    s = str(deadstock_status)
    if s == STATUS_NEVER_SOLD:
        return "Never Sold"
    if s != STATUS_CANDIDATE:
        return None
    d = days_since_last_sale
    if d is None or pd.isna(d):
        return None
    d = int(d)
    if d < i:
        return None
    if d <= i + 29:
        return f"{i}–{i + 29}d"
    if d <= i + 89:
        return f"{i + 30}–{i + 89}d"
    if d <= i + 274:
        return f"{i + 90}–{i + 274}d"
    return f"{i + 275}d+"


def deadstock_aging_summary(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    """Per-bucket product count / stock units / estimated value over Candidate + Never Sold rows,
    in fixed bucket order (all five buckets present, zeros where empty)."""
    labels = aging_bucket_labels(interval)
    rows = []
    conf = (df[df["deadstock_status"].isin([STATUS_CANDIDATE, STATUS_NEVER_SOLD])].copy()
            if df is not None and not df.empty else pd.DataFrame(columns=OUTPUT_COLUMNS))
    if not conf.empty:
        conf["_bucket"] = [aging_bucket(dd, ss, interval)
                           for dd, ss in zip(conf["days_since_last_sale"], conf["deadstock_status"])]
    for lab in labels:
        sub = conf[conf["_bucket"] == lab] if not conf.empty else conf
        rows.append({
            "bucket": lab, "products": int(len(sub)),
            "units": float(pd.to_numeric(sub["stock_on_hand"], errors="coerce").sum()) if len(sub) else 0.0,
            "value": float(pd.to_numeric(sub["estimated_deadstock_value"], errors="coerce").dropna().sum()) if len(sub) else 0.0,
        })
    return pd.DataFrame(rows, columns=["bucket", "products", "units", "value"])


def analysis_inputs_changed(meta: "dict | None", category, interval) -> bool:
    """True when the current form inputs differ from the completed analysis metadata, so the
    dashboard can warn that a fresh Analyse is needed. False when there is no completed analysis."""
    if not meta:
        return False
    try:
        same_cat = str(meta.get("analysis_category")) == str(category)
        same_int = int(meta.get("analysis_inactivity_days")) == int(interval)
    except (TypeError, ValueError):
        return True
    return not (same_cat and same_int)


# User-facing column order for the complete table + export (technical fields kept separate).
EXPORT_COLUMNS = [
    "Status", "Product", "SKU", "Category", "Brand", "Current Stock", "Last Sale Date",
    "Inactive Days", "Product Age", "Configured Interval", "Unit Cost", "Cost Source",
    "Estimated Deadstock Value", "Snapshot Date",
]


def deadstock_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the user-facing complete-dataset / export frame for the CURRENTLY FILTERED rows.

    Keeps ALL rows passed in (never a visible-page slice), preserves full product names, and
    leaves null cost/value/date as null (blank on export — never zero). Raw numeric/date values
    are retained so the export utility can format and nulls render blank.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS)
    def _n(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([None] * len(df))
    out = pd.DataFrame({
        "Status": df["deadstock_status"].astype(str),
        "Product": df["sku_name"],                       # complete, untruncated
        "SKU": df["sku"].astype(str),
        "Category": df.get("category"),
        "Brand": df.get("brand"),
        "Current Stock": _n("stock_on_hand"),
        "Last Sale Date": df.get("last_sale_date"),
        "Inactive Days": _n("days_since_last_sale").astype("Int64"),
        "Product Age": _n("product_age_days").astype("Int64"),
        "Configured Interval": _n("inactivity_interval_days").astype("Int64"),
        "Unit Cost": _n("unit_cost").round(2),
        "Cost Source": df.get("cost_source"),
        "Estimated Deadstock Value": _n("estimated_deadstock_value").round(2),
        "Snapshot Date": df.get("snapshot_date"),
    })
    return out.reset_index(drop=True)
