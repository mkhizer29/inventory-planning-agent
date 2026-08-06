"""Extract raw frames from a Magento `pg_1` source.

Every optional/custom table is guarded by table_exists so the same code runs
against both the staging DB and the production backup. Missing tables yield an
empty DataFrame (and a warning) rather than crashing the run.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from . import config, db

log = logging.getLogger("etl.extract")


def _scope_placeholders() -> dict:
    scope = config.settings()["scope"]
    types = ",".join(f"'{t}'" for t in scope["include_product_types"])
    status_filter = "" if scope.get("include_disabled") else "AND (st.value = 1 OR st.value IS NULL)"
    return {"TYPES": types, "STATUS_FILTER": status_filter}


def _guarded(engine: Engine, out: dict, key: str, table: str, sql_file: str, **ph) -> None:
    """Run an extract only if its primary table exists; else empty frame + warn."""
    if db.table_exists(engine, table):
        log.info("extract: %s (%s)", key, table)
        out[key] = db.read_sql(engine, db.load_sql(sql_file, **ph))
    else:
        log.warning("%s absent -> skipping %s", table, key)
        out[key] = pd.DataFrame()


# Primary-key span per sales page. ~150k item_ids yields a query of a few tens of
# seconds on the staging link -- short enough that a dropped connection costs one
# page, not the whole extract. See db.read_sql_key_ranges for the full rationale.
SALES_PAGE_KEYS = 150_000


def _sales_cache_dir(sales_since: str | None) -> Path:
    """Per-range part directory for one sales window, under the (gitignored) output dir."""
    key = sales_since or "ALL"
    return config.target_sqlite_path().parent / ".extract_cache" / f"sales_{key}"


def _extract_sales(engine: Engine, since_filter: str, sales_since: str | None,
                   resume_cache: bool = False) -> pd.DataFrame:
    """Read the sales extract in primary-key ranges instead of one streaming pass.

    The item_id lower bound is resolved from `sales_since` once, so the ~1.7M
    pre-window rows are never paged over at all.

    With `resume_cache`, EACH range is written to its own parquet part as soon as
    it arrives, and a rerun skips ranges whose part already exists. This extract
    takes ~65min over the staging link and was killed twice (2026-08-05) after
    finishing but before load_all() wrote inventory.db, losing everything; part
    files make a rerun continue instead of restarting. Assembling from parts also
    avoids the peak of concatenating 120 in-memory frames at once, which matters
    on an 8GB machine.

    The cache is explicitly opt-in because it is NOT invalidated by new rows
    landing in the source: use it to resume an interrupted run, not as a general
    speed-up. Delete the directory to force a fresh pull.
    """
    parts_dir = _sales_cache_dir(sales_since)

    lo_sql = "SELECT MIN(item_id) FROM sales_order_item"
    if sales_since:
        lo_sql += f" WHERE created_at >= '{sales_since}'"
    lo = db.scalar(engine, lo_sql)
    hi = db.scalar(engine, "SELECT MAX(item_id) FROM sales_order_item")
    if lo is None or hi is None:  # no rows in window
        return db.read_sql(engine, db.load_sql("sales.sql", SINCE_FILTER=since_filter,
                                               PAGE_FILTER="AND 1=0"))
    log.info("sales: paging item_id %d..%d in steps of %d", lo, hi, SALES_PAGE_KEYS)

    def make_sql(range_lo: int, range_hi: int) -> str:
        return db.load_sql(
            "sales.sql", SINCE_FILTER=since_filter,
            PAGE_FILTER=f"AND oi.item_id > {range_lo} AND oi.item_id <= {range_hi}",
        )

    # lo-1 so the first range's exclusive lower bound still includes item_id == lo
    if not resume_cache:
        return db.read_sql_key_ranges(engine, make_sql, int(lo) - 1, int(hi),
                                      SALES_PAGE_KEYS, label="sales")
    return db.read_sql_key_ranges_cached(engine, make_sql, int(lo) - 1, int(hi),
                                         SALES_PAGE_KEYS, parts_dir, label="sales")


def extract_all(engine: Engine, sales_since: str | None = None,
                skip_support: bool = False,
                resume_cache: bool = False) -> dict[str, pd.DataFrame]:
    """Return a dict of raw DataFrames keyed by logical name.

    ``skip_support`` omits the analytical *signal* tables (shipments, returns,
    promotions, geography, related products, views, alerts, searches), leaving
    them as empty frames. The canonical §9 tables the forecasting pipeline reads
    -- sku_master, sales_transactions, inventory_snapshot, external_signals --
    are unaffected, as is the product/cost enrichment that feeds sku_master.
    Use it when the link to staging is unreliable: it removes ~2.3M rows of
    read time (shipments + geography) that forecasting never touches.
    """
    out: dict[str, pd.DataFrame] = {}

    # --- always-present core ---
    ph = _scope_placeholders()
    log.info("extract: products_raw")
    out["products"] = db.read_sql(engine, db.load_sql("products_raw.sql", **ph))

    log.info("extract: stock_item")
    out["stock"] = db.read_sql(engine, db.load_sql("stock_item.sql"))

    since_filter = f"AND oi.created_at >= '{sales_since}'" if sales_since else ""
    log.info("extract: sales (since=%s)", sales_since or "ALL")
    out["sales"] = _extract_sales(engine, since_filter, sales_since, resume_cache=resume_cache)

    # --- guarded product enrichment / cost ---
    _guarded(engine, out, "product_ops", "nhd_product_flat", "product_ops.sql")
    _guarded(engine, out, "cost_rows", "staging_margin", "cost_rows.sql")
    if config.settings()["cost"].get("fallback_to_flat"):
        _guarded(engine, out, "flat_cost", "catalog_product_flat_1", "flat_cost.sql")
    else:
        out["flat_cost"] = pd.DataFrame()
    _guarded(engine, out, "box_products", "nhd_box_products", "box_products.sql")

    # --- merged-in supporting signals (all guarded) ---
    support_keys = ("shipments", "returns", "promotions_catalog", "promotions_cart",
                    "delivery_geography", "related_products", "product_views",
                    "stock_alerts", "search_queries")
    if skip_support:
        log.warning("skip_support -> omitting %s (empty frames)", ", ".join(support_keys))
        for key in support_keys:
            out[key] = pd.DataFrame()
        return out

    # shipments/returns carry their own dates, so they honour the same window as
    # sales -- rows outside it have no order line in the warehouse to attach to.
    ship_since = f"WHERE ss.created_at >= '{sales_since}'" if sales_since else ""
    ret_since = f"WHERE scm.created_at >= '{sales_since}'" if sales_since else ""
    _guarded(engine, out, "shipments", "sales_shipment_item", "shipments.sql",
             SINCE_FILTER=ship_since)
    _guarded(engine, out, "returns", "sales_creditmemo_item", "returns.sql",
             SINCE_FILTER=ret_since)
    _guarded(engine, out, "promotions_catalog", "catalogrule", "promotions_catalog.sql")
    _guarded(engine, out, "promotions_cart", "salesrule", "promotions_cart.sql")
    _guarded(engine, out, "delivery_geography", "sales_order_address", "delivery_geography.sql")
    _guarded(engine, out, "related_products", "catalog_product_link", "related_products.sql")
    _guarded(engine, out, "product_views", "report_viewed_product_aggregated_daily", "product_views.sql")
    _guarded(engine, out, "stock_alerts", "product_alert_stock", "stock_alerts.sql")

    # search table name varies by install (search_query vs search_query_1);
    # pick whichever exists AND carries the most rows (most demand signal).
    candidates = [t for t in ("search_query", "search_query_1") if db.table_exists(engine, t)]
    search_table = None
    if candidates:
        with engine.connect() as c:
            counts = {t: c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() for t in candidates}
        search_table = max(counts, key=counts.get)
    if search_table:
        log.info("extract: search_queries (%s)", search_table)
        out["search_queries"] = db.read_sql(engine, db.load_sql("search_queries.sql", TABLE=search_table))
    else:
        log.warning("no search_query table -> skipping search_queries")
        out["search_queries"] = pd.DataFrame()

    return out
