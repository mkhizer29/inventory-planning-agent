"""Extract raw frames from a Magento `pg_1` source.

Every optional/custom table is guarded by table_exists so the same code runs
against both the staging DB and the production backup. Missing tables yield an
empty DataFrame (and a warning) rather than crashing the run.
"""
from __future__ import annotations

import logging

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


def extract_all(engine: Engine, sales_since: str | None = None) -> dict[str, pd.DataFrame]:
    """Return a dict of raw DataFrames keyed by logical name."""
    out: dict[str, pd.DataFrame] = {}

    # --- always-present core ---
    ph = _scope_placeholders()
    log.info("extract: products_raw")
    out["products"] = db.read_sql(engine, db.load_sql("products_raw.sql", **ph))

    log.info("extract: stock_item")
    out["stock"] = db.read_sql(engine, db.load_sql("stock_item.sql"))

    since_filter = f"AND oi.created_at >= '{sales_since}'" if sales_since else ""
    log.info("extract: sales (since=%s)", sales_since or "ALL")
    out["sales"] = db.read_sql(engine, db.load_sql("sales.sql", SINCE_FILTER=since_filter))

    # --- guarded product enrichment / cost ---
    _guarded(engine, out, "product_ops", "nhd_product_flat", "product_ops.sql")
    _guarded(engine, out, "cost_rows", "staging_margin", "cost_rows.sql")
    if config.settings()["cost"].get("fallback_to_flat"):
        _guarded(engine, out, "flat_cost", "catalog_product_flat_1", "flat_cost.sql")
    else:
        out["flat_cost"] = pd.DataFrame()
    _guarded(engine, out, "box_products", "nhd_box_products", "box_products.sql")

    # --- merged-in supporting signals (all guarded) ---
    _guarded(engine, out, "shipments", "sales_shipment_item", "shipments.sql")
    _guarded(engine, out, "returns", "sales_creditmemo_item", "returns.sql")
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
