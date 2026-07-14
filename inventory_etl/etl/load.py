"""Load canonical frames into the SQLite warehouse with a stable schema + indexes.

Also exports each canonical table to CSV for spreadsheet / hand-off use.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

log = logging.getLogger("etl.load")

# canonical table -> (index name, columns) to create after load
_INDEXES = {
    "sku_master": [("ix_sku_master_pid", "product_id")],
    "sales_transactions": [
        ("ix_sales_sku", "sku_id"),
        ("ix_sales_pid", "product_id"),
        ("ix_sales_date", "transaction_date"),
        ("ix_sales_channel", "channel"),
    ],
    "inventory_snapshot": [
        ("ix_snap_pid", "product_id"),
        ("ix_snap_loc", "location_id"),
    ],
    "external_signals": [("ix_signal_date", "signal_date")],
    "channel_master": [],
    "shipments": [("ix_ship_sku", "sku"), ("ix_ship_order", "order_id")],
    "returns": [("ix_ret_sku", "sku")],
    "promotions_catalog": [("ix_promocat_pid", "product_id")],
    "product_views": [("ix_views_pid", "product_id"), ("ix_views_date", "view_date")],
    "delivery_geography": [("ix_geo_order", "order_id")],
    "stock_alerts": [("ix_alert_pid", "product_id")],
    "inventory_snapshot_history": [
        ("ix_snaphist_pid", "product_id"),
        ("ix_snaphist_date", "snapshot_date"),
    ],
}


def load_all(engine: Engine, frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Replace each canonical table and (re)build indexes. Returns row counts."""
    counts: dict[str, int] = {}
    with engine.begin() as conn:
        for name, df in frames.items():
            if df.shape[1] == 0:
                log.warning("load: %-22s skipped (table absent in source)", name)
                counts[name] = 0
                continue
            df.to_sql(name, conn, if_exists="replace", index=False)
            counts[name] = len(df)
            log.info("load: %-22s %8d rows", name, len(df))
            for ix_name, cols in _INDEXES.get(name, []):
                conn.execute(text(f"DROP INDEX IF EXISTS {ix_name}"))
                conn.execute(text(f"CREATE INDEX {ix_name} ON {name} ({cols})"))
    return counts


def append_inventory_history(engine: Engine, snap: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """Accumulate a dated stock history across runs (Magento keeps none natively).

    Re-running on the same date replaces that date's rows (no duplicates), so a
    daily schedule builds a real time series. Returns the full history frame.
    """
    from sqlalchemy import inspect
    table = "inventory_snapshot_history"
    if table in inspect(engine).get_table_names():
        existing = pd.read_sql(f"SELECT * FROM {table}", engine)
        existing = existing[existing["snapshot_date"].astype(str) != str(run_date)]
        combined = pd.concat([existing, snap], ignore_index=True)
    else:
        combined = snap.copy()
    with engine.begin() as conn:
        combined.to_sql(table, conn, if_exists="replace", index=False)
        for ix_name, cols in _INDEXES.get(table, []):
            conn.execute(text(f"DROP INDEX IF EXISTS {ix_name}"))
            conn.execute(text(f"CREATE INDEX {ix_name} ON {table} ({cols})"))
    log.info("history: %-22s %8d rows (all dates)", table, len(combined))
    return combined


def export_csvs(frames: dict[str, pd.DataFrame], csv_dir: Path) -> None:
    """Write each canonical table to output/csv/<table>.csv (Excel-friendly UTF-8)."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        if df.shape[1] == 0:
            continue  # table absent in this source; nothing to export
        path = csv_dir / f"{name}.csv"
        # utf-8-sig adds a BOM so Excel renders brand/product names correctly
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log.info("csv:  %-22s -> %s", name, path.name)
