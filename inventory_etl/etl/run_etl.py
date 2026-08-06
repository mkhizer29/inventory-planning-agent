"""Orchestrator CLI: extract -> transform -> load -> quality report.

Usage:
  python -m etl.run_etl --source staging
  python -m etl.run_etl --source local_backup --sales-since 2024-01-01
  python -m etl.run_etl --source staging --run-date 2026-07-13
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

from . import config, db, extract, external_signals, load, quality_report, transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("etl")


def run(source: str | None, sales_since: str | None, run_date: str | None,
        skip_support: bool = False, resume_cache: bool = False) -> int:
    run_date = run_date or dt.date.today().isoformat()
    profile = config.source_profile(source)["profile"]
    log.info("=== ETL start | source=%s | run_date=%s%s%s ===", profile, run_date,
             " | skip_support" if skip_support else "",
             " | resume_cache" if resume_cache else "")

    src = db.source_engine(source)
    raw = extract.extract_all(src, sales_since=sales_since, skip_support=skip_support,
                              resume_cache=resume_cache)

    # canonical spec §9 tables
    frames = {
        "sku_master": transform.build_sku_master(raw),
        "sales_transactions": transform.build_sales_transactions(raw),
        "inventory_snapshot": transform.build_inventory_snapshot(raw, run_date),
        "channel_master": transform.build_channel_master(),
        "external_signals": external_signals.build_external_signals(end_date=run_date),
    }
    # supporting signal tables (shipments, returns, promotions, views, etc.)
    frames.update(transform.build_support_tables(raw))

    tgt = db.target_engine()
    counts = load.load_all(tgt, frames)

    # accumulate a dated stock history across runs, then include it in outputs
    history = load.append_inventory_history(tgt, frames["inventory_snapshot"], run_date)
    frames["inventory_snapshot_history"] = history

    # keep DB, CSVs and report co-located with the SQLite target
    out_dir = config.target_sqlite_path().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    load.export_csvs(frames, out_dir / "csv")

    report = quality_report.build_report(frames, profile, run_date)
    report_path = out_dir / "data_quality_report.md"
    report_path.write_text(report, encoding="utf-8")
    log.info("quality report -> %s", report_path)

    log.info("=== ETL done | %s ===", {k: counts[k] for k in counts})
    print("\n" + report)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Inventory Planning Agent ETL")
    ap.add_argument("--source", default=None, help="staging | local_backup (default from .env)")
    ap.add_argument("--sales-since", default=None, help="YYYY-MM-DD lower bound on sales history")
    ap.add_argument("--run-date", default=None, help="snapshot/run date (default: today)")
    ap.add_argument("--skip-support", action="store_true",
                    help="omit the analytical signal tables (shipments, geography, promos, "
                         "views, searches). Canonical tables used by forecasting are unaffected; "
                         "use when the staging link is unreliable.")
    ap.add_argument("--resume-cache", action="store_true",
                    help="cache the sales extract to inventory_etl/output/.extract_cache and "
                         "reuse it if present. For resuming an interrupted run -- the cache is "
                         "NOT invalidated by new source rows; delete it to force a fresh pull.")
    args = ap.parse_args()
    try:
        return run(args.source, args.sales_since, args.run_date, args.skip_support,
                   args.resume_cache)
    except Exception as exc:  # surface a clean error, non-zero exit
        log.exception("ETL failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
