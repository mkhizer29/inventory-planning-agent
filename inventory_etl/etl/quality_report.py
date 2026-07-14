"""Data-quality report (FR-A4 / FR-A7): coverage, nulls, and cleansing counts."""
from __future__ import annotations

import pandas as pd
from tabulate import tabulate


def build_report(frames: dict[str, pd.DataFrame], source_profile: str, run_date: str) -> str:
    sku = frames["sku_master"]
    sales = frames["sales_transactions"]
    snap = frames["inventory_snapshot"]
    sig = frames["external_signals"]

    lines: list[str] = []
    lines.append(f"# ETL Data-Quality Report")
    lines.append(f"- source profile : **{source_profile}**")
    lines.append(f"- run date       : {run_date}")
    lines.append("")

    # row counts
    lines.append("## Row counts")
    lines.append(tabulate(
        [[k, len(v)] for k, v in frames.items()],
        headers=["table", "rows"], tablefmt="github"))
    lines.append("")

    # sku_master coverage
    n = len(sku) or 1
    def pct(mask): return f"{100 * mask.sum() / n:.1f}%"
    lines.append("## sku_master field coverage")
    cov = [
        ["unit_cost present", pct(sku["unit_cost"].notna())],
        ["price present", pct(sku["price"].notna())],
        ["brand present", pct(sku["brand"].notna())],
        ["category present", pct(sku["category"].notna())],
        ["is_perishable = True", pct(sku["is_perishable"] == True)],  # noqa: E712
        ["is_dropship = True", pct(sku["is_dropship"] == True)],      # noqa: E712
        ["pack_size > 1", pct(pd.to_numeric(sku["pack_size"], errors="coerce") > 1)],
    ]
    lines.append(tabulate(cov, headers=["check", "coverage"], tablefmt="github"))
    lines.append("")

    # inventory cleansing flags
    lines.append("## inventory_snapshot cleansing flags")
    flag_counts = snap["stock_flag"].value_counts().reset_index()
    flag_counts.columns = ["flag", "rows"]
    lines.append(tabulate(flag_counts.values.tolist(),
                          headers=["flag", "rows"], tablefmt="github"))
    loc_counts = snap["location_id"].value_counts().reset_index()
    loc_counts.columns = ["location_id", "rows"]
    lines.append("")
    lines.append("### stock rows by location")
    lines.append(tabulate(loc_counts.values.tolist(),
                          headers=["location_id", "rows"], tablefmt="github"))
    lines.append("")

    # sales demand summary
    lines.append("## sales_transactions summary")
    if not sales.empty:
        by_ch = (sales.groupby("channel")["quantity_sold"].agg(["count", "sum"])
                 .reset_index())
        lines.append(tabulate(by_ch.values.tolist(),
                              headers=["channel", "lines", "net_units"], tablefmt="github"))
        lines.append(f"\n- date range: {sales['transaction_date'].min()} → {sales['transaction_date'].max()}")
        lines.append(f"- distinct SKUs sold: {sales['sku_id'].nunique()}")
    lines.append("")

    # external signals
    lines.append("## external_signals")
    lines.append(f"- days: {len(sig)}  | holidays: {int(sig['is_public_holiday'].sum())}"
                 f"  | payday days: {int(sig['is_payday_window'].sum())}")
    lines.append("")

    # supporting signal tables (merged) — populated vs empty
    support_names = ["shipments", "returns", "promotions_catalog", "promotions_cart",
                     "delivery_geography", "related_products", "product_views",
                     "stock_alerts", "search_queries", "inventory_snapshot_history"]
    rows = []
    for nm in support_names:
        df = frames.get(nm)
        if df is None:
            continue
        state = "populated" if len(df) > 0 else "empty (present, no rows)"
        rows.append([nm, len(df), state])
    if rows:
        lines.append("## Supporting signal tables (merged)")
        lines.append(tabulate(rows, headers=["table", "rows", "status"], tablefmt="github"))
        lines.append("")

    # warnings
    warns = []
    if sku["unit_cost"].notna().mean() < 0.5:
        warns.append("Cost coverage < 50% — reorder/margin math will be sparse. "
                     "Confirm cost source with the buying team.")
    if (snap["location_id"] == "ALL").all():
        warns.append("All inventory is single-pool ('ALL') — per-warehouse columns "
                     "are unpopulated in this source (expected on staging).")
    if sku["supplier_lead_time_days"].nunique() <= 1:
        warns.append("supplier_lead_time_days is a single assumed constant "
                     "(no lead-time data in DB).")
    if warns:
        lines.append("## ⚠ Warnings")
        for w in warns:
            lines.append(f"- {w}")
    return "\n".join(lines)
