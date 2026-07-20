"""prepare_pilot_data.py — builds the two SHARED input files the whole pilot uses.

Reads the ETL warehouse (inventory_etl/output/inventory.db) and writes:
    data/processed/weekly_sales.parquet    (one row per pilot-SKU per week)
    data/processed/weekly_signals.parquet  (one row per week)

These are the locked inputs that src/evaluation.py loads and all three models
(baselines / holtwinters / lgbm) import. Run this ONCE (by the pipeline owner);
teammates then just read the committed parquet files.

Pilot SKU selection (frozen): the top-15 best-selling SKUs (by net units) in each
of the two most-popular categories = 30 SKUs. The chosen list is written to
`pilot_skus.csv` at the repo root and reused on later runs so the set never drifts.

Weekly rules (must match every model's assumptions):
  * week_start = the Monday of each ISO week.
  * Only COMPLETE Mon–Sun weeks inside the data window are kept (partial first/last
    weeks are dropped so no week has artificially low units — important for the
    5-week test set).
  * Every SKU gets a row for every week; weeks with no sale are zero-filled
    (a zero is real demand information).
  * on_promo = 1 if that SKU had any discounted sale that week (discount_amount>0).

Run from the repo root:
    python src/prepare_pilot_data.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ── paths (this file lives in <repo>/src) ────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "inventory_etl" / "output" / "inventory.db"
PROC_DIR = REPO_ROOT / "data" / "processed"
PILOT_LIST = REPO_ROOT / "pilot_skus.csv"

TOP_N_PER_CATEGORY = 15          # 15 SKUs from each of the 2 categories = 30
N_CATEGORIES = 2

sys.path.append(str(Path(__file__).resolve().parent))
try:
    from evaluation import TEST_WEEKS          # keep the split in sync
except Exception:
    TEST_WEEKS = 5


# ── helpers ──────────────────────────────────────────────────────────────────
def _monday(s: pd.Series) -> pd.Series:
    """Return the Monday (00:00) of the ISO week each date falls in."""
    s = pd.to_datetime(s)
    return (s - pd.to_timedelta(s.dt.weekday, unit="D")).dt.normalize()


def select_pilot_skus(con: sqlite3.Connection) -> pd.DataFrame:
    """Freeze the 30 pilot SKUs: top-15 by net units in each of the top-2 categories.
    Reuses pilot_skus.csv if it already exists (so the locked set never drifts)."""
    if PILOT_LIST.exists():
        print(f"[skus] using existing frozen list: {PILOT_LIST.name}")
        return pd.read_csv(PILOT_LIST)

    print("[skus] pilot_skus.csv not found -> selecting fresh (top-2 categories, top-15 each)")
    # exclude free giveaways / bundle skus and rows with no category
    base = """
        FROM sales_transactions st
        JOIN sku_master s ON s.sku_id = st.sku_id
        WHERE s.category IS NOT NULL AND TRIM(s.category) <> ''
          AND s.sku_id NOT LIKE 'Free%' AND s.sku_id NOT LIKE 'PACK%'
    """
    top_cats = pd.read_sql(
        f"SELECT s.category, SUM(st.quantity_sold) units {base} "
        f"GROUP BY s.category ORDER BY units DESC LIMIT {N_CATEGORIES}", con)["category"].tolist()
    print(f"[skus] top {N_CATEGORIES} categories: {top_cats}")

    rows = []
    for cat in top_cats:
        df = pd.read_sql(
            f"SELECT s.sku_id AS sku, s.category, s.brand, s.sku_name AS name, "
            f"       ROUND(SUM(st.quantity_sold)) units "
            f"{base} AND s.category = :c "
            f"GROUP BY s.sku_id ORDER BY units DESC LIMIT {TOP_N_PER_CATEGORY}",
            con, params={"c": cat})
        rows.append(df)
    pilot = pd.concat(rows, ignore_index=True)
    pilot.to_csv(PILOT_LIST, index=False, encoding="utf-8-sig")
    print(f"[skus] wrote frozen list -> {PILOT_LIST.name} ({len(pilot)} SKUs)")
    return pilot


def build_week_grid(dmin: pd.Timestamp, dmax: pd.Timestamp) -> pd.DatetimeIndex:
    """All Mondays whose full Mon–Sun week sits inside [dmin, dmax]."""
    first = dmin + pd.Timedelta(days=(7 - dmin.weekday()) % 7)     # first Monday >= dmin
    last_wk_monday = dmax - pd.Timedelta(days=dmax.weekday())      # Monday of dmax's week
    if last_wk_monday + pd.Timedelta(days=6) > dmax:              # that week is partial
        last_wk_monday -= pd.Timedelta(days=7)
    return pd.date_range(first, last_wk_monday, freq="7D")


# ── main build ────────────────────────────────────────────────────────────────
def main() -> int:
    if not DB_PATH.exists():
        sys.exit(f"ETL warehouse not found: {DB_PATH}\nRun the ETL first "
                 f"(see TEAMMATE_SETUP.md), then re-run this script.")
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)

    pilot = select_pilot_skus(con)
    skus = pilot["sku"].tolist()
    placeholders = ",".join("?" * len(skus))

    # --- pull raw sales for the pilot SKUs ---
    sales = pd.read_sql(
        f"SELECT sku_id AS sku, transaction_date, quantity_sold, discount_amount "
        f"FROM sales_transactions WHERE sku_id IN ({placeholders})", con, params=skus)
    sales["transaction_date"] = pd.to_datetime(sales["transaction_date"])
    sales["week_start"] = _monday(sales["transaction_date"])

    # --- complete-week grid (drop partial first/last weeks) ---
    grid = build_week_grid(sales["transaction_date"].min(), sales["transaction_date"].max())
    sales = sales[sales["week_start"].isin(grid)]

    # --- aggregate to SKU × week ---
    agg = (sales.groupby(["sku", "week_start"])
                .agg(units=("quantity_sold", "sum"),
                     disc=("discount_amount", "sum"))
                .reset_index())
    agg["units"] = agg["units"].round().astype(int)
    agg["on_promo"] = (agg["disc"] > 0).astype(int)

    # --- rectangular panel: every SKU × every week, zero-filled ---
    panel = (pd.MultiIndex.from_product([skus, grid], names=["sku", "week_start"])
               .to_frame(index=False))
    panel = panel.merge(agg[["sku", "week_start", "units", "on_promo"]],
                        on=["sku", "week_start"], how="left")
    panel["units"] = panel["units"].fillna(0).astype(int)
    panel["on_promo"] = panel["on_promo"].fillna(0).astype(int)

    # --- attach product attributes (category, brand, price) ---
    attrs = pd.read_sql(
        f"SELECT sku_id AS sku, category, brand, price FROM sku_master "
        f"WHERE sku_id IN ({placeholders})", con, params=skus)
    weekly_sales = panel.merge(attrs, on="sku", how="left")[
        ["sku", "category", "brand", "price", "week_start", "units", "on_promo"]
    ].sort_values(["sku", "week_start"]).reset_index(drop=True)

    # --- weekly signals (holiday/payday day-counts per week) ---
    sig = pd.read_sql(
        "SELECT signal_date, is_public_holiday, is_payday_window FROM external_signals", con)
    sig["week_start"] = _monday(sig["signal_date"])
    sig = sig[sig["week_start"].isin(grid)]
    weekly_signals = (sig.groupby("week_start")
                        .agg(holiday_days=("is_public_holiday", "sum"),
                             payday_days=("is_payday_window", "sum"))
                        .reindex(grid, fill_value=0)
                        .rename_axis("week_start").reset_index())
    weekly_signals["holiday_days"] = weekly_signals["holiday_days"].astype(int)
    weekly_signals["payday_days"] = weekly_signals["payday_days"].astype(int)
    con.close()

    # --- write parquet ---
    ws_path = PROC_DIR / "weekly_sales.parquet"
    wsig_path = PROC_DIR / "weekly_signals.parquet"
    weekly_sales.to_parquet(ws_path, index=False)
    weekly_signals.to_parquet(wsig_path, index=False)

    # --- summary ---
    n_weeks = len(grid)
    print("\n================ pilot data built ================")
    print(f"SKUs                : {len(skus)}  ({N_CATEGORIES} categories x {TOP_N_PER_CATEGORY})")
    print(f"weeks (complete)    : {n_weeks}   ->  train {n_weeks - TEST_WEEKS} / test {TEST_WEEKS}")
    print(f"week range          : {grid.min().date()}  ..  {grid.max().date()}")
    print(f"weekly_sales rows   : {len(weekly_sales)}  (= {len(skus)} x {n_weeks})")
    print(f"promo SKU-weeks     : {int(weekly_sales.on_promo.sum())}")
    print(f"zero-sales SKU-weeks: {int((weekly_sales.units == 0).sum())} "
          f"({(weekly_sales.units == 0).mean():.0%})")
    print(f"holiday weeks       : {int((weekly_signals.holiday_days > 0).sum())}  |  "
          f"payday weeks: {int((weekly_signals.payday_days > 0).sum())}")
    print(f"\nwrote: {ws_path.relative_to(REPO_ROOT)}")
    print(f"wrote: {wsig_path.relative_to(REPO_ROOT)}")
    print(f"frozen SKU list: {PILOT_LIST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
