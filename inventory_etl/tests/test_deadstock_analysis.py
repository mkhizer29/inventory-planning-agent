"""Deadstock analysis tests — src/deadstock_analysis.py on tiny temporary SQLite databases.

Never touches the real warehouse. Each test builds a minimal sku_master / inventory_snapshot /
sales_transactions with the real column names and asserts one classification/aggregation rule.
"""
import hashlib
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import deadstock_analysis as da     # noqa: E402

SNAP = "2026-06-30"
ECOM = "online_delivery"           # an ecommerce channel from pilot.ecommerce_channel_map


def _d(days_before: int) -> str:
    return (pd.Timestamp(SNAP) - pd.Timedelta(days=int(days_before))).strftime("%Y-%m-%d")


def _sku(sku="A", pid=1, cat="Cat", brand="B", name="Prod", cost=10.0, cost_src="eav",
         dropship=0, status=1, created="2020-01-01"):
    return (sku, pid, name, cat, brand, cost, cost_src, dropship, status, created)


def _build_db(path: Path, skus, inv, sales) -> Path:
    if path.exists():
        path.unlink()                       # fresh DB even when a test builds twice
    con = sqlite3.connect(path)
    c = con.cursor()
    c.execute("CREATE TABLE sku_master (sku_id TEXT, product_id INTEGER, sku_name TEXT, "
              "category TEXT, brand TEXT, unit_cost REAL, cost_source TEXT, is_dropship INTEGER, "
              "status INTEGER, created_at TEXT)")
    c.execute("CREATE TABLE inventory_snapshot (product_id INTEGER, location_id TEXT, "
              "stock_on_hand REAL, snapshot_date TEXT)")
    c.execute("CREATE TABLE sales_transactions (sku_id TEXT, channel TEXT, "
              "transaction_date TEXT, quantity_sold REAL)")
    c.executemany("INSERT INTO sku_master VALUES (?,?,?,?,?,?,?,?,?,?)", skus)
    c.executemany("INSERT INTO inventory_snapshot VALUES (?,?,?,?)", inv)
    c.executemany("INSERT INTO sales_transactions VALUES (?,?,?,?)", sales)
    con.commit(); con.close()
    return path


def _run(tmp, skus, inv, sales, *, inactivity_days=90, category=None, include_not_deadstock=False):
    db = _build_db(tmp / "wh.db", skus, inv, sales)
    return da.analyse_deadstock(db_path=db, inactivity_days=inactivity_days, category=category,
                                include_not_deadstock=include_not_deadstock)


def _status(df, sku):
    row = df[df["sku"] == sku]
    return None if row.empty else row.iloc[0]["deadstock_status"]


# 1 — old sale + positive stock -> Deadstock Candidate
def test_01_old_sale_positive_stock_is_candidate(tmp_path):
    df, _ = _run(tmp_path, [_sku()], [(1, "ALL", 10.0, SNAP)], [("A", ECOM, _d(180), 5.0)])
    assert _status(df, "A") == da.STATUS_CANDIDATE


# 2 — recent sale -> Not Deadstock (absent from the default deadstock frame)
def test_02_recent_sale_is_not_deadstock(tmp_path):
    df, _ = _run(tmp_path, [_sku()], [(1, "ALL", 10.0, SNAP)], [("A", ECOM, _d(10), 5.0)])
    assert df[df["sku"] == "A"].empty
    df_all, _ = _run(tmp_path, [_sku()], [(1, "ALL", 10.0, SNAP)], [("A", ECOM, _d(10), 5.0)],
                     include_not_deadstock=True)
    assert _status(df_all, "A") == da.STATUS_NOT


# 3 — zero stock is excluded from the scan
def test_03_zero_stock_excluded(tmp_path):
    df, summ = _run(tmp_path, [_sku()], [(1, "ALL", 0.0, SNAP)], [("A", ECOM, _d(180), 5.0)])
    assert df.empty and summ["products_scanned"] == 0


# 4 — negative net stock is excluded
def test_04_negative_stock_excluded(tmp_path):
    df, summ = _run(tmp_path, [_sku()], [(1, "ALL", -5.0, SNAP)], [("A", ECOM, _d(180), 5.0)])
    assert df.empty and summ["products_scanned"] == 0


# 5 — never-sold old product -> Never Sold
def test_05_never_sold_old_is_never_sold(tmp_path):
    df, _ = _run(tmp_path, [_sku(created="2020-01-01")], [(1, "ALL", 10.0, SNAP)], [])
    r = df[df["sku"] == "A"].iloc[0]
    assert r["deadstock_status"] == da.STATUS_NEVER_SOLD and pd.isna(r["last_sale_date"])


# 6 — newly created never-sold product is not deadstock
def test_06_new_never_sold_is_not_deadstock(tmp_path):
    df, _ = _run(tmp_path, [_sku(created=_d(10))], [(1, "ALL", 10.0, SNAP)], [], inactivity_days=90)
    assert df[df["sku"] == "A"].empty


# 7 — changing the interval changes the classification
def test_07_interval_changes_classification(tmp_path):
    skus, inv, sales = [_sku()], [(1, "ALL", 10.0, SNAP)], [("A", ECOM, _d(100), 5.0)]
    df90, _ = _run(tmp_path, skus, inv, sales, inactivity_days=90)
    df120, _ = _run(tmp_path, skus, inv, sales, inactivity_days=120, include_not_deadstock=True)
    assert _status(df90, "A") == da.STATUS_CANDIDATE
    assert _status(df120, "A") == da.STATUS_NOT


# 8 — exact interval boundary is included (days_since == interval)
def test_08_exact_boundary_included(tmp_path):
    df, _ = _run(tmp_path, [_sku()], [(1, "ALL", 10.0, SNAP)], [("A", ECOM, _d(90), 5.0)],
                 inactivity_days=90)
    r = df[df["sku"] == "A"].iloc[0]
    assert int(r["days_since_last_sale"]) == 90 and r["deadstock_status"] == da.STATUS_CANDIDATE


# 9 — the latest inventory snapshot is used (value + date)
def test_09_latest_snapshot_used(tmp_path):
    inv = [(1, "ALL", 999.0, "2026-05-01"), (1, "ALL", 10.0, SNAP)]      # older 999, latest 10
    df, summ = _run(tmp_path, [_sku()], inv, [("A", ECOM, _d(180), 5.0)])
    r = df[df["sku"] == "A"].iloc[0]
    assert float(r["stock_on_hand"]) == 10.0 and str(r["snapshot_date"]) == SNAP
    assert summ["snapshot_date"] == SNAP


# 10 — products present only in an older snapshot are ignored
def test_10_older_snapshot_ignored(tmp_path):
    skus = [_sku(sku="X", pid=1), _sku(sku="Y", pid=2)]
    inv = [(1, "ALL", 50.0, "2026-05-01"),      # X only in the OLD snapshot -> excluded
           (2, "ALL", 7.0, SNAP)]               # Y in the latest snapshot -> included
    df, _ = _run(tmp_path, skus, inv, [], include_not_deadstock=True)
    assert "X" not in set(df["sku"]) and "Y" in set(df["sku"])


# 11 — multiple inventory locations are summed
def test_11_locations_summed(tmp_path):
    inv = [(1, "BHD", 3.0, SNAP), (1, "MLR", 4.0, SNAP)]
    df, _ = _run(tmp_path, [_sku()], inv, [("A", ECOM, _d(180), 5.0)])
    assert float(df[df["sku"] == "A"].iloc[0]["stock_on_hand"]) == 7.0


# 12 — dropship products are excluded
def test_12_dropship_excluded(tmp_path):
    df, summ = _run(tmp_path, [_sku(dropship=1)], [(1, "ALL", 10.0, SNAP)], [("A", ECOM, _d(180), 5.0)])
    assert df.empty and summ["products_scanned"] == 0


# 13 — inactive (disabled) products are excluded
def test_13_inactive_excluded(tmp_path):
    df, summ = _run(tmp_path, [_sku(status=2)], [(1, "ALL", 10.0, SNAP)], [("A", ECOM, _d(180), 5.0)])
    assert df.empty and summ["products_scanned"] == 0


# 14 — category filter works
def test_14_category_filter(tmp_path):
    skus = [_sku(sku="A", pid=1, cat="Alpha"), _sku(sku="B", pid=2, cat="Beta")]
    inv = [(1, "ALL", 10.0, SNAP), (2, "ALL", 10.0, SNAP)]
    df, summ = _run(tmp_path, skus, inv, [], category="Alpha")
    assert set(df["sku"]) == {"A"} and summ["category"] == "Alpha"


# 15 — All Categories includes every category
def test_15_all_categories(tmp_path):
    skus = [_sku(sku="A", pid=1, cat="Alpha"), _sku(sku="B", pid=2, cat="Beta")]
    inv = [(1, "ALL", 10.0, SNAP), (2, "ALL", 10.0, SNAP)]
    df, summ = _run(tmp_path, skus, inv, [], category=None)
    assert {"A", "B"} <= set(df["sku"]) and summ["category"] == "All Categories"


# 16 — only positive quantity_sold counts as a sale
def test_16_only_positive_quantity_sold(tmp_path):
    # A's only "sale" is a zero-qty row -> treated as never sold (old product -> Never Sold)
    df, _ = _run(tmp_path, [_sku(created="2020-01-01")], [(1, "ALL", 10.0, SNAP)],
                 [("A", ECOM, _d(5), 0.0)])
    r = df[df["sku"] == "A"].iloc[0]
    assert r["deadstock_status"] == da.STATUS_NEVER_SOLD and pd.isna(r["last_sale_date"])


# 17 — sales after the snapshot date are ignored
def test_17_post_snapshot_sales_ignored(tmp_path):
    sales = [("A", ECOM, _d(180), 5.0), ("A", ECOM, "2026-08-01", 9.0)]   # future sale must be ignored
    df, _ = _run(tmp_path, [_sku()], [(1, "ALL", 10.0, SNAP)], sales)
    r = df[df["sku"] == "A"].iloc[0]
    assert str(r["last_sale_date"]) == _d(180) and r["deadstock_status"] == da.STATUS_CANDIDATE


# 18 — estimated value == stock × valid unit cost
def test_18_estimated_value(tmp_path):
    df, _ = _run(tmp_path, [_sku(cost=25.0)], [(1, "ALL", 10.0, SNAP)], [("A", ECOM, _d(180), 5.0)])
    assert float(df[df["sku"] == "A"].iloc[0]["estimated_deadstock_value"]) == 250.0


# 19 — missing/invalid cost -> null value, counted in missing_cost_count
def test_19_missing_cost_null_and_counted(tmp_path):
    df, summ = _run(tmp_path, [_sku(cost=None)], [(1, "ALL", 10.0, SNAP)], [("A", ECOM, _d(180), 5.0)])
    r = df[df["sku"] == "A"].iloc[0]
    assert pd.isna(r["estimated_deadstock_value"]) and summ["missing_cost_count"] == 1


# 20 — summary totals reconcile to the detail rows
def test_20_summary_reconciles(tmp_path):
    skus = [_sku(sku="A", pid=1, cost=10.0, created="2020-01-01"),       # candidate, value 100
            _sku(sku="B", pid=2, cost=5.0, created="2020-01-01"),        # never-sold, value 40
            _sku(sku="C", pid=3, cost=None, created="2020-01-01")]       # never-sold, missing cost
    inv = [(1, "ALL", 10.0, SNAP), (2, "ALL", 8.0, SNAP), (3, "ALL", 4.0, SNAP)]
    sales = [("A", ECOM, _d(180), 5.0)]                                   # only A ever sold
    df, summ = _run(tmp_path, skus, inv, sales)
    dead = df[df["deadstock_status"].isin([da.STATUS_CANDIDATE, da.STATUS_NEVER_SOLD])]
    assert summ["deadstock_candidate_count"] == 1 and summ["never_sold_count"] == 2
    assert summ["deadstock_units"] == float(dead["stock_on_hand"].sum()) == 22.0
    assert summ["estimated_deadstock_value"] == 140.0    # 100 + 40 ; C's null cost excluded, not zeroed
    assert summ["missing_cost_count"] == 1


# 21 — the database file is never modified (read-only)
def test_21_db_hash_unchanged(tmp_path):
    db = _build_db(tmp_path / "wh.db", [_sku()], [(1, "ALL", 10.0, SNAP)], [("A", ECOM, _d(180), 5.0)])
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    da.analyse_deadstock(db_path=db, inactivity_days=90)
    da.list_deadstock_categories(db)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


# 22 — app.py compiles
def test_22_app_compiles():
    import py_compile
    py_compile.compile(str(REPO_ROOT / "dashboard" / "app.py"), doraise=True)


# 23 — styles.py compiles
def test_23_styles_compiles():
    import py_compile
    py_compile.compile(str(REPO_ROOT / "dashboard" / "styles.py"), doraise=True)


# ══════════════════════════════════════════════════════════════════════════════════════════
# Pure dashboard DISPLAY helpers (sort / filter / aging / export / stale-input)
# ══════════════════════════════════════════════════════════════════════════════════════════
def _dead_df():
    """A small mixed-status frame for the display helpers (names shuffled vs. value).
    Distinctive SKU codes so a SKU search cannot collide with letters inside product names."""
    return pd.DataFrame({
        "sku": ["SKA", "SKB", "SKC", "SKD", "SKE"],
        "sku_name": ["Zeta Cleanser", "Alpha Serum", "Beta Balm", "Gamma Oil", "Delta Wax"],
        "category": ["X"] * 5, "brand": ["b"] * 5,
        "stock_on_hand": [10, 5, 8, 3, 7],
        "snapshot_date": ["2026-07-26"] * 5,
        "last_sale_date": ["2026-01-01", "2026-02-01", None, "2026-03-01", None],
        "days_since_last_sale": pd.array([206, 175, None, 145, None], dtype="Int64"),
        "product_created_date": ["2020-01-01"] * 5, "product_age_days": pd.array([2000] * 5, dtype="Int64"),
        "inactivity_interval_days": [90] * 5,
        "deadstock_status": [da.STATUS_CANDIDATE, da.STATUS_CANDIDATE, da.STATUS_NEVER_SOLD,
                             da.STATUS_CANDIDATE, da.STATUS_REVIEW],
        "unit_cost": [10.0, None, 5.0, 2.0, None], "cost_source": ["eav"] * 5,
        "estimated_deadstock_value": [100.0, None, 40.0, 6.0, None], "is_dropship": [False] * 5,
    })


# 24 — deterministic status ordering (candidate -> never sold -> manual review -> not)
def test_24_status_ordering(tmp_path):
    assert da.status_rank(da.STATUS_CANDIDATE) < da.status_rank(da.STATUS_NEVER_SOLD) \
        < da.status_rank(da.STATUS_REVIEW) < da.status_rank(da.STATUS_NOT)
    ranked = da.sort_deadstock_queue(_dead_df(), "Highest Value")
    statuses = list(ranked["deadstock_status"])
    assert statuses == sorted(statuses, key=da.status_rank)          # grouped by status priority
    # candidates come before never-sold before manual-review
    assert list(ranked["sku"])[:3] == ["SKA", "SKD", "SKB"]          # candidates, value desc, null last


# 25 — value sorting places null estimated values last within a status
def test_25_value_sort_nulls_last(tmp_path):
    ranked = da.sort_deadstock_queue(_dead_df(), "Highest Value")
    cands = ranked[ranked["deadstock_status"] == da.STATUS_CANDIDATE]
    assert list(cands["sku"]) == ["SKA", "SKD", "SKB"]               # 100, 6, then null(SKB) last


# 26 — search by product name
def test_26_search_by_name(tmp_path):
    assert set(da.filter_deadstock(_dead_df(), query="alpha")["sku"]) == {"SKB"}


# 27 — search by SKU
def test_27_search_by_sku(tmp_path):
    assert set(da.filter_deadstock(_dead_df(), query="skd")["sku"]) == {"SKD"}   # SKU code, no name collision


# 28 — status filtering
def test_28_status_filter(tmp_path):
    out = da.filter_deadstock(_dead_df(), statuses=[da.STATUS_CANDIDATE])
    assert set(out["sku"]) == {"SKA", "SKB", "SKD"} and set(out["deadstock_status"]) == {da.STATUS_CANDIDATE}


# 29 — Never Sold retains a null last_sale_date through the helpers
def test_29_never_sold_null_last_sale_preserved(tmp_path):
    ranked = da.sort_deadstock_queue(_dead_df(), "Highest Value")
    c = ranked[ranked["sku"] == "SKC"].iloc[0]
    assert c["deadstock_status"] == da.STATUS_NEVER_SOLD and pd.isna(c["last_sale_date"])


# 30 — null cost stays null in the export frame (never zeroed)
def test_30_export_null_cost_stays_null(tmp_path):
    ex = da.deadstock_export_frame(_dead_df())
    assert ex.loc[ex["SKU"] == "SKB", "Unit Cost"].isna().all()


# 31 — null estimated value stays null in the export frame
def test_31_export_null_value_stays_null(tmp_path):
    ex = da.deadstock_export_frame(_dead_df())
    assert ex.loc[ex["SKU"] == "SKB", "Estimated Deadstock Value"].isna().all()


# 32 — aging buckets are mutually exclusive (each candidate maps to exactly one)
def test_32_aging_buckets_mutually_exclusive(tmp_path):
    labels = da.aging_bucket_labels(90)
    assert labels == ["90–119d", "120–179d", "180–364d", "365d+", "Never Sold"]
    day_buckets = labels[:4]
    for d in (90, 119, 120, 179, 180, 364, 365, 900):
        b = da.aging_bucket(d, da.STATUS_CANDIDATE, 90)
        assert b in day_buckets and sum(da.aging_bucket(d, da.STATUS_CANDIDATE, 90) == x for x in day_buckets) == 1
    assert da.aging_bucket(90, da.STATUS_CANDIDATE, 90) == "90–119d"
    assert da.aging_bucket(120, da.STATUS_CANDIDATE, 90) == "120–179d"
    # the first bucket adapts when the interval exceeds 119
    assert da.aging_bucket_labels(150)[0] == "150–179d"


# 33 — Never Sold uses its own bucket regardless of days
def test_33_never_sold_own_bucket(tmp_path):
    assert da.aging_bucket(None, da.STATUS_NEVER_SOLD, 90) == "Never Sold"
    assert da.aging_bucket(None, da.STATUS_REVIEW, 90) is None      # review is not aged
    summ = da.deadstock_aging_summary(_dead_df(), 90)
    ns = summ[summ["bucket"] == "Never Sold"].iloc[0]
    assert int(ns["products"]) == 1 and float(ns["units"]) == 8.0   # only C, never-sold


# 34 — completed-analysis metadata stays separate from changed form inputs
def test_34_inputs_changed_detection(tmp_path):
    meta = {"analysis_category": "All Categories", "analysis_inactivity_days": 90}
    assert da.analysis_inputs_changed(meta, "All Categories", 90) is False
    assert da.analysis_inputs_changed(meta, "Groceries & Pets", 90) is True
    assert da.analysis_inputs_changed(meta, "All Categories", 120) is True
    assert da.analysis_inputs_changed(None, "All Categories", 90) is False   # no analysis yet


# 35 — export uses the FULL filtered rows, not only the visible queue page of 8
def test_35_export_full_filtered_rows(tmp_path):
    big = pd.concat([_dead_df()] * 6, ignore_index=True)            # 30 rows
    big["sku"] = [f"S{i}" for i in range(len(big))]
    filtered = da.filter_deadstock(big, statuses=[da.STATUS_CANDIDATE])   # 18 candidate rows
    ex = da.deadstock_export_frame(filtered)
    assert len(ex) == len(filtered) == 18 and len(ex) > 8


# 36 — full product names are never truncated by the helpers
def test_36_full_names_unchanged(tmp_path):
    long_name = "Extra Gentle Micellar Cleansing Water for Sensitive Skin 400ml Twin Pack"
    df = _dead_df(); df.loc[df["sku"] == "SKA", "sku_name"] = long_name
    ex = da.deadstock_export_frame(da.sort_deadstock_queue(df, "Highest Value"))
    assert long_name in set(ex["Product"])
