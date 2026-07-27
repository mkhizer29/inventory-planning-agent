"""Focused unit tests for src/dynamic_selection.py (Phase 1 dynamic selection).

Everything runs against a throwaway temp SQLite DB built to mimic the warehouse
schema (sku_master + sales_transactions). The real Magento DB and the real
inventory.db are never touched.
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

# src/ is not on the packaged pythonpath — add it (mirrors test_prepare_pilot_data.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import dynamic_selection as ds  # noqa: E402

CUTOFF = "2026-06-30"
ECOM = "online_delivery"        # an ecommerce SOURCE channel key from config.yaml
PHYS = "store"                  # a physical channel (must be excluded)
UNKNOWN = "weirdchan"           # an unknown channel (must be excluded)


# ── temp warehouse builder ──────────────────────────────────────────────────────────
def _make_db(tmp_path: Path, skus: list[dict], sales: list[dict]) -> Path:
    """Create a minimal warehouse. `skus`: sku_id/sku_name/category/sub_category/brand.
    `sales`: sku_id/channel/transaction_date/quantity_sold."""
    db = tmp_path / "wh.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE sku_master (sku_id TEXT, product_id INTEGER, sku_name TEXT, "
        "category TEXT, sub_category TEXT, brand TEXT)"
    )
    con.execute(
        "CREATE TABLE sales_transactions (sku_id TEXT, channel TEXT, "
        "transaction_date DATE, quantity_sold REAL)"
    )
    for i, s in enumerate(skus):
        con.execute(
            "INSERT INTO sku_master (sku_id, product_id, sku_name, category, sub_category, brand) "
            "VALUES (?,?,?,?,?,?)",
            (s["sku_id"], 1000 + i, s.get("sku_name"), s.get("category"),
             s.get("sub_category"), s.get("brand")),
        )
    for t in sales:
        con.execute(
            "INSERT INTO sales_transactions (sku_id, channel, transaction_date, quantity_sold) "
            "VALUES (?,?,?,?)",
            (t["sku_id"], t.get("channel", ECOM), t["transaction_date"], t["quantity_sold"]),
        )
    con.commit()
    con.close()
    return db


def _daily_sales(sku_id: str, start: str, days: int, qty: float, channel: str = ECOM) -> list[dict]:
    """`days` consecutive daily ecommerce rows of `qty` units each."""
    d0 = pd.Timestamp(start)
    return [{"sku_id": sku_id, "channel": channel,
             "transaction_date": (d0 + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
             "quantity_sold": qty} for i in range(days)]


def _standard_db(tmp_path: Path) -> Path:
    """A category 'Groceries' with 3 SKUs of differing volume, plus a thin SKU and
    a second category, plus physical/unknown-channel noise."""
    skus = [
        {"sku_id": "IC-A", "sku_name": "Prod A", "category": "Groceries", "sub_category": "Milk", "brand": "BrandA"},
        {"sku_id": "IC-B", "sku_name": "Prod B", "category": "Groceries", "sub_category": None, "brand": None},
        {"sku_id": "IC-C", "sku_name": "Prod C", "category": "Groceries", "sub_category": "Snacks", "brand": "BrandC"},
        {"sku_id": "IC-THIN", "sku_name": "Thin", "category": "Groceries", "sub_category": None, "brand": None},
        {"sku_id": "IC-BEA", "sku_name": "Beauty 1", "category": "Beauty", "sub_category": None, "brand": "B"},
    ]
    sales: list[dict] = []
    sales += _daily_sales("IC-A", "2026-02-01", 40, 10)   # 400 units, 40 active days
    sales += _daily_sales("IC-B", "2026-02-01", 40, 5)    # 200 units, 40 active days
    sales += _daily_sales("IC-C", "2026-02-01", 30, 5)    # 150 units, 30 active days
    sales += _daily_sales("IC-THIN", "2026-02-01", 10, 9) # 10 active days -> below min-history 28
    sales += _daily_sales("IC-BEA", "2026-02-01", 35, 3)  # Beauty: 105 units, 35 days
    return _make_db(tmp_path, skus, sales)


def _checksum(db: Path) -> tuple[int, int, str]:
    con = sqlite3.connect(db)
    n_sku = con.execute("SELECT COUNT(*) FROM sku_master").fetchone()[0]
    n_tx = con.execute("SELECT COUNT(*) FROM sales_transactions").fetchone()[0]
    con.close()
    digest = hashlib.sha256(db.read_bytes()).hexdigest()
    return n_sku, n_tx, digest


# ── selection: exact category, ordering, counts ─────────────────────────────────────
def test_exact_category_filtering(tmp_path):
    db = _standard_db(tmp_path)
    sel, _ = ds.select_top_skus(db, "Groceries", 10, CUTOFF, 28)
    assert set(sel["category"]) == {"Groceries"}
    assert "IC-BEA" not in set(sel["sku"])  # Beauty excluded


def test_category_whitespace_normalization(tmp_path):
    db = _standard_db(tmp_path)
    sel, _ = ds.select_top_skus(db, "  Groceries  ", 10, CUTOFF, 28)
    assert set(sel["sku"]) == {"IC-A", "IC-B", "IC-C"}


def test_ecommerce_only_and_physical_excluded(tmp_path):
    # IC-A has ecom + physical sales; physical must not add units/days.
    skus = [{"sku_id": "IC-A", "sku_name": "A", "category": "G", "sub_category": None, "brand": None}]
    sales = _daily_sales("IC-A", "2026-02-01", 30, 10, channel=ECOM)
    sales += _daily_sales("IC-A", "2026-02-01", 30, 100, channel=PHYS)   # noise
    db = _make_db(tmp_path, skus, sales)
    sel, _ = ds.select_top_skus(db, "G", 5, CUTOFF, 28)
    assert int(sel.iloc[0]["historical_units"]) == 300   # 30*10 ecom only, not 30*110
    assert int(sel.iloc[0]["active_days"]) == 30


def test_unknown_channel_excluded(tmp_path):
    skus = [{"sku_id": "IC-A", "sku_name": "A", "category": "G", "sub_category": None, "brand": None}]
    sales = _daily_sales("IC-A", "2026-02-01", 30, 10, channel=ECOM)
    sales += _daily_sales("IC-A", "2026-05-01", 30, 50, channel=UNKNOWN)
    db = _make_db(tmp_path, skus, sales)
    sel, _ = ds.select_top_skus(db, "G", 5, CUTOFF, 28)
    assert int(sel.iloc[0]["historical_units"]) == 300


# ── cutoff leakage protection ────────────────────────────────────────────────────────
def _post_cutoff_db(tmp_path):
    skus = [
        {"sku_id": "IC-A", "sku_name": "A", "category": "G", "sub_category": None, "brand": None},
        {"sku_id": "IC-B", "sku_name": "B", "category": "G", "sub_category": None, "brand": None},
    ]
    sales = _daily_sales("IC-A", "2026-02-01", 30, 10)                 # 300 units by cutoff
    sales += _daily_sales("IC-B", "2026-02-01", 30, 9)                 # 270 units by cutoff
    # After cutoff: huge spike for B that would overtake A and add active days if leaked.
    sales += _daily_sales("IC-B", "2026-07-01", 20, 1000)
    return _make_db(tmp_path, skus, sales)


def test_cutoff_does_not_affect_units(tmp_path):
    db = _post_cutoff_db(tmp_path)
    sel, _ = ds.select_top_skus(db, "G", 5, CUTOFF, 28)
    units = dict(zip(sel["sku"], sel["historical_units"]))
    assert int(units["IC-A"]) == 300 and int(units["IC-B"]) == 270


def test_cutoff_does_not_affect_active_days(tmp_path):
    db = _post_cutoff_db(tmp_path)
    sel, _ = ds.select_top_skus(db, "G", 5, CUTOFF, 28)
    days = dict(zip(sel["sku"], sel["active_days"]))
    assert int(days["IC-B"]) == 30            # not 30+20


def test_cutoff_does_not_affect_ranking(tmp_path):
    db = _post_cutoff_db(tmp_path)
    sel, _ = ds.select_top_skus(db, "G", 5, CUTOFF, 28)
    assert list(sel["sku"]) == ["IC-A", "IC-B"]   # A stays #1 despite B's post-cutoff spike


# ── eligibility, ranking, ranks ──────────────────────────────────────────────────────
def test_min_history_filtering(tmp_path):
    db = _standard_db(tmp_path)
    sel, _ = ds.select_top_skus(db, "Groceries", 10, CUTOFF, 28)
    assert "IC-THIN" not in set(sel["sku"])   # only 10 active days


def test_correct_top_n(tmp_path):
    db = _standard_db(tmp_path)
    sel, _ = ds.select_top_skus(db, "Groceries", 2, CUTOFF, 28)
    assert list(sel["sku"]) == ["IC-A", "IC-B"]
    assert len(sel) == 2


def test_deterministic_ranking_order(tmp_path):
    # Two SKUs tie on units; tie broken by active_days desc, then sku asc.
    skus = [
        {"sku_id": "IC-Z", "sku_name": "Z", "category": "G", "sub_category": None, "brand": None},
        {"sku_id": "IC-Y", "sku_name": "Y", "category": "G", "sub_category": None, "brand": None},
        {"sku_id": "IC-X", "sku_name": "X", "category": "G", "sub_category": None, "brand": None},
    ]
    sales = _daily_sales("IC-Z", "2026-02-01", 30, 10)          # 300 units, 30 days
    sales += _daily_sales("IC-Y", "2026-02-01", 40, 5) + _daily_sales("IC-Y", "2026-04-01", 0, 0)
    # IC-Y: 40 days * 5 = 200. Make units tie between two SKUs on 40 days:
    skus.append({"sku_id": "IC-W", "sku_name": "W", "category": "G", "sub_category": None, "brand": None})
    sales += _daily_sales("IC-X", "2026-02-01", 40, 5)          # 200 units, 40 days
    sales += _daily_sales("IC-W", "2026-02-01", 40, 5)          # 200 units, 40 days (tie with IC-X)
    db = _make_db(tmp_path, skus, sales)
    sel, _ = ds.select_top_skus(db, "G", 10, CUTOFF, 28)
    # Z first (300). Then the 200-unit group: IC-Y/X/W all 200 & 40 days -> sku asc: W, X, Y
    assert list(sel["sku"]) == ["IC-Z", "IC-W", "IC-X", "IC-Y"]


def test_rank_starts_at_one(tmp_path):
    db = _standard_db(tmp_path)
    sel, _ = ds.select_top_skus(db, "Groceries", 3, CUTOFF, 28)
    assert list(sel["rank"]) == [1, 2, 3]


# ── names / nullability ──────────────────────────────────────────────────────────────
def test_product_names_returned(tmp_path):
    db = _standard_db(tmp_path)
    sel, _ = ds.select_top_skus(db, "Groceries", 3, CUTOFF, 28)
    assert sel.loc[sel["sku"] == "IC-A", "sku_name"].iloc[0] == "Prod A"


def test_null_name_ok(tmp_path):
    skus = [{"sku_id": "IC-A", "sku_name": None, "category": "G", "sub_category": None, "brand": None}]
    db = _make_db(tmp_path, skus, _daily_sales("IC-A", "2026-02-01", 30, 10))
    sel, _ = ds.select_top_skus(db, "G", 5, CUTOFF, 28)
    assert list(sel["sku"]) == ["IC-A"]
    assert pd.isna(sel.iloc[0]["sku_name"])


def test_blank_name_ok(tmp_path):
    skus = [{"sku_id": "IC-A", "sku_name": "   ", "category": "G", "sub_category": None, "brand": None}]
    db = _make_db(tmp_path, skus, _daily_sales("IC-A", "2026-02-01", 30, 10))
    sel, _ = ds.select_top_skus(db, "G", 5, CUTOFF, 28)
    assert list(sel["sku"]) == ["IC-A"]


def test_null_brand_ok(tmp_path):
    db = _standard_db(tmp_path)
    sel, _ = ds.select_top_skus(db, "Groceries", 3, CUTOFF, 28)
    assert pd.isna(sel.loc[sel["sku"] == "IC-B", "brand"].iloc[0])


def test_null_sub_category_ok(tmp_path):
    db = _standard_db(tmp_path)
    sel, _ = ds.select_top_skus(db, "Groceries", 3, CUTOFF, 28)
    assert pd.isna(sel.loc[sel["sku"] == "IC-B", "sub_category"].iloc[0])


# ── Free/PACK exclusion ───────────────────────────────────────────────────────────────
def test_free_prefixed_excluded(tmp_path):
    skus = [
        {"sku_id": "IC-A", "sku_name": "A", "category": "G", "sub_category": None, "brand": None},
        {"sku_id": "Freebie1", "sku_name": "F", "category": "G", "sub_category": None, "brand": None},
    ]
    sales = _daily_sales("IC-A", "2026-02-01", 30, 10) + _daily_sales("Freebie1", "2026-02-01", 30, 99)
    db = _make_db(tmp_path, skus, sales)
    sel, _ = ds.select_top_skus(db, "G", 5, CUTOFF, 28)
    assert list(sel["sku"]) == ["IC-A"]


def test_pack_prefixed_excluded(tmp_path):
    skus = [
        {"sku_id": "IC-A", "sku_name": "A", "category": "G", "sub_category": None, "brand": None},
        {"sku_id": "PACK-9", "sku_name": "P", "category": "G", "sub_category": None, "brand": None},
    ]
    sales = _daily_sales("IC-A", "2026-02-01", 30, 10) + _daily_sales("PACK-9", "2026-02-01", 30, 99)
    db = _make_db(tmp_path, skus, sales)
    sel, _ = ds.select_top_skus(db, "G", 5, CUTOFF, 28)
    assert list(sel["sku"]) == ["IC-A"]


# ── fewer-than-requested / error conditions ──────────────────────────────────────────
def test_fewer_than_requested_warns(tmp_path):
    db = _standard_db(tmp_path)
    sel, warnings = ds.select_top_skus(db, "Groceries", 10, CUTOFF, 28)
    assert len(sel) == 3                       # only 3 eligible (THIN excluded)
    assert len(warnings) == 1
    w = warnings[0]
    assert "10" in w and "3" in w              # requested + eligible/selected counts


def test_no_eligible_raises(tmp_path):
    # Category exists but all SKUs are below min-history.
    skus = [{"sku_id": "IC-A", "sku_name": "A", "category": "G", "sub_category": None, "brand": None}]
    db = _make_db(tmp_path, skus, _daily_sales("IC-A", "2026-02-01", 5, 10))
    with pytest.raises(ds.CategoryEligibilityError):
        ds.select_top_skus(db, "G", 5, CUTOFF, 28)


def test_nonexistent_category_raises(tmp_path):
    db = _standard_db(tmp_path)
    with pytest.raises(ds.CategoryNotFoundError):
        ds.select_top_skus(db, "DoesNotExist", 5, CUTOFF, 28)


# ── top_n validation ──────────────────────────────────────────────────────────────────
def test_top_n_one_ok(tmp_path):
    db = _standard_db(tmp_path)
    sel, _ = ds.select_top_skus(db, "Groceries", 1, CUTOFF, 28)
    assert list(sel["sku"]) == ["IC-A"]


def test_top_n_hundred_ok(tmp_path):
    db = _standard_db(tmp_path)
    sel, warnings = ds.select_top_skus(db, "Groceries", 100, CUTOFF, 28)
    assert len(sel) == 3 and len(warnings) == 1     # capped by eligibility, warns


def test_top_n_zero_fails(tmp_path):
    db = _standard_db(tmp_path)
    with pytest.raises(ds.InvalidTopNError):
        ds.select_top_skus(db, "Groceries", 0, CUTOFF, 28)


def test_top_n_negative_fails(tmp_path):
    db = _standard_db(tmp_path)
    with pytest.raises(ds.InvalidTopNError):
        ds.select_top_skus(db, "Groceries", -3, CUTOFF, 28)


def test_top_n_over_100_fails(tmp_path):
    db = _standard_db(tmp_path)
    with pytest.raises(ds.InvalidTopNError):
        ds.select_top_skus(db, "Groceries", 101, CUTOFF, 28)


def test_top_n_noninteger_fails(tmp_path):
    db = _standard_db(tmp_path)
    with pytest.raises(ds.InvalidTopNError):
        ds.select_top_skus(db, "Groceries", 5.0, CUTOFF, 28)


def test_top_n_boolean_fails(tmp_path):
    db = _standard_db(tmp_path)
    with pytest.raises(ds.InvalidTopNError):
        ds.select_top_skus(db, "Groceries", True, CUTOFF, 28)


def test_unsupported_ranking_metric_fails(tmp_path):
    db = _standard_db(tmp_path)
    with pytest.raises(ds.UnsupportedRankingMetricError):
        ds.select_top_skus(db, "Groceries", 3, CUTOFF, 28, ranking_metric="revenue")


# ── category listing ──────────────────────────────────────────────────────────────────
def test_category_listing_counts(tmp_path):
    db = _standard_db(tmp_path)
    cats = ds.list_eligible_categories(db, CUTOFF, 28)
    assert list(cats.columns) == ds.CATEGORY_COLUMNS
    counts = dict(zip(cats["category"], cats["eligible_sku_count"]))
    assert counts["Groceries"] == 3 and counts["Beauty"] == 1   # THIN excluded


def test_category_listing_unit_totals(tmp_path):
    db = _standard_db(tmp_path)
    cats = ds.list_eligible_categories(db, CUTOFF, 28)
    units = dict(zip(cats["category"], cats["historical_units"]))
    assert int(units["Groceries"]) == 750   # 400+200+150 (THIN's 90 excluded)
    assert int(units["Beauty"]) == 105
    # deterministic order: Groceries (750) before Beauty (105)
    assert list(cats["category"]) == ["Groceries", "Beauty"]


def test_category_listing_history_bounds(tmp_path):
    db = _standard_db(tmp_path)
    cats = ds.list_eligible_categories(db, CUTOFF, 28)
    row = cats.loc[cats["category"] == "Groceries"].iloc[0]
    assert row["history_start"] == "2026-02-01"
    assert row["history_end"] == "2026-03-12"   # IC-A: 40 days from 02-01


# ── determinism / output / safety ─────────────────────────────────────────────────────
def test_repeated_calls_identical(tmp_path):
    db = _standard_db(tmp_path)
    a, _ = ds.select_top_skus(db, "Groceries", 3, CUTOFF, 28)
    b, _ = ds.select_top_skus(db, "Groceries", 3, CUTOFF, 28)
    pd.testing.assert_frame_equal(a, b)


def test_output_csv_column_order(tmp_path):
    db = _standard_db(tmp_path)
    sel, _ = ds.select_top_skus(db, "Groceries", 3, CUTOFF, 28)
    out = tmp_path / "sub" / "sel.csv"
    ds._atomic_write_csv(sel, out, overwrite=False)
    written = pd.read_csv(out)
    assert list(written.columns) == ds.SELECTION_COLUMNS


def test_selection_does_not_modify_warehouse(tmp_path):
    db = _standard_db(tmp_path)
    before = _checksum(db)
    ds.list_eligible_categories(db, CUTOFF, 28)
    ds.select_top_skus(db, "Groceries", 3, CUTOFF, 28)
    after = _checksum(db)
    assert before == after                      # row counts + byte checksum unchanged


def test_row_counts_and_checksum_unchanged(tmp_path):
    db = _standard_db(tmp_path)
    n_sku0, n_tx0, digest0 = _checksum(db)
    for _ in range(3):
        ds.select_top_skus(db, "Groceries", 2, CUTOFF, 28)
    n_sku1, n_tx1, digest1 = _checksum(db)
    assert (n_sku0, n_tx0, digest0) == (n_sku1, n_tx1, digest1)


def test_sql_injection_category_is_data(tmp_path):
    db = _standard_db(tmp_path)
    before = _checksum(db)
    malicious = "Groceries'); DROP TABLE sku_master;--"
    # Treated purely as a (non-matching) category value -> no such category, no DB harm.
    with pytest.raises(ds.CategoryNotFoundError):
        ds.select_top_skus(db, malicious, 3, CUTOFF, 28)
    after = _checksum(db)
    assert before == after
    # table still present and intact
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM sku_master").fetchone()[0] == 5
    con.close()


# ── date / db-path validation ─────────────────────────────────────────────────────────
def test_invalid_cutoff_fails(tmp_path):
    db = _standard_db(tmp_path)
    with pytest.raises(ds.InvalidDateError):
        ds.select_top_skus(db, "Groceries", 3, "2026-13-40", 28)
    with pytest.raises(ds.InvalidDateError):
        ds.list_eligible_categories(db, "not-a-date", 28)


def test_missing_db_fails(tmp_path):
    missing = tmp_path / "nope.db"
    with pytest.raises(ds.MissingWarehouseError):
        ds.list_eligible_categories(missing, CUTOFF, 28)
    with pytest.raises(ds.MissingWarehouseError):
        ds.select_top_skus(missing, "Groceries", 3, CUTOFF, 28)


def test_cutoff_accepts_multiple_types(tmp_path):
    db = _standard_db(tmp_path)
    for cutoff in ("2026-06-30", pd.Timestamp("2026-06-30"), pd.Timestamp("2026-06-30").date()):
        sel, _ = ds.select_top_skus(db, "Groceries", 3, cutoff, 28)
        assert len(sel) == 3
