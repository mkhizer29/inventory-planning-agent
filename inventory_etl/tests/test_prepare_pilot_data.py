"""Focused tests for src/prepare_pilot_data.py (daily, ecommerce-only pilot).

Builds a tiny temporary SQLite warehouse and exercises the highest-risk behaviours:
channel scope, unique daily keys, no pre-activation zero-fill, missing-vs-zero stock,
stockout censoring, leakage-free future features, assumption flags, schema + manifest.
"""
import sqlite3
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import prepare_pilot_data as prep  # noqa: E402


def _make_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE sku_master (sku_id TEXT, product_id INT, sku_name TEXT,
        category TEXT, sub_category TEXT, brand TEXT, price REAL, pack_size INT,
        is_perishable INT, shelf_life_days REAL, unit_cost REAL, is_dropship INT,
        created_at TEXT)""")
    con.executemany("INSERT INTO sku_master VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        ("A", 1, "Prod A", "Cat1", None, "BrandA", 100.0, 1, 0, None, None, 0, "2026-02-01"),
        ("B", 2, "Prod B", "Cat2", None, "BrandB", 50.0, 1, 0, None, None, 0, "2026-02-01"),
    ])
    con.execute("""CREATE TABLE sales_transactions (sku_id TEXT, channel TEXT,
        transaction_date TEXT, quantity_sold REAL, discount_amount REAL, row_total REAL)""")
    con.executemany("INSERT INTO sales_transactions VALUES (?,?,?,?,?,?)", [
        ("A", "online_delivery", "2026-03-01", 5, 0, 500),   # active start for A
        ("A", "online_delivery", "2026-03-03", 3, 30, 270),  # promo day (discount>0)
        ("A", "store",           "2026-03-01", 9, 0, 900),   # physical -> excluded
        ("A", "weirdchan",       "2026-03-01", 1, 0, 100),   # unknown channel
        ("B", "online_delivery", "2026-03-05", 2, 0, 100),   # B active start = 03-05
    ])
    con.execute("""CREATE TABLE inventory_snapshot_history (product_id INT,
        snapshot_date TEXT, stock_on_hand REAL, stock_flag TEXT, location_id TEXT)""")
    con.executemany("INSERT INTO inventory_snapshot_history VALUES (?,?,?,?,?)", [
        (1, "2026-03-01", 10, "ok", "ALL"),   # A available
        (1, "2026-03-02", 0,  "ok", "ALL"),   # A reliable zero -> censored
    ])
    con.commit()
    con.close()


@pytest.fixture()
def ctx(tmp_path):
    db = tmp_path / "inv.db"
    _make_db(db)
    con = sqlite3.connect(db)
    cfg = prep.load_config()
    pilot = pd.DataFrame({"sku": ["A", "B"], "category": ["Cat1", "Cat2"]})
    panel, stats = prep.build_model_panel(con, pilot, cfg, None, "2026-03-06")
    yield types.SimpleNamespace(con=con, cfg=cfg, pilot=pilot, panel=panel, stats=stats,
                                as_of=pd.Timestamp("2026-03-06"), db=db)
    con.close()


def test_physical_store_rows_excluded(ctx):
    assert ctx.stats["physical_store_rows_excluded"] == 1          # the one 'store' row
    assert "store" not in set(ctx.panel["channel"])


def test_online_delivery_normalised_to_naheed_web(ctx):
    assert set(ctx.panel["channel"]) == {"naheed_web"}


def test_unknown_channel_counted_not_mapped(ctx):
    assert "weirdchan" in ctx.stats["unknown_channel_rows"]         # logged, not silently mapped


def test_unique_daily_keys(ctx):
    assert not ctx.panel.duplicated(["sku", "channel", "date"]).any()


def test_no_pre_activation_zero_fill(ctx):
    b = ctx.panel[ctx.panel.sku == "B"]
    assert b["date"].min() == pd.Timestamp("2026-03-05")           # B not filled before first sale
    a = ctx.panel[ctx.panel.sku == "A"]
    assert a["date"].min() == pd.Timestamp("2026-03-01")


def test_missing_stock_is_not_zero(ctx):
    b = ctx.panel[ctx.panel.sku == "B"]
    assert (~b["stock_observation_available"]).all()               # B has no snapshot
    assert b["stock_on_hand"].isna().all()                         # missing, NOT 0
    assert not b["is_stockout"].any()


def test_reliable_zero_stock_marks_censored(ctx):
    row = ctx.panel[(ctx.panel.sku == "A") & (ctx.panel.date == "2026-03-02")].iloc[0]
    assert row["is_stockout"] and row["demand_censored"]
    assert not row["training_eligible"]
    assert row["units_observed"] == 0                              # observed value preserved


def test_forecast_features_no_leakage(ctx):
    ff = prep.build_forecast_features(ctx.panel, ctx.con, ctx.pilot, ctx.cfg, ctx.as_of)
    assert "units_observed" not in ff.columns                      # no actual sales
    assert (ff["planned_promo"] == 0).all()
    assert ff["feature_availability_flag"].str.contains("planned_promo_unavailable").all()
    assert (ff["date"] > ctx.as_of).all()                          # future only
    assert (ff.groupby(["sku", "channel"])["date"].nunique() == 14).all()


def test_lead_time_and_moq_are_flagged_assumptions(ctx):
    inv = prep.build_inventory_context(ctx.con, ctx.pilot, ctx.cfg, ctx.as_of)
    assert inv["lead_time_is_assumed"].all() and inv["moq_is_assumed"].all()
    assert (inv["supplier_lead_time_days"] == 7).all() and (inv["moq"] == 1).all()
    assert inv["stock_in_transit_is_assumed"].all() and inv["perishability_is_assumed"].all()
    assert (inv["pack_size"] >= 1).all()
    assert inv["location_id"].nunique() == 1                       # no channel duplication


def test_schemas_validate_and_manifest_passes(ctx):
    ff = prep.build_forecast_features(ctx.panel, ctx.con, ctx.pilot, ctx.cfg, ctx.as_of)
    inv = prep.build_inventory_context(ctx.con, ctx.pilot, ctx.cfg, ctx.as_of)
    problems = prep.validate_outputs(ctx.panel, ff, inv, ctx.cfg)
    assert problems == []
    assert list(ctx.panel.columns) == prep.MODEL_PANEL_COLS
    args = types.SimpleNamespace(generated_at="t", db_path=str(ctx.db), selection_cutoff=None)
    man = prep.build_manifest(ctx.panel, ff, inv, ctx.pilot, ctx.cfg, args, ctx.stats,
                              ctx.as_of, [], problems)
    assert man["validation_status"] == "passed"
    assert man["data_frequency"] == "daily"
    assert man["physical_store_rows_excluded"] == 1
    assert man["ecommerce_channels"] == ["naheed_web"]
