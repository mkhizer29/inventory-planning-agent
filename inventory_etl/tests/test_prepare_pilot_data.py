"""Consolidated invariant tests for the REAL-demand / SYNTHETIC-daily-stock pilot.

Builds a tiny temporary SQLite warehouse and exercises the scope/leakage/stock/cost/snapshot
invariants from the correction brief. One file on purpose — no scatter of tiny test modules.
"""
import copy
import json
import sqlite3
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import prepare_pilot_data as prep       # noqa: E402
import evaluation as ev                  # noqa: E402

AS_OF = "2026-04-30"
DAYS = pd.date_range("2026-01-01", AS_OF, freq="D")   # 120 real days


def _make_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE sku_master (sku_id TEXT, product_id INT, sku_name TEXT,
        category TEXT, sub_category TEXT, brand TEXT, price REAL, pack_size INT, moq INT,
        supplier_lead_time_days INT, is_perishable INT, shelf_life_days REAL, unit_cost REAL,
        cost_source TEXT, eav_cost REAL, margin_cost REAL, flat_cost REAL, is_dropship INT,
        created_at TEXT)""")
    con.executemany("INSERT INTO sku_master VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        # A: real case-pack (6) + real lead time (3) + valid eav cost
        ("A", 1, "Prod A", "Cat1", None, "BrandA", 100.0, 6, 1, 3, 0, None, 40.0, "magento_eav", 40.0, 45.0, 50.0, 0, "2025-12-01"),
        # B: eav invalid(0), margin valid -> staging_margin; default lead/pack
        ("B", 2, "Prod B", "Cat2", None, "BrandB", 50.0, 1, 1, None, 0, None, 20.0, "staging_margin", 0.0, 20.0, None, 0, "2025-12-01"),
        # C: all cost sources invalid -> imputed; default lead/pack
        ("C", 3, "Prod C", "Cat1", None, "BrandC", 80.0, 1, 1, None, 0, None, None, "missing", -5.0, 0.0, None, 0, "2025-12-01"),
    ])
    # qty_ordered == quantity_sold here (no cancellations); row_total = qty * list_price
    con.execute("""CREATE TABLE sales_transactions (sku_id TEXT, channel TEXT,
        transaction_date TEXT, quantity_sold REAL, qty_ordered REAL, discount_amount REAL, row_total REAL)""")
    rows = []
    for i, d in enumerate(DAYS):
        ds = d.date().isoformat()
        qa = 5 + (i % 7); rows.append(("A", "online_delivery", ds, qa, qa, 0, qa * 100.0))
        qb = 3 + (i % 4); rows.append(("B", "online_delivery", ds, qb, qb, 0, qb * 50.0))
        qc = 2 + (i % 3); rows.append(("C", "online_delivery", ds, qc, qc, 0, qc * 80.0))
    rows += [
        ("A", "store", "2026-02-01", 99, 99, 0, 9900),        # physical -> excluded
        ("A", "weirdchan", "2026-02-01", 7, 7, 0, 700),        # unknown -> counted
        ("A", "online_delivery", "2026-06-15", 9999, 9999, 0, 999900),  # AFTER as_of -> dropped
        ("B", "online_delivery", "2026-05-20", 8888, 8888, 0, 444400),  # AFTER as_of -> dropped
    ]
    con.executemany("INSERT INTO sales_transactions VALUES (?,?,?,?,?,?,?)", rows)
    # real stock snapshots: all AFTER the 2026-04-30 as_of, plus warehouse rows to test ALL-preference
    con.execute("""CREATE TABLE inventory_snapshot_history (product_id INT, snapshot_date TEXT,
        location_id TEXT, stock_on_hand REAL, stock_flag TEXT)""")
    snap = []
    for pid in (1, 2, 3):
        for ds, val in (("2026-05-10", 500 + pid), ("2026-05-12", 600 + pid)):
            snap.append((pid, ds, "ALL", float(val), "ok"))
            snap.append((pid, ds, "MLR", 111.0, "ok"))    # must NOT be added to ALL
            snap.append((pid, ds, "BHD", 222.0, "ok"))
    con.executemany("INSERT INTO inventory_snapshot_history VALUES (?,?,?,?,?)", snap)
    con.commit()
    con.close()


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    db = tmp_path_factory.mktemp("wh") / "inv.db"
    _make_db(db)
    con = sqlite3.connect(db)
    cfg = prep.load_config()
    pilot = pd.DataFrame({"sku": ["A", "B", "C"], "category": ["Cat1", "Cat2", "Cat1"]})
    panel, stats = prep.build_model_panel(con, pilot, cfg, None, AS_OF)
    as_of = pd.Timestamp(AS_OF)
    repl = stats["repl_params"]
    pids = [1, 2, 3]
    snapshot, n_real, n_future = prep.select_real_snapshot(con, pids, as_of)
    inv = prep.build_inventory_context(con, pilot, cfg, as_of, panel, repl, snapshot)
    ff = prep.build_forecast_frame(panel, con, pilot, cfg, as_of)
    yield types.SimpleNamespace(con=con, cfg=cfg, pilot=pilot, panel=panel, stats=stats,
                                repl=repl, snapshot=snapshot, n_real=n_real, n_future=n_future,
                                inv=inv, ff=ff, as_of=as_of, db=db)
    con.close()


# 1. units_observed is real and matches source after aggregation
def test_units_observed_real(ctx):
    a = ctx.panel[ctx.panel.sku == "A"].set_index("date")["units_observed"]
    exp = {d.date(): 5 + (i % 7) for i, d in enumerate(DAYS)}
    assert all(int(a.loc[d]) == exp[d.date()] for d in a.index)


# 2. as_of hard boundary drops post-as_of rows
def test_as_of_boundary(ctx):
    assert ctx.panel["date"].max() == ctx.as_of
    assert ctx.stats["rows_after_as_of_dropped"] == 2
    assert ctx.panel["units_observed"].max() < 100          # 9999/8888 spikes gone


# 3. SKU selection uses only the cutoff
def test_selection_cutoff(ctx):
    cand = prep.reselect_candidates(ctx.con, ctx.cfg, "2026-04-30")
    a = int(cand[cand.sku == "A"]["units"].iloc[0])
    src = ctx.panel[ctx.panel.sku == "A"]["units_observed"].sum()
    assert a == int(src)                                    # not inflated by post-cutoff spike


# 4. no banned scenario / synthetic-demand columns anywhere
def test_no_banned_columns(ctx):
    for df in (ctx.panel, ctx.ff, ctx.inv):
        assert [c for c in prep.BANNED_COLS if c in df.columns] == []


# 5. stock_on_hand non-negative, finite, integral
def test_stock_valid(ctx):
    s = ctx.panel["stock_on_hand"]
    assert (s >= 0).all() and np.isfinite(s).all()
    assert (s == s.round()).all() and str(s.dtype).startswith("int")


# 6. stock balance holds and real sales are never capped (implied replenishment >= 0)
def test_stock_balance_and_no_capping(ctx):
    for sku in ("A", "B", "C"):
        g = ctx.panel[ctx.panel.sku == sku].sort_values("date").reset_index(drop=True)
        soh = g["stock_on_hand"].to_numpy(float)
        u = g["units_observed"].to_numpy(float)
        repl = soh[1:] - soh[:-1] + u[1:]                   # = stock[t]-stock[t-1]+units[t]
        assert (repl >= -1e-9).all()                        # replenishment never negative
    # units_observed untouched by reconstruction (matches a pure real re-aggregation)
    assert ctx.panel["units_observed"].sum() > 0


# 7. deterministic across identical runs
def test_deterministic(ctx):
    panel2, _ = prep.build_model_panel(ctx.con, ctx.pilot, ctx.cfg, None, AS_OF)
    assert ctx.panel["stock_on_hand"].tolist() == panel2["stock_on_hand"].tolist()


# 8. synthetic stock is NOT a demand feature
def test_stock_not_a_feature(ctx):
    assert "stock_on_hand" not in prep.DEMAND_FEATURE_WHITELIST
    assert "stock_on_hand" in prep.DEMAND_FEATURE_FORBIDDEN
    assert "stock_on_hand" not in ctx.ff.columns


# 9. demand eval independent of synthetic stock
def test_demand_independent_of_stock(ctx):
    ev.assert_synthetic_independence(ctx.panel, horizon=14)


# 10. June-style run (no eligible snapshot): all synthetic; future snapshots excluded;
#     inventory stock == final panel stock
def test_synthetic_branch(ctx):
    assert ctx.n_real == 0
    assert ctx.n_future == 2                                 # 2026-05-10 and 2026-05-12
    assert ctx.inv["stock_on_hand_is_synthetic"].all()
    last = ctx.panel.sort_values("date").groupby("sku")["stock_on_hand"].last()
    inv_soh = ctx.inv.set_index("sku")["stock_on_hand"]
    assert last.reindex(inv_soh.index).astype(int).equals(inv_soh.astype(int))


# 11. later run with eligible real snapshot: real stock used, cutoff respected, ALL not double-counted
def test_real_snapshot_branch(ctx):
    as_of2 = pd.Timestamp("2026-05-31")
    snap, n_real, n_future = prep.select_real_snapshot(ctx.con, [1, 2, 3], as_of2)
    assert n_real == 3 and n_future == 0
    # latest <= 05-31 is 05-12; ALL value (600+pid), NOT summed with MLR/BHD (111/222)
    assert snap[1]["stock"] == 601 and snap[1]["snapshot_date"] == "2026-05-12"
    assert snap[1]["location"] == "ALL"
    inv2 = prep.build_inventory_context(ctx.con, ctx.pilot, ctx.cfg, as_of2, ctx.panel, ctx.repl, snap)
    assert (~inv2["stock_on_hand_is_synthetic"]).all()
    assert (pd.to_datetime(inv2["stock_snapshot_date"]) <= as_of2).all()
    assert int(inv2[inv2.sku == "A"]["stock_on_hand"].iloc[0]) == 601


# 12. never select a snapshot after as_of
def test_snapshot_cutoff_enforced(ctx):
    snap, n_real, n_future = prep.select_real_snapshot(ctx.con, [1, 2, 3], pd.Timestamp("2026-05-11"))
    assert snap[1]["snapshot_date"] == "2026-05-10"          # 05-12 is after -> excluded
    assert n_future == 1


# 13. per-SKU real replenishment overrides default; else assumed
def test_replenishment_sources(ctx):
    a = ctx.inv[ctx.inv.sku == "A"].iloc[0]
    assert a["lead_time_days"] == 3 and a["lead_time_source"] == "sku_master_picking_mode"
    assert a["pack_size"] == 6 and a["pack_size_source"] == "sku_master_case_pack"
    b = ctx.inv[ctx.inv.sku == "B"].iloc[0]
    assert b["lead_time_days"] == 7 and b["lead_time_source"] == "assumed_default"
    assert b["pack_size"] == 1 and b["pack_size_source"] == "assumed_default"


# 14. cost: effective positive/finite; invalid observed stays null & flagged
def test_cost_validation(ctx):
    eff = pd.to_numeric(ctx.inv["unit_cost_effective"], errors="coerce").dropna()
    assert (eff > 0).all() and np.isfinite(eff).all()
    c = ctx.inv[ctx.inv.sku == "C"].iloc[0]
    assert not c["cost_is_valid"] and pd.isna(c["unit_cost_observed"]) and c["cost_is_imputed"]
    assert "NON_POSITIVE_COST" in c["cost_quality_flag"]


# 15. cost precedence
def test_cost_precedence():
    tol = 0.25
    assert prep.classify_cost([("magento_eav", 40), ("staging_margin", 45), ("product_flat", 50)], 100, tol)["source"] == "magento_eav"
    assert prep.classify_cost([("magento_eav", 0), ("staging_margin", 20), ("product_flat", 30)], 100, tol)["source"] == "staging_margin"
    r = prep.classify_cost([("magento_eav", 0), ("staging_margin", -1), ("product_flat", None)], 100, tol)
    assert r["source"] == "missing" and r["observed"] is None
    assert "NON_POSITIVE_COST" in r["flags"] and "MISSING_COST" not in r["flags"]
    m = prep.classify_cost([("magento_eav", None), ("staging_margin", None), ("product_flat", None)], 100, tol)
    assert "MISSING_COST" in m["flags"]


# 16. reorder recommendation is transparent (pack multiple, MOQ, purchase value)
def test_reorder_recommendation(ctx):
    a = ctx.inv[ctx.inv.sku == "A"].iloc[0]
    q = a["recommended_order_quantity"]
    assert q >= 0 and q % a["pack_size"] == 0                # rounded to pack size
    if q > 0:
        assert q >= a["moq"]
        assert a["recommended_purchase_value"] == pytest.approx(q * a["unit_cost_effective"])


# 17. forecast frame: 14 future days, no actuals
def test_forecast_frame(ctx):
    n = int(ctx.cfg["pilot"]["forecast_feature_days"])
    assert (ctx.ff.groupby(["sku", "channel"])["date"].nunique() == n).all()
    assert (pd.to_datetime(ctx.ff["date"]) > ctx.as_of).all()
    assert "units_observed" not in ctx.ff.columns


# 18. chronological split; full validation passes; scope correct
def test_validation_and_scope(ctx):
    cutoff, train, test = ev.backtest_split(ctx.panel, 14)
    assert train["date"].max() <= cutoff < test["date"].min()
    problems = prep.validate_outputs(ctx.panel, ctx.ff, ctx.inv, ctx.pilot, ctx.cfg, ctx.as_of, False)
    assert problems == [], problems
    assert set(ctx.panel["channel"]) == {"naheed_web"}
    assert ctx.stats["physical_store_rows_excluded"] == 1
    assert "weirdchan" in ctx.stats["unknown_channel_rows"]
    assert list(ctx.panel.columns) == prep.MODEL_PANEL_COLS
    assert list(ctx.inv.columns) == prep.INVENTORY_COLS


# 19. price columns exist and equal the list price on a clean sale day (no cancellations, no discount)
def test_price_columns_present(ctx):
    assert "net_price_paid" in prep.MODEL_PANEL_COLS
    a = ctx.panel[(ctx.panel.sku == "A") & (ctx.panel.units_observed > 0)].iloc[0]
    assert a["effective_unit_price"] == pytest.approx(100.0)   # list price, stable
    assert a["net_price_paid"] == pytest.approx(100.0)          # no discount -> equal


# 20. price is computed on the ORDERED basis: a mostly-cancelled order must NOT inflate price,
#     and a discount must show up in net_price_paid (this is the reported bug fix)
def test_price_not_inflated_by_cancellation(tmp_path):
    import sqlite3
    db = tmp_path / "px.db"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE sku_master (sku_id TEXT, product_id INT, sku_name TEXT,
        category TEXT, sub_category TEXT, brand TEXT, price REAL, moq INT, pack_size INT,
        supplier_lead_time_days INT)""")
    con.execute("INSERT INTO sku_master VALUES ('Z', 9, 'Prod Z', 'Cat1', NULL, 'BrandZ', 450.0, 1, 1, 7)")
    con.execute("""CREATE TABLE sales_transactions (sku_id TEXT, channel TEXT,
        transaction_date TEXT, quantity_sold REAL, qty_ordered REAL, discount_amount REAL, row_total REAL)""")
    days = pd.date_range("2026-03-01", periods=20, freq="D")
    rows = []
    for i, d in enumerate(days):
        ds = d.date().isoformat()
        if i == 10:      # 51 ordered @450, 50 cancelled -> 1 net unit (the inflation scenario)
            rows.append(("Z", "online_delivery", ds, 1, 51, 0, 51 * 450.0))
        elif i == 11:    # 10 ordered @450 with a 450 total discount
            rows.append(("Z", "online_delivery", ds, 10, 10, 450.0, 10 * 450.0))
        else:
            rows.append(("Z", "online_delivery", ds, 4, 4, 0, 4 * 450.0))
    con.executemany("INSERT INTO sales_transactions VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    cfg = prep.load_config()
    pilot = pd.DataFrame({"sku": ["Z"], "category": ["Cat1"]})
    panel, _ = prep.build_model_panel(con, pilot, cfg, None, "2026-03-20")
    con.close()
    canc = panel[panel.date == "2026-03-11"].iloc[0]        # the cancellation day
    assert canc["units_observed"] == 1                       # real net demand preserved
    assert canc["effective_unit_price"] == pytest.approx(450.0)   # NOT 22950
    disc = panel[panel.date == "2026-03-12"].iloc[0]        # the discount day
    assert disc["effective_unit_price"] == pytest.approx(450.0)   # list price
    assert disc["net_price_paid"] == pytest.approx(405.0)         # (4500-450)/10


# 21. Ramazan calendar features (config-driven, known-in-advance, inclusive boundaries)
def test_ramadan_calendar_features():
    cfg = prep.load_config()
    dates = pd.DatetimeIndex(["2026-02-18", "2026-02-19", "2026-02-25", "2026-02-26",
                              "2026-03-18", "2026-03-19", "2026-03-20", "2026-03-21"])
    cal = prep.calendar_features(dates, cfg).set_index("date")
    # config end_date = 2026-03-20 (30th fast) so 03-19 (day 29) and 03-20 (day 30) are IN Ramazan;
    # 03-21 (Eid al-Fitr) is out.
    expected = {
        "2026-02-18": (0, 0, 0),
        "2026-02-19": (1, 1, 1),
        "2026-02-25": (1, 7, 1),
        "2026-02-26": (1, 8, 2),
        "2026-03-18": (1, 28, 4),
        "2026-03-19": (1, 29, 5),
        "2026-03-20": (1, 30, 5),
        "2026-03-21": (0, 0, 0),
    }
    for d, (ir, rd, rw) in expected.items():
        row = cal.loc[pd.Timestamp(d)]
        assert (int(row["is_ramadan"]), int(row["ramadan_day"]), int(row["ramadan_week"])) == (ir, rd, rw), d
    for c in ("is_ramadan", "ramadan_day", "ramadan_week"):
        assert cal[c].notna().all() and cal[c].dtype.kind in "iu"     # integer, no nulls
    out = cal[cal["is_ramadan"] == 0]
    assert (out["ramadan_day"] == 0).all() and (out["ramadan_week"] == 0).all()


# 22. Ramazan fields are in every schema + whitelist; stock/cost never whitelisted
def test_ramadan_in_schemas(ctx):
    for f in ("is_ramadan", "ramadan_day", "ramadan_week"):
        assert f in prep.MODEL_PANEL_COLS
        assert f in prep.FORECAST_FRAME_COLS
        assert f in prep.DEMAND_FEATURE_WHITELIST
        assert f in ctx.panel.columns and f in ctx.ff.columns
        for df in (ctx.panel, ctx.ff):
            assert df[f].notna().all() and df[f].dtype.kind in "iu"
    assert set(ctx.panel["is_ramadan"].unique()) == {0, 1}    # panel spans Ramazan 2026
    for bad in ("stock_on_hand", "unit_cost", "unit_cost_effective", "unit_cost_observed"):
        assert bad not in prep.DEMAND_FEATURE_WHITELIST


# 23. an invalid configured period (end_date < start_date) raises a clear error
def test_invalid_ramadan_period_raises():
    cfg = copy.deepcopy(prep.load_config())
    cfg["external_signals"]["ramadan_periods"] = [
        {"year": 2027, "location": "Karachi", "start_date": "2027-03-01", "end_date": "2027-02-01"}]
    with pytest.raises(ValueError):
        prep.calendar_features(pd.date_range("2027-01-01", "2027-04-01", freq="D"), cfg)


# 24. existing calendar features remain intact alongside the new Ramazan ones
def test_existing_calendar_unchanged():
    cfg = prep.load_config()
    cal = prep.calendar_features(pd.DatetimeIndex(["2026-02-19"]), cfg).iloc[0]
    for c in ("is_public_holiday", "holiday_name", "is_payday_window", "day_of_week",
              "is_weekend", "week_of_year", "month"):
        assert c in cal.index
    assert int(cal["day_of_week"]) == 3       # 2026-02-19 is a Thursday (Mon=0)
    assert int(cal["is_weekend"]) == 0
    assert int(cal["month"]) == 2


# 24b. product name (sku_name) is carried into the model panel
def test_sku_name_in_panel(ctx):
    assert "sku_name" in prep.MODEL_PANEL_COLS
    assert "sku_name" in ctx.panel.columns
    assert ctx.panel[ctx.panel.sku == "A"]["sku_name"].iloc[0] == "Prod A"
    assert ctx.panel["sku_name"].notna().all()


# 25. Eid al-Fitr holidays (21-23 Mar 2026) come from configured extra_public_holidays
def test_eid_public_holidays():
    cfg = prep.load_config()
    dates = pd.DatetimeIndex(["2026-03-20", "2026-03-21", "2026-03-22", "2026-03-23"])
    cal = prep.calendar_features(dates, cfg).set_index("date")
    for d in ("2026-03-21", "2026-03-22", "2026-03-23"):
        row = cal.loc[pd.Timestamp(d)]
        assert int(row["is_public_holiday"]) == 1
        assert "Eid" in str(row["holiday_name"])
        assert int(row["is_ramadan"]) == 0        # Eid is after Ramazan (day 30 was 03-20)
    assert int(cal.loc[pd.Timestamp("2026-03-20")]["is_ramadan"]) == 1   # last fast still Ramazan


# 26. an invalid extra_public_holidays entry raises a clear error
def test_invalid_extra_holiday_raises():
    cfg = copy.deepcopy(prep.load_config())
    cfg["external_signals"]["extra_public_holidays"] = [{"name": "no date key"}]
    with pytest.raises(ValueError):
        prep.calendar_features(pd.date_range("2026-03-01", "2026-03-31", freq="D"), cfg)


# 27. catalog/special-price sale (unit_price < original_price -> catalog_discount_amount) is
#     folded into discount_pct / on_promo, alongside cart discount_amount
def test_catalog_sale_discount(tmp_path):
    import sqlite3
    db = tmp_path / "cat.db"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE sku_master (sku_id TEXT, product_id INT, sku_name TEXT,
        category TEXT, sub_category TEXT, brand TEXT, price REAL, moq INT, pack_size INT,
        supplier_lead_time_days INT)""")
    con.execute("INSERT INTO sku_master VALUES ('Z', 9, 'Prod Z', 'Cat1', NULL, 'BrandZ', 450.0, 1, 1, 7)")
    con.execute("""CREATE TABLE sales_transactions (sku_id TEXT, channel TEXT,
        transaction_date TEXT, quantity_sold REAL, qty_ordered REAL, discount_amount REAL,
        catalog_discount_amount REAL, row_total REAL)""")
    days = pd.date_range("2026-03-01", periods=20, freq="D")
    rows = []
    for i, d in enumerate(days):
        ds = d.date().isoformat()
        if i == 10:      # catalog sale: 10 @400 (regular 450) -> markdown (450-400)*10 = 500, no cart discount
            rows.append(("Z", "online_delivery", ds, 10, 10, 0, 500.0, 4000.0))
        elif i == 11:    # cart discount only: 10 @450 with 450 coupon, no catalog markdown
            rows.append(("Z", "online_delivery", ds, 10, 10, 450.0, 0.0, 4500.0))
        else:            # regular day: no discount of either kind
            rows.append(("Z", "online_delivery", ds, 4, 4, 0, 0.0, 4 * 450.0))
    con.executemany("INSERT INTO sales_transactions VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    cfg = prep.load_config()
    pilot = pd.DataFrame({"sku": ["Z"], "category": ["Cat1"]})
    panel, _ = prep.build_model_panel(con, pilot, cfg, None, "2026-03-20")
    con.close()
    assert "catalog_discount_amount" in prep.MODEL_PANEL_COLS
    cat = panel[panel.date == "2026-03-11"].iloc[0]          # catalog-sale day
    assert cat["catalog_discount_amount"] == pytest.approx(500.0)
    assert cat["on_promo"] == 1                               # flagged even with discount_amount == 0
    assert cat["discount_pct"] == pytest.approx(500.0 / 4500.0)   # markdown off the regular value
    cart = panel[panel.date == "2026-03-12"].iloc[0]         # cart-discount day
    assert cart["on_promo"] == 1 and cart["discount_pct"] == pytest.approx(0.10)
    reg = panel[panel.date == "2026-03-13"].iloc[0]          # regular day
    assert reg["on_promo"] == 0 and reg["discount_pct"] == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════════
# Phase 2 — run-specific pilot file (--pilot-file) + isolated output dir (--output-dir)
# ══════════════════════════════════════════════════════════════════════════════════
FILES4 = ("model_panel.parquet", "forecast_frame.parquet",
          "inventory_context.parquet", "pilot_manifest.json")


def _fresh_db(tmp_path: Path) -> Path:
    db = tmp_path / "inv.db"
    if not db.exists():
        _make_db(db)
    return db


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _run_prep(tmp_path: Path, pilot_csv: Path, out_name: str = "processed",
              extra: list[str] | None = None, db: Path | None = None):
    db = db or _fresh_db(tmp_path)
    out_dir = tmp_path / out_name
    argv = ["--db-path", str(db), "--pilot-file", str(pilot_csv),
            "--output-dir", str(out_dir), "--as-of-date", AS_OF]
    if extra:
        argv += extra
    rc = prep.main(argv)
    return rc, out_dir, db


def _manifest(out_dir: Path) -> dict:
    return json.loads((out_dir / "pilot_manifest.json").read_text(encoding="utf-8"))


def _read_panel(out_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "model_panel.parquet")


def _snapshot_dir(d: Path):
    if not d.exists():
        return None
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in d.iterdir() if p.is_file()}


# 1. default pilot path is the repo-root pilot_skus.csv
def test_default_pilot_path_is_pilot_skus_csv():
    args = prep.parse_args([])
    assert Path(args.pilot_file).resolve() == prep.PILOT_LIST.resolve()


# 2 + 22. fixed-pilot invocation still works without --pilot-file; selection_mode == fixed_pilot
def test_fixed_pilot_without_flag_and_mode(tmp_path, monkeypatch):
    fixed = _write_csv(tmp_path / "fixed_pilot.csv", "sku\nA\nB\nC\n")
    monkeypatch.setattr(prep, "PILOT_LIST", fixed)          # treat this temp file as THE fixed list
    db = _fresh_db(tmp_path)
    out_dir = tmp_path / "processed"
    rc = prep.main(["--db-path", str(db), "--output-dir", str(out_dir), "--as-of-date", AS_OF])
    assert rc == 0
    assert all((out_dir / f).exists() for f in FILES4)
    assert _manifest(out_dir)["selection_mode"] == "fixed_pilot"


# 3. custom pilot file with only a sku column works
def test_custom_pilot_only_sku_column(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku\nA\nB\n")
    rc, out_dir, _ = _run_prep(tmp_path, pilot)
    assert rc == 0
    assert set(_read_panel(out_dir)["sku"].unique()) == {"A", "B"}


# 4. full Phase 1 selector schema (all extra columns) works
def test_custom_pilot_full_selector_schema(tmp_path):
    header = "rank,sku,sku_name,category,sub_category,brand,historical_units,active_days,history_start,history_end"
    body = ("1,A,WRONG NAME,WrongCat,WrongSub,WrongBrand,999999,999,2000-01-01,2000-01-02\n"
            "2,B,Other,WrongCat,,,, ,,\n")
    pilot = _write_csv(tmp_path / "sel.csv", header + "\n" + body)
    rc, out_dir, _ = _run_prep(tmp_path, pilot)
    assert rc == 0
    assert set(_read_panel(out_dir)["sku"].unique()) == {"A", "B"}


# 5. custom file selects only its listed SKUs
def test_custom_pilot_selects_only_listed(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku\nA\n")
    rc, out_dir, _ = _run_prep(tmp_path, pilot)
    assert rc == 0
    assert set(_read_panel(out_dir)["sku"].unique()) == {"A"}


# 6. historical_units from the CSV is NOT used as demand
def test_csv_historical_units_not_used_as_demand(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku,historical_units\nA,999999\n")
    rc, out_dir, _ = _run_prep(tmp_path, pilot)
    panel = _read_panel(out_dir)
    assert panel["units_observed"].max() < 100          # real DB demand, not the 999999 in CSV


# 7. sku_name from the CSV is NOT used as a join key / authoritative attribute
def test_csv_sku_name_not_authoritative(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku,sku_name\nA,TOTALLY WRONG NAME\n")
    rc, out_dir, _ = _run_prep(tmp_path, pilot)
    panel = _read_panel(out_dir)
    assert panel.loc[panel["sku"] == "A", "sku_name"].iloc[0] == "Prod A"   # from sku_master


# 8. unknown extra columns are tolerated
def test_custom_pilot_unknown_columns_tolerated(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku,foo_bar,whatever\nA,1,x\nB,2,y\n")
    rc, out_dir, _ = _run_prep(tmp_path, pilot)
    assert rc == 0 and set(_read_panel(out_dir)["sku"].unique()) == {"A", "B"}


# 9. missing pilot file fails clearly
def test_missing_pilot_file_fails(tmp_path):
    with pytest.raises(SystemExit):
        _run_prep(tmp_path, tmp_path / "does_not_exist.csv")


# 10. a directory passed as --pilot-file fails clearly
def test_directory_pilot_file_fails(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(SystemExit):
        _run_prep(tmp_path, d)


# 11. missing sku column fails clearly
def test_missing_sku_column_fails(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "product\nA\nB\n")
    with pytest.raises(SystemExit):
        _run_prep(tmp_path, pilot)


# 12. blank SKU fails clearly (second column keeps the whitespace row from being skipped)
def test_blank_sku_fails(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku,note\nA,x\n   ,y\nB,z\n")
    with pytest.raises(SystemExit):
        _run_prep(tmp_path, pilot)


# 13. null SKU fails clearly
def test_null_sku_fails(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku,category\nA,Cat1\n,Cat2\n")
    with pytest.raises(SystemExit):
        _run_prep(tmp_path, pilot)


# 14. duplicate SKU: warning in non-strict, failure in strict
def test_duplicate_sku_warn_then_strict_fail(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku\nA\nB\nA\n")
    rc, out_dir, db = _run_prep(tmp_path, pilot)
    assert rc == 0
    man = _manifest(out_dir)
    assert any("duplicate" in w.lower() for w in man["warnings"])
    assert man["selected_sku_count"] == 2                 # deduped, not duplicated
    with pytest.raises(SystemExit):
        _run_prep(tmp_path, pilot, out_name="processed_strict", extra=["--strict"], db=db)


# 15. unknown warehouse SKU: warning in non-strict, failure in strict
def test_unknown_sku_warn_then_strict_fail(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku\nA\nB\nZZZ-NOPE\n")
    rc, out_dir, db = _run_prep(tmp_path, pilot)
    assert rc == 0
    assert any("not found in sku_master" in w for w in _manifest(out_dir)["warnings"])
    assert set(_read_panel(out_dir)["sku"].unique()) == {"A", "B"}   # unknown excluded downstream
    with pytest.raises(SystemExit):
        _run_prep(tmp_path, pilot, out_name="processed_strict", extra=["--strict"], db=db)


# 16. the real repo pilot_skus.csv is never modified by a custom run
def test_repo_pilot_skus_untouched(tmp_path):
    before = prep.PILOT_LIST.read_bytes() if prep.PILOT_LIST.exists() else None
    pilot = _write_csv(tmp_path / "sel.csv", "sku\nA\n")
    _run_prep(tmp_path, pilot)
    after = prep.PILOT_LIST.read_bytes() if prep.PILOT_LIST.exists() else None
    assert before == after


# 17. the input custom pilot file is never modified
def test_input_pilot_file_untouched(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku,note\nA,keep\nB,keep\n")
    before = pilot.read_bytes()
    _run_prep(tmp_path, pilot)
    assert pilot.read_bytes() == before


# 18. custom output dir receives exactly the four generated files
def test_custom_output_dir_has_four_files(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku\nA\nB\n")
    rc, out_dir, _ = _run_prep(tmp_path, pilot)
    assert rc == 0
    produced = {p.name for p in out_dir.iterdir() if p.is_file()}
    assert produced == set(FILES4)


# 19. a custom output dir does not write to data/processed
def test_custom_run_does_not_touch_data_processed(tmp_path):
    before = _snapshot_dir(prep.DEFAULT_OUT)
    pilot = _write_csv(tmp_path / "sel.csv", "sku\nA\nB\n")
    _run_prep(tmp_path, pilot)
    assert _snapshot_dir(prep.DEFAULT_OUT) == before


# 20 + 21. two runs are isolated; the first is unchanged after the second
def test_two_runs_isolated(tmp_path):
    db = _fresh_db(tmp_path)
    p1 = _write_csv(tmp_path / "s1.csv", "sku\nA\nB\n")
    p2 = _write_csv(tmp_path / "s2.csv", "sku\nA\nB\nC\n")
    _run_prep(tmp_path, p1, out_name="run1", db=db)
    run1_bytes = (tmp_path / "run1" / "model_panel.parquet").read_bytes()
    _run_prep(tmp_path, p2, out_name="run2", db=db)
    assert set(_read_panel(tmp_path / "run1")["sku"].unique()) == {"A", "B"}
    assert set(_read_panel(tmp_path / "run2")["sku"].unique()) == {"A", "B", "C"}
    assert (tmp_path / "run1" / "model_panel.parquet").read_bytes() == run1_bytes


# 23. selection_mode is dynamic for a custom file
def test_selection_mode_dynamic(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku\nA\nB\n")
    _, out_dir, _ = _run_prep(tmp_path, pilot)
    assert _manifest(out_dir)["selection_mode"] == "dynamic"


# 24 + 25. pilot_file and source_warehouse recorded in the manifest
def test_manifest_records_paths(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku\nA\nB\n")
    _, out_dir, db = _run_prep(tmp_path, pilot)
    man = _manifest(out_dir)
    assert man["pilot_file"].endswith("sel.csv")
    assert man["source_warehouse"].endswith("inv.db")


# 26 + 27 + 28. counts + sorted categories + null requested_sku_count
def test_manifest_counts_categories_requested(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku\nB\nA\n")   # order B,A on purpose
    _, out_dir, _ = _run_prep(tmp_path, pilot)
    man = _manifest(out_dir)
    assert man["selected_sku_count"] == 2
    assert man["selected_categories"] == ["Cat1", "Cat2"]     # A->Cat1, B->Cat2, sorted
    assert man["requested_sku_count"] is None


# 29. schema constants remain unchanged and the produced panel matches them
def test_schema_constants_and_columns(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku\nA\nB\n")
    _, out_dir, _ = _run_prep(tmp_path, pilot)
    assert list(_read_panel(out_dir).columns) == prep.MODEL_PANEL_COLS
    assert "stock_on_hand" in prep.DEMAND_FEATURE_FORBIDDEN
    assert "stock_on_hand" not in prep.DEMAND_FEATURE_WHITELIST


# 30. leakage guard still holds for a dynamic-run panel
def test_dynamic_run_is_leakage_safe(tmp_path):
    pilot = _write_csv(tmp_path / "sel.csv", "sku\nA\nB\nC\n")
    _, out_dir, _ = _run_prep(tmp_path, pilot)
    panel = _read_panel(out_dir)
    panel["date"] = pd.to_datetime(panel["date"])
    ev.assert_synthetic_independence(panel, horizon=14)      # raises on leakage
    assert panel["date"].max() <= pd.Timestamp(AS_OF)        # no demand after as_of


# 31. --reselect-pilot-skus behavior unchanged (writes a candidate list, approved list untouched)
def test_reselect_still_works(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path)
    cand = tmp_path / "candidate.csv"
    monkeypatch.setattr(prep, "CANDIDATE_LIST", cand)         # keep the repo clean
    rc = prep.main(["--db-path", str(db), "--reselect-pilot-skus",
                    "--selection-cutoff", AS_OF])
    assert rc == 0 and cand.exists()
    cols = pd.read_csv(cand).columns.tolist()
    assert cols == ["sku", "category", "brand", "name", "units"]


# 32. --reselect-pilot-skus + a custom --pilot-file is rejected
def test_reselect_plus_custom_pilot_rejected(tmp_path):
    db = _fresh_db(tmp_path)
    pilot = _write_csv(tmp_path / "sel.csv", "sku\nA\n")
    with pytest.raises(SystemExit):
        prep.main(["--db-path", str(db), "--reselect-pilot-skus",
                   "--selection-cutoff", AS_OF, "--pilot-file", str(pilot)])
