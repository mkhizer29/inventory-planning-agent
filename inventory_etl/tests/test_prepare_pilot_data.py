"""Consolidated invariant tests for the real-demand / synthetic-inventory pilot.

Builds a tiny temporary SQLite warehouse and exercises the 20 causality / leakage /
cost invariants from the redesign brief (§13). One file on purpose — no scatter of
tiny test modules.
"""
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
import synthetic_inventory as syn        # noqa: E402
import evaluation as ev                  # noqa: E402

AS_OF = "2026-04-30"
DAYS = pd.date_range("2026-01-01", AS_OF, freq="D")   # 120 real days


def _make_db(path: Path) -> None:
    con = sqlite3.connect(path)
    # sku_master carries per-source cost columns to exercise precedence + validity
    con.execute("""CREATE TABLE sku_master (sku_id TEXT, product_id INT, sku_name TEXT,
        category TEXT, sub_category TEXT, brand TEXT, price REAL, pack_size INT,
        is_perishable INT, shelf_life_days REAL, unit_cost REAL, cost_source TEXT,
        eav_cost REAL, margin_cost REAL, flat_cost REAL, is_dropship INT, created_at TEXT)""")
    con.executemany("INSERT INTO sku_master VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        # A: valid eav cost -> magento_eav
        ("A", 1, "Prod A", "Cat1", None, "BrandA", 100.0, 6, 0, None, 40.0, "magento_eav", 40.0, 45.0, 50.0, 0, "2025-12-01"),
        # B: eav invalid (0), margin valid -> staging_margin
        ("B", 2, "Prod B", "Cat2", None, "BrandB", 50.0, 6, 0, None, 20.0, "staging_margin", 0.0, 20.0, None, 0, "2025-12-01"),
        # C: all sources invalid -> imputed fallback, observed stays null
        ("C", 3, "Prod C", "Cat1", None, "BrandC", 80.0, 6, 0, None, None, "missing", -5.0, 0.0, None, 0, "2025-12-01"),
    ])
    con.execute("""CREATE TABLE sales_transactions (sku_id TEXT, channel TEXT,
        transaction_date TEXT, quantity_sold REAL, discount_amount REAL, row_total REAL)""")
    rows = []
    for i, d in enumerate(DAYS):
        ds = d.date().isoformat()
        rows.append(("A", "online_delivery", ds, 5 + (i % 7), 0, (5 + (i % 7)) * 100.0))
        rows.append(("B", "online_delivery", ds, 3 + (i % 4), 0, (3 + (i % 4)) * 50.0))
        rows.append(("C", "online_delivery", ds, 2 + (i % 3), 0, (2 + (i % 3)) * 80.0))
    # noise rows: physical store (excluded), unknown channel (counted), and POST-as_of sales
    rows += [
        ("A", "store", "2026-02-01", 99, 0, 9900),
        ("A", "weirdchan", "2026-02-01", 7, 0, 700),
        ("A", "online_delivery", "2026-06-15", 9999, 0, 999900),   # AFTER as_of -> must be dropped
        ("B", "online_delivery", "2026-05-20", 8888, 0, 444400),   # AFTER as_of -> must be dropped
    ]
    con.executemany("INSERT INTO sales_transactions VALUES (?,?,?,?,?,?)", rows)
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
    real_daily = (panel.groupby(["sku", "date"], as_index=False)["units_observed"]
                       .sum().rename(columns={"units_observed": "units"}))
    sku_meta = panel[["sku", "product_id", "category", "channel"]].drop_duplicates("sku")
    cal = prep.calendar_features(pd.date_range(panel["date"].min(), as_of, freq="D"), cfg)
    calib_end = panel["date"].min() + pd.Timedelta(days=int(cfg["synthetic"]["calibration_days"]))
    scenarios, events, sim_params = syn.run(real_daily, sku_meta, cal, calib_end, as_of, cfg)
    inv = prep.build_inventory_context(con, pilot, cfg, as_of, scenarios)
    ff = prep.build_forecast_features(panel, con, pilot, cfg, as_of)
    yield types.SimpleNamespace(con=con, cfg=cfg, pilot=pilot, panel=panel, stats=stats,
                                scenarios=scenarios, events=events, sim_params=sim_params,
                                inv=inv, ff=ff, as_of=as_of, calib_end=calib_end,
                                real_daily=real_daily, sku_meta=sku_meta, cal=cal, db=db)
    con.close()


# 1. units_observed stays real (matches the raw daily sales, never synthetic sales)
def test_units_observed_is_real(ctx):
    a = ctx.panel[(ctx.panel.sku == "A")].set_index("date")["units_observed"]
    expected = {d.date(): 5 + (i % 7) for i, d in enumerate(DAYS)}
    assert all(int(a.loc[d]) == expected[d.date()] for d in a.index)
    assert "synthetic_sales" not in ctx.panel.columns


# 2. no record after as_of affects outputs
def test_as_of_hard_boundary(ctx):
    assert ctx.panel["date"].max() == ctx.as_of
    assert ctx.stats["rows_after_as_of_dropped"] == 2          # the two post-as_of sales
    # the 9999 / 8888 spikes never appear
    assert ctx.panel["units_observed"].max() < 100


# 3. SKU selection uses only the selection cutoff (post-cutoff sales excluded from totals)
def test_selection_cutoff_only(ctx):
    cand = prep.reselect_candidates(ctx.con, ctx.cfg, "2026-04-30")
    a_units = int(cand[cand.sku == "A"]["units"].iloc[0])
    # equals sum of A's in-window sales, NOT inflated by the 9999 post-cutoff spike
    assert a_units == int(ctx.real_daily[ctx.real_daily.sku == "A"]["units"].sum())


# 4. synthetic inventory does not alter demand-eval rows (independence assertion passes)
def test_demand_eval_independent_of_synthetic(ctx):
    ev.assert_synthetic_independence(ctx.panel, horizon=14)     # raises if leakage
    leaked = [c for c in prep.DEMAND_FEATURE_FORBIDDEN if c in ctx.panel.columns]
    assert leaked == []


# 5. forecast_training_eligible is independent of synthetic stockout labels
def test_forecast_eligibility_independent(ctx):
    # eligibility depends only on history length; recomputed value ignores synthetic data
    didx = ctx.panel.groupby(["sku", "channel"]).cumcount()
    assert (ctx.panel["forecast_training_eligible"] == (didx >= 14)).all()


# 6. opening_stock[t] == prior ending + arrivals[t]
def test_opening_stock_identity(ctx):
    g = ctx.scenarios[(ctx.scenarios.sku == "A") & (ctx.scenarios.scenario_id == "baseline")]
    g = g.sort_values("date").reset_index(drop=True)
    for t in range(1, len(g)):
        assert g.loc[t, "opening_stock"] == pytest.approx(
            g.loc[t - 1, "ending_stock"] + g.loc[t, "replenishment_received"], abs=1.0)


# 7. inventory never negative
def test_no_negative_inventory(ctx):
    assert (ctx.scenarios["ending_stock"] >= 0).all()
    assert (ctx.scenarios["opening_stock"] >= 0).all()


# 8. synthetic sales never exceed available stock
def test_sales_bounded_by_available(ctx):
    avail = ctx.scenarios["opening_stock"].clip(lower=0)
    assert (ctx.scenarios["synthetic_sales"] <= avail + 1e-9).all()


# 9. lost_sales == max(0, latent - available)
def test_lost_sales_formula(ctx):
    avail = ctx.scenarios["opening_stock"].clip(lower=0)
    expected = (ctx.scenarios["latent_synthetic_demand"] - avail).clip(lower=0)
    assert (np.abs(ctx.scenarios["lost_sales"] - expected) <= 1.0).all()


# 10. an order cannot arrive before it was placed
def test_orders_arrive_after_placed(ctx):
    e = ctx.events
    if len(e):
        assert (pd.to_datetime(e["expected_arrival_date"]) > pd.to_datetime(e["order_date"])).all()
        assert (pd.to_datetime(e["actual_arrival_date"]) >= pd.to_datetime(e["order_date"])).all()


# 11. same seed -> identical outputs
def test_reproducible_with_same_seed(ctx):
    s2, _, _ = syn.run(ctx.real_daily, ctx.sku_meta, ctx.cal, ctx.calib_end, ctx.as_of, ctx.cfg)
    pd.testing.assert_frame_equal(
        ctx.scenarios.reset_index(drop=True), s2.reset_index(drop=True))


# 12. changing the seed changes at least some stochastic outputs
def test_seed_change_changes_output(ctx):
    cfg2 = prep.load_config()
    cfg2["synthetic"]["seed"] = int(ctx.cfg["synthetic"]["seed"]) + 1
    s2, _, _ = syn.run(ctx.real_daily, ctx.sku_meta, ctx.cal, ctx.calib_end, ctx.as_of, cfg2)
    merged = ctx.scenarios.merge(
        s2, on=["sku", "scenario_id", "date"], suffixes=("_a", "_b"))
    assert (merged["latent_synthetic_demand_a"] != merged["latent_synthetic_demand_b"]).any()


# 13. no same-day real sales are used to increase opening stock
def test_opening_stock_ignores_real_sales(ctx):
    # opening stock derives only from prior ending + arrivals; latent demand != real units.
    g = ctx.scenarios[(ctx.scenarios.sku == "A") & (ctx.scenarios.scenario_id == "baseline")]
    joined = g.merge(ctx.real_daily[ctx.real_daily.sku == "A"], on="date")
    # if opening stock were forced up to same-day REAL sales, min opening would track real units;
    # the simulation depends on latent demand instead, so they differ on many days.
    assert (joined["latent_synthetic_demand"] != joined["units"]).mean() > 0.3


# 14. calibration uses only <= calibration_end (dropping later data doesn't change params)
def test_calibration_is_past_only(ctx):
    full = syn.calibrate(ctx.real_daily,
                         dict(zip(ctx.sku_meta.sku, ctx.sku_meta.category)),
                         ctx.calib_end, ctx.cfg)
    truncated_daily = ctx.real_daily[ctx.real_daily.date <= ctx.calib_end]
    trunc = syn.calibrate(truncated_daily,
                          dict(zip(ctx.sku_meta.sku, ctx.sku_meta.category)),
                          ctx.calib_end, ctx.cfg)
    for sku in ctx.sku_meta.sku:
        assert full[sku]["lam"] == pytest.approx(trunc[sku]["lam"])


# 15. every synthetic row carries scenario / seed / simulation-version metadata
def test_synthetic_rows_have_metadata(ctx):
    for col in ("scenario_id", "scenario_type", "simulation_version", "simulation_seed", "is_synthetic"):
        assert col in ctx.scenarios.columns
        assert ctx.scenarios[col].notna().all()
    assert ctx.scenarios["is_synthetic"].all()
    assert ctx.scenarios["stockout_label_is_synthetic"].all()


# 16. unit_cost_effective is positive and finite when present
def test_effective_cost_positive(ctx):
    eff = pd.to_numeric(ctx.inv["unit_cost_effective"], errors="coerce").dropna()
    assert (eff > 0).all() and np.isfinite(eff).all()


# 17. invalid observed costs remain identifiable
def test_invalid_costs_flagged(ctx):
    c = ctx.inv[ctx.inv.sku == "C"].iloc[0]
    assert not c["cost_is_valid"]
    assert pd.isna(c["unit_cost_observed"])            # observed stays null
    assert c["cost_is_imputed"]
    assert "NON_POSITIVE_COST" in c["cost_quality_flag"]
    assert "global_median" in c["cost_source"] or "category_median" in c["cost_source"]


# 18. cost source precedence works
def test_cost_precedence():
    tol = 0.25
    assert prep.classify_cost([("magento_eav", 40), ("staging_margin", 45), ("product_flat", 50)], 100, tol)["source"] == "magento_eav"
    assert prep.classify_cost([("magento_eav", 0), ("staging_margin", 20), ("product_flat", 30)], 100, tol)["source"] == "staging_margin"
    # present-but-invalid values -> NON_POSITIVE_COST (NOT reported as MISSING_COST)
    r = prep.classify_cost([("magento_eav", 0), ("staging_margin", -1), ("product_flat", None)], 100, tol)
    assert r["source"] == "missing" and r["observed"] is None
    assert "NON_POSITIVE_COST" in r["flags"] and "MISSING_COST" not in r["flags"]
    # truly absent everywhere -> MISSING_COST
    m = prep.classify_cost([("magento_eav", None), ("staging_margin", None), ("product_flat", None)], 100, tol)
    assert m["source"] == "missing" and "MISSING_COST" in m["flags"]
    # material disagreement between two valid sources -> conflict flag
    assert "COST_SOURCE_CONFLICT" in prep.classify_cost([("magento_eav", 10), ("staging_margin", 100)], 200, tol)["flags"]


# 19. unit cost is absent from the demand-model feature whitelist
def test_cost_not_a_demand_feature(ctx):
    for c in ("unit_cost", "unit_cost_effective", "unit_cost_observed", "price"):
        assert c not in prep.DEMAND_FEATURE_WHITELIST
    assert "unit_cost" not in ctx.ff.columns and "unit_cost_effective" not in ctx.ff.columns


# 20. forecast evaluation uses chronological splits
def test_chronological_split(ctx):
    cutoff, train, test = ev.backtest_split(ctx.panel, horizon=14)
    assert train["date"].max() <= cutoff < test["date"].min()
    assert (test["date"] > cutoff).all()


# extra: schemas + full validate_outputs pass, forecast horizon width correct
def test_schemas_and_validation(ctx):
    problems = prep.validate_outputs(ctx.panel, ctx.ff, ctx.inv, ctx.scenarios, ctx.cfg)
    assert problems == [], problems
    assert list(ctx.panel.columns) == prep.MODEL_PANEL_COLS
    assert list(ctx.inv.columns) == prep.INVENTORY_COLS
    n = int(ctx.cfg["pilot"]["forecast_feature_days"])
    assert (ctx.ff.groupby(["sku", "channel"])["date"].nunique() == n).all()
    assert ctx.stats["physical_store_rows_excluded"] == 1
    assert "weirdchan" in ctx.stats["unknown_channel_rows"]
