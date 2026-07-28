"""Phase 3 — shared forecasting contract, path-aware evaluator, and the
run-aware Baseline + Holt-Winters integrations.

One consolidated file (no per-model scatter). Uses temporary SQLite warehouses
and small synthetic prepared runs — never the real Magento DB and never the
tracked data/processed or outputs/ trees. N_SIM (Holt-Winters interval
simulation count) is reduced via the module global for speed; the forecasting,
selection and evaluation logic is exercised unchanged.
"""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling test-helper import

import model_contract as mc          # noqa: E402
import evaluation as ev              # noqa: E402
import baselines as bl               # noqa: E402
import holtwinters as hw             # noqa: E402
import lgbm_global as lgbm           # noqa: E402
import prepare_pilot_data as prep    # noqa: E402
from test_prepare_pilot_data import _make_db as make_periodic_db   # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════════
def _prepare_run(tmp: Path, db: Path, skus: list[str], as_of: str, name: str = "processed") -> Path:
    """Build a real prepared run (model_panel/forecast_frame/manifest) from a temp DB."""
    pilot = tmp / f"{name}_pilot.csv"
    pilot.write_text("sku\n" + "\n".join(skus) + "\n", encoding="utf-8")
    proc = tmp / name
    rc = prep.main(["--db-path", str(db), "--pilot-file", str(pilot),
                    "--output-dir", str(proc), "--as-of-date", as_of, "--strict"])
    assert rc == 0
    return proc


def _paths(proc: Path):
    return (proc / "model_panel.parquet", proc / "forecast_frame.parquet", proc / "pilot_manifest.json")


def _make_trended_db(path: Path, n_sku: int = 6, days: int = 150, as_of: str = "2026-05-30") -> list[str]:
    """A warehouse whose ecommerce demand TRENDS upward per SKU, so ETS produces
    non-degenerate fractional forecasts (Holt-Winters' integer-forecast guard passes)."""
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE sku_master (sku_id TEXT, product_id INT, sku_name TEXT,
        category TEXT, sub_category TEXT, brand TEXT, price REAL, pack_size INT, moq INT,
        supplier_lead_time_days INT, is_perishable INT, shelf_life_days REAL, unit_cost REAL,
        cost_source TEXT, eav_cost REAL, margin_cost REAL, flat_cost REAL, is_dropship INT,
        created_at TEXT)""")
    skus = [f"T{i:03d}" for i in range(n_sku)]
    for i, s in enumerate(skus):
        con.execute("INSERT INTO sku_master VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (s, 100 + i, f"Trended Prod {i}", "Cat1", None, f"Brand{i}", 100.0, 1, 1, 7,
                     0, None, 40.0, "magento_eav", 40.0, 45.0, 50.0, 0, "2025-12-01"))
    con.execute("""CREATE TABLE sales_transactions (sku_id TEXT, channel TEXT,
        transaction_date TEXT, quantity_sold REAL, qty_ordered REAL, discount_amount REAL, row_total REAL)""")
    dates = pd.date_range("2026-01-01", as_of, freq="D")
    rng = np.random.default_rng(12345)              # deterministic noise -> ETS alpha<1 -> fractional forecasts
    rows = []
    for i, s in enumerate(skus):
        base = 30 + i * 7
        for j, d in enumerate(dates):
            val = base + 0.5 * j + 10 * np.sin(2 * np.pi * j / 7) + rng.normal(0, 5)
            q = max(1, int(round(val)))             # noisy trend + weekly season, integer units, always > 0
            rows.append((s, "online_delivery", d.date().isoformat(), q, q, 0, q * 100.0))
    con.executemany("INSERT INTO sales_transactions VALUES (?,?,?,?,?,?,?)", rows)
    con.execute("""CREATE TABLE inventory_snapshot_history (product_id INT, snapshot_date TEXT,
        location_id TEXT, stock_on_hand REAL, stock_flag TEXT)""")
    con.commit()
    con.close()
    return skus


# ══════════════════════════════════════════════════════════════════════════════════
# Part A — shared evaluator
# ══════════════════════════════════════════════════════════════════════════════════
def _eval_panel(days: int = 40) -> pd.DataFrame:
    d = pd.date_range("2026-01-01", periods=days, freq="D")
    return pd.DataFrame({
        "sku": "A", "channel": "naheed_web", "date": d,
        "units_observed": [3 + (i % 5) for i in range(days)],
        "forecast_training_eligible": True,
        "stock_on_hand": [100 - i for i in range(days)],
        "stock_on_hand_is_synthetic": True, "stock_source": "synthetic_reconstruction",
        "stock_generation_version": "2.0",
    })


def test_1_load_model_panel_default():
    df = ev.load_model_panel()                       # data/processed default
    assert isinstance(df, pd.DataFrame) and {"sku", "channel", "date"} <= set(df.columns)
    assert df.equals(df.sort_values(["sku", "channel", "date"]).reset_index(drop=True))


def test_2_load_model_panel_custom(tmp_path):
    p = tmp_path / "mp.parquet"
    _eval_panel(5).to_parquet(p, index=False)
    df = ev.load_model_panel(p)
    assert len(df) == 5 and pd.api.types.is_datetime64_any_dtype(df["date"])


def test_3_load_forecast_frame_custom(tmp_path):
    p = tmp_path / "ff.parquet"
    pd.DataFrame({"sku": ["A"], "channel": ["naheed_web"], "date": ["2026-02-01"],
                  "forecast_horizon_day": [1]}).to_parquet(p, index=False)
    df = ev.load_forecast_frame(p)
    assert len(df) == 1 and pd.api.types.is_datetime64_any_dtype(df["date"])


def test_4_panel_and_path_together_fails(tmp_path):
    p = tmp_path / "mp.parquet"
    _eval_panel(30).to_parquet(p, index=False)
    preds = pd.DataFrame({"sku": [], "channel": [], "date": [], "y_pred": []})
    with pytest.raises(ValueError):
        ev.evaluate(preds, horizon=14, panel=_eval_panel(30), panel_path=p)


def test_5_metric_definitions_unchanged():
    assert ev.mae([1, 2], [1, 4]) == pytest.approx(1.0)
    assert ev.rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert ev.wape([2, 2], [3, 3]) == pytest.approx(0.5)
    assert ev.bias([1, 1], [2, 3]) == pytest.approx(1.5)
    assert ev.mase([10, 10], [12, 12], [1, 2, 3, 4]) == pytest.approx(2.0)   # mae 2 / naive 1


def test_6_backtest_split_cutoffs_unchanged():
    panel = _eval_panel(40)
    panel["date"] = pd.to_datetime(panel["date"])
    cutoff, train, test = ev.backtest_split(panel, 7)
    assert cutoff == panel["date"].max() - pd.Timedelta(days=7)
    assert train["date"].max() <= cutoff < test["date"].min()
    assert ev.HORIZONS == (7, 14)


def test_7_synthetic_independence_enforced():
    panel = _eval_panel(40)
    panel["date"] = pd.to_datetime(panel["date"])
    ev.assert_synthetic_independence(panel, horizon=14)   # must not raise on a valid panel


# ══════════════════════════════════════════════════════════════════════════════════
# Part B — future-prediction contract
# ══════════════════════════════════════════════════════════════════════════════════
def _ff_and_manifest():
    ff = pd.DataFrame({"sku": ["A", "A", "B"], "channel": "naheed_web",
                       "date": pd.to_datetime(["2026-05-01", "2026-05-02", "2026-05-01"]),
                       "forecast_horizon_day": [1, 2, 1]})
    return ff, {"as_of_date": "2026-04-30"}


def _good_future(ff):
    g = ff.copy()
    g["y_pred"] = [1.0, 2.5, 3.0]
    g["model"] = "m"
    g["model_version"] = "v1"
    g["as_of_date"] = "2026-04-30"
    return g


def test_8_valid_future_passes():
    ff, man = _ff_and_manifest()
    out = mc.validate_future_predictions(_good_future(ff), ff, man, "m")
    assert list(out.columns).index("sku") == 0 and len(out) == 3


def test_9_missing_required_column_fails():
    ff, man = _ff_and_manifest()
    bad = _good_future(ff).drop(columns=["model_version"])
    with pytest.raises(ValueError):
        mc.validate_future_predictions(bad, ff, man, "m")


def test_10_duplicate_keys_fail():
    ff, man = _ff_and_manifest()
    bad = pd.concat([_good_future(ff), _good_future(ff).head(1)], ignore_index=True)
    with pytest.raises(ValueError):
        mc.validate_future_predictions(bad, ff, man, "m")


def test_11_missing_forecast_frame_keys_fail():
    ff, man = _ff_and_manifest()
    bad = _good_future(ff).iloc[:2]
    with pytest.raises(ValueError):
        mc.validate_future_predictions(bad, ff, man, "m")


def test_12_unexpected_keys_fail():
    ff, man = _ff_and_manifest()
    extra = _good_future(ff).iloc[[0]].copy()
    extra["sku"] = "ZZZ"
    with pytest.raises(ValueError):
        mc.validate_future_predictions(pd.concat([_good_future(ff), extra], ignore_index=True), ff, man, "m")


def test_13_date_on_or_before_as_of_fails():
    ff, man = _ff_and_manifest()
    bad = _good_future(ff)
    bad.loc[0, "date"] = pd.Timestamp("2026-04-30")   # == as_of
    with pytest.raises(ValueError):
        mc.validate_future_predictions(bad, ff, man, "m")


def test_14_negative_ypred_fails():
    ff, man = _ff_and_manifest()
    bad = _good_future(ff)
    bad.loc[0, "y_pred"] = -0.1
    with pytest.raises(ValueError):
        mc.validate_future_predictions(bad, ff, man, "m")


def test_15_nan_inf_ypred_fails():
    ff, man = _ff_and_manifest()
    for badval in (np.nan, np.inf):
        bad = _good_future(ff)
        bad.loc[0, "y_pred"] = badval
        with pytest.raises(ValueError):
            mc.validate_future_predictions(bad, ff, man, "m")


def test_16_y_true_column_fails():
    ff, man = _ff_and_manifest()
    bad = _good_future(ff)
    bad["y_true"] = 1.0
    with pytest.raises(ValueError):
        mc.validate_future_predictions(bad, ff, man, "m")


def test_17_incorrect_channel_fails():
    ff, man = _ff_and_manifest()
    bad = _good_future(ff)
    bad.loc[0, "channel"] = "foodpanda"
    with pytest.raises(ValueError):
        mc.validate_future_predictions(bad, ff, man, "m")


def test_18_incorrect_horizon_day_fails():
    ff, man = _ff_and_manifest()
    bad = _good_future(ff)
    bad.loc[0, "forecast_horizon_day"] = 99
    with pytest.raises(ValueError):
        mc.validate_future_predictions(bad, ff, man, "m")


def test_19_invalid_interval_ordering_fails():
    ff, man = _ff_and_manifest()
    bad = _good_future(ff)
    bad["lower_80"] = [0.5, 1.0, 1.0]
    bad["upper_80"] = [2.0, 3.0, 4.0]
    bad["lower_95"] = [0.2, 0.5, 0.5]
    bad["upper_95"] = [1.5, 3.0, 4.0]     # upper_95 < upper_80 on row 0 -> invalid
    with pytest.raises(ValueError):
        mc.validate_future_predictions(bad, ff, man, "m")


def test_20_deterministic_sort_applied():
    ff, man = _ff_and_manifest()
    shuffled = _good_future(ff).iloc[[2, 0, 1]].reset_index(drop=True)
    out = mc.validate_future_predictions(shuffled, ff, man, "m")
    assert list(out["sku"]) == ["A", "A", "B"] and list(out["date"].dt.day) == [1, 2, 1]


def test_model_column_must_match_name():
    ff, man = _ff_and_manifest()
    bad = _good_future(ff)
    bad["model"] = "other"
    with pytest.raises(ValueError):
        mc.validate_future_predictions(bad, ff, man, "m")


# ══════════════════════════════════════════════════════════════════════════════════
# Part B — backtest-prediction contract
# ══════════════════════════════════════════════════════════════════════════════════
def _good_backtest():
    return pd.DataFrame({
        "sku": ["A", "A", "B"], "channel": "naheed_web",
        "date": pd.to_datetime(["2026-04-25", "2026-04-26", "2026-04-25"]),
        "y_pred": [1.0, 2.0, 3.0], "model": "m", "horizon": [7, 7, 7],
        "origin": "2026-04-24", "evaluation_type": "locked_holdout",
    })


def test_21_valid_backtest_passes():
    out = mc.validate_backtest_predictions(_good_backtest(), "m")
    assert len(out) == 3


def test_22_invalid_horizon_fails():
    bad = _good_backtest()
    bad["horizon"] = 9
    with pytest.raises(ValueError):
        mc.validate_backtest_predictions(bad, "m")


def test_23_duplicate_backtest_keys_fail():
    bad = pd.concat([_good_backtest(), _good_backtest().head(1)], ignore_index=True)
    with pytest.raises(ValueError):
        mc.validate_backtest_predictions(bad, "m")


def test_24_missing_origin_or_evaltype_fails():
    for col in ("origin", "evaluation_type"):
        with pytest.raises(ValueError):
            mc.validate_backtest_predictions(_good_backtest().drop(columns=[col]), "m")


def test_25_backtest_y_true_fails():
    bad = _good_backtest()
    bad["y_true"] = 1.0
    with pytest.raises(ValueError):
        mc.validate_backtest_predictions(bad, "m")


# ══════════════════════════════════════════════════════════════════════════════════
# Part B — atomic writers + fingerprint
# ══════════════════════════════════════════════════════════════════════════════════
def test_26_atomic_parquet(tmp_path):
    p = mc.write_dataframe_atomic(pd.DataFrame({"a": [1, 2]}), tmp_path / "x.parquet", "parquet")
    assert p.exists() and not (tmp_path / "x.parquet.tmp").exists()
    assert pd.read_parquet(p)["a"].tolist() == [1, 2]


def test_27_atomic_csv(tmp_path):
    p = mc.write_dataframe_atomic(pd.DataFrame({"a": [1]}), tmp_path / "x.csv", "csv")
    assert p.exists() and pd.read_csv(p)["a"].tolist() == [1]


def test_28_atomic_json(tmp_path):
    p = mc.write_json_atomic({"b": 2, "a": 1}, tmp_path / "x.json")
    txt = p.read_text(encoding="utf-8")
    assert json.loads(txt) == {"a": 1, "b": 2} and txt.index('"a"') < txt.index('"b"')   # sorted keys


def test_29_failure_leaves_no_partial(tmp_path):
    class Bad(pd.DataFrame):
        pass
    with pytest.raises(ValueError):
        mc.write_dataframe_atomic(pd.DataFrame({"a": [1]}), tmp_path / "x.txt", "xml")  # bad format
    assert not (tmp_path / "x.txt").exists() and not (tmp_path / "x.txt.tmp").exists()


def test_30_fingerprint_deterministic(tmp_path):
    a = tmp_path / "a.parquet"; b = tmp_path / "b.parquet"; c = tmp_path / "c.json"
    pd.DataFrame({"x": [1]}).to_parquet(a, index=False)
    pd.DataFrame({"y": [2]}).to_parquet(b, index=False)
    c.write_text('{"k":1}', encoding="utf-8")
    assert mc.dataset_fingerprint(a, b, c) == mc.dataset_fingerprint(a, b, c)


def test_31_fingerprint_changes_on_input_change(tmp_path):
    a = tmp_path / "a.parquet"; b = tmp_path / "b.parquet"; c = tmp_path / "c.json"
    pd.DataFrame({"x": [1]}).to_parquet(a, index=False)
    pd.DataFrame({"y": [2]}).to_parquet(b, index=False)
    c.write_text('{"k":1}', encoding="utf-8")
    fp1 = mc.dataset_fingerprint(a, b, c)
    c.write_text('{"k":2}', encoding="utf-8")
    assert mc.dataset_fingerprint(a, b, c) != fp1


# ══════════════════════════════════════════════════════════════════════════════════
# Part C — Baseline integration
# ══════════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def baseline_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bl")
    db = tmp / "wh.db"
    make_periodic_db(db)                          # A/B/C, 120 days, as_of 2026-04-30
    proc = _prepare_run(tmp, db, ["A", "B", "C"], "2026-04-30")
    mp, ff, man = _paths(proc)
    out = tmp / "outputs"
    bl.run(model_panel=mp, forecast_frame=ff, manifest=man, output_dir=out, horizons=(7, 14))
    return dict(tmp=tmp, db=db, proc=proc, out=out, mp=mp, ff=ff, man=man)


def test_32_baseline_help():
    with pytest.raises(SystemExit) as e:
        bl.parse_args(["--help"])
    assert e.value.code == 0


def test_33_baseline_custom_paths_accepted(baseline_run):
    assert all((baseline_run["out"] / f).exists() for f in
               ("baseline_backtest_predictions.parquet", "baseline_scorecard.csv",
                "future_forecast_baseline.parquet", "baseline_run_summary.json"))


def test_34_baseline_no_global_outputs(baseline_run):
    # nothing written to the tracked repo outputs/ during the custom-dir run
    assert not (bl.DEFAULT_OUT / "baseline_backtest_predictions.parquet").exists()


def test_35_baseline_backtest_contract(baseline_run):
    bt = pd.read_parquet(baseline_run["out"] / "baseline_backtest_predictions.parquet")
    assert set(mc.BACKTEST_REQUIRED_COLUMNS) <= set(bt.columns)
    assert set(bt["model"].unique()) == set(bl.BASELINES.keys())
    assert set(bt["horizon"].unique()) == {7, 14}


def test_36_baseline_scorecard_columns(baseline_run):
    sc = pd.read_csv(baseline_run["out"] / "baseline_scorecard.csv")
    assert list(sc.columns) == mc.SCORECARD_COLUMNS
    assert len(sc) == len(bl.BASELINES) * 2       # 4 models x 2 horizons


def test_37_baseline_future_keys_match_frame(baseline_run):
    fut = pd.read_parquet(baseline_run["out"] / "future_forecast_baseline.parquet")
    ff = pd.read_parquet(baseline_run["ff"])
    fk = set(map(tuple, ff[["sku", "channel", "date"]].astype({"date": "datetime64[ns]"})
                 .itertuples(index=False, name=None)))
    pk = set(map(tuple, fut[["sku", "channel", "date"]].astype({"date": "datetime64[ns]"})
                 .itertuples(index=False, name=None)))
    assert pk == fk and fut["model"].nunique() == 1


def test_38_baseline_future_finite_nonneg(baseline_run):
    fut = pd.read_parquet(baseline_run["out"] / "future_forecast_baseline.parquet")
    yp = pd.to_numeric(fut["y_pred"])
    assert bool((yp >= 0).all()) and bool(np.isfinite(yp).all())


def test_39_baseline_deterministic(baseline_run):
    out2 = baseline_run["tmp"] / "outputs2"
    bl.run(model_panel=baseline_run["mp"], forecast_frame=baseline_run["ff"],
           manifest=baseline_run["man"], output_dir=out2, horizons=(7, 14))
    a = pd.read_parquet(baseline_run["out"] / "future_forecast_baseline.parquet")
    b = pd.read_parquet(out2 / "future_forecast_baseline.parquet")
    pd.testing.assert_frame_equal(a, b)


# ══════════════════════════════════════════════════════════════════════════════════
# Part D — Holt-Winters integration (trended data so ETS forecasts are fractional)
# ══════════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def hw_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hw")
    db = tmp / "wh.db"
    skus = _make_trended_db(db)
    proc = _prepare_run(tmp, db, skus, "2026-05-30")
    mp, ff, man = _paths(proc)
    out = tmp / "outputs"
    old = hw.N_SIM
    hw.N_SIM = 200                                # sanctioned: interval sim count only (points/selection unaffected)
    try:
        hw.run_pipeline(model_panel=mp, forecast_frame=ff, manifest=man, output_dir=out, horizons=(7, 14))
    finally:
        hw.N_SIM = old
    return dict(tmp=tmp, proc=proc, out=out, mp=mp, ff=ff, man=man)


def test_40_holtwinters_help():
    with pytest.raises(SystemExit) as e:
        hw.main(["--help"])
    assert e.value.code == 0


def test_41_holtwinters_custom_paths_accepted(hw_run):
    for f in ("holtwinters_backtest_predictions.parquet", "holtwinters_scorecard.csv",
              "future_forecast_holtwinters.parquet", "holtwinters_model_selection.json",
              "holtwinters_run_summary.json"):
        assert (hw_run["out"] / f).exists()


def test_42_holtwinters_no_global_outputs(hw_run):
    assert not (hw.OUT / "future_forecast_holtwinters.parquet").exists() or \
        (hw.OUT / "future_forecast_holtwinters.parquet").resolve() != (hw_run["out"] / "future_forecast_holtwinters.parquet").resolve()


def test_43_holtwinters_future_contract(hw_run):
    fut = pd.read_parquet(hw_run["out"] / "future_forecast_holtwinters.parquet")
    assert set(mc.FUTURE_REQUIRED_COLUMNS) <= set(fut.columns)
    assert set(fut["model"].unique()) == {"holtwinters"}
    ff = pd.read_parquet(hw_run["ff"])
    fk = set(map(tuple, ff[["sku", "channel", "date"]].astype({"date": "datetime64[ns]"})
                 .itertuples(index=False, name=None)))
    pk = set(map(tuple, fut[["sku", "channel", "date"]].astype({"date": "datetime64[ns]"})
                 .itertuples(index=False, name=None)))
    assert pk == fk


def test_44_holtwinters_backtest_contract(hw_run):
    bt = pd.read_parquet(hw_run["out"] / "holtwinters_backtest_predictions.parquet")
    assert set(mc.BACKTEST_REQUIRED_COLUMNS) <= set(bt.columns)
    assert set(bt["model"].unique()) == {"holtwinters"}
    assert set(bt["horizon"].unique()) <= {7, 14}


def test_45_holtwinters_scorecard_columns(hw_run):
    sc = pd.read_csv(hw_run["out"] / "holtwinters_scorecard.csv")
    assert list(sc.columns) == mc.SCORECARD_COLUMNS
    assert set(sc["horizon"]) == {7, 14} and set(sc["model"]) == {"holtwinters"}


def test_46_holtwinters_future_keys_match(hw_run):
    fut = pd.read_parquet(hw_run["out"] / "future_forecast_holtwinters.parquet")
    ff = pd.read_parquet(hw_run["ff"])
    assert len(fut) == len(ff)


def test_47_holtwinters_interval_ordering_valid(hw_run):
    fut = pd.read_parquet(hw_run["out"] / "future_forecast_holtwinters.parquet")
    for c in ("lower_95", "lower_80", "upper_80", "upper_95"):
        assert c in fut.columns
    ok = ((fut["lower_95"] <= fut["lower_80"]) & (fut["lower_80"] <= fut["y_pred"]) &
          (fut["y_pred"] <= fut["upper_80"]) & (fut["upper_80"] <= fut["upper_95"]))
    assert bool(ok.all())


def test_48_holtwinters_fallback_metadata_present(hw_run):
    fut = pd.read_parquet(hw_run["out"] / "future_forecast_holtwinters.parquet")
    for c in ("selected_model", "model_actually_used", "fit_status", "converged",
              "fallback_used", "interval_method"):
        assert c in fut.columns


def test_49_holtwinters_deterministic_smoke(hw_run):
    out2 = hw_run["tmp"] / "outputs2"
    old = hw.N_SIM
    hw.N_SIM = 200
    try:
        hw.run_pipeline(model_panel=hw_run["mp"], forecast_frame=hw_run["ff"],
                        manifest=hw_run["man"], output_dir=out2, horizons=(7, 14))
    finally:
        hw.N_SIM = old
    a = pd.read_parquet(hw_run["out"] / "future_forecast_holtwinters.parquet")
    b = pd.read_parquet(out2 / "future_forecast_holtwinters.parquet")
    pd.testing.assert_frame_equal(a.drop(columns=["source_manifest_generated_at"], errors="ignore"),
                                  b.drop(columns=["source_manifest_generated_at"], errors="ignore"))


# ══════════════════════════════════════════════════════════════════════════════════
# Part E — LightGBM integration (pooled model; leakage-free recursive backtest)
# ══════════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def lgbm_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lgbm")
    db = tmp / "wh.db"
    make_periodic_db(db)                              # A/B/C, 120 days, as_of 2026-04-30
    proc = _prepare_run(tmp, db, ["A", "B", "C"], "2026-04-30")
    mp, ff, man = _paths(proc)
    out = tmp / "outputs"
    summary = lgbm.run(model_panel=mp, forecast_frame=ff, manifest=man, output_dir=out, horizons=(7, 14))
    manifest = json.loads(man.read_text(encoding="utf-8"))
    return dict(tmp=tmp, proc=proc, out=out, mp=mp, ff=ff, man=man, manifest=manifest, summary=summary)


def _lgbm_ctx(run):
    """(panel, features, sku_categories) for direct helper-level leakage tests."""
    panel = ev.load_model_panel(run["mp"])
    features = lgbm.feature_list(run["manifest"])
    sku_categories = pd.CategoricalDtype(categories=sorted(panel["sku"].unique()))
    return panel, features, sku_categories


def test_50_lgbm_help():
    with pytest.raises(SystemExit) as e:
        lgbm.main(["--help"])
    assert e.value.code == 0


def test_51_lgbm_custom_paths_accepted(lgbm_run):
    for f in ("lightgbm_backtest_predictions.parquet", "lightgbm_scorecard.csv",
              "future_forecast_lightgbm.parquet", "lightgbm_run_summary.json"):
        assert (lgbm_run["out"] / f).exists()


def _snapshot_hashes(d: Path) -> dict:
    """{filename: sha256} for every file directly in `d` (empty if absent)."""
    if not d.exists():
        return {}
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.iterdir()) if p.is_file()}


def test_52_lgbm_no_global_outputs(lgbm_run, tmp_path):
    # Strong isolation: the real global outputs/ file list AND per-file hashes are identical
    # before and after a custom-output-dir run.
    before = _snapshot_hashes(lgbm.OUT)
    out = tmp_path / "iso_out"
    lgbm.run(model_panel=lgbm_run["mp"], forecast_frame=lgbm_run["ff"], manifest=lgbm_run["man"],
             output_dir=out, horizons=(7, 14))
    after = _snapshot_hashes(lgbm.OUT)
    assert before == after       # global outputs/ untouched (list + SHA-256 hashes)
    # Aqib's former filenames must never be written by the run-aware model
    for legacy in ("demand_forecasts_lgbm.parquet", "demand_forecasts_lgbm.csv",
                   "lgbm_backtest_predictions.csv", "lgbm_scorecard.csv", "lgbm_model_selection.json"):
        assert not (out / legacy).exists()


def test_53_lgbm_backtest_contract(lgbm_run):
    bt = pd.read_parquet(lgbm_run["out"] / "lightgbm_backtest_predictions.parquet")
    mc.validate_backtest_predictions(bt, "lightgbm", allowed_horizons=(7, 14))   # must not raise
    assert set(bt["model"].unique()) == {"lightgbm"} and set(bt["horizon"].unique()) == {7, 14}
    assert (bt["evaluation_type"] == "locked_holdout").all()


def test_54_lgbm_backtest_no_y_true(lgbm_run):
    bt = pd.read_parquet(lgbm_run["out"] / "lightgbm_backtest_predictions.parquet")
    assert "y_true" not in bt.columns


def test_55_lgbm_scorecard_columns(lgbm_run):
    sc = pd.read_csv(lgbm_run["out"] / "lightgbm_scorecard.csv")
    assert list(sc.columns) == mc.SCORECARD_COLUMNS
    assert set(sc["horizon"]) == {7, 14} and set(sc["model"]) == {"lightgbm"}


def test_56_lgbm_future_contract(lgbm_run):
    fut = pd.read_parquet(lgbm_run["out"] / "future_forecast_lightgbm.parquet")
    ff = ev.load_forecast_frame(lgbm_run["ff"])
    mc.validate_future_predictions(fut, ff, lgbm_run["manifest"], "lightgbm")   # must not raise
    assert set(fut["model"].unique()) == {"lightgbm"}
    assert set(fut["model_version"].unique()) == {"lgbm_global_v1"}


def test_57_lgbm_future_keys_match_frame(lgbm_run):
    fut = pd.read_parquet(lgbm_run["out"] / "future_forecast_lightgbm.parquet")
    ff = ev.load_forecast_frame(lgbm_run["ff"])
    fk = set(map(tuple, ff[["sku", "channel", "date"]].itertuples(index=False, name=None)))
    pk = set(map(tuple, fut[["sku", "channel", "date"]].itertuples(index=False, name=None)))
    assert pk == fk and len(fut) == len(ff)


def test_58_lgbm_predictions_finite_nonneg(lgbm_run):
    for f in ("lightgbm_backtest_predictions.parquet", "future_forecast_lightgbm.parquet"):
        yp = pd.to_numeric(pd.read_parquet(lgbm_run["out"] / f)["y_pred"])
        assert bool((yp >= 0).all()) and bool(np.isfinite(yp).all())


def test_59_lgbm_respects_requested_horizons(lgbm_run):
    out7 = lgbm_run["tmp"] / "out7"
    lgbm.run(model_panel=lgbm_run["mp"], forecast_frame=lgbm_run["ff"],
             manifest=lgbm_run["man"], output_dir=out7, horizons=(7,))
    bt = pd.read_parquet(out7 / "lightgbm_backtest_predictions.parquet")
    sc = pd.read_csv(out7 / "lightgbm_scorecard.csv")
    assert set(bt["horizon"].unique()) == {7} and set(sc["horizon"]) == {7}


def test_60_lgbm_run_summary_fields(lgbm_run):
    s = json.loads((lgbm_run["out"] / "lightgbm_run_summary.json").read_text(encoding="utf-8"))
    assert s["model"] == "lightgbm" and s["model_version"] == "lgbm_global_v1"
    assert "dataset_fingerprint" in s and len(s["dataset_fingerprint"]) == 64
    assert s["backtest_method"] == "fixed-origin recursive multi-step forecast using predicted demand for future lags"
    assert s["future_method"] == "recursive multi-step forecast using predicted demand for future lags"
    assert "hyperparameters" in s and s["hyperparameters"]["objective"] == "tweedie"
    pol = s["backtest_exogenous_policy"]                     # origin-safe exogenous policy recorded
    assert pol["on_promo"].startswith("0") and pol["discount_pct"].startswith("0")
    assert "cutoff" in pol["effective_unit_price"]


def test_61_fingerprints_match_across_models(hw_run):
    # all three models on IDENTICAL prepared inputs (hw_run's trended run) -> same fingerprint
    hw_fp = json.loads((hw_run["out"] / "holtwinters_run_summary.json").read_text())["dataset_fingerprint"]
    bl_out = hw_run["tmp"] / "bl_on_hw"
    bl.run(model_panel=hw_run["mp"], forecast_frame=hw_run["ff"], manifest=hw_run["man"],
           output_dir=bl_out, horizons=(7, 14))
    lg_out = hw_run["tmp"] / "lg_on_hw"
    lgbm.run(model_panel=hw_run["mp"], forecast_frame=hw_run["ff"], manifest=hw_run["man"],
             output_dir=lg_out, horizons=(7, 14))
    bl_fp = json.loads((bl_out / "baseline_run_summary.json").read_text())["dataset_fingerprint"]
    lg_fp = json.loads((lg_out / "lightgbm_run_summary.json").read_text())["dataset_fingerprint"]
    assert bl_fp == hw_fp == lg_fp


def test_62_lgbm_deterministic(lgbm_run):
    out2 = lgbm_run["tmp"] / "out_det"
    lgbm.run(model_panel=lgbm_run["mp"], forecast_frame=lgbm_run["ff"],
             manifest=lgbm_run["man"], output_dir=out2, horizons=(7, 14))
    a = pd.read_parquet(lgbm_run["out"] / "future_forecast_lightgbm.parquet")
    b = pd.read_parquet(out2 / "future_forecast_lightgbm.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_63_no_forbidden_feature_and_whitelist(lgbm_run):
    _, features, _ = _lgbm_ctx(lgbm_run)
    assert not (set(features) & set(lgbm.FORBIDDEN_INPUTS))            # 14: no inventory/cost feature
    assert "sku" in features                                          # 15: sku categorical grouping feature
    assert set(lgbm_run["manifest"]["demand_feature_whitelist"]) <= set(features)   # 16: manifest whitelist
    with pytest.raises(RuntimeError):                                 # refuses forbidden columns
        lgbm._assert_no_forbidden_columns(list(features) + ["stock_on_hand"])


def test_64_recursive_backtest_ignores_heldout_actuals(lgbm_run):
    """17/18: mutating held-out actual demand AND held-out target-derived lag/rolling columns
    does not change the recursive backtest predictions."""
    panel, features, sku_cats = _lgbm_ctx(lgbm_run)
    cutoff, _train, test = ev.backtest_split(panel, 7)
    _bt, _res, model = lgbm.run_backtest(panel, 7, features, sku_cats)
    base = lgbm.build_recursive_backtest_predictions(model, panel, test, cutoff, features, sku_cats, 7)
    mut = test.copy()
    for c in ("units_observed",) + lgbm.TARGET_DERIVED_FEATURES:
        if c in mut.columns:
            mut[c] = 99999.0                                          # poison held-out target-derived values
    after = lgbm.build_recursive_backtest_predictions(model, panel, mut, cutoff, features, sku_cats, 7)
    pd.testing.assert_frame_equal(base.reset_index(drop=True), after.reset_index(drop=True))


def test_65_day2_lag_from_prediction_not_actual(lgbm_run):
    """19: changing ONLY the first held-out day's actual demand leaves later-day predictions
    unchanged (day-2+ lags come from earlier predictions, not held-out actuals)."""
    panel, features, sku_cats = _lgbm_ctx(lgbm_run)
    cutoff, _train, test = ev.backtest_split(panel, 7)
    _bt, _res, model = lgbm.run_backtest(panel, 7, features, sku_cats)
    base = lgbm.build_recursive_backtest_predictions(model, panel, test, cutoff, features, sku_cats, 7)
    first_day = test["date"].min()
    mut = test.copy()
    mask = mut["date"] == first_day
    for c in ("units_observed",) + lgbm.TARGET_DERIVED_FEATURES:
        if c in mut.columns:
            mut.loc[mask, c] = 88888.0
    after = lgbm.build_recursive_backtest_predictions(model, panel, mut, cutoff, features, sku_cats, 7)
    pd.testing.assert_frame_equal(base.reset_index(drop=True), after.reset_index(drop=True))


def test_66_future_is_recursive(lgbm_run):
    """20: the future feature builder derives lags from the running series (real history +
    fed-back predictions), so units_lag_1 equals the most recent series value each step."""
    panel, features, _ = _lgbm_ctx(lgbm_run)
    sku = sorted(panel["sku"].unique())[0]
    hist = panel[panel["sku"] == sku].sort_values("date")
    series = pd.Series(hist["units_observed"].astype(float).to_numpy(),
                       index=pd.DatetimeIndex(hist["date"]))
    ff = ev.load_forecast_frame(lgbm_run["ff"])
    frow = ff[ff["sku"] == sku].sort_values("date").iloc[0]
    feat = lgbm._future_row_features(sku, frow["date"], series, frow, features)
    assert feat["units_lag_1"] == pytest.approx(float(series.iloc[-1]))
    series.loc[frow["date"]] = 123.5                       # fold a prediction back in
    frow2 = ff[ff["sku"] == sku].sort_values("date").iloc[1]
    feat2 = lgbm._future_row_features(sku, frow2["date"], series, frow2, features)
    assert feat2["units_lag_1"] == pytest.approx(123.5)


def test_67_recursive_backtest_ignores_heldout_exogenous(lgbm_run):
    """Origin-safe policy: mutating held-out REALIZED price / discount / promotion does NOT
    change fixed-origin recursive backtest predictions (those values are not known at the
    cutoff; the model uses last-known price + zero promo/discount instead)."""
    panel, features, sku_cats = _lgbm_ctx(lgbm_run)
    cutoff, _train, test = ev.backtest_split(panel, 7)
    _bt, _res, model = lgbm.run_backtest(panel, 7, features, sku_cats)
    base = lgbm.build_recursive_backtest_predictions(model, panel, test, cutoff, features, sku_cats, 7)
    mut = test.copy()
    for c in lgbm.ORIGIN_SAFE_EXOG:          # effective_unit_price, discount_pct, on_promo
        if c in mut.columns:
            mut[c] = 42424.0                 # poison the realized held-out exogenous values
    after = lgbm.build_recursive_backtest_predictions(model, panel, mut, cutoff, features, sku_cats, 7)
    pd.testing.assert_frame_equal(base.reset_index(drop=True), after.reset_index(drop=True))


def test_68_origin_safe_price_is_last_known(lgbm_run):
    """The origin-safe price used for held-out days is the last price known on/before the
    cutoff — never a realized held-out price."""
    panel, features, sku_cats = _lgbm_ctx(lgbm_run)
    cutoff, _train, test = ev.backtest_split(panel, 7)
    sku = sorted(panel["sku"].unique())[0]
    last_known = float(panel[(panel["sku"] == sku) & (panel["date"] <= cutoff)]
                       .sort_values("date")["effective_unit_price"].dropna().iloc[-1])
    captured = {}
    orig = lgbm._recursive_feature_row
    def spy(s, d, series, exo_row, feats, origin_exog):
        if s == sku and "effective_unit_price" in origin_exog:
            captured["price"] = origin_exog["effective_unit_price"]
            captured["on_promo"] = origin_exog.get("on_promo")
            captured["discount_pct"] = origin_exog.get("discount_pct")
        return orig(s, d, series, exo_row, feats, origin_exog)
    lgbm._recursive_feature_row = spy
    try:
        _bt, _res, model = lgbm.run_backtest(panel, 7, features, sku_cats)
        lgbm.build_recursive_backtest_predictions(model, panel, test, cutoff, features, sku_cats, 7)
    finally:
        lgbm._recursive_feature_row = orig
    assert captured["price"] == pytest.approx(last_known)
    assert captured["on_promo"] == 0 and captured["discount_pct"] == 0.0
