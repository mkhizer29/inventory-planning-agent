"""Phase 4 — run-aware forecasting orchestrator tests.

Real selection + real preparation run against a TEMPORARY sqlite warehouse; the three
model runners are monkeypatched with fast, contract-valid fakes for orchestration-logic
tests (selection/preparation/ranking correctness is never bypassed). One real end-to-end
smoke test runs all three actual models on a small trended run with Holt-Winters'
interval-simulation count reduced and restored.
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_contract as mc                # noqa: E402
import forecast_orchestrator as orch       # noqa: E402
import baselines, holtwinters, lgbm_global  # noqa: E402


# ── temporary warehouse (trended so the real Holt-Winters smoke fits) ─────────────────
def _make_db(path: Path, n_sku: int = 6, as_of: str = "2026-05-30", category: str = "Groceries & Pets"):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE sku_master (sku_id TEXT, product_id INT, sku_name TEXT,
        category TEXT, sub_category TEXT, brand TEXT, price REAL, pack_size INT, moq INT,
        supplier_lead_time_days INT, is_perishable INT, shelf_life_days REAL, unit_cost REAL,
        cost_source TEXT, eav_cost REAL, margin_cost REAL, flat_cost REAL, is_dropship INT,
        created_at TEXT)""")
    skus = [f"T{i:03d}" for i in range(n_sku)]
    for i, s in enumerate(skus):
        con.execute("INSERT INTO sku_master VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (s, 100 + i, f"Prod {i}", category, None, f"Brand{i}", 100.0, 1, 1, 7,
                     0, None, 40.0, "magento_eav", 40.0, 45.0, 50.0, 0, "2025-12-01"))
    con.execute("""CREATE TABLE sales_transactions (sku_id TEXT, channel TEXT,
        transaction_date TEXT, quantity_sold REAL, qty_ordered REAL, discount_amount REAL, row_total REAL)""")
    dates = pd.date_range("2026-01-01", as_of, freq="D")
    rng = np.random.default_rng(7)
    rows = []
    for i, s in enumerate(skus):
        for j, d in enumerate(dates):
            q = max(1, int(round(30 + i * 6 + 0.5 * j + 10 * np.sin(2 * np.pi * j / 7) + rng.normal(0, 5))))
            rows.append((s, "online_delivery", d.date().isoformat(), q, q, 0, q * 100.0))
    con.executemany("INSERT INTO sales_transactions VALUES (?,?,?,?,?,?,?)", rows)
    con.execute("""CREATE TABLE inventory_snapshot_history (product_id INT, snapshot_date TEXT,
        location_id TEXT, stock_on_hand REAL, stock_flag TEXT)""")
    con.commit()
    con.close()
    return skus


CATEGORY = "Groceries & Pets"
AS_OF = "2026-05-30"
BASE = dict(category=CATEGORY, top_n=3, as_of_date=AS_OF, selection_cutoff=AS_OF,
            min_history_days=28, horizons=(7, 14))


# ── fast contract-valid fake model runners ────────────────────────────────────────────
DEFAULT_WAPE = {"last_day_naive": 0.90, "seasonal_naive_7": 0.85, "moving_average_7": 0.80,
                "moving_average_14": 0.75, "holtwinters": 0.70, "lightgbm": 0.60}


def _manifest_of(man_path):
    return json.loads(Path(man_path).read_text(encoding="utf-8"))


def _fake_scorecard_rows(models, panel, horizons, wape):
    n_sku = int(panel["sku"].nunique())
    rows = []
    for m in models:
        for h in horizons:
            cutoff = str((panel["date"].max() - pd.Timedelta(days=int(h))).date())
            w = float(wape[m])
            rows.append({"model": m, "horizon": int(h), "wape": w, "mase": w + 0.1,
                         "mae": w * 10, "rmse": w * 12, "bias": 0.0,
                         "n_rows": n_sku * int(h), "n_skus": n_sku, "n_channels": 1,
                         "cutoff": cutoff, "evaluation_type": LOCKED})
    return rows


LOCKED = "locked_holdout"


def _fake_future(ff, man, model, mv="fake_v1"):
    fut = ff[["sku", "channel", "date"]].copy()
    if "product_id" in ff.columns:
        fut["product_id"] = ff["product_id"].values
    fut["forecast_horizon_day"] = ff["forecast_horizon_day"].values
    fut["y_pred"] = 1.0
    fut["model"] = model
    fut["model_version"] = mv
    fut["as_of_date"] = man["as_of_date"]
    return fut


def _fake_backtest(models, panel, horizons):
    """Contract-valid backtest predictions: one row per (model, horizon, sku); unique keys."""
    rows = []
    skus = sorted(panel["sku"].astype(str).unique())
    d = panel["date"].max()
    for m in models:
        for h in horizons:
            for s in skus:
                rows.append({"sku": s, "channel": "naheed_web", "date": d, "y_pred": 1.0,
                             "model": m, "horizon": int(h), "origin": "2026-01-01",
                             "evaluation_type": LOCKED})
    return pd.DataFrame(rows)


def _write_fake_model(name, model_ids, out, panel, ff, man, horizons, wape, fingerprint,
                      corrupt_kind=None, baseline_future_model=None):
    """Write one model's four/five contract-valid artifacts (optionally corrupted for tests)."""
    arts = orch.MODEL_ARTIFACTS[name]
    sc = pd.DataFrame(_fake_scorecard_rows(model_ids, panel, horizons, wape))[mc.SCORECARD_COLUMNS]
    bt = _fake_backtest(model_ids, panel, horizons)
    if name == "baseline":
        fut_model = baseline_future_model or min(model_ids, key=lambda m: wape[m])
    else:
        fut_model = name
    fut = _fake_future(ff, man, fut_model)
    summary = {"model": ("baselines" if name == "baseline" else name),
               "model_version": f"{name}_v1", "as_of_date": man["as_of_date"],
               "horizons": list(horizons), "dataset_fingerprint": fingerprint}
    if name == "lightgbm":
        summary["n_skus"] = int(panel["sku"].nunique())
    if name == "baseline":
        summary["official_future_model"] = fut_model

    if corrupt_kind == "future_missing_key":
        fut = fut.iloc[:-1]
    elif corrupt_kind == "future_dup_key":
        fut = pd.concat([fut, fut.iloc[[0]]], ignore_index=True)
    elif corrupt_kind == "future_negative":
        fut.loc[fut.index[0], "y_pred"] = -1.0
    elif corrupt_kind == "future_wrong_model":
        fut["model"] = "totally_wrong"
    elif corrupt_kind == "backtest_y_true":
        bt["y_true"] = 1.0
    elif corrupt_kind == "backtest_bad_horizon":
        bt.loc[bt.index[0], "horizon"] = 9
    elif corrupt_kind == "scorecard_bad_columns":
        sc = sc.drop(columns=["wape"])
    elif corrupt_kind == "bad_fingerprint":
        summary["dataset_fingerprint"] = "Z" * 64          # non-hex
    elif corrupt_kind == "bad_n_skus":
        summary["n_skus"] = 999

    mc.write_dataframe_atomic(sc, out / arts["scorecard"], "csv")
    mc.write_dataframe_atomic(bt, out / arts["backtest"], "parquet")
    mc.write_dataframe_atomic(fut, out / arts["future"], "parquet")
    if name == "holtwinters":
        mc.write_json_atomic({"model": "holtwinters"}, out / arts["selection"])
    mc.write_json_atomic(summary, out / arts["summary"])


def install_fakes(monkeypatch, *, wape=None, fail=(), fp_override=None,
                  baseline_future_model=None, corrupt=None):
    """Patch the three model entrypoints with fast fakes that write contract-valid outputs.

    corrupt = (model_name, kind) injects one bad artifact for that model (see _write_fake_model).
    """
    wape = {**DEFAULT_WAPE, **(wape or {})}
    ck = {corrupt[0]: corrupt[1]} if corrupt else {}

    def fp(mp, ff, man_path):
        return fp_override if fp_override else mc.dataset_fingerprint(mp, ff, man_path)

    def _mk(name, ids):
        def _runner(*, model_panel, forecast_frame, manifest, output_dir, horizons):
            if name in fail:
                raise RuntimeError(f"fake {name} failure")
            out = Path(output_dir)
            panel = pd.read_parquet(model_panel); ff = pd.read_parquet(forecast_frame)
            man = _manifest_of(manifest)
            _write_fake_model(name, ids, out, panel, ff, man, horizons, wape,
                              fp(model_panel, forecast_frame, manifest),
                              corrupt_kind=ck.get(name), baseline_future_model=baseline_future_model)
        return _runner

    monkeypatch.setattr(baselines, "run", _mk("baseline", list(orch.BASELINE_METHODS)))
    monkeypatch.setattr(holtwinters, "run_pipeline", _mk("holtwinters", ["holtwinters"]))
    monkeypatch.setattr(lgbm_global, "run", _mk("lightgbm", ["lightgbm"]))


def _run(tmp, monkeypatch, **overrides):
    db = tmp / "wh.db"
    if not db.exists():
        _make_db(db)
    kw = {**BASE, "runs_dir": tmp / "runs", "db_path": db}
    kw.update(overrides)
    return orch.run_forecast_pipeline(**kw)


# ══════════════════════════════════════════════════════════════════════════════════
# request validation / CLI (no model runs needed)
# ══════════════════════════════════════════════════════════════════════════════════
def test_01_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        orch.parse_args(["--help"])
    assert e.value.code == 0


def test_06_blank_category_rejected(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    with pytest.raises(orch.RequestError):
        _run(tmp_path, monkeypatch, category="   ")


def test_07_invalid_top_n_rejected(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    for bad in (0, -1, 101, True):
        with pytest.raises(orch.RequestError):
            _run(tmp_path, monkeypatch, top_n=bad)


def test_08_invalid_date_rejected(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    with pytest.raises(orch.RequestError):
        _run(tmp_path, monkeypatch, as_of_date="2026-13-40")


def test_09_cutoff_after_asof_rejected(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    with pytest.raises(orch.RequestError):
        _run(tmp_path, monkeypatch, selection_cutoff="2026-07-01")


def test_10_invalid_horizons_rejected(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    with pytest.raises(orch.RequestError):
        _run(tmp_path, monkeypatch, horizons=(3,))


def test_11_invalid_skip_model_rejected(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    with pytest.raises(orch.RequestError):
        _run(tmp_path, monkeypatch, skip_models=("bogus",))


def test_12_missing_database_rejected(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    with pytest.raises(orch.RequestError):
        orch.run_forecast_pipeline(**BASE, runs_dir=tmp_path / "runs", db_path=tmp_path / "nope.db")


def test_13_path_traversal_run_id_rejected(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    for bad in ("../evil", "a/b", "a\\b", ".."):
        with pytest.raises(orch.RequestError):
            _run(tmp_path, monkeypatch, run_id=bad)


def test_39_no_model_requested_fails_cleanly(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    with pytest.raises(orch.RequestError):
        _run(tmp_path, monkeypatch, skip_models=("baseline", "holtwinters", "lightgbm"))


def test_43_validation_failure_creates_no_run_dir(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    with pytest.raises(orch.RequestError):
        _run(tmp_path, monkeypatch, top_n=0)
    runs = tmp_path / "runs"
    assert not runs.exists() or not any(runs.iterdir())


# ══════════════════════════════════════════════════════════════════════════════════
# happy path (fakes)
# ══════════════════════════════════════════════════════════════════════════════════
@pytest.fixture()
def good(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    m = _run(tmp_path, monkeypatch, run_id="run_good")
    return dict(tmp=tmp_path, manifest=m, run_dir=tmp_path / "runs" / "run_good")


def test_02_15_valid_request_creates_tree_and_completes(good):
    rd = good["run_dir"]
    for f in ("request.json", "selected_skus.csv", "status.json", "pipeline.log",
              "combined_scorecard.csv", "model_ranking.csv", "selected_forecasts.parquet",
              "run_manifest.json"):
        assert (rd / f).exists(), f
    assert (rd / "processed").is_dir() and (rd / "outputs").is_dir()
    assert good["manifest"]["status"] == "completed"
    status = json.loads((rd / "status.json").read_text())
    assert status["status"] == "completed" and status["progress_pct"] == 100


def test_03_explicit_run_id(good):
    assert good["manifest"]["run_id"] == "run_good"


def test_04_duplicate_run_id_rejected(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    _run(tmp_path, monkeypatch, run_id="dup")
    with pytest.raises(orch.RequestError):
        _run(tmp_path, monkeypatch, run_id="dup")


def test_05_generated_run_id_is_path_safe(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    m = _run(tmp_path, monkeypatch)
    assert orch.RUN_ID_RE.match(m["run_id"]) and "top3" in m["run_id"]


def test_14_request_json_normalized(good):
    r = json.loads((good["run_dir"] / "request.json").read_text())
    assert r["category"] == CATEGORY and r["horizons"] == [7, 14]
    assert r["selection_cutoff"] == AS_OF and r["requested_models"] == ["baseline", "holtwinters", "lightgbm"]


def test_16_selected_sku_file_unique_ranked(good):
    df = pd.read_csv(good["run_dir"] / "selected_skus.csv")
    assert list(df["rank"]) == list(range(1, len(df) + 1))
    assert df["sku"].is_unique and len(df) == 3


def test_18_19_preparation_isolated_and_matches_selection(good):
    proc = good["run_dir"] / "processed"
    for f in ("model_panel.parquet", "forecast_frame.parquet", "inventory_context.parquet", "pilot_manifest.json"):
        assert (proc / f).exists()
    man = json.loads((proc / "pilot_manifest.json").read_text())
    sel = set(pd.read_csv(good["run_dir"] / "selected_skus.csv")["sku"].astype(str))
    assert set(map(str, man["selected_skus"])) == sel and man["validation_status"] == "passed"


def test_21_model_outputs_under_run(good):
    outs = good["run_dir"] / "outputs"
    for f in ("baseline_scorecard.csv", "holtwinters_scorecard.csv", "lightgbm_scorecard.csv",
              "future_forecast_baseline.parquet", "future_forecast_holtwinters.parquet",
              "future_forecast_lightgbm.parquet"):
        assert (outs / f).exists()


def test_22_fingerprints_match(good):
    assert good["manifest"]["dataset_fingerprint"] and len(good["manifest"]["dataset_fingerprint"]) == 64


def test_23_combined_scorecard_schema(good):
    sc = pd.read_csv(good["run_dir"] / "combined_scorecard.csv")
    assert list(sc.columns) == mc.SCORECARD_COLUMNS
    assert set(sc["model"]) == set(orch.BASELINE_METHODS) | {"holtwinters", "lightgbm"}


def test_20_all_models_same_inputs(tmp_path, monkeypatch):
    seen = {}
    orig = orch._run_model

    def spy(name, proc, outputs, horizons, logger):
        seen.setdefault(name, (str(proc), str(outputs), tuple(horizons)))
        return orig(name, proc, outputs, horizons, logger)
    install_fakes(monkeypatch)
    monkeypatch.setattr(orch, "_run_model", spy)
    _run(tmp_path, monkeypatch, run_id="inp")
    assert len(set(seen.values())) == 1 and set(seen) == {"baseline", "holtwinters", "lightgbm"}


def test_40_skipped_models_recorded(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    m = _run(tmp_path, monkeypatch, run_id="skip", skip_models=("lightgbm",))
    assert m["skipped_models"] == ["lightgbm"] and set(m["completed_models"]) == {"baseline", "holtwinters"}


def test_44_custom_runs_dir_respected(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    custom = tmp_path / "my_runs"
    _run(tmp_path, monkeypatch, run_id="c1", runs_dir=custom)
    assert (custom / "c1" / "run_manifest.json").exists()


# ══════════════════════════════════════════════════════════════════════════════════
# ranking / operational selection
# ══════════════════════════════════════════════════════════════════════════════════
def test_26_28_29_ranking_and_operational(good):
    rk = pd.read_csv(good["run_dir"] / "model_ranking.csv")
    assert list(rk.columns) == ["horizon", "rank", "model", "wape", "mase", "mae", "rmse", "bias", "selection_reason"]
    for h in (7, 14):
        sub = rk[rk["horizon"] == h].sort_values("rank")
        assert sub.iloc[0]["model"] == "lightgbm"            # lowest WAPE 0.60
        assert list(sub["wape"]) == sorted(sub["wape"])      # ascending
    assert good["manifest"]["operational_horizon"] == 14
    assert good["manifest"]["operational_model"] == "lightgbm"
    assert good["manifest"]["winners_by_horizon"] == {"7": "lightgbm", "14": "lightgbm"}


def test_26b_ranking_tiebreak_order():
    # equal WAPE -> MASE -> MAE -> |bias| -> name
    rows = [
        {"model": "b", "horizon": 7, "wape": 0.5, "mase": 0.9, "mae": 5, "rmse": 6, "bias": -1, "n_rows": 1, "n_skus": 1, "n_channels": 1, "cutoff": "x", "evaluation_type": LOCKED},
        {"model": "a", "horizon": 7, "wape": 0.5, "mase": 0.9, "mae": 5, "rmse": 6, "bias": 1, "n_rows": 1, "n_skus": 1, "n_channels": 1, "cutoff": "x", "evaluation_type": LOCKED},
        {"model": "c", "horizon": 7, "wape": 0.5, "mase": 0.8, "mae": 5, "rmse": 6, "bias": 0, "n_rows": 1, "n_skus": 1, "n_channels": 1, "cutoff": "x", "evaluation_type": LOCKED},
    ]
    rk = orch.rank_models(pd.DataFrame(rows), (7,))
    assert list(rk["model"]) == ["c", "a", "b"]              # c best MASE; a<b by |bias| tie then name


def test_27_ranking_ignores_non_locked_rows():
    rows = [
        {"model": "lightgbm", "horizon": 7, "wape": 0.9, "mase": 1, "mae": 1, "rmse": 1, "bias": 0, "n_rows": 1, "n_skus": 1, "n_channels": 1, "cutoff": "x", "evaluation_type": LOCKED},
        {"model": "seasonal_naive_7", "horizon": 7, "wape": 0.1, "mase": 1, "mae": 1, "rmse": 1, "bias": 0, "n_rows": 1, "n_skus": 1, "n_channels": 1, "cutoff": "x", "evaluation_type": "seasonal_naive_locked"},
    ]
    rk = orch.rank_models(pd.DataFrame(rows), (7,))
    assert list(rk["model"]) == ["lightgbm"]                 # diagnostic row excluded despite lower WAPE


def test_49_ranking_deterministic():
    rows = _fake_scorecard_rows(list(orch.BASELINE_METHODS) + ["holtwinters", "lightgbm"],
                                pd.DataFrame({"sku": ["A", "B"], "date": pd.to_datetime(["2026-01-01", "2026-01-02"])}),
                                (7, 14), DEFAULT_WAPE)
    a = orch.rank_models(pd.DataFrame(rows), (7, 14))
    b = orch.rank_models(pd.DataFrame(rows), (7, 14))
    pd.testing.assert_frame_equal(a, b)


def test_30_31_selected_forecast_keys_and_baseline_winner(tmp_path, monkeypatch):
    # make a baseline method the overall winner for horizon 14
    install_fakes(monkeypatch, wape={"moving_average_14": 0.30, "holtwinters": 0.70, "lightgbm": 0.65})
    m = _run(tmp_path, monkeypatch, run_id="blwin")
    assert m["operational_model"] == "moving_average_14"
    sel = pd.read_parquet(tmp_path / "runs" / "blwin" / "selected_forecasts.parquet")
    ff = pd.read_parquet(tmp_path / "runs" / "blwin" / "processed" / "forecast_frame.parquet")
    key = ["sku", "channel", "date"]
    sel["date"] = pd.to_datetime(sel["date"]); ff["date"] = pd.to_datetime(ff["date"])
    assert set(map(tuple, sel[key].itertuples(index=False, name=None))) == \
        set(map(tuple, ff[key].itertuples(index=False, name=None)))
    assert set(sel["model"].unique()) == {"moving_average_14"}
    assert set(sel["selection_horizon"]) == {14} and set(sel["selection_rank"]) == {1}


def test_31b_baseline_future_mismatch_fails(tmp_path, monkeypatch):
    # baseline is the winner but its future file carries the WRONG method -> fail
    install_fakes(monkeypatch, wape={"moving_average_14": 0.30, "holtwinters": 0.70, "lightgbm": 0.65},
                  baseline_future_model="last_day_naive")
    m = _run(tmp_path, monkeypatch, run_id="blbad")
    assert m["status"] == "failed"


# ══════════════════════════════════════════════════════════════════════════════════
# comparability / fingerprint failures
# ══════════════════════════════════════════════════════════════════════════════════
# These test the PRODUCTION helper orch.validate_combined_scorecard directly.
def _combined(models, horizons=(7, 14), wape=None):
    panel = pd.DataFrame({"sku": ["A", "B"], "date": pd.to_datetime(["2026-01-01", "2026-01-02"])})
    rows = _fake_scorecard_rows(models, panel, horizons, {**DEFAULT_WAPE, **(wape or {})})
    return pd.DataFrame(rows)[mc.SCORECARD_COLUMNS]


def test_23b_valid_combined_passes():
    orch.validate_combined_scorecard(_combined(list(orch.BASELINE_METHODS) + ["holtwinters", "lightgbm"]), (7, 14))


def test_24_mismatched_cutoff_rejected():
    c = _combined(["holtwinters", "lightgbm"], (7,))
    c.loc[0, "cutoff"] = "2020-01-01"
    with pytest.raises(RuntimeError):
        orch.validate_combined_scorecard(c, (7,))


def test_25_mismatched_n_rows_rejected():
    c = _combined(["holtwinters", "lightgbm"], (7,))
    c.loc[0, "n_rows"] = 999
    with pytest.raises(RuntimeError):
        orch.validate_combined_scorecard(c, (7,))


def test_25b_mismatched_n_skus_rejected():
    c = _combined(["holtwinters", "lightgbm"], (7,))
    c.loc[0, "n_skus"] = 999
    with pytest.raises(RuntimeError):
        orch.validate_combined_scorecard(c, (7,))


def test_25c_mismatched_n_channels_rejected():
    c = _combined(["holtwinters", "lightgbm"], (7,))
    c.loc[0, "n_channels"] = 5
    with pytest.raises(RuntimeError):
        orch.validate_combined_scorecard(c, (7,))


def test_25d_duplicate_rows_rejected():
    c = _combined(["holtwinters"], (7,))
    c = pd.concat([c, c], ignore_index=True)     # duplicate model/horizon/evaluation_type
    with pytest.raises(RuntimeError):
        orch.validate_combined_scorecard(c, (7,))


def test_25e_unrequested_horizon_rejected():
    c = _combined(["holtwinters", "lightgbm"], (7, 14))
    with pytest.raises(RuntimeError):
        orch.validate_combined_scorecard(c, (7,))     # 14 present but only 7 requested


def test_25f_missing_requested_horizon_rejected():
    c = _combined(["holtwinters", "lightgbm"], (7,))
    with pytest.raises(RuntimeError):
        orch.validate_combined_scorecard(c, (7, 14))  # no candidates for horizon 14


def test_38_fingerprint_mismatch_fails_even_partial(tmp_path, monkeypatch):
    # lightgbm writes valid outputs but a DIFFERENT valid-hex fingerprint
    install_fakes(monkeypatch)

    def diff_lgbm(*, model_panel, forecast_frame, manifest, output_dir, horizons):
        out = Path(output_dir); panel = pd.read_parquet(model_panel); ff = pd.read_parquet(forecast_frame)
        man = _manifest_of(manifest)
        _write_fake_model("lightgbm", ["lightgbm"], out, panel, ff, man, horizons, DEFAULT_WAPE,
                          fingerprint="a" * 64)          # valid hex, but differs from baseline/hw
    monkeypatch.setattr(lgbm_global, "run", diff_lgbm)
    m = _run(tmp_path, monkeypatch, run_id="fpmix", allow_partial_success=True)
    assert m["status"] == "failed"


# ══════════════════════════════════════════════════════════════════════════════════
# failure / partial-success
# ══════════════════════════════════════════════════════════════════════════════════
def test_32_33_34_failed_model_default_fails(tmp_path, monkeypatch):
    install_fakes(monkeypatch, fail=("holtwinters",))
    m = _run(tmp_path, monkeypatch, run_id="fail1")
    assert m["status"] == "failed"
    rd = tmp_path / "runs" / "fail1"
    status = json.loads((rd / "status.json").read_text())
    assert status["model_status"]["holtwinters"]["status"] == "failed"
    assert status["model_status"]["holtwinters"]["error"]
    assert m["failed_models"] == ["holtwinters"]
    assert (rd / "pipeline.log").exists()                       # logs preserved
    assert (rd / "outputs" / "baseline_scorecard.csv").exists()  # partial artifact preserved
    assert not (rd / "model_ranking.csv").exists()               # no ranking on failure
    assert not (rd / "selected_forecasts.parquet").exists()


def test_35_36_partial_success(tmp_path, monkeypatch):
    install_fakes(monkeypatch, fail=("holtwinters",))
    m = _run(tmp_path, monkeypatch, run_id="partial", allow_partial_success=True)
    assert m["status"] == "completed_with_warnings"
    assert m["failed_models"] == ["holtwinters"] and set(m["completed_models"]) == {"baseline", "lightgbm"}
    rk = pd.read_csv(tmp_path / "runs" / "partial" / "model_ranking.csv")
    assert "holtwinters" not in set(rk["model"])                 # excluded from ranking


def test_37_all_model_failure_stays_failed(tmp_path, monkeypatch):
    install_fakes(monkeypatch, fail=("baseline", "holtwinters", "lightgbm"))
    m = _run(tmp_path, monkeypatch, run_id="allfail", allow_partial_success=True)
    assert m["status"] == "failed" and m["completed_models"] == []


# ══════════════════════════════════════════════════════════════════════════════════
# artifacts / atomicity / isolation
# ══════════════════════════════════════════════════════════════════════════════════
def test_41_status_writes_atomic_no_tmp(good):
    assert not list(good["run_dir"].glob("*.tmp")) and not list((good["run_dir"] / "outputs").glob("*.tmp"))


def test_42_no_temp_files_left(good):
    assert not list(good["run_dir"].rglob("*.tmp"))


def test_48_manifest_artifact_hashes_match(good):
    m = good["manifest"]
    for art in m["artifact_inventory"]:
        p = good["run_dir"] / art["path"]
        assert p.exists() and orch._sha256(p) == art["sha256"] and p.stat().st_size == art["size_bytes"]
    names = {a["path"] for a in m["artifact_inventory"]}
    assert "pipeline.log" not in names and "run_manifest.json" not in names and "status.json" not in names


def test_50_pipeline_log_has_steps(good):
    log = (good["run_dir"] / "pipeline.log").read_text(encoding="utf-8")
    for token in ("selecting_skus", "preparing_data", "running_baseline", "ranking_models", "final status"):
        assert token in log


def test_45_46_47_tracked_paths_unchanged(tmp_path, monkeypatch):
    import hashlib

    def snap(d):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in d.iterdir() if p.is_file()} if d.exists() else {}
    proc_before = snap(REPO_ROOT / "data" / "processed")
    out_before = snap(REPO_ROOT / "outputs")
    db = REPO_ROOT / "inventory_etl" / "output" / "inventory.db"
    db_before = hashlib.sha256(db.read_bytes()).hexdigest() if db.exists() else None
    install_fakes(monkeypatch)
    _run(tmp_path, monkeypatch, run_id="iso")
    assert snap(REPO_ROOT / "data" / "processed") == proc_before
    assert snap(REPO_ROOT / "outputs") == out_before
    assert (hashlib.sha256(db.read_bytes()).hexdigest() if db.exists() else None) == db_before


def test_17_fewer_eligible_warns(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    # only 6 SKUs exist; request 50 -> selects <=6 with a warning, not a failure
    m = _run(tmp_path, monkeypatch, run_id="fewer", top_n=50)
    assert m["status"] in ("completed", "completed_with_warnings")
    assert m["selected_sku_count"] < 50 and m["selection_warning"]


# ══════════════════════════════════════════════════════════════════════════════════
# ONE real end-to-end smoke — all three actual models (Holt-Winters N_SIM reduced)
# ══════════════════════════════════════════════════════════════════════════════════
def test_real_end_to_end_top3(tmp_path):
    db = tmp_path / "wh.db"
    _make_db(db)
    old = holtwinters.N_SIM
    holtwinters.N_SIM = 200
    try:
        m = orch.run_forecast_pipeline(category=CATEGORY, top_n=3, as_of_date=AS_OF,
                                       selection_cutoff=AS_OF, min_history_days=28, horizons=(7, 14),
                                       runs_dir=tmp_path / "runs", run_id="real_top3", db_path=db)
    finally:
        holtwinters.N_SIM = old
    assert m["status"] == "completed", m.get("errors")
    assert set(m["completed_models"]) == {"baseline", "holtwinters", "lightgbm"}
    assert m["selected_sku_count"] == 3
    assert m["dataset_fingerprint"] and len(m["dataset_fingerprint"]) == 64
    assert m["operational_horizon"] == 14 and m["operational_model"] in \
        set(orch.BASELINE_METHODS) | {"holtwinters", "lightgbm"}
    rd = tmp_path / "runs" / "real_top3"
    sel = pd.read_parquet(rd / "selected_forecasts.parquet")
    ff = pd.read_parquet(rd / "processed" / "forecast_frame.parquet")
    assert len(sel) == 42 == len(ff)                              # 3 SKUs x 14 days
    sel["date"] = pd.to_datetime(sel["date"]); ff["date"] = pd.to_datetime(ff["date"])
    key = ["sku", "channel", "date"]
    assert set(map(tuple, sel[key].itertuples(index=False, name=None))) == \
        set(map(tuple, ff[key].itertuples(index=False, name=None)))
    rk = pd.read_csv(rd / "model_ranking.csv")
    assert set(rk["horizon"]) == {7, 14}


# ══════════════════════════════════════════════════════════════════════════════════
# model-output corruption — every case must FAIL before ranking (default mode)
# ══════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("kind", [
    "future_missing_key", "future_dup_key", "future_negative", "future_wrong_model",
    "backtest_y_true", "backtest_bad_horizon", "scorecard_bad_columns",
    "bad_fingerprint", "bad_n_skus",
])
def test_model_output_corruption_fails_before_ranking(tmp_path, monkeypatch, kind):
    install_fakes(monkeypatch, corrupt=("lightgbm", kind))
    m = _run(tmp_path, monkeypatch, run_id=f"corrupt_{kind}")
    assert m["status"] == "failed", kind
    rd = tmp_path / "runs" / f"corrupt_{kind}"
    assert not (rd / "model_ranking.csv").exists()            # never ranked
    assert not (rd / "selected_forecasts.parquet").exists()   # never selected


# ══════════════════════════════════════════════════════════════════════════════════
# status lifecycle + success/failure timestamp semantics
# ══════════════════════════════════════════════════════════════════════════════════
def test_status_lifecycle_states_are_real(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    seen = []
    orig = orch._write_status

    def spy(run_dir, status):
        seen.append(status["status"])
        return orig(run_dir, status)
    monkeypatch.setattr(orch, "_write_status", spy)
    _run(tmp_path, monkeypatch, run_id="lifecycle")
    for s in ("selecting_skus", "preparing_data", "running_baseline", "running_holtwinters",
              "running_lightgbm", "validating_outputs", "ranking_models", "completed"):
        assert s in seen, s
    assert "running" not in seen           # the generic placeholder is never used


def test_success_timestamps(good):
    st = json.loads((good["run_dir"] / "status.json").read_text())
    assert st["completed_at"] and st["failed_at"] is None
    m = good["manifest"]
    assert m["completed_at"] and m["failed_at"] is None


def test_failure_timestamps(tmp_path, monkeypatch):
    install_fakes(monkeypatch, fail=("holtwinters",))
    m = _run(tmp_path, monkeypatch, run_id="ftime")
    assert m["status"] == "failed"
    st = json.loads((tmp_path / "runs" / "ftime" / "status.json").read_text())
    assert st["failed_at"] and st["completed_at"] is None
    assert m["failed_at"] and m["completed_at"] is None


def test_independent_output_validation_catches_bad_future(tmp_path, monkeypatch):
    # even though the file EXISTS, a contract-invalid future forecast fails the run
    install_fakes(monkeypatch, corrupt=("holtwinters", "future_negative"))
    m = _run(tmp_path, monkeypatch, run_id="badhwfut")
    assert m["status"] == "failed"
    assert any("holtwinters" in e for e in m["errors"])


# ══════════════════════════════════════════════════════════════════════════════════
# Phase B — forecast-driven stockout risk
# ══════════════════════════════════════════════════════════════════════════════════
def test_phase_b_writes_artifacts_and_manifest(good):
    rd, m = good["run_dir"], good["manifest"]
    assert (rd / "decisions" / "stockout_risk.parquet").exists()
    assert (rd / "decisions" / "stockout_trajectory.parquet").exists()
    assert m["status"] == "completed"
    assert m["decisioning_status"] == "completed"
    assert m["stockout_risk_file"] == "decisions/stockout_risk.parquet"
    assert m["stockout_trajectory_file"] == "decisions/stockout_trajectory.parquet"
    assert m["stockout_validation_summary"]["risk_rows"] >= 1
    risk = pd.read_parquet(rd / "decisions" / "stockout_risk.parquet")
    assert "y_true" not in risk.columns and "units_observed" not in risk.columns
    sel = pd.read_parquet(rd / "selected_forecasts.parquet")
    assert len(risk) == sel[["sku", "channel"]].drop_duplicates().shape[0]      # one row per selected key
    inv_paths = {a["path"] for a in m["artifact_inventory"]}
    assert {"decisions/stockout_risk.parquet", "decisions/stockout_trajectory.parquet"} <= inv_paths


def test_phase_b_failure_fails_whole_run_even_partial(tmp_path, monkeypatch):
    install_fakes(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("phase b boom")

    monkeypatch.setattr(orch.stockout_risk, "compute_stockout_risk", _boom)
    # allow_partial_success tolerates a MODEL failure, but NEVER a Phase B failure
    m = _run(tmp_path, monkeypatch, run_id="pbfail", allow_partial_success=True)
    assert m["status"] == "failed"
    assert any("stockout" in e.lower() for e in m["errors"])
    status = json.loads((tmp_path / "runs" / "pbfail" / "status.json").read_text())
    assert status["status"] == "failed"
    assert m.get("decisioning_status") in (None, "failed")


# ══════════════════════════════════════════════════════════════════════════════════
# Phase C — forecast-driven reorder recommendations
# ══════════════════════════════════════════════════════════════════════════════════
# 32 — Phase C runs only AFTER Phase B: if Phase B fails, no Phase C artifacts are written
def test_c_runs_only_after_phase_b(tmp_path, monkeypatch):
    install_fakes(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("phase b boom")

    monkeypatch.setattr(orch.stockout_risk, "compute_stockout_risk", _boom)
    m = _run(tmp_path, monkeypatch, run_id="cnob")
    assert m["status"] == "failed"
    rd = tmp_path / "runs" / "cnob"
    assert not (rd / "decisions" / "reorder_recommendations.parquet").exists()
    assert not (rd / "decisions" / "reorder_summary.json").exists()


# 33 — a Phase C failure fails the whole run
def test_c_failure_fails_run(tmp_path, monkeypatch):
    install_fakes(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("phase c boom")

    monkeypatch.setattr(orch.reorder_recommendations, "compute_reorder_recommendations", _boom)
    m = _run(tmp_path, monkeypatch, run_id="cfail")
    assert m["status"] == "failed"
    assert any("reorder" in e.lower() for e in m["errors"])
    status = json.loads((tmp_path / "runs" / "cfail" / "status.json").read_text())
    assert status["status"] == "failed"


# 34 — allow_partial_success does NOT hide a Phase C failure
def test_c_failure_not_hidden_by_partial(tmp_path, monkeypatch):
    install_fakes(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("phase c boom")

    monkeypatch.setattr(orch.reorder_recommendations, "compute_reorder_recommendations", _boom)
    m = _run(tmp_path, monkeypatch, run_id="cpart", allow_partial_success=True)
    assert m["status"] == "failed"
    assert any("reorder" in e.lower() for e in m["errors"])


# 35 — a completed run contains valid Phase C artifacts + manifest fields
def test_c_completed_run_has_artifacts(good):
    rd, m = good["run_dir"], good["manifest"]
    assert m["status"] == "completed"
    assert (rd / "decisions" / "reorder_recommendations.parquet").exists()
    assert (rd / "decisions" / "reorder_summary.json").exists()
    assert m["reorder_recommendations_file"] == "decisions/reorder_recommendations.parquet"
    assert m["reorder_summary_file"] == "decisions/reorder_summary.json"
    assert m["reorder_policy_version"] == orch.dc.REORDER_POLICY_VERSION
    assert m["stockout_policy_version"] == orch.dc.STOCKOUT_POLICY_VERSION
    assert m["reorder_validation_summary"]["count_by_action"] is not None
    reco = pd.read_parquet(rd / "decisions" / "reorder_recommendations.parquet")
    assert list(reco.columns) == orch.dc.REORDER_RECOMMENDATION_COLUMNS
    assert not reco["order_placed"].any()
    sel = pd.read_parquet(rd / "selected_forecasts.parquet")
    assert len(reco) == sel[["sku", "channel"]].drop_duplicates().shape[0]


# 36 — Phase C artifact hashes in the manifest match the files on disk
def test_c_artifact_hashes_match(good):
    import hashlib
    rd, m = good["run_dir"], good["manifest"]
    inv = {a["path"]: a for a in m["artifact_inventory"]}
    for rel in ("decisions/reorder_recommendations.parquet", "decisions/reorder_summary.json"):
        assert rel in inv, rel
        actual = hashlib.sha256((rd / rel).read_bytes()).hexdigest()
        assert inv[rel]["sha256"] == actual
        assert inv[rel]["size_bytes"] == (rd / rel).stat().st_size


# 37 — manifest artifact paths are run-relative, never absolute
def test_c_manifest_paths_relative(good):
    m = good["manifest"]
    assert not any(ch in m["reorder_recommendations_file"] for ch in (":", "\\"))
    assert m["reorder_recommendations_file"].startswith("decisions/")
    assert m["reorder_summary_file"].startswith("decisions/")


# 38 — the status lifecycle includes calculating_reorder_recommendations, after stockout risk
def test_c_lifecycle_step_present(good):
    assert "calculating_reorder_recommendations" in orch.STEP_PROGRESS
    log = (good["run_dir"] / "pipeline.log").read_text(encoding="utf-8")
    assert "calculating_reorder_recommendations" in log
    assert "calculating_stockout_risk" in log
    assert log.index("calculating_stockout_risk") < log.index("calculating_reorder_recommendations")


# 39 — an older completed run WITHOUT Phase C stays valid/browsable (Phase C never mutated Phase B)
def test_c_older_run_without_phase_c_still_valid(good):
    rd = good["run_dir"]
    # simulate a pre-Phase-C run by removing the Phase C artifacts
    (rd / "decisions" / "reorder_recommendations.parquet").unlink()
    (rd / "decisions" / "reorder_summary.json").unlink()
    # the Phase B artifacts + manifest remain independently valid and readable
    risk = pd.read_parquet(rd / "decisions" / "stockout_risk.parquet")
    sel = pd.read_parquet(rd / "selected_forecasts.parquet")
    orch.dc.validate_stockout_risk(risk, sel[["sku", "channel"]].drop_duplicates(),
                                   good["manifest"]["run_id"])
    assert json.loads((rd / "run_manifest.json").read_text())["status"] == "completed"
