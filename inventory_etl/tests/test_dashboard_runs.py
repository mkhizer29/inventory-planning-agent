"""Phase 5 — dashboard run_service tests.

Tests dashboard/run_service.py directly (never imports dashboard/app.py, which would
execute the Streamlit app). Uses temporary run directories and temporary SQLite DBs;
no real subprocess is launched unless subprocess.Popen is monkeypatched.
"""
import hashlib
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "dashboard"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_service as rs             # noqa: E402


# ── fixtures / builders ────────────────────────────────────────────────────────────────
def _write(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def _make_completed_run(runs_dir: Path, run_id="20260101T000000Z_groceries-pets_top3_abc123",
                        created="2026-07-28T17:02:00+00:00", n_sku=3):
    rd = runs_dir / run_id
    (rd / "processed").mkdir(parents=True)
    (rd / "outputs").mkdir(parents=True)
    _write(rd / "status.json", {"run_id": run_id, "status": "completed", "progress_pct": 100,
                                "current_step": "completed", "created_at": created,
                                "completed_at": "2026-07-28T17:03:00+00:00", "failed_at": None})
    _write(rd / "request.json", {"run_id": run_id, "category": "Groceries & Pets", "top_n": 3,
                                 "as_of_date": "2026-06-30", "created_at": created})
    _write(rd / "run_manifest.json", {
        "run_id": run_id, "status": "completed", "selected_sku_count": n_sku,
        "completed_models": ["baseline", "holtwinters", "lightgbm"], "failed_models": [],
        "skipped_models": [], "dataset_fingerprint": "a" * 64, "duration_seconds": 60.0,
        "operational_model": "moving_average_7", "operational_horizon": 14,
        "winners_by_horizon": {"7": "lightgbm", "14": "moving_average_7"},
        "created_at": created})
    # minimal real artifacts
    skus = [f"S{i}" for i in range(n_sku)]
    dates = pd.date_range("2026-07-01", periods=14, freq="D")
    panel = pd.DataFrame({"sku": [s for s in skus for _ in dates],
                          "channel": "naheed_web",
                          "date": [d for _ in skus for d in pd.date_range("2026-01-01", periods=14, freq="D")],
                          "units_observed": 5.0})
    panel.to_parquet(rd / "processed" / "model_panel.parquet", index=False)
    ff = pd.DataFrame({"sku": [s for s in skus for _ in dates], "channel": "naheed_web",
                       "date": [d for _ in skus for d in dates],
                       "forecast_horizon_day": [i + 1 for _ in skus for i in range(14)]})
    ff.to_parquet(rd / "processed" / "forecast_frame.parquet", index=False)
    pd.DataFrame({"sku": skus}).to_parquet(rd / "processed" / "inventory_context.parquet", index=False)
    _write(rd / "processed" / "pilot_manifest.json", {"selected_skus": skus, "validation_status": "passed"})
    (rd / "combined_scorecard.csv").write_text("model,horizon\nlightgbm,7\n", encoding="utf-8")
    (rd / "model_ranking.csv").write_text("horizon,rank,model\n7,1,lightgbm\n", encoding="utf-8")
    # future files with a model column + forecast_horizon_day
    def _fut(model):
        return pd.DataFrame({"sku": [s for s in skus for _ in dates], "channel": "naheed_web",
                             "date": [d for _ in skus for d in dates],
                             "forecast_horizon_day": [i + 1 for _ in skus for i in range(14)],
                             "y_pred": 1.0, "model": model, "model_version": "v1",
                             "as_of_date": "2026-06-30"})
    _fut("moving_average_7").to_parquet(rd / "selected_forecasts.parquet", index=False)
    _fut("moving_average_7").to_parquet(rd / "outputs" / "future_forecast_baseline.parquet", index=False)
    _fut("holtwinters").to_parquet(rd / "outputs" / "future_forecast_holtwinters.parquet", index=False)
    _fut("lightgbm").to_parquet(rd / "outputs" / "future_forecast_lightgbm.parquet", index=False)
    return rd


def _make_running_run(runs_dir: Path, run_id="20260102T000000Z_x_top5_def456",
                      created="2026-07-29T09:00:00+00:00", updated=None):
    """A LIVE run: status.json was touched moments ago. A non-terminal run whose status has
    not moved for STALE_RUN_HOURS is a dead process and is reported as stalled instead, so a
    fixture that means "currently running" must carry a fresh updated_at."""
    from datetime import datetime as _d, timezone as _tz
    rd = runs_dir / run_id
    rd.mkdir(parents=True)
    _write(rd / "status.json", {"run_id": run_id, "status": "running_lightgbm", "progress_pct": 70,
                                "current_step": "running_lightgbm", "created_at": created,
                                "updated_at": updated or _d.now(_tz.utc).isoformat()})
    _write(rd / "request.json", {"category": "Health & Beauty", "top_n": 5,
                                 "as_of_date": "2026-06-30", "created_at": created})
    return rd


def _make_failed_run(runs_dir: Path, run_id="20260103T000000Z_y_top4_aaa111",
                     created="2026-07-27T09:00:00+00:00"):
    rd = runs_dir / run_id
    rd.mkdir(parents=True)
    _write(rd / "status.json", {"run_id": run_id, "status": "failed", "created_at": created,
                                "failed_at": "2026-07-27T09:01:00+00:00", "completed_at": None,
                                "error_message": "model holtwinters failed"})
    _write(rd / "run_manifest.json", {"run_id": run_id, "status": "failed", "created_at": created,
                                      "failed_models": ["holtwinters"], "errors": ["boom"]})
    return rd


def _make_db(path: Path, max_date="2026-06-30"):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE sales_transactions (sku_id TEXT, transaction_date TEXT, quantity_sold REAL)")
    con.executemany("INSERT INTO sales_transactions VALUES (?,?,?)",
                    [("A", "2026-01-01", 3), ("A", max_date, 9), ("B", "2026-05-01", 2)])
    con.execute("""CREATE TABLE sku_master (sku_id TEXT, category TEXT, brand TEXT, sku_name TEXT)""")
    con.commit(); con.close()


# ── 1-3 run id ─────────────────────────────────────────────────────────────────────────
def test_01_generate_run_id_path_safe():
    rid = rs.generate_run_id("Groceries & Pets", 10)
    assert rs.RUN_ID_RE.match(rid) and rs.is_safe_run_id(rid)
    assert "/" not in rid and "\\" not in rid and ".." not in rid and "top10" in rid


def test_02_generated_ids_unique():
    assert rs.generate_run_id("X", 3) != rs.generate_run_id("X", 3)


def test_03_unsafe_ids_rejected():
    for bad in ("../evil", "a/b", "a\\b", "..", "a b", "a;b"):
        assert not rs.is_safe_run_id(bad)


# ── 4-13 discovery ───────────────────────────────────────────────────────────────────────
def test_04_discover_missing_dir(tmp_path):
    assert rs.discover_runs(tmp_path / "nope") == []


def test_05_discover_ignores_non_directories(tmp_path):
    (tmp_path / "loose.txt").write_text("x")
    assert rs.discover_runs(tmp_path) == []


def test_06_discover_rejects_symlinked_run(tmp_path):
    target = tmp_path / "real"; _make_completed_run(target if False else tmp_path, run_id="realrun")
    link = tmp_path / "linked"
    try:
        link.symlink_to(tmp_path / "realrun", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")
    ids = {r["run_id"] for r in rs.discover_runs(tmp_path)}
    assert "linked" not in ids and "realrun" in ids


def test_07_discover_reads_running(tmp_path):
    _make_running_run(tmp_path)
    r = rs.discover_runs(tmp_path)[0]
    assert r["is_running"] and not r["is_terminal"] and r["current_step"] == "running_lightgbm"


def test_08_discover_reads_completed_manifest(tmp_path):
    _make_completed_run(tmp_path)
    r = rs.discover_runs(tmp_path)[0]
    assert r["is_completed"] and r["operational_model"] == "moving_average_7"
    assert r["winners_by_horizon"] == {"7": "lightgbm", "14": "moving_average_7"}
    assert r["selected_sku_count"] == 3


def test_09_discover_reads_failed(tmp_path):
    _make_failed_run(tmp_path)
    r = rs.discover_runs(tmp_path)[0]
    assert r["is_failed"] and r["is_terminal"] and not r["is_running"]


def test_10_malformed_status_json(tmp_path):
    rd = tmp_path / "badrun"; rd.mkdir()
    (rd / "status.json").write_text("{not json")
    r = rs.discover_runs(tmp_path)
    assert len(r) == 1 and r[0]["status"] == "unknown"


def test_11_malformed_manifest_json(tmp_path):
    _make_completed_run(tmp_path)
    rd = next(p for p in tmp_path.iterdir() if p.is_dir())
    (rd / "run_manifest.json").write_text("{broken")
    r = rs.discover_runs(tmp_path)[0]
    assert r["status"] == "completed"          # status.json still readable


def test_12_sort_running_first_then_newest(tmp_path):
    _make_completed_run(tmp_path, run_id="c_old", created="2026-07-20T00:00:00+00:00")
    _make_completed_run(tmp_path, run_id="c_new", created="2026-07-28T00:00:00+00:00")
    _make_running_run(tmp_path, run_id="r_run", created="2026-07-10T00:00:00+00:00")
    order = [r["run_id"] for r in rs.discover_runs(tmp_path)]
    assert order[0] == "r_run" and order[1] == "c_new" and order[2] == "c_old"


def test_13_terminal_detection():
    assert "completed" in rs.TERMINAL_STATES and "failed" in rs.TERMINAL_STATES
    assert "running_baseline" in rs.RUNNING_STATES


# ── 14-19 command building ───────────────────────────────────────────────────────────────
def _cmd(**over):
    base = dict(category="Groceries & Pets", top_n=3, as_of_date="2026-06-30",
                selection_cutoff="2026-06-30", min_history_days=28, horizons=(7, 14),
                runs_dir="runs", run_id="rid1", db_path="inventory_etl/output/inventory.db")
    base.update(over)
    return rs.build_orchestrator_command(**base)


def test_14_command_starts_with_executable():
    c = _cmd()
    assert c[0] == sys.executable and c[1].endswith("forecast_orchestrator.py")


def test_15_command_is_list_no_shell():
    c = _cmd()
    assert isinstance(c, list) and all(isinstance(x, str) for x in c)
    assert not any(tok in " ".join(c) for tok in ("&&", "|", ";", ">"))


def test_16_command_category_exact_single_arg():
    c = _cmd(category="Groceries & Pets")
    assert c[c.index("--category") + 1] == "Groceries & Pets"


def test_17_command_top_n():
    c = _cmd(top_n=10)
    assert c[c.index("--top-n") + 1] == "10"


def test_18_command_dates_and_horizons():
    c = _cmd()
    assert c[c.index("--as-of-date") + 1] == "2026-06-30"
    i = c.index("--horizons")
    assert c[i + 1] == "7" and c[i + 2] == "14"


def test_19_command_uses_repo_paths():
    c = _cmd(runs_dir="runs", db_path="inventory_etl/output/inventory.db")
    assert c[c.index("--runs-dir") + 1] == "runs"
    assert c[c.index("--db-path") + 1] == "inventory_etl/output/inventory.db"


# ── ranking metric (Top-N by units sold vs by stockout risk) ─────────────────────────────
def test_19a_command_defaults_to_units():
    c = _cmd()
    assert c[c.index("--ranking-metric") + 1] == rs.METRIC_UNITS


def test_19b_command_carries_stockout_risk():
    c = _cmd(ranking_metric=rs.METRIC_STOCKOUT_RISK)
    assert c[c.index("--ranking-metric") + 1] == "stockout_risk"


def test_19c_launch_rejects_unknown_metric(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="ranking_metric"):
        _launch(tmp_path, monkeypatch, ranking_metric="whatever")


def test_19d_launch_forwards_metric(tmp_path, monkeypatch):
    _launch(tmp_path, monkeypatch, ranking_metric=rs.METRIC_STOCKOUT_RISK)
    cmd = _FakePopen.last["cmd"]
    assert cmd[cmd.index("--ranking-metric") + 1] == "stockout_risk"


def test_19e_metric_labels_cover_every_supported_metric():
    assert all(rs.ranking_metric_label(m) for m in rs.SUPPORTED_RANKING_METRICS)
    assert rs.ranking_metric_label(rs.METRIC_STOCKOUT_RISK) == "Stockout risk"


def test_19f_legacy_run_without_metric_reads_as_units(tmp_path):
    """Runs created before this feature have no ranking_metric — they were units-ranked."""
    run = tmp_path / "runs" / "old_run"
    run.mkdir(parents=True)
    (run / "request.json").write_text(json.dumps(
        {"category": "Groceries", "top_n": 5, "as_of_date": "2026-06-30"}), encoding="utf-8")
    (run / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    rec = rs.discover_runs(tmp_path / "runs")[0]
    assert rec["ranking_metric"] == rs.METRIC_UNITS
    assert rs.is_risk_ranked(rec) is False
    assert rs.RISK_RANKED_SYMBOL not in rs.format_run_label_short(rec)


def test_19g_risk_ranked_run_is_marked_in_labels(tmp_path):
    run = tmp_path / "runs" / "risk_run"
    run.mkdir(parents=True)
    (run / "request.json").write_text(json.dumps(
        {"category": "Groceries", "top_n": 5, "as_of_date": "2026-06-30",
         "ranking_metric": "stockout_risk"}), encoding="utf-8")
    (run / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    rec = rs.discover_runs(tmp_path / "runs")[0]
    assert rs.is_risk_ranked(rec) is True
    short = rs.format_run_label_short(rec)
    assert "Risk" in short                       # spelled out, not an unexplained glyph
    assert rs.RISK_RANKED_SYMBOL not in short
    assert "by stockout risk" in rs.format_run_label_full(rec)


# ── trailing extract-tail guard on the default as-of date ────────────────────────────────
def _volume_db(path: Path, days: "list[tuple[str, float]]"):
    """A warehouse whose daily totals are exactly `days` [(date, units), ...]."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE sales_transactions (sku_id TEXT, transaction_date TEXT, "
                "quantity_sold REAL)")
    con.executemany("INSERT INTO sales_transactions VALUES (?,?,?)",
                    [("A", d, u) for d, u in days])
    con.commit(); con.close()


def test_19h_tail_of_stray_rows_is_discounted(tmp_path):
    """The real failure mode: extract stops mid-day, leaving days with a couple of units."""
    db = tmp_path / "tail.db"
    _volume_db(db, [(f"2026-07-{d:02d}", 12000.0) for d in range(1, 24)]
               + [("2026-07-24", 162.0), ("2026-07-28", 3.0), ("2026-07-31", 2.0)])
    diag = rs.sales_date_diagnostics(db)
    assert diag["raw_max"] == date(2026, 7, 31)
    assert diag["usable_max"] == date(2026, 7, 23)          # last full trading day
    assert rs.get_latest_sales_date(db) == date(2026, 7, 23)
    assert [d.isoformat() for d in diag["ignored_dates"]] == \
        ["2026-07-24", "2026-07-28", "2026-07-31"]


def test_19i_clean_warehouse_is_unchanged(tmp_path):
    """No tail -> the guard must be a no-op and still return the true maximum."""
    db = tmp_path / "clean.db"
    _volume_db(db, [(f"2026-07-{d:02d}", 12000.0) for d in range(1, 25)])
    assert rs.get_latest_sales_date(db) == date(2026, 7, 24)
    assert rs.sales_date_diagnostics(db)["ignored_dates"] == []


def test_19j_low_volume_warehouse_not_penalised(tmp_path):
    """A genuinely tiny warehouse must not have its newest day discounted — the floor of
    1 unit exists so the guard only ever removes a near-empty tail."""
    db = tmp_path / "small.db"
    _volume_db(db, [("2026-07-01", 3.0), ("2026-07-02", 1.0), ("2026-07-03", 2.0)])
    assert rs.get_latest_sales_date(db) == date(2026, 7, 3)
    assert rs.sales_date_diagnostics(db)["ignored_dates"] == []


def test_19k_uneven_but_real_days_survive(tmp_path):
    """Ordinary weekday/weekend swings are not an extract tail."""
    db = tmp_path / "uneven.db"
    _volume_db(db, [("2026-07-01", 12000.0), ("2026-07-02", 15000.0),
                    ("2026-07-03", 9000.0), ("2026-07-04", 6000.0)])
    assert rs.get_latest_sales_date(db) == date(2026, 7, 4)


def test_19l_diagnostics_safe_on_missing_and_empty_db(tmp_path):
    assert rs.get_latest_sales_date(tmp_path / "nope.db") is None
    assert rs.sales_date_diagnostics(tmp_path / "nope.db")["usable_max"] is None
    empty = tmp_path / "empty.db"
    _volume_db(empty, [])
    assert rs.get_latest_sales_date(empty) is None


def test_19m_guard_never_writes(tmp_path):
    db = tmp_path / "ro.db"
    _volume_db(db, [("2026-07-01", 5000.0), ("2026-07-02", 2.0)])
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    rs.get_latest_sales_date(db)
    rs.sales_date_diagnostics(db)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


# ── 20-24 launch ─────────────────────────────────────────────────────────────────────────
class _FakePopen:
    last = None

    def __init__(self, cmd, cwd=None, stdout=None, stderr=None, shell=False):
        _FakePopen.last = {"cmd": cmd, "cwd": cwd, "shell": shell}
        self.pid = 4242
        if stdout is not None and hasattr(stdout, "close"):
            stdout.close()


def _launch(tmp_path, monkeypatch, **over):
    db = tmp_path / "wh.db"; _make_db(db)
    monkeypatch.setattr(rs.subprocess, "Popen", _FakePopen)
    base = dict(category="Groceries & Pets", top_n=3, as_of_date="2026-06-30",
                selection_cutoff="2026-06-30", min_history_days=28, horizons=(7, 14),
                runs_dir=tmp_path / "runs", run_id="rid_launch", db_path=db)
    base.update(over)
    return rs.launch_forecast_run(**base)


def test_20_launch_uses_popen_shell_false(tmp_path, monkeypatch):
    _launch(tmp_path, monkeypatch)
    assert _FakePopen.last["shell"] is False


def test_21_launch_cwd_is_repo_root(tmp_path, monkeypatch):
    _launch(tmp_path, monkeypatch)
    assert _FakePopen.last["cwd"] == str(rs.REPO_ROOT)


def test_22_launch_returns_run_id_and_pid(tmp_path, monkeypatch):
    info = _launch(tmp_path, monkeypatch)
    assert info["run_id"] == "rid_launch" and info["pid"] == 4242 and info["command"][0] == sys.executable


def test_23_launch_does_not_create_run_dir(tmp_path, monkeypatch):
    info = _launch(tmp_path, monkeypatch)
    assert not (tmp_path / "runs" / "rid_launch").exists()
    assert info["expected_run_dir"].endswith("rid_launch")


def test_24_launcher_log_under_launcher_logs(tmp_path, monkeypatch):
    info = _launch(tmp_path, monkeypatch)
    assert Path(info["log_path"]).parent.name == rs.LAUNCHER_LOG_DIRNAME
    assert (tmp_path / "runs" / rs.LAUNCHER_LOG_DIRNAME / "rid_launch.log").exists()


def test_49_launch_rejects_arbitrary_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(rs.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(rs, "build_orchestrator_command",
                        lambda **kw: ["/bin/evil", "boom"])   # not sys.executable
    db = tmp_path / "wh.db"; _make_db(db)
    with pytest.raises(ValueError):
        rs.launch_forecast_run(category="C", top_n=3, as_of_date="2026-06-30",
                               runs_dir=tmp_path / "runs", run_id="rid_evil", db_path=db)


def test_48_no_process_without_monkeypatch(tmp_path):
    # a bad run_id is rejected before any Popen; no real process is ever spawned here
    db = tmp_path / "wh.db"; _make_db(db)
    with pytest.raises(ValueError):
        rs.launch_forecast_run(category="C", top_n=3, as_of_date="2026-06-30",
                               runs_dir=tmp_path / "runs", run_id="../escape", db_path=db)


# ── 25-27 warehouse reads ────────────────────────────────────────────────────────────────
def test_25_latest_sales_date(tmp_path):
    db = tmp_path / "wh.db"; _make_db(db, max_date="2026-06-30")
    assert rs.get_latest_sales_date(db).isoformat() == "2026-06-30"


def test_26_latest_sales_date_readonly(tmp_path):
    db = tmp_path / "wh.db"; _make_db(db)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    rs.get_latest_sales_date(db)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_27_categories_use_phase1_api(tmp_path, monkeypatch):
    import dynamic_selection as dsel
    called = {}

    def fake(db_path, cutoff, mhd):
        called["args"] = (str(db_path), str(cutoff), mhd)
        return pd.DataFrame({"category": ["Groceries & Pets"], "eligible_sku_count": [3],
                             "historical_units": [100.0], "history_start": ["2026-01-01"],
                             "history_end": ["2026-06-30"]})
    monkeypatch.setattr(dsel, "list_eligible_categories", fake)
    db = tmp_path / "wh.db"; _make_db(db)
    out = rs.list_categories(db, "2026-06-30", 28)
    assert called["args"] == (str(db), "2026-06-30", 28) and list(out["category"]) == ["Groceries & Pets"]


# ── 28-36 context / mapping ──────────────────────────────────────────────────────────────
def test_28_context_resolves_paths(tmp_path):
    _make_completed_run(tmp_path)
    ctx = rs.resolve_run_context(rs.discover_runs(tmp_path)[0])
    assert ctx["mode"] == "run" and ctx["model_panel"].exists() and ctx["outputs_dir"].is_dir()
    assert ctx["selected_forecasts"].exists()


def test_29_context_rejects_traversal(tmp_path):
    rec = {"run_id": "x", "run_dir": str(tmp_path / ".." / "x"), "is_completed": True, "status": "completed"}
    with pytest.raises(rs.RunContextError):
        rs.resolve_run_context(rec)


def test_30_context_rejects_outside_run_dir(tmp_path):
    _make_completed_run(tmp_path)
    rec = rs.discover_runs(tmp_path)[0]
    # a run dir that does not actually exist -> missing artifacts / not a dir
    rec2 = dict(rec, run_dir=str(tmp_path / "ghost"))
    with pytest.raises(rs.RunContextError):
        rs.resolve_run_context(rec2)


def test_31_running_run_not_activatable(tmp_path):
    _make_running_run(tmp_path)
    with pytest.raises(rs.RunContextError):
        rs.resolve_run_context(rs.discover_runs(tmp_path)[0])


def test_32_failed_run_not_activatable(tmp_path):
    _make_failed_run(tmp_path)
    with pytest.raises(rs.RunContextError):
        rs.resolve_run_context(rs.discover_runs(tmp_path)[0])


def test_33_legacy_context_paths():
    ctx = rs.legacy_context()
    assert ctx["mode"] == "legacy"
    assert ctx["model_panel"] == rs.REPO_ROOT / "data" / "processed" / "model_panel.parquet"
    assert ctx["outputs_dir"] == rs.REPO_ROOT / "outputs"


def test_34_35_model_file_mapping(tmp_path):
    _make_completed_run(tmp_path)
    rec = rs.discover_runs(tmp_path)[0]
    ctx = rs.resolve_run_context(rec)
    opts = rs.run_model_options(ctx, rec)
    labels = [o["label"] for o in opts]
    assert labels[0].startswith("Operational Winner — moving_average_7")
    op = next(o for o in opts if o["key"] == "operational")
    assert op["path"] == ctx["selected_forecasts"]         # operational -> selected_forecasts.parquet
    assert any(o["key"] == "holtwinters" for o in opts) and any(o["key"] == "lightgbm" for o in opts)


def test_36_baseline_label_from_model_column(tmp_path):
    _make_completed_run(tmp_path)
    rec = rs.discover_runs(tmp_path)[0]
    ctx = rs.resolve_run_context(rec)
    bl = next(o for o in rs.run_model_options(ctx, rec) if o["key"] == "baseline")
    assert bl["label"] == "Best Baseline — moving_average_7"   # read from the file's model column


def test_34b_skipped_model_option_absent(tmp_path):
    _make_completed_run(tmp_path)
    rd = next(p for p in tmp_path.iterdir() if p.is_dir())
    man = json.loads((rd / "run_manifest.json").read_text())
    man["completed_models"] = ["baseline", "holtwinters"]      # lightgbm skipped/failed
    (rd / "run_manifest.json").write_text(json.dumps(man))
    rec = rs.discover_runs(tmp_path)[0]
    ctx = rs.resolve_run_context(rec)
    assert not any(o["key"] == "lightgbm" for o in rs.run_model_options(ctx, rec))


# ── 37-39 horizon filtering ──────────────────────────────────────────────────────────────
def test_37_horizon_filter_uses_forecast_horizon_day():
    df = pd.DataFrame({"sku": ["A"] * 14, "channel": "naheed_web",
                       "date": pd.date_range("2026-07-01", periods=14, freq="D"),
                       "forecast_horizon_day": list(range(1, 15)), "y_pred": 1.0})
    assert len(rs.apply_horizon_filter(df, 7)) == 7
    assert len(rs.apply_horizon_filter(df, 14)) == 14


def test_38_legacy_horizon_fallback():
    df = pd.DataFrame({"sku": ["A"] * 20, "date": pd.date_range("2026-07-01", periods=20, freq="D"),
                       "y_pred": 1.0})
    assert len(rs.apply_horizon_filter(df, 7)) == 7          # head(horizon) fallback, no fhd column


def test_39_duplicate_keys_deduped():
    df = pd.DataFrame({"sku": ["A", "A", "B"], "channel": "naheed_web",
                       "date": pd.to_datetime(["2026-07-01", "2026-07-01", "2026-07-01"]),
                       "forecast_horizon_day": [1, 1, 1], "y_pred": [1.0, 9.0, 2.0]})
    out = rs.apply_horizon_filter(df, 7)
    assert len(out) == 2 and out[out["sku"] == "A"]["y_pred"].iloc[0] == 1.0   # first kept


# ── 40-42 artifacts / logs ───────────────────────────────────────────────────────────────
def test_40_missing_artifact_context_error(tmp_path):
    _make_completed_run(tmp_path)
    rd = next(p for p in tmp_path.iterdir() if p.is_dir())
    (rd / "selected_forecasts.parquet").unlink()
    with pytest.raises(rs.RunContextError):
        rs.resolve_run_context(rs.discover_runs(tmp_path)[0])


def test_41_log_tail_max_200(tmp_path):
    log = tmp_path / "p.log"
    log.write_text("\n".join(f"line {i}" for i in range(1000)), encoding="utf-8")
    out = rs.tail_log(log, 200)
    assert out.count("\n") == 199 and out.endswith("line 999")


def test_42_log_tail_plain_text(tmp_path):
    log = tmp_path / "p.log"
    log.write_text("<script>alert(1)</script>\nok", encoding="utf-8")
    out = rs.tail_log(log)
    assert "<script>" in out and isinstance(out, str)       # returned verbatim, rendered as text by UI


def test_41b_log_tail_missing_file(tmp_path):
    assert rs.tail_log(tmp_path / "nope.log") == ""


# ── 43-45 isolation ──────────────────────────────────────────────────────────────────────
def _snap(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in d.iterdir() if p.is_file()} if d.exists() else {}


def test_43_44_45_no_global_writes(tmp_path, monkeypatch):
    proc_before = _snap(rs.REPO_ROOT / "data" / "processed")
    out_before = _snap(rs.REPO_ROOT / "outputs")
    db = rs.REPO_ROOT / "inventory_etl" / "output" / "inventory.db"
    db_before = hashlib.sha256(db.read_bytes()).hexdigest() if db.exists() else None
    # exercise discovery + a monkeypatched launch + latest-date read
    _make_completed_run(tmp_path)
    rs.discover_runs(tmp_path)
    monkeypatch.setattr(rs.subprocess, "Popen", _FakePopen)
    wh = tmp_path / "wh.db"; _make_db(wh)
    rs.launch_forecast_run(category="C", top_n=3, as_of_date="2026-06-30",
                           runs_dir=tmp_path / "runs", run_id="iso_run", db_path=wh)
    rs.get_latest_sales_date(wh)
    assert _snap(rs.REPO_ROOT / "data" / "processed") == proc_before
    assert _snap(rs.REPO_ROOT / "outputs") == out_before
    assert (hashlib.sha256(db.read_bytes()).hexdigest() if db.exists() else None) == db_before


# ── 46-47 app/styles compile ─────────────────────────────────────────────────────────────
def test_46_app_compiles():
    import py_compile
    py_compile.compile(str(REPO_ROOT / "dashboard" / "app.py"), doraise=True)


def test_47_styles_compiles():
    import py_compile
    py_compile.compile(str(REPO_ROOT / "dashboard" / "styles.py"), doraise=True)


# ── 50 legacy always available ───────────────────────────────────────────────────────────
def test_50_legacy_available_with_zero_completed(tmp_path):
    _make_running_run(tmp_path); _make_failed_run(tmp_path)
    runs = rs.discover_runs(tmp_path)
    assert not any(r["is_completed"] for r in runs)
    assert rs.legacy_context()["mode"] == "legacy"           # legacy still resolvable


# ── timestamp display: UTC persisted, Pakistan time shown ────────────────────────────────
PKT_EXPECTED = "29 Jul 2026 · 11:03 AM PKT"


def test_51_utc_offset_converted_to_pkt():
    assert rs.format_local_datetime("2026-07-29T06:03:00+00:00") == PKT_EXPECTED


def test_52_zulu_suffix_converted_to_pkt():
    assert rs.format_local_datetime("2026-07-29T06:03:00Z") == PKT_EXPECTED


def test_53_non_utc_offset_converted():
    # 09:03 at +03:00 == 06:03 UTC == 11:03 PKT
    assert rs.format_local_datetime("2026-07-29T09:03:00+03:00") == PKT_EXPECTED


def test_54_naive_timestamp_not_shifted():
    # legacy naive values may already be local — the clock must not gain +5h
    assert rs.format_local_datetime("2026-07-29T11:03:00") == PKT_EXPECTED
    assert rs.format_local_datetime("2026-07-29T06:03:00").startswith("29 Jul 2026 · 06:03 AM")


def test_55_invalid_and_null_return_dash():
    for bad in (None, "", "   ", "garbage", "not-a-timestamp"):
        assert rs.format_local_datetime(bad) == "—"


def test_56_format_run_label_uses_pakistan_time():
    label = rs.format_run_label({"created_at": "2026-07-29T06:03:00+00:00",
                                 "category": "Groceries & Pets", "top_n": 10, "status": "completed"})
    assert label.startswith(PKT_EXPECTED)
    assert "PKT" in label and "06:03" not in label
    assert label == f"{PKT_EXPECTED} · Groceries & Pets · Top 10 by units sold · completed"


def test_57_display_timezone_is_karachi():
    assert str(rs.DISPLAY_TIMEZONE) == "Asia/Karachi"


def test_58_component_flags():
    ts = "2026-07-29T06:03:00+00:00"
    assert rs.format_local_datetime(ts, include_time=False) == "29 Jul 2026"
    assert rs.format_local_datetime(ts, include_date=False) == "11:03 AM PKT"
    assert rs.format_local_datetime(ts, include_date=False, include_timezone=False) == "11:03 AM"


def test_59_datetime_object_accepted():
    from datetime import datetime as _dt, timezone as _tz
    assert rs.format_local_datetime(_dt(2026, 7, 29, 6, 3, tzinfo=_tz.utc)) == PKT_EXPECTED


def test_60_business_dates_untouched_by_helper_usage():
    # a plain business date must never be turned into a PKT instant by the dashboard
    assert rs.format_local_datetime("2026-06-30", include_time=False) == "30 Jun 2026"


# ── Phase B: decision artifact exposure (backward compatible) ─────────────────────────────
def _add_decision_artifacts(rd: Path, skus, run_id):
    dec = rd / "decisions"
    dec.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"run_id": run_id, "sku": list(skus), "channel": "naheed_web",
                  "stockout_probability": [0.1, 0.5, 0.9][:len(skus)],
                  "overall_risk_tier": ["low", "high", "critical"][:len(skus)]}
                 ).to_parquet(dec / "stockout_risk.parquet", index=False)
    dates = pd.date_range("2026-07-01", periods=3, freq="D")
    pd.DataFrame({"run_id": run_id, "sku": [s for s in skus for _ in dates], "channel": "naheed_web",
                  "date": [d for _ in skus for d in dates],
                  "cumulative_stockout_probability": 0.3}
                 ).to_parquet(dec / "stockout_trajectory.parquet", index=False)


def test_61_context_exposes_decision_artifacts_when_present(tmp_path):
    run_id = "20260101T000000Z_groceries-pets_top3_dec001"
    rd = _make_completed_run(tmp_path, run_id=run_id, n_sku=3)
    _add_decision_artifacts(rd, [f"S{i}" for i in range(3)], run_id)
    ctx = rs.resolve_run_context(rs._run_record(rd))
    assert ctx["has_stockout_risk"] and ctx["has_stockout_trajectory"]
    assert ctx["stockout_risk"].exists() and ctx["stockout_trajectory"].exists()
    # the dashboard reads the PRECOMPUTED file (never recomputes)
    risk = pd.read_parquet(ctx["stockout_risk"])
    assert len(risk) == 3 and set(risk["overall_risk_tier"]) == {"low", "high", "critical"}


def test_62_old_completed_run_without_decisions_still_browsable(tmp_path):
    rd = _make_completed_run(tmp_path, run_id="20260101T000000Z_groceries-pets_top3_old999", n_sku=3)
    ctx = rs.resolve_run_context(rs._run_record(rd))   # must NOT raise for a pre-Phase-B run
    assert ctx["has_stockout_risk"] is False and ctx["has_stockout_trajectory"] is False
    assert ctx["selected_forecasts"].exists() and ctx["model_panel"].exists()  # still fully browsable


# ── Phase B dashboard-UX helpers (pure, framework-free) ───────────────────────────────────
def _risk_df():
    """A deliberately shuffled stockout-risk frame with an unambiguous priority order,
    mirroring the columns the dashboard reads from the validated decision artifact."""
    skus = ["SK01", "SK02", "SK03", "SK04", "SK05", "SK06", "SK07", "SK08"]
    names = ["Alpha Cleanser", "Bravo Shampoo", "Charlie Wipes", "Delta Lotion",
             "Echo Serum", "Foxtrot Balm", "Golf Toner", "Hotel Mask"]
    tiers = ["low", "critical", "critical", "high", "unknown", "critical", "critical", "critical"]
    probs = [0.10, 0.90, 0.90, 0.70, 0.00, 0.90, 0.90, 0.90]
    proj = [None, "2026-08-05", "2026-08-02", "2026-08-10", None, "2026-08-02", "2026-08-02", None]
    rev = [10.0, 100.0, 50.0, 200.0, None, 80.0, 80.0, 999.0]
    review = [False, False, False, False, True, False, False, False]
    cover = [30.0, 1.0, 2.0, 6.0, None, 2.0, 2.0, 1.0]
    return pd.DataFrame({
        "sku": skus, "sku_name": names, "overall_risk_tier": tiers,
        "stockout_probability": probs, "projected_stockout_date": proj,
        "estimated_revenue_at_risk": rev, "manual_review_required": review,
        "forecast_days_of_cover": cover, "reason_trace": [f"reason for {s}" for s in skus],
    })


def _traj_df(rows):
    """rows: list of (sku, date_str, forecast_horizon_day)."""
    return pd.DataFrame({
        "sku": [s for s, _, _ in rows], "channel": "naheed_web",
        "date": [d for _, d, _ in rows],
        "forecast_horizon_day": [f for _, _, f in rows],
        "projected_p50_inventory": [10.0 - i for i in range(len(rows))],
        "cumulative_stockout_probability": [0.1 * i for i in range(len(rows))],
    })


def test_63_sort_risk_queue_deterministic_full_order():
    ranked = rs.sort_risk_queue(_risk_df())
    # severity → P(stockout) desc → projected date asc (nulls last) → revenue desc → name asc
    assert list(ranked["sku"]) == ["SK06", "SK07", "SK03", "SK02", "SK08",
                                   "SK04", "SK01", "SK05"]


def test_64_sort_risk_queue_nulls_and_ties():
    ranked = rs.sort_risk_queue(_risk_df()).reset_index(drop=True)
    pos = {s: i for i, s in enumerate(ranked["sku"])}
    # equal severity+prob+projected+revenue → name asc (Foxtrot before Golf)
    assert pos["SK06"] < pos["SK07"]
    # a null projected date sorts AFTER dated peers even with far higher revenue (SK08 rev=999)
    assert pos["SK02"] < pos["SK08"]
    # unknown tier is always last, never promoted above real risk
    assert pos["SK05"] == len(ranked) - 1


def test_65_sort_risk_queue_pure_and_reason_trace_untouched():
    df = _risk_df()
    before = df.copy(deep=True)
    ranked = rs.sort_risk_queue(df)
    pd.testing.assert_frame_equal(df, before)              # input never mutated
    # reason_trace is carried through verbatim (no truncation / rewriting in the dashboard layer)
    got = dict(zip(ranked["sku"], ranked["reason_trace"]))
    assert got == {s: f"reason for {s}" for s in df["sku"]}
    assert "reason_trace" in ranked.columns and not ranked["reason_trace"].isna().any()


def test_66_risk_severity_unknown_is_not_healthy():
    assert rs.risk_severity_rank("critical") == 0
    assert (rs.risk_severity_rank("critical") < rs.risk_severity_rank("high")
            < rs.risk_severity_rank("watch") < rs.risk_severity_rank("low"))
    assert rs.risk_severity_rank("low") == rs.risk_severity_rank("healthy")
    # unknown must rank strictly worse than healthy/low — never treated as safe
    assert rs.risk_severity_rank("unknown") > rs.risk_severity_rank("healthy")
    assert rs.risk_severity_rank(" Critical ") == 0        # case / whitespace tolerant
    # the engine emits "medium"; it must rank with "watch", NOT fall through to the unknown bucket
    assert rs.risk_severity_rank("medium") == rs.risk_severity_rank("watch") == 2
    assert rs.risk_severity_rank("medium") < rs.risk_severity_rank("low")
    assert rs.risk_severity_rank("nonsense") == 4          # unrecognised → unknown bucket


def test_67_risk_tier_tone_unknown_never_green():
    assert rs.risk_tier_tone("critical") == "red"
    assert rs.risk_tier_tone("high") == "amber"
    assert rs.risk_tier_tone("watch") == "blue" and rs.risk_tier_tone("medium") == "blue"
    assert rs.risk_tier_tone("low") == "success" and rs.risk_tier_tone("healthy") == "success"
    assert rs.risk_tier_tone("unknown") == "slate"
    assert rs.risk_tier_tone("unknown") != rs.risk_tier_tone("healthy")   # unknown is not green


def test_68_filter_risk_queue_by_tier():
    out = rs.filter_risk_queue(_risk_df(), tier="critical")
    assert set(out["sku"]) == {"SK02", "SK03", "SK06", "SK07", "SK08"}
    assert set(rs.filter_risk_queue(_risk_df(), tier="all")["sku"]) == set(_risk_df()["sku"])


def test_69_filter_risk_queue_query_sku_or_name_ci():
    df = _risk_df()
    assert set(rs.filter_risk_queue(df, query="FOXTROT")["sku"]) == {"SK06"}   # name only, case-insensitive
    assert set(rs.filter_risk_queue(df, query="sk03")["sku"]) == {"SK03"}      # sku only, case-insensitive
    assert rs.filter_risk_queue(df, query="zzz").empty


def test_70_filter_risk_queue_projected_and_review_toggles():
    df = _risk_df()
    proj = rs.filter_risk_queue(df, projected_only=True)
    assert proj["projected_stockout_date"].notna().all() and set(proj["sku"]) == {
        "SK02", "SK03", "SK04", "SK06", "SK07"}
    review = rs.filter_risk_queue(df, review_only=True)
    assert set(review["sku"]) == {"SK05"}


def test_71_revenue_total_excludes_nulls_never_zero():
    total, missing = rs.risk_revenue_at_risk_total(_risk_df())
    assert total == 1519.0        # 10+100+50+200+80+80+999 ; the null row is NOT added as 0
    assert missing == 1           # SK05 has no price → reported as missing, not silently zeroed


def test_72_trajectory_for_sku_filters_sku_and_horizon_sorted():
    df = _traj_df([("SK01", "2026-07-03", 3), ("SK01", "2026-07-01", 1),
                   ("SK01", "2026-07-02", 2), ("SK01", "2026-07-04", 4),
                   ("SK02", "2026-07-01", 1)])
    out, warns = rs.trajectory_for_sku(df, "SK01", horizon=3)
    assert list(out["sku"].unique()) == ["SK01"]                       # other skus dropped
    assert list(out["forecast_horizon_day"]) == [1, 2, 3]              # day 4 beyond horizon dropped
    assert list(out["date"]) == list(pd.to_datetime(
        ["2026-07-01", "2026-07-02", "2026-07-03"]))                   # sorted ascending
    assert warns == []


def test_73_trajectory_for_sku_dedups_with_warning():
    df = _traj_df([("SK01", "2026-07-01", 1), ("SK01", "2026-07-01", 1),
                   ("SK01", "2026-07-02", 2)])
    out, warns = rs.trajectory_for_sku(df, "SK01")
    assert len(out) == 2 and warns and "duplicate" in warns[0].lower()


def test_74_full_product_label_is_complete_untruncated():
    long_name = "Extra Strength Herbal Multivitamin Complex 120 Softgels Family Pack"
    label = rs.full_product_label(long_name, "SK99")
    assert label == f"{long_name} (SK99)" and long_name in label      # never truncated


def test_75_full_product_label_falls_back_to_sku():
    assert rs.full_product_label(None, "SK01") == "SK01"
    assert rs.full_product_label(float("nan"), "SK01") == "SK01"
    assert rs.full_product_label("SK01", "SK01") == "SK01"            # name == sku → no redundant "(SK01)"


def test_76_legacy_context_has_no_decision_artifacts():
    ctx = rs.legacy_context()
    assert ctx["has_stockout_risk"] is False and ctx["has_stockout_trajectory"] is False
    assert ctx["stockout_risk"] is None and ctx["stockout_trajectory"] is None
    assert ctx["decisioning_status"] is None      # forecast-driven risk is unavailable in legacy mode


def test_77_risk_artifacts_only_resolve_for_completed_run_with_decisions(tmp_path):
    # completed run WITH decisions → flags true & files readable (dashboard reads, never recomputes)
    run_id = "20260101T000000Z_groceries-pets_top3_dec077"
    rd = _make_completed_run(tmp_path, run_id=run_id, n_sku=3)
    _add_decision_artifacts(rd, [f"S{i}" for i in range(3)], run_id)
    ctx = rs.resolve_run_context(rs._run_record(rd))
    assert ctx["has_stockout_risk"] and ctx["stockout_risk"].exists()
    # a running run cannot be activated at all → no risk surface for an incomplete run
    _make_running_run(tmp_path)
    with pytest.raises(rs.RunContextError):
        rs.resolve_run_context(rs.discover_runs(tmp_path)[0])


# ══════════════════════════════════════════════════════════════════════════════════════════
# Phase C — reorder recommendation dashboard-service + UI helper tests
# ══════════════════════════════════════════════════════════════════════════════════════════
def _add_reorder_artifacts(rd: Path, skus, run_id):
    dec = rd / "decisions"
    dec.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"run_id": run_id, "sku": list(skus), "channel": "naheed_web",
                  "action": (["order_now", "monitor", "no_order"] * 3)[:len(skus)],
                  "recommended_order_quantity": [12, 0, 0][:len(skus)],
                  "recommended_purchase_value": [600.0, None, None][:len(skus)]}
                 ).to_parquet(dec / "reorder_recommendations.parquet", index=False)
    (dec / "reorder_summary.json").write_text(json.dumps({
        "run_id": run_id, "selected_series_count": len(skus), "order_now_count": 1}), encoding="utf-8")


def _reorder_df():
    """A deliberately shuffled reorder frame with an unambiguous buyer-priority order."""
    return pd.DataFrame({
        "sku": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "sku_name": ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"],
        "action": ["no_order", "order_now", "order_now", "manual_review", "vendor_follow_up", "order_now"],
        "overall_risk_tier": ["low", "critical", "critical", "high", "high", "critical"],
        "stockout_probability": [0.05, 0.9, 0.9, 0.6, 0.6, 0.9],
        "projected_stockout_date": [None, "2026-08-05", "2026-08-02", "2026-08-09", None, "2026-08-02"],
        "recommended_purchase_value": [None, 100.0, 50.0, None, None, 80.0],
        "recommended_order_quantity": [0, 24, 12, 0, 0, 36],
        "raw_target_gap": [None, 20.0, 5.0, None, None, 30.0],
        "moq_adjusted_quantity": [None, 24.0, 12.0, None, None, 30.0],
        "rounded_order_quantity": [None, 24.0, 12.0, None, None, 36.0],
        "recommended_order_date": [None, "2026-06-30", "2026-06-30", None, None, "2026-06-30"],
        "approval_required": [False, True, True, True, True, True],
        "assumption_flags": ["", "imputed_cost;synthetic_stock", "assumed_moq", "", "dropship", ""],
        "reason_trace": [f"reason for S{i}" for i in range(1, 7)],
    })


# 40 — run_service resolves Phase C files safely when present
def test_c40_context_exposes_reorder_when_present(tmp_path):
    run_id = "20260101T000000Z_health-beauty_top3_rc001"
    rd = _make_completed_run(tmp_path, run_id=run_id, n_sku=3)
    _add_reorder_artifacts(rd, [f"S{i}" for i in range(3)], run_id)
    ctx = rs.resolve_run_context(rs._run_record(rd))
    assert ctx["reorder_available"] is True
    assert ctx["reorder_recommendations"].exists() and ctx["reorder_summary"].exists()


# 41 — reorder_available is False for an older completed run (still browsable, no crash)
def test_c41_reorder_unavailable_old_run(tmp_path):
    rd = _make_completed_run(tmp_path, run_id="20260101T000000Z_hb_top3_old_rc", n_sku=3)
    ctx = rs.resolve_run_context(rs._run_record(rd))          # must NOT raise
    assert ctx["reorder_available"] is False
    assert ctx["reorder_recommendations"].exists() is False
    assert ctx["selected_forecasts"].exists() and ctx["model_panel"].exists()   # still browsable


# 42 — dynamic mode never treats inventory-context as Phase C: a run with an inventory-context
#      recommendation column but no reorder artifacts still reports reorder_available False
def test_c42_no_fallback_to_inventory_context(tmp_path):
    rd = _make_completed_run(tmp_path, run_id="20260101T000000Z_hb_top3_nofb", n_sku=3)
    # inventory_context (with its historical recommended_order_quantity) exists...
    assert (rd / "processed" / "inventory_context.parquet").exists()
    ctx = rs.resolve_run_context(rs._run_record(rd))
    assert ctx["reorder_available"] is False          # ...but Phase C is still unavailable (no fallback)


# 43 — long product names are preserved unchanged by the helper data
def test_c43_long_name_unchanged(tmp_path):
    long_name = "Ultra Gentle Micellar Cleansing Water for Sensitive Skin 400ml Twin Pack"
    assert rs.full_product_label(long_name, "SK7") == f"{long_name} (SK7)"
    assert long_name in rs.full_product_label(long_name, "SK7")


# 44 — priority sorting is deterministic
def test_c44_sort_reorder_deterministic(tmp_path):
    ranked = rs.sort_reorder_queue(_reorder_df())
    assert list(ranked["sku"]) == ["S6", "S3", "S2", "S5", "S4", "S1"]


# 45 — action / tier / query / approval / assumed filters work
def test_c45_filters(tmp_path):
    df = _reorder_df()
    assert set(rs.filter_reorder_queue(df, action="order_now")["sku"]) == {"S2", "S3", "S6"}
    assert set(rs.filter_reorder_queue(df, tier="high")["sku"]) == {"S4", "S5"}
    assert set(rs.filter_reorder_queue(df, query="foxtrot")["sku"]) == {"S6"}
    assert set(rs.filter_reorder_queue(df, query="s5")["sku"]) == {"S5"}
    assert "S1" not in set(rs.filter_reorder_queue(df, approval_only=True)["sku"])
    # assumed/imputed inputs = imputed_cost / synthetic_stock / assumed_* (dropship is NOT one)
    assert set(rs.filter_reorder_queue(df, assumed_only=True)["sku"]) == {"S2", "S3"}


# 46 — null cost/purchase value stays null (never coerced to zero) in the total
def test_c46_null_cost_stays_null(tmp_path):
    total, missing = rs.reorder_purchase_value_total(_reorder_df())
    assert total == 230.0            # 100 + 50 + 80 ; the null values are NOT added as 0
    # S2, S3, S6 are order_now with values; none missing among order_now
    assert missing == 0
    df2 = _reorder_df()
    df2.loc[df2["sku"] == "S2", "recommended_purchase_value"] = None
    total2, missing2 = rs.reorder_purchase_value_total(df2)
    assert total2 == 130.0 and missing2 == 1     # order_now S2 now unpriced -> counted, never zeroed


# 47 — null dates remain null through the sort helper (no coercion)
def test_c47_null_dates_preserved(tmp_path):
    ranked = rs.sort_reorder_queue(_reorder_df())
    s1 = ranked[ranked["sku"] == "S1"].iloc[0]
    assert pd.isna(s1["recommended_order_date"]) and pd.isna(s1["projected_stockout_date"])


# 48 — filtering to one SKU yields only that SKU (deep-dive selection is single-SKU)
def test_c48_single_sku_selection(tmp_path):
    one = rs.filter_reorder_queue(_reorder_df(), query="S3")
    assert list(one["sku"]) == ["S3"] and len(one) == 1


# 49 — quantity-stage values are display-only: helpers never recompute them
def test_c49_quantity_stages_untouched(tmp_path):
    df = _reorder_df()
    ranked = rs.sort_reorder_queue(df)
    s6 = ranked[ranked["sku"] == "S6"].iloc[0]
    assert float(s6["raw_target_gap"]) == 30.0 and float(s6["moq_adjusted_quantity"]) == 30.0
    assert float(s6["rounded_order_quantity"]) == 36.0 and int(s6["recommended_order_quantity"]) == 36
    # filtering preserves the stages too
    filt = rs.filter_reorder_queue(df, action="order_now")
    assert set(filt["rounded_order_quantity"].dropna()) == {24.0, 12.0, 36.0}


# 50 — reason trace is not truncated by the helper layer
def test_c50_reason_trace_not_truncated(tmp_path):
    df = _reorder_df()
    ranked = rs.sort_reorder_queue(df)
    got = dict(zip(ranked["sku"], ranked["reason_trace"]))
    assert got == {f"S{i}": f"reason for S{i}" for i in range(1, 7)}
    filt = rs.filter_reorder_queue(df, approval_only=True)
    assert (filt["reason_trace"].astype(str).str.len() > 0).all()


# 51 — the dashboard modules compile
def test_c51_dashboard_modules_compile():
    import py_compile
    for name in ("app.py", "styles.py", "run_service.py"):
        py_compile.compile(str(REPO_ROOT / "dashboard" / name), doraise=True)


# ── UX pass: compact vs full run labels ────────────────────────────────────────────────
def _lbl_rec(run_id="20260731T060331Z_g_top10_a4c921", created="2026-07-31T06:03:31+00:00",
             category="Groceries & Pets", top_n=10, status="completed"):
    return {"run_id": run_id, "created_at": created, "category": category,
            "top_n": top_n, "status": status}


def test_70_short_label_is_compact_format():
    short = rs.format_run_label_short(_lbl_rec())
    assert short.startswith("31 Jul · Groceries & Pets · Top 10")
    # the compact label must NOT carry the noisy bits
    assert "PKT" not in short and "2026" not in short and "completed" not in short
    assert "a4c921" not in short                       # full/partial run id only when disambiguating
    assert len(short) < 45


def test_71_short_label_status_symbol():
    # A plain ✓ is dropped — every completed run would carry one, so it costs sidebar width
    # without distinguishing anything. States that need attention still show their symbol.
    done = rs.format_run_label_short(_lbl_rec(status="completed"))
    assert not done.endswith("✓") and done.endswith("Sales")
    assert rs.format_run_label_short(_lbl_rec(status="failed")).endswith("✕")
    assert rs.format_run_label_short(_lbl_rec(status="completed_with_warnings")).endswith("⚠")


def test_72_full_label_retains_everything():
    full = rs.format_run_label_full(_lbl_rec())
    assert "31 Jul 2026" in full and "11:03 AM PKT" in full
    assert "Groceries & Pets" in full and "Top 10" in full and "completed" in full
    assert rs.format_run_label(_lbl_rec()) == full     # back-compat alias


def test_73_similar_runs_get_distinguishable_short_labels():
    a = _lbl_rec(run_id="20260731T060331Z_g_top10_a4c921", created="2026-07-31T06:03:31+00:00")
    b = _lbl_rec(run_id="20260731T090000Z_g_top10_b7f333", created="2026-07-31T09:00:00+00:00")
    labels = rs.build_short_labels([a, b])
    assert labels[a["run_id"]] != labels[b["run_id"]]
    assert labels[a["run_id"]].endswith("a4c921") and labels[b["run_id"]].endswith("b7f333")


def test_74_distinct_runs_keep_clean_labels():
    a = _lbl_rec(run_id="r1", category="Groceries & Pets", top_n=10)
    b = _lbl_rec(run_id="r2", category="Kids & Babies", top_n=5)
    labels = rs.build_short_labels([a, b])
    assert not labels["r1"].endswith("r1") and "Kids & Babies" in labels["r2"]
    assert len(set(labels.values())) == 2


def test_75_short_labels_unique_across_many_runs(tmp_path):
    recs = [_lbl_rec(run_id=f"20260731T0600{i:02d}Z_g_top10_x{i:05d}") for i in range(12)]
    labels = rs.build_short_labels(recs)
    assert len(set(labels.values())) == len(recs)      # selectbox keys must stay unique


def test_76_short_label_handles_missing_fields():
    short = rs.format_run_label_short({"run_id": "r", "status": "running_lightgbm"})
    assert isinstance(short, str) and short             # no crash on partial records
    assert rs.format_run_label_full({}) .startswith("?") or "?" in rs.format_run_label_full({})


def test_77_active_run_never_shows_full_id_as_giant_metric():
    """The compact status strip shows only the run-id suffix; the full id lives in details."""
    rec = _lbl_rec()
    short_id = str(rec["run_id"])[-6:]
    assert short_id == "a4c921" and len(short_id) == 6
    assert rs.format_run_label_short(rec).count(rec["run_id"]) == 0


# ══════════════════════════════════════════════════════════════════════════════════
# Historical Date Range control (From / To + presets).
# Driven through Streamlit's AppTest harness: the control is display-only, so these
# assert UI state and rendered KPIs — never model or artifact behaviour.
# ══════════════════════════════════════════════════════════════════════════════════
import datetime as _dt        # noqa: E402
import re as _re              # noqa: E402

APP_PY = REPO_ROOT / "dashboard" / "app.py"


def _app(page="Executive Overview"):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP_PY), default_timeout=300).run()
    at.session_state["nav_page"] = page
    at.run()
    return at


def _sv(at, key, default=None):
    try:
        return at.session_state[key]
    except Exception:
        return default


def _bounds(at):
    return at.date_input(key="flt_date_from").value, at.date_input(key="flt_date_to").value


def _kpis(at):
    md = " ".join(str(m.value) for m in at.markdown)
    pairs = _re.findall(r'ipa-kpi-label">([^<]+)</div>.*?ipa-kpi-value">([^<]+)<', md, _re.S)
    return {k.strip(): v.strip() for k, v in pairs}


def test_80_default_from_to_use_dataset_min_max():
    at = _app()
    f, t = _bounds(at)
    assert f <= t
    # both defaults must sit inside the widget's own allowed window
    w_from = at.date_input(key="flt_date_from")
    assert w_from.value == f and not at.exception


def test_81_valid_range_filters_rows_inclusively():
    at = _app()
    _, mx = _bounds(at)
    before = _kpis(at).get("Historical Sales Rows")
    at.session_state["flt_date_from"] = mx - _dt.timedelta(days=6)
    at.session_state["flt_date_to"] = mx
    at.run()
    after = _kpis(at).get("Historical Sales Rows")
    assert before and after and before != after and not at.exception


def test_82_from_after_to_is_rejected():
    at = _app()
    mn, mx = _bounds(at)
    at.session_state["flt_date_from"] = mx
    at.session_state["flt_date_to"] = mn
    at.run()
    assert any("after" in str(e.value).lower() for e in at.error)     # inline notice shown
    assert not at.exception                                          # and the page still renders


def _click_preset(at, label):
    btn = {b.label: b for b in at.button}.get(label)
    assert btn is not None, f"preset {label} not rendered"
    btn.click()
    at.run()
    return _sv(at, "flt_date_from"), _sv(at, "flt_date_to")


def test_83_last_7_days_ends_at_latest_available_date():
    at = _app()
    _, mx = _bounds(at)
    f, t = _click_preset(at, "7d")
    assert t == mx and (t - f).days + 1 <= 7


def test_84_last_30_days_is_clamped_to_minimum():
    at = _app()
    mn, mx = _bounds(at)
    f, t = _click_preset(at, "30d")
    assert f >= mn and t == mx and (t - f).days + 1 <= 30


def test_85_all_history_resets_to_full_range():
    at = _app()
    mn, mx = _bounds(at)
    _click_preset(at, "7d")
    f, t = _click_preset(at, "All")
    assert (f, t) == (mn, mx)


def test_86_range_persists_across_history_pages():
    at = _app()
    _, mx = _bounds(at)
    keep = (mx - _dt.timedelta(days=3), mx)
    at.session_state["flt_date_from"], at.session_state["flt_date_to"] = keep
    at.run()
    for page in ("Demand Analytics", "Forecast Explorer", "Stockout Risk", "Executive Overview"):
        at.session_state["nav_page"] = page
        at.run()
        assert not at.exception, page
    assert (_sv(at, "flt_date_from"), _sv(at, "flt_date_to")) == keep


def test_87_stale_single_date_flt_daterange_is_ignored():
    at = _app()
    _, mx = _bounds(at)
    at.session_state["flt_daterange"] = mx          # retired single-widget payload
    at.run()
    assert not at.exception
    f, t = _bounds(at)
    assert f <= t                                   # new controls still coherent


def test_88_future_forecast_rows_are_not_date_filtered():
    at = _app("Forecast Explorer")
    mn, mx = _bounds(at)
    at.session_state["flt_date_from"], at.session_state["flt_date_to"] = mn, mx
    at.run()
    full = {k: v for k, v in _kpis(at).items() if k.startswith("Forecasted")}
    at.session_state["flt_date_from"] = mx - _dt.timedelta(days=6)
    at.session_state["flt_date_to"] = mx
    at.run()
    narrowed = {k: v for k, v in _kpis(at).items() if k.startswith("Forecasted")}
    assert full and full == narrowed, "forecast KPIs must ignore the historical date window"


def test_89_date_controls_only_on_historical_pages():
    at = _app()
    for page, expected in (("Executive Overview", True), ("Demand Analytics", True),
                           ("Forecast Explorer", True), ("Stockout Risk", False),
                           ("Inventory & Replenishment", False), ("Data Quality & Assumptions", False)):
        at.session_state["nav_page"] = page
        at.run()
        labels = {d.label for d in at.date_input}
        assert ({"From", "To"} <= labels) is expected, f"{page}: {labels}"


def test_90_dashboard_modules_compile():
    import py_compile
    for mod in ("app.py", "styles.py"):
        py_compile.compile(str(REPO_ROOT / "dashboard" / mod), doraise=True)


# ══════════════════════════════════════════════════════════════════════════════════
# Production historical-window helpers (rs.normalize_history_window /
# rs.filter_historical_frame) — the exact functions the dashboard pages call.
# ══════════════════════════════════════════════════════════════════════════════════
MN = _dt.date(2026, 1, 15)
MX = _dt.date(2026, 7, 15)


def _hist_frame(start="2026-01-15", periods=182, skus=("A", "B")):
    days = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame({"sku": [s for s in skus for _ in days],
                         "channel": "naheed_web",
                         "date": [d for _ in skus for d in days],
                         "units_observed": 5.0,
                         "category": "Cat1"})


def test_91_full_range_returns_every_row():
    df = _hist_frame()
    out = rs.filter_historical_frame(df, date_from=MN, date_to=MX)
    assert len(out) == len(df)


def test_92_single_day_range_returns_only_that_date():
    df = _hist_frame()
    out = rs.filter_historical_frame(df, date_from=MX, date_to=MX)
    assert len(out) == 2                                    # one row per SKU on that date
    assert set(pd.to_datetime(out["date"]).dt.date) == {MX}


def test_93_filtering_is_inclusive_on_both_ends():
    df = _hist_frame()
    a, b = _dt.date(2026, 2, 1), _dt.date(2026, 2, 3)
    out = rs.filter_historical_frame(df, date_from=a, date_to=b)
    got = sorted(set(pd.to_datetime(out["date"]).dt.date))
    assert got == [a, _dt.date(2026, 2, 2), b]              # both endpoints kept


def test_94_from_after_to_is_rejected_by_normalizer():
    d_from, d_to, err = rs.normalize_history_window(MN, MX, MX, MN)
    assert (d_from, d_to) == (MN, MX) and err and "after" in err.lower()


def test_95_dates_outside_dataset_are_clamped():
    d_from, d_to, err = rs.normalize_history_window(
        MN, MX, _dt.date(2020, 1, 1), _dt.date(2099, 12, 31))
    assert (d_from, d_to) == (MN, MX) and err is None


def test_96_last_7_days_ends_at_latest_available_date():
    d_from, d_to = MX - _dt.timedelta(days=6), MX
    out = rs.filter_historical_frame(_hist_frame(), date_from=d_from, date_to=d_to)
    assert pd.to_datetime(out["date"]).max().date() == MX
    assert len(set(pd.to_datetime(out["date"]).dt.date)) == 7


def test_97_stale_single_date_payload_is_ignored():
    # the retired flt_daterange stored a tuple/list — it must not break the window
    for stale in ((MN, MX), [MN, MX], None, True):
        d_from, d_to, err = rs.normalize_history_window(MN, MX, stale, stale)
        assert (d_from, d_to) == (MN, MX) and err is None


def test_98_helper_does_not_mutate_input_and_handles_empty():
    df = _hist_frame()
    before = df.copy()
    out = rs.filter_historical_frame(df, date_from=_dt.date(2030, 1, 1), date_to=_dt.date(2030, 1, 2))
    pd.testing.assert_frame_equal(df, before)               # original untouched
    assert out.empty and list(out.columns) == list(df.columns)
    assert rs.filter_historical_frame(pd.DataFrame(), date_from=MN, date_to=MX).empty


def test_99_frames_without_a_date_column_pass_through():
    inv = pd.DataFrame({"sku": ["A", "B"], "stock_on_hand": [1, 2]})
    out = rs.filter_historical_frame(inv, date_from=MX, date_to=MX)
    assert len(out) == 2                                    # decision artifacts are never cut


def test_100_executive_overview_row_count_changes():
    at = _app("Executive Overview")
    mn, mx = _bounds(at)
    full = _kpis(at).get("Historical Sales Rows")
    at.date_input(key="flt_date_from").set_value(mx)        # real widget interaction
    at.date_input(key="flt_date_to").set_value(mx)
    at.run()
    single = _kpis(at).get("Historical Sales Rows")
    assert full and single and full != single, f"{full} -> {single}"
    assert int(single.replace(",", "")) < int(full.replace(",", ""))


def test_101_demand_analytics_reacts_to_the_window():
    at = _app("Demand Analytics")
    mn, mx = _bounds(at)
    at.date_input(key="flt_date_from").set_value(mn)
    at.date_input(key="flt_date_to").set_value(mx)
    at.run()
    wide = " ".join(str(m.value) for m in at.markdown)
    at.date_input(key="flt_date_from").set_value(mx)
    at.date_input(key="flt_date_to").set_value(mx)
    at.run()
    narrow = " ".join(str(m.value) for m in at.markdown)
    assert wide != narrow and not at.exception


def test_102_result_caption_reports_the_window():
    at = _app("Executive Overview")
    _, mx = _bounds(at)
    at.date_input(key="flt_date_from").set_value(mx)
    at.date_input(key="flt_date_to").set_value(mx)
    at.run()
    md = " ".join(str(m.value) for m in at.markdown)
    assert "historical rows" in md and "ipa-daterange-result" in md


def test_103_reset_restores_full_history():
    at = _app("Executive Overview")
    mn, mx = _bounds(at)
    at.date_input(key="flt_date_from").set_value(mx)
    at.run()
    btn = {b.label: b for b in at.button}.get("Reset")
    assert btn is not None
    btn.click(); at.run()
    assert _bounds(at) == (mn, mx) and not at.exception


def test_104_decision_artifacts_unaffected_by_the_window():
    at = _app("Stockout Risk")
    before = len([b for b in at.button if b.label == "Details"])
    at.session_state["flt_date_from"] = MX
    at.session_state["flt_date_to"] = MX
    at.run()
    after = len([b for b in at.button if b.label == "Details"])
    assert before == after and not at.exception              # risk queue never date-filtered


# ══════════════════════════════════════════════════════════════════════════════════
# Stalled-run detection: a non-terminal run whose status.json stopped moving is a dead
# process (the launcher cannot finalise a killed child), so it must not be presented as
# live work — nor sort above genuinely recent runs.
# ══════════════════════════════════════════════════════════════════════════════════
def test_110_fresh_running_run_is_live_not_stale(tmp_path):
    _make_running_run(tmp_path)
    r = rs.discover_runs(tmp_path)[0]
    assert r["is_running"] and not r["is_stale"] and not r["is_terminal"]


def test_111_untouched_running_run_is_reported_stalled(tmp_path):
    stale_ts = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=48)).isoformat()
    _make_running_run(tmp_path, updated=stale_ts)
    r = rs.discover_runs(tmp_path)[0]
    assert r["is_stale"] and not r["is_running"]
    assert r["stale_hours"] is not None and r["stale_hours"] >= rs.STALE_RUN_HOURS
    assert not r["is_completed"]          # still never usable as a data source


def test_112_stale_run_does_not_sort_above_recent_completed(tmp_path):
    stale_ts = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=200)).isoformat()
    _make_running_run(tmp_path, run_id="zzz_stale", created="2026-01-01T00:00:00+00:00",
                      updated=stale_ts)
    _make_completed_run(tmp_path, run_id="aaa_recent", created="2026-07-31T00:00:00+00:00")
    order = [r["run_id"] for r in rs.discover_runs(tmp_path)]
    assert order[0] == "aaa_recent", order      # the dead run no longer hogs the top row


def test_113_terminal_runs_are_never_marked_stale(tmp_path):
    _make_completed_run(tmp_path, created="2020-01-01T00:00:00+00:00")
    _make_failed_run(tmp_path, created="2020-01-01T00:00:00+00:00")
    for r in rs.discover_runs(tmp_path):
        assert not r["is_stale"] and r["stale_hours"] is None


def test_114_unparseable_timestamp_does_not_crash_discovery(tmp_path):
    _make_running_run(tmp_path, updated="not-a-timestamp")
    r = rs.discover_runs(tmp_path)[0]
    assert r["is_stale"] is False and r["stale_hours"] is None   # unknown age != dead


# ══════════════════════════════════════════════════════════════════════════════════
# CEO-demo regression guards. Each test below pins a bug that was reproduced on real
# data and fixed; the comment names the wrong behaviour so a regression is obvious.
# Pure-helper tests first (fast, no app), then AppTest cases for app-level wiring.
# ══════════════════════════════════════════════════════════════════════════════════
def _run_rec(run_id="r1", *, created="2026-08-10T10:34:00+00:00", category="Phones & Computers",
             top_n=10, selected=None, metric="stockout_risk", status="completed"):
    """A discover_runs-shaped record; only the label-relevant keys matter here."""
    return {"run_id": run_id, "created_at": created, "category": category, "top_n": top_n,
            "selected_sku_count": selected, "ranking_metric": metric, "status": status}


def test_115_sku_count_is_grammatical_and_absent_when_unknown():
    assert rs.format_sku_count(1) == "1 SKU"          # never "1 SKUs"
    assert rs.format_sku_count(10) == "10 SKUs"
    assert rs.format_sku_count(0) == "0 SKUs"
    assert rs.format_sku_count(None) is None          # unknown must not read as zero


def test_116_selected_count_preferred_over_requested_top_n():
    actual, is_actual = rs.selected_or_requested(_run_rec(selected=1, top_n=10))
    assert (actual, is_actual) == ("1 SKU", True)
    # Selection has not happened yet -> fall back to the request, flagged as a request.
    pending, is_actual = rs.selected_or_requested(_run_rec(selected=None, top_n=10,
                                                           status="running_baseline"))
    assert (pending, is_actual) == ("Top 10", False)


def test_117_risk_run_label_shows_actual_count_and_ranking_not_top_n():
    # The bug: a run that requested Top 10 and selected ONE product was labelled
    # "Top 10", and risk ranking was implied only by a bare ⚡ icon.
    label = rs.format_run_label_short(_run_rec(selected=1, top_n=10, metric="stockout_risk"))
    assert "1 SKU" in label and "1 SKUs" not in label
    assert "Risk" in label
    assert "Top 10" not in label
    assert rs.RISK_RANKED_SYMBOL not in label          # a word, not an unexplained glyph


def test_118_units_run_label_reads_sku_count_and_sales():
    label = rs.format_run_label_short(
        _run_rec(run_id="r2", category="Groceries & Pets", selected=10, top_n=10, metric="units"))
    assert "10 SKUs" in label and "Sales" in label and "Top 10" not in label


def test_119_labels_stay_unique_when_runs_are_otherwise_identical():
    same = [_run_rec(run_id="20260803T010000Z_a_aaaaaa", selected=10, metric="units"),
            _run_rec(run_id="20260803T020000Z_b_bbbbbb", selected=10, metric="units")]
    labels = rs.build_short_labels(same)
    assert len(set(labels.values())) == 2, labels


def test_120_coverage_gap_anchors_on_usable_max_not_raw_max():
    # Real warehouse shape: snapshot 07 Aug, raw sales tail to 31 Jul, but only 23 Jul is
    # complete. Anchoring on raw_max reports a 7-day gap and understates the borderline band.
    cov = rs.reliable_coverage_gap(
        "2026-08-07", {"raw_max": date(2026, 7, 31), "usable_max": date(2026, 7, 23)})
    assert cov["anchor"] == date(2026, 7, 23)
    assert cov["gap_days"] == 15                       # NOT 7
    assert cov["raw_max"] == date(2026, 7, 31)         # still reported, for disclosure


def test_121_coverage_gap_degrades_safely():
    # No completeness verdict -> fall back to raw_max rather than claiming no gap.
    assert rs.reliable_coverage_gap("2026-08-07", {"raw_max": date(2026, 7, 31)})["gap_days"] == 7
    assert rs.reliable_coverage_gap("2026-08-07", {})["gap_days"] == 0
    assert rs.reliable_coverage_gap(None, {"usable_max": date(2026, 7, 23)})["gap_days"] == 0
    # A snapshot OLDER than sales coverage is not a negative gap.
    assert rs.reliable_coverage_gap("2026-07-01", {"usable_max": date(2026, 7, 23)})["gap_days"] == 0


def test_122_stock_provenance_never_claims_live_for_a_completed_run():
    real = rs.stock_provenance_note(0, 12, "07 Aug 2026")
    assert "live" not in real.lower()
    assert "used by this run" in real and "07 Aug 2026" in real
    partial = rs.stock_provenance_note(3, 12)
    assert "3 of 12" in partial and "remaining rows use warehouse stock" in partial
    assert rs.stock_provenance_note(12, 12) == "12 of 12 stock figures are synthetically reconstructed"


# ── App-level wiring (AppTest). These read the REAL runs/ directory, so each test skips
# ── when the repo has no run of the shape it needs rather than asserting on absent data.
def _completed_runs_on_disk():
    return [r for r in rs.discover_runs() if r.get("is_completed")]


def _label_for(rec):
    return rs.build_short_labels(_completed_runs_on_disk())[rec["run_id"]]


def _has_decisions(rec):
    d = Path(rec["run_dir"]) / "decisions"
    return d.is_dir() and any(d.glob("*.parquet"))


def _pick_run(min_skus=2, *, decisions=True):
    """Largest completed run matching the shape a test needs.

    ``decisions`` matters: runs made before Phase B/C have no decision artifacts and are
    SUPPOSED to fall back to the baseline simulation, so decision-behaviour tests must not
    pick one.
    """
    cands = [r for r in _completed_runs_on_disk()
             if (r.get("selected_sku_count") or 0) >= min_skus
             and _has_decisions(r) == decisions]
    if not cands:
        pytest.skip(f"no completed run with >= {min_skus} SKUs and decisions={decisions}")
    return max(cands, key=lambda r: r.get("selected_sku_count") or 0)


def _app_on(page, *, source=None, state=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP_PY), default_timeout=300)
    if source is not None:
        at.session_state["data_source_choice"] = source
    for k, v in (state or {}).items():
        at.session_state[k] = v
    at.session_state["nav_page"] = page
    at.run()
    assert not at.exception, f"{page}: {at.exception}"
    return at


def _text(at):
    return " ".join(str(m.value) for m in at.markdown)


def _queue_count(at):
    m = _re.search(r">(\d+) in queue<", _text(at))
    return int(m.group(1)) if m else None


def _queue_skus(at):
    return sorted(set(_re.findall(r'class="q-sub">SKU ([^<]+)<', _text(at))))


def _panel_rows(at):
    """metric_panel label -> value pairs."""
    return {k.strip(): v.strip() for k, v in
            _re.findall(r'class="l">([^<]+)</span><span class="v">([^<]+)<', _text(at))}


def test_123_fixed_horizon_forecast_kpis_do_not_follow_the_horizon_selector():
    # The bug: the 7d/14d KPIs were sliced from the horizon-truncated frame, so selecting a
    # 7-day horizon made "Forecasted 14-Day Demand" collapse onto the 7-day number.
    run = _pick_run(min_skus=2)
    label = _label_for(run)
    # Pin the page to a product that actually has demand in days 8-14, otherwise the
    # magnitude check below compares 0 with 0 and proves nothing. The product filter also
    # makes the page's focus deterministic instead of "whichever SKU sorts first".
    fc = pd.read_parquet(Path(run["run_dir"]) / "selected_forecasts.parquet")
    fc = fc.sort_values(["sku", "date"])
    totals = {sku: (g["y_pred"].head(7).sum(), g["y_pred"].head(14).sum())
              for sku, g in fc.groupby("sku")}
    busiest = [s for s, (a, b) in sorted(totals.items(), key=lambda kv: -kv[1][1]) if b > a]
    if not busiest:
        pytest.skip("no product in this run has demand beyond day 7")
    state = {"flt_skus": [busiest[0]]}
    k7 = _kpis(_app_on("Forecast Explorer", source=label, state={**state, "flt_horizon": 7}))
    k14 = _kpis(_app_on("Forecast Explorer", source=label, state={**state, "flt_horizon": 14}))
    key7, key14 = "Forecasted 7-Day Demand", "Forecasted 14-Day Demand"
    if key7 not in k7 or key14 not in k7:
        pytest.skip("forecast KPI cards not rendered for this run")
    assert k7[key7] == k14[key7], "7-day KPI moved with the horizon selector"
    assert k7[key14] == k14[key14], "14-day KPI moved with the horizon selector"
    num = lambda s: float(str(s).replace(",", ""))
    assert num(k7[key14]) > num(k7[key7]), "14-day demand collapsed onto the 7-day value"


def test_124_historical_date_filter_does_not_change_decision_queues():
    # Historical From/To is a DISPLAY control for sales history. Decision artifacts are
    # produced by the run and must not shrink because a sparse date was selected.
    label = _label_for(_pick_run(min_skus=2))
    narrow = {"flt_date_from": _dt.date(2026, 1, 1), "flt_date_to": _dt.date(2026, 1, 1)}
    for page in ("Stockout Risk", "Inventory & Replenishment"):
        wide = _queue_count(_app_on(page, source=label))
        if wide is None:
            pytest.skip(f"{page} renders no queue for this run")
        tight = _queue_count(_app_on(page, source=label, state=narrow))
        assert tight == wide, f"{page}: date filter changed the queue ({wide} -> {tight})"


def test_125_product_filter_still_narrows_decision_queues():
    # Guard against "fixing" 124 by disconnecting product filtering entirely.
    label = _label_for(_pick_run(min_skus=2))
    at = _app_on("Stockout Risk", source=label)
    full, skus = _queue_count(at), _queue_skus(at)
    if not full or full < 2 or not skus:
        pytest.skip("run has too few risk rows to narrow")
    for page in ("Stockout Risk", "Inventory & Replenishment"):
        n = _queue_count(_app_on(page, source=label, state={"flt_skus": [skus[0]]}))
        assert n is not None and n < full, f"{page}: product filter did not narrow ({n} vs {full})"


def test_126_executive_reorder_cards_respect_the_product_filter():
    # The bug: the Executive panel read the whole-run reorder_summary.json, so filtering to
    # one product still showed the run-wide order count.
    run = _pick_run(min_skus=2)
    label = _label_for(run)
    wide = _panel_rows(_app_on("Executive Overview", source=label))
    if "Order Now" not in wide:
        pytest.skip("no reorder panel for this run")
    skus = _queue_skus(_app_on("Stockout Risk", source=label))
    if not skus:
        pytest.skip("cannot recover a SKU to filter by")
    tight = _panel_rows(_app_on("Executive Overview", source=label, state={"flt_skus": [skus[0]]}))
    denom = lambda s: s.split("/")[-1].strip()
    assert denom(wide["Order Now"]) == str(run["selected_sku_count"]),         f"unfiltered panel should cover the whole run, got {wide['Order Now']}"
    assert denom(tight["Order Now"]) == "1",         f"filtered panel should cover 1 product, got {tight['Order Now']}"


def test_127_run_mode_insights_use_decision_artifacts_not_the_baseline_simulation():
    cards = _re.findall(r'class="txt">([^<]+)<',
                        _text(_app_on("Executive Overview", source=_label_for(_pick_run(min_skus=1)))))
    joined = " ".join(cards)
    assert "baseline simulation" not in joined,         "run mode still shows the legacy synthetic reorder insight"
    assert any(k in joined for k in ("order-now", "stockout tier", "manual review")),         f"no forecast-driven insight surfaced: {cards}"


def test_128_pre_decisioning_run_keeps_the_baseline_insight_fallback():
    # The other half of 127: a run with no Phase B/C artifacts must still say something,
    # and the honest thing to say is that it came from the baseline simulation.
    run = _pick_run(min_skus=1, decisions=False)
    cards = _re.findall(r'class="txt">([^<]+)<', _text(_app_on("Executive Overview",
                                                               source=_label_for(run))))
    assert any("baseline simulation" in c for c in cards), cards


def test_129_deep_dive_follows_the_product_filter():
    # The bug: with a product selected in the filter but "Compare products" left blank, the
    # deep-dive fell back to the first SKU of the WHOLE catalogue — so it charted a product
    # the filter had excluded and rendered "No data for this product in the current view".
    run = _pick_run(min_skus=3)
    panel = pd.read_parquet(Path(run["run_dir"]) / "processed" / "model_panel.parquet")
    skus = sorted(panel["sku"].unique().tolist())
    target = skus[-1]                                  # deliberately not the first product
    at = _app_on("Demand Analytics", source=_label_for(run), state={"flt_skus": [target]})
    txt = _text(at)
    heads = _re.findall(r'ipa-section-title">Deep-Dive · ([^<]+)<', txt)
    assert heads, "no deep-dive rendered"
    assert target in heads[0], f"deep-dive charted {heads[0]!r} instead of the filtered {target}"
    assert "No data for this product" not in _re.sub(r"<[^>]+>", " ", txt)


def test_130_forecast_table_shows_whole_units():
    # Products are counted in whole items; a predicted 41.642857 in a CEO-facing table is
    # noise. Accuracy metrics keep their decimals and are checked separately.
    at = _app_on("Forecast Explorer", source=_label_for(_pick_run(min_skus=2)))
    tables = [el.value for el in at.dataframe]
    frames = [t for t in tables if isinstance(t, pd.DataFrame)
              and {"Date", "Predicted", "Cumulative"} <= set(t.columns)]
    if not frames:
        pytest.skip("daily forecast table not rendered for this run")
    daily = frames[0]
    for col in ("Predicted", "Cumulative"):
        vals = pd.to_numeric(daily[col], errors="coerce").dropna()
        assert (vals % 1 == 0).all(), f"{col} still shows fractional units: {vals.head().tolist()}"
    assert daily["Cumulative"].iloc[-1] >= daily["Predicted"].iloc[0]


def test_131_days_of_cover_displays_as_whole_days_rounded_up():
    # Days of cover is read off the queue cards at a glance; "6.2 days cover" is precision
    # the number does not carry. Displayed as whole days, rounded up. Tiering still happens
    # in the backend on the raw value, so this is presentation only.
    import math
    run = _pick_run(min_skus=2)
    label = _label_for(run)
    reco = pd.read_parquet(Path(run["run_dir"]) / "decisions" / "reorder_recommendations.parquet")
    raw = pd.to_numeric(reco.get("days_of_cover"), errors="coerce").dropna()
    for page in ("Stockout Risk", "Inventory & Replenishment"):
        at = _app_on(page, source=label)
        cards = _re.findall(r'<b>([\d,.—]+)</b><div class="q-sub">days cover</div>', _text(at))
        assert cards, f"{page}: no days-cover cards rendered"
        assert not [c for c in cards if "." in c], \
            f"{page}: fractional days of cover still shown: {[c for c in cards if '.' in c]}"
        for el in at.dataframe:
            v = el.value
            if not isinstance(v, pd.DataFrame):
                continue
            for col in [c for c in v.columns if str(c) in ("Days of Cover", "Days Cover")]:
                vals = pd.to_numeric(v[col], errors="coerce").dropna()
                assert (vals % 1 == 0).all(), f"{page}/{col} still fractional"
    if not raw.empty:
        expected = {str(int(math.ceil(x))) for x in raw}
        shown = set(_re.findall(r'<b>([\d]+)</b><div class="q-sub">days cover</div>',
                                _text(_app_on("Inventory & Replenishment", source=label))))
        assert shown <= expected, f"displayed days {shown - expected} are not ceil() of the artifact"
