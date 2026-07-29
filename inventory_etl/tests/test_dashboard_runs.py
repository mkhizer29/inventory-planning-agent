"""Phase 5 — dashboard run_service tests.

Tests dashboard/run_service.py directly (never imports dashboard/app.py, which would
execute the Streamlit app). Uses temporary run directories and temporary SQLite DBs;
no real subprocess is launched unless subprocess.Popen is monkeypatched.
"""
import hashlib
import json
import sqlite3
import sys
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
                      created="2026-07-29T09:00:00+00:00"):
    rd = runs_dir / run_id
    rd.mkdir(parents=True)
    _write(rd / "status.json", {"run_id": run_id, "status": "running_lightgbm", "progress_pct": 70,
                                "current_step": "running_lightgbm", "created_at": created})
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
    assert label == f"{PKT_EXPECTED} · Groceries & Pets · Top 10 · completed"


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
