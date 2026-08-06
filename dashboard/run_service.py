"""dashboard/run_service.py — Phase 5 backend service for the dashboard.

Function-based, framework-free glue between the Streamlit UI and the Phase 4
run-aware orchestrator. It is importable and unit-testable WITHOUT importing
dashboard/app.py (no `import streamlit` here).

Responsibilities:
  * discover completed / running / failed runs under runs/
  * generate path-safe run ids and build a safe orchestrator argv
  * launch the orchestrator as a non-blocking subprocess (shell=False)
  * resolve the exact artifact paths for a completed run (context)
  * expose the legacy fixed-pilot context
  * read the latest warehouse sales date (read-only) and eligible categories
  * small pure helpers for horizon filtering / labels / log tails

Safety: never uses shell=True / os.system / string commands; never interpolates
user input into a command; never writes to data/processed or global outputs/.
"""
from __future__ import annotations

import json
import re
import secrets
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo          # stdlib (Python 3.9+) — no new dependency

import pandas as pd

# Persisted timestamps stay UTC (written by the orchestrator); we convert ONLY for display.
DISPLAY_TIMEZONE = ZoneInfo("Asia/Karachi")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_DB_PATH = REPO_ROOT / "inventory_etl" / "output" / "inventory.db"
ORCHESTRATOR_PATH = REPO_ROOT / "src" / "forecast_orchestrator.py"
LAUNCHER_LOG_DIRNAME = ".launcher_logs"

# Top-N ranking metrics, mirrored from src/dynamic_selection so the UI never invents a value.
METRIC_UNITS = "units"
METRIC_STOCKOUT_RISK = "stockout_risk"
SUPPORTED_RANKING_METRICS = (METRIC_UNITS, METRIC_STOCKOUT_RISK)
RANKING_METRIC_LABELS = {
    METRIC_UNITS: "Units sold",
    METRIC_STOCKOUT_RISK: "Stockout risk",
}

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
TERMINAL_STATES = {"completed", "completed_with_warnings", "failed"}
RUNNING_STATES = {"created", "selecting_skus", "preparing_data", "running_baseline",
                  "running_holtwinters", "running_lightgbm", "validating_outputs", "ranking_models",
                  "calculating_stockout_risk", "calculating_reorder_recommendations"}
STEP_LABELS = {
    "created": "Creating run", "selecting_skus": "Selecting products",
    "preparing_data": "Preparing forecasting data", "running_baseline": "Running baseline models",
    "running_holtwinters": "Running Holt-Winters", "running_lightgbm": "Running LightGBM",
    "validating_outputs": "Validating model outputs", "ranking_models": "Ranking models",
    "calculating_stockout_risk": "Calculating stockout risk",
    "calculating_reorder_recommendations": "Calculating reorder recommendations",
    "completed": "Completed", "completed_with_warnings": "Completed with warnings",
    "failed": "Failed",
}
PROGRESS_PCT = {
    "created": 0, "selecting_skus": 10, "preparing_data": 25, "running_baseline": 40,
    "running_holtwinters": 55, "running_lightgbm": 70, "validating_outputs": 85,
    "ranking_models": 92, "calculating_stockout_risk": 96,
    "calculating_reorder_recommendations": 98,
    "completed": 100, "completed_with_warnings": 100, "failed": 100,
}
BASELINE_METHODS = ("last_day_naive", "seasonal_naive_7", "moving_average_7", "moving_average_14")


class RunContextError(RuntimeError):
    """A completed run's artifacts could not be safely resolved."""


# ── id / path helpers ─────────────────────────────────────────────────────────────────
def is_safe_run_id(run_id: str) -> bool:
    return (isinstance(run_id, str) and bool(RUN_ID_RE.match(run_id))
            and run_id not in (".", "..") and not any(c in run_id for c in "/\\")
            and not any(ord(c) < 32 for c in run_id))


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s or "run"


def generate_run_id(category: str, top_n: int, runs_dir: "str | Path" = DEFAULT_RUNS_DIR) -> str:
    """Path-safe unique id: YYYYMMDDTHHMMSSZ_<category-slug>_top<N>_<6-hex>."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{ts}_{_slug(category)}_top{int(top_n)}"
    runs_dir = Path(runs_dir)
    for _ in range(100):
        rid = f"{base}_{secrets.token_hex(3)}"
        if is_safe_run_id(rid) and not (runs_dir / rid).exists():
            return rid
    raise RuntimeError("could not generate a unique run id")   # pragma: no cover


def _within(base: Path, child: Path) -> bool:
    try:
        base_r = base.resolve()
        child_r = child.resolve()
    except OSError:
        return False
    return base_r == child_r or base_r in child_r.parents


# ── JSON-tolerant reading ─────────────────────────────────────────────────────────────
def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:      # missing, partially written, or malformed — never crash discovery
        return {}


# ── run discovery ─────────────────────────────────────────────────────────────────────
def _run_record(run_dir: Path) -> dict:
    status = _read_json(run_dir / "status.json")
    request = _read_json(run_dir / "request.json")
    manifest = _read_json(run_dir / "run_manifest.json")

    state = status.get("status") or manifest.get("status") or "unknown"
    winners = manifest.get("winners_by_horizon") or {}
    rec = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "status": state,
        "progress_pct": status.get("progress_pct", PROGRESS_PCT.get(state, 0)),
        "current_step": status.get("current_step", state),
        "category": request.get("category") or manifest.get("request", {}).get("category"),
        "top_n": request.get("top_n") or manifest.get("request", {}).get("top_n"),
        "selected_sku_count": manifest.get("selected_sku_count"),
        "as_of_date": request.get("as_of_date") or manifest.get("request", {}).get("as_of_date"),
        "created_at": status.get("created_at") or request.get("created_at") or manifest.get("created_at"),
        "completed_at": status.get("completed_at") or manifest.get("completed_at"),
        "failed_at": status.get("failed_at") or manifest.get("failed_at"),
        "operational_model": manifest.get("operational_model"),
        "operational_horizon": manifest.get("operational_horizon"),
        "winners_by_horizon": winners,
        "completed_models": manifest.get("completed_models", []),
        "failed_models": manifest.get("failed_models", []),
        "skipped_models": manifest.get("skipped_models", []),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "duration_seconds": manifest.get("duration_seconds"),
        "error_message": status.get("error_message"),
        "decisioning_status": manifest.get("decisioning_status"),
        # How the Top-N was chosen. Runs created before this feature have no field, and
        # they were all units-ranked, so that is the correct default for them.
        "ranking_metric": (request.get("ranking_metric")
                           or manifest.get("request", {}).get("ranking_metric")
                           or METRIC_UNITS),
        "selection": manifest.get("selection") or {},
    }
    rec["is_terminal"] = state in TERMINAL_STATES
    rec["is_completed"] = state in ("completed", "completed_with_warnings")
    rec["is_failed"] = state == "failed"
    rec["is_running"] = state in RUNNING_STATES
    return rec


def discover_runs(runs_dir: "str | Path" = DEFAULT_RUNS_DIR) -> list[dict]:
    """List immediate run directories under runs_dir (never crashing on bad/partial data).
    Sorted running-first, then newest created_at descending."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []
    records = []
    for child in runs_dir.iterdir():
        if child.name == LAUNCHER_LOG_DIRNAME:
            continue
        try:
            if child.is_symlink() or not child.is_dir():
                continue
        except OSError:
            continue
        if not is_safe_run_id(child.name):
            continue
        records.append(_run_record(child))
    records.sort(key=lambda r: (r.get("created_at") or ""), reverse=True)   # newest first
    records.sort(key=lambda r: 0 if r["is_running"] else 1)                  # running first (stable)
    return records


# ── run context (completed only) + legacy ──────────────────────────────────────────────
def resolve_run_context(run_record: dict) -> dict:
    """Exact, path-safe artifact locations for a COMPLETED run. Raises RunContextError for
    running/failed runs, path traversal, or missing artifacts. Never returns partial data."""
    if not run_record.get("is_completed"):
        raise RunContextError(f"run {run_record.get('run_id')!r} is not completed "
                              f"(status={run_record.get('status')!r})")
    run_dir = Path(run_record["run_dir"])
    if ".." in run_dir.parts:
        raise RunContextError("run directory contains a path-traversal component")
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RunContextError("run directory is missing or a symlink")

    processed = run_dir / "processed"
    outputs = run_dir / "outputs"
    ctx = {
        "mode": "run", "run_id": run_dir.name, "run_dir": run_dir,
        "processed_dir": processed, "outputs_dir": outputs,
        "model_panel": processed / "model_panel.parquet",
        "forecast_frame": processed / "forecast_frame.parquet",
        "inventory_context": processed / "inventory_context.parquet",
        "pilot_manifest": processed / "pilot_manifest.json",
        "combined_scorecard": run_dir / "combined_scorecard.csv",
        "model_ranking": run_dir / "model_ranking.csv",
        "selected_forecasts": run_dir / "selected_forecasts.parquet",
        "status": run_dir / "status.json",
        "request": run_dir / "request.json",
        "run_manifest": run_dir / "run_manifest.json",
        # per-model future files (present only for completed models)
        "future_baseline": outputs / "future_forecast_baseline.parquet",
        "future_holtwinters": outputs / "future_forecast_holtwinters.parquet",
        "future_lightgbm": outputs / "future_forecast_lightgbm.parquet",
        # Phase B decision artifacts (OPTIONAL — absent on runs created before Phase B)
        "decisions_dir": run_dir / "decisions",
        "stockout_risk": run_dir / "decisions" / "stockout_risk.parquet",
        "stockout_trajectory": run_dir / "decisions" / "stockout_trajectory.parquet",
        # Phase C decision artifacts (OPTIONAL — absent on runs created before Phase C)
        "reorder_recommendations": run_dir / "decisions" / "reorder_recommendations.parquet",
        "reorder_summary": run_dir / "decisions" / "reorder_summary.json",
    }
    # availability flags let the UI show a graceful "unavailable" notice for older runs
    ctx["has_stockout_risk"] = ctx["stockout_risk"].exists()
    ctx["has_stockout_trajectory"] = ctx["stockout_trajectory"].exists()
    ctx["reorder_available"] = ctx["reorder_recommendations"].exists() and ctx["reorder_summary"].exists()
    ctx["decisioning_status"] = run_record.get("decisioning_status")
    required = ["model_panel", "forecast_frame", "inventory_context", "pilot_manifest",
               "combined_scorecard", "model_ranking", "selected_forecasts", "run_manifest"]
    for key, p in ctx.items():
        if isinstance(p, Path) and not _within(run_dir, p):
            raise RunContextError(f"resolved path escapes run directory: {key}")
    missing = [k for k in required if not ctx[k].exists()]
    if missing:
        raise RunContextError(f"run {run_dir.name} is missing artifact(s): {missing}")
    return ctx


def legacy_context() -> dict:
    proc = REPO_ROOT / "data" / "processed"
    outputs = REPO_ROOT / "outputs"
    return {
        "mode": "legacy", "run_id": None, "run_dir": None,
        "processed_dir": proc, "outputs_dir": outputs,
        "model_panel": proc / "model_panel.parquet",
        "forecast_frame": proc / "forecast_frame.parquet",
        "inventory_context": proc / "inventory_context.parquet",
        "pilot_manifest": proc / "pilot_manifest.json",
        "combined_scorecard": None, "model_ranking": None, "selected_forecasts": None,
        "run_manifest": None,
        # Phase B/C are run-scoped; the legacy fixed-pilot context never has decision artifacts
        "decisions_dir": None, "stockout_risk": None, "stockout_trajectory": None,
        "reorder_recommendations": None, "reorder_summary": None,
        "has_stockout_risk": False, "has_stockout_trajectory": False,
        "reorder_available": False, "decisioning_status": None,
    }


# ── warehouse reads (read-only) ─────────────────────────────────────────────────────────
DEFAULT_MIN_SHARE_OF_MEDIAN_DAILY_UNITS = 0.10


def _min_share_of_median_daily_units() -> float:
    """Configured extract-tail threshold; falls back to the module default on any problem."""
    try:
        import yaml                                        # noqa: PLC0415 - optional at import time
        cfg = yaml.safe_load(
            (REPO_ROOT / "inventory_etl" / "config" / "config.yaml").read_text(encoding="utf-8")) or {}
        share = float((cfg.get("sales_calendar") or {})
                      .get("min_share_of_median_daily_units",
                           DEFAULT_MIN_SHARE_OF_MEDIAN_DAILY_UNITS))
        return share if 0.0 <= share < 1.0 else DEFAULT_MIN_SHARE_OF_MEDIAN_DAILY_UNITS
    except Exception:
        return DEFAULT_MIN_SHARE_OF_MEDIAN_DAILY_UNITS


def _daily_totals(db_path: "str | Path") -> "list[tuple[str, float]]":
    """[(date, total units)] ascending, read-only. Empty list on any problem."""
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    con = None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT transaction_date, COALESCE(SUM(quantity_sold), 0) FROM sales_transactions "
            "WHERE transaction_date IS NOT NULL GROUP BY transaction_date "
            "ORDER BY transaction_date").fetchall()
        return [(str(d)[:10], float(u or 0.0)) for d, u in rows if d]
    except Exception:
        return []
    finally:
        if con is not None:
            con.close()


def sales_date_diagnostics(db_path: "str | Path" = DEFAULT_DB_PATH) -> dict:
    """Describe the usable sales window and any trailing extract tail that was discounted.

    Keys: ``raw_max`` (plain MAX(transaction_date)), ``usable_max`` (latest day with a
    plausible full day of demand), ``ignored_dates``, ``median_daily_units``, ``threshold``.
    All dates are ``datetime.date`` or None. Never raises.
    """
    empty = {"raw_max": None, "usable_max": None, "ignored_dates": [],
             "median_daily_units": None, "threshold": None}
    totals = _daily_totals(db_path)
    if not totals:
        return empty

    def _d(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            return None

    units = sorted(u for _, u in totals)
    n = len(units)
    median = units[n // 2] if n % 2 else (units[n // 2 - 1] + units[n // 2]) / 2.0
    # Floor of 1 unit keeps tiny fixtures and genuinely low-volume warehouses intact:
    # the guard should only ever discount a near-empty tail, never a real trading day.
    threshold = max(1.0, _min_share_of_median_daily_units() * median)

    raw_max = _d(totals[-1][0])
    usable_max, ignored = None, []
    for d, u in reversed(totals):                          # walk back from the newest day
        if u >= threshold:
            usable_max = _d(d)
            break
        ignored.append(_d(d))
    return {"raw_max": raw_max, "usable_max": usable_max,
            "ignored_dates": [d for d in reversed(ignored) if d is not None],
            "median_daily_units": float(median), "threshold": float(threshold)}


def get_latest_sales_date(db_path: "str | Path" = DEFAULT_DB_PATH) -> "date | None":
    """Latest transaction_date carrying a plausible full day of sales. None on any problem.

    NOT a plain MAX(transaction_date): a partially-extracted warehouse leaves trailing dates
    holding a few stray rows, and defaulting the as-of date onto one of those produces a
    locked-holdout window with zero actual demand — an undefined WAPE, and a run that fails
    at model ranking. Trailing days below ``sales_calendar.min_share_of_median_daily_units``
    of the median daily volume are treated as extract tail. Use
    :func:`sales_date_diagnostics` to show what was discounted and why.
    """
    return sales_date_diagnostics(db_path)["usable_max"]


def list_categories(db_path: "str | Path" = DEFAULT_DB_PATH, selection_cutoff=None,
                    min_history_days: int = 28) -> pd.DataFrame:
    """Warehouse categories eligible for selection, via the Phase 1 API (read-only)."""
    import dynamic_selection as dsel
    cutoff = selection_cutoff or get_latest_sales_date(db_path)
    if cutoff is None:
        return pd.DataFrame(columns=["category", "eligible_sku_count", "historical_units",
                                     "history_start", "history_end"])
    return dsel.list_eligible_categories(db_path, str(cutoff), int(min_history_days))


# ── command building + launch ───────────────────────────────────────────────────────────
def build_orchestrator_command(*, category, top_n, as_of_date, selection_cutoff,
                               min_history_days, horizons, runs_dir, run_id, db_path,
                               allow_partial_success: bool = False,
                               ranking_metric: str = METRIC_UNITS) -> list[str]:
    """A safe argv (list) for the orchestrator. Always begins with the active interpreter
    and the orchestrator script; never a shell string; user input only as list elements."""
    cmd = [
        sys.executable, str(ORCHESTRATOR_PATH),
        "--category", str(category),
        "--top-n", str(int(top_n)),
        "--as-of-date", str(as_of_date),
        "--selection-cutoff", str(selection_cutoff),
        "--min-history-days", str(int(min_history_days)),
        "--horizons", *[str(int(h)) for h in horizons],
        "--runs-dir", str(runs_dir),
        "--run-id", str(run_id),
        "--db-path", str(db_path),
        "--ranking-metric", str(ranking_metric),
    ]
    if allow_partial_success:
        cmd.append("--allow-partial-success")
    return cmd


def _validate_launch_inputs(category, top_n, as_of_date, selection_cutoff, min_history_days,
                            horizons, ranking_metric=METRIC_UNITS):
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category must be a non-blank string")
    if ranking_metric not in SUPPORTED_RANKING_METRICS:
        raise ValueError(f"ranking_metric must be one of {SUPPORTED_RANKING_METRICS}")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not (1 <= top_n <= 100):
        raise ValueError("top_n must be an integer in 1..100")
    a = datetime.strptime(str(as_of_date), "%Y-%m-%d").date()
    c = a if selection_cutoff in (None, "") else datetime.strptime(str(selection_cutoff), "%Y-%m-%d").date()
    if c > a:
        raise ValueError("selection_cutoff must not be after as_of_date")
    if isinstance(min_history_days, bool) or not isinstance(min_history_days, int) or min_history_days < 1:
        raise ValueError("min_history_days must be an integer >= 1")
    hz = tuple(int(h) for h in horizons)
    if not hz or any(h not in (7, 14) for h in hz):
        raise ValueError("horizons must be a non-empty subset of {7, 14}")
    return a.isoformat(), c.isoformat(), tuple(sorted(set(hz))), str(ranking_metric)


def launch_forecast_run(*, category, top_n, as_of_date, selection_cutoff=None,
                        min_history_days=28, horizons=(7, 14), runs_dir=DEFAULT_RUNS_DIR,
                        run_id=None, db_path=DEFAULT_DB_PATH,
                        allow_partial_success: bool = False,
                        ranking_metric: str = METRIC_UNITS) -> dict:
    """Validate inputs and launch the orchestrator as a non-blocking subprocess (shell=False).
    Returns run_id/pid/command/launched_at/expected_run_dir. Does NOT create the run dir."""
    as_of, cutoff, hz, metric = _validate_launch_inputs(
        category, top_n, as_of_date, selection_cutoff, min_history_days, horizons,
        ranking_metric)
    runs_dir = Path(runs_dir)
    db_path = Path(db_path)
    if not db_path.exists():
        raise ValueError(f"database not found: {db_path}")
    rid = run_id or generate_run_id(category, top_n, runs_dir)
    if not is_safe_run_id(rid):
        raise ValueError(f"unsafe run_id: {run_id!r}")
    run_dir = runs_dir / rid
    if run_dir.exists():
        raise ValueError(f"run_id already exists: {run_dir}")   # orchestrator would reject it too

    cmd = build_orchestrator_command(
        category=category, top_n=top_n, as_of_date=as_of, selection_cutoff=cutoff,
        min_history_days=min_history_days, horizons=hz, runs_dir=runs_dir, run_id=rid,
        db_path=db_path, allow_partial_success=allow_partial_success,
        ranking_metric=metric)
    if cmd[0] != sys.executable:                                # never an arbitrary executable
        raise ValueError("refusing to launch: interpreter is not sys.executable")

    log_dir = runs_dir / LAUNCHER_LOG_DIRNAME                    # runs/ is gitignored; never outputs/
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{rid}.log"
    logf = open(log_path, "w", encoding="utf-8")               # noqa: SIM115 — handed to the child process
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=logf,
                            stderr=subprocess.STDOUT, shell=False)
    return {
        "run_id": rid, "pid": proc.pid, "command": cmd,
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "expected_run_dir": str(run_dir), "log_path": str(log_path),
    }


# ── model-file mapping for run-mode Forecast Explorer ────────────────────────────────────
def read_model_column(path: "str | Path") -> "str | None":
    """The single `model` value in a forecast parquet (e.g. the baseline method actually used)."""
    try:
        df = pd.read_parquet(path, columns=["model"])
    except Exception:
        return None
    vals = [str(v) for v in df["model"].dropna().unique()]
    return vals[0] if len(vals) == 1 else None


def run_model_options(context: dict, run_record: dict) -> list[dict]:
    """Ordered Forecast-Explorer model options for a completed run. Only includes a model
    whose future file exists. Each option: {label, key, path, model}."""
    outputs = context["outputs_dir"]
    completed = set(run_record.get("completed_models") or [])
    opts: list[dict] = []
    op_model = run_record.get("operational_model")
    sel = context["selected_forecasts"]
    if sel.exists() and op_model:
        opts.append({"label": f"Operational Winner — {op_model}", "key": "operational",
                     "path": sel, "model": op_model})
    if "baseline" in completed and context["future_baseline"].exists():
        method = read_model_column(context["future_baseline"]) or "baseline"
        opts.append({"label": f"Best Baseline — {method}", "key": "baseline",
                     "path": context["future_baseline"], "model": method})
    if "holtwinters" in completed and context["future_holtwinters"].exists():
        opts.append({"label": "Holt-Winters", "key": "holtwinters",
                     "path": context["future_holtwinters"], "model": "holtwinters"})
    if "lightgbm" in completed and context["future_lightgbm"].exists():
        opts.append({"label": "LightGBM", "key": "lightgbm",
                     "path": context["future_lightgbm"], "model": "lightgbm"})
    return opts


# ── forecast horizon filtering (run vs legacy) ───────────────────────────────────────────
def normalize_history_window(min_date, max_date, raw_from, raw_to):
    """Clamp a requested From/To into the dataset window.

    Returns ``(date_from, date_to, error)`` as plain ``datetime.date`` values, both inside
    [min_date, max_date] and satisfying ``date_from <= date_to``. Null / boolean / stale
    payloads (e.g. the retired single-widget tuple) fall back to the full window. When the
    user picks From after To, the full window is returned together with an error message so
    the caller can show a notice instead of applying an invalid filter.
    """
    def _coerce(v):
        if v is None or isinstance(v, bool) or isinstance(v, (tuple, list)):
            return None
        try:
            ts = pd.to_datetime(v, errors="coerce")
        except (TypeError, ValueError):
            return None
        return None if pd.isna(ts) else ts.date()

    lo, hi = _coerce(min_date), _coerce(max_date)
    if lo is None or hi is None:
        return None, None, None
    if lo > hi:
        lo, hi = hi, lo
    d_from = _coerce(raw_from) or lo
    d_to = _coerce(raw_to) or hi
    d_from = min(max(d_from, lo), hi)
    d_to = min(max(d_to, lo), hi)
    if d_from > d_to:
        return lo, hi, "“From” is after “To” — showing the full available period instead."
    return d_from, d_to, None


def filter_historical_frame(df: pd.DataFrame, *, date_from, date_to) -> pd.DataFrame:
    """Inclusively restrict a HISTORICAL frame to [date_from, date_to] on its ``date`` column.

    Display-only: it never touches the run, its as-of date, the horizons or any artifact, and
    it is only ever applied to historical frames (model_panel) — never to future forecasts,
    stockout risk or reorder recommendations. Returns a new frame; the input is not mutated.
    An empty result is returned as an empty frame with the original columns.
    """
    if df is None or getattr(df, "empty", True):
        return df
    if "date" not in getattr(df, "columns", []) or date_from is None or date_to is None:
        return df
    out = df.copy()                                  # never mutate the caller's frame
    dates = pd.to_datetime(out["date"], errors="coerce")
    start, end = pd.Timestamp(date_from), pd.Timestamp(date_to)
    if start > end:                                  # defensive: caller should have normalized
        start, end = end, start
    keep = dates.notna() & (dates >= start) & (dates <= end.normalize() + pd.Timedelta(days=1)
                                               - pd.Timedelta(nanoseconds=1))
    return out.loc[keep].reset_index(drop=True)


def apply_horizon_filter(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Restrict a future-forecast frame to `horizon` days.

    Run-mode files carry forecast_horizon_day -> filter <= horizon (primary path).
    Legacy files without it fall back to per-SKU head(horizon) by date. Guarantees at most
    one row per (sku, date); duplicates are dropped deterministically with the frame's order.
    """
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    if "forecast_horizon_day" in out.columns:
        out = out[pd.to_numeric(out["forecast_horizon_day"], errors="coerce") <= int(horizon)]
    else:
        out = (out.sort_values("date").groupby("sku", group_keys=False).head(int(horizon))
               if "sku" in out.columns else out.sort_values("date").head(int(horizon)))
    if {"sku", "date"}.issubset(out.columns):
        out = out.drop_duplicates(subset=["sku", "date"], keep="first")
    return out.reset_index(drop=True)


# ── misc UI helpers ──────────────────────────────────────────────────────────────────────
def step_label(step: str) -> str:
    return STEP_LABELS.get(step, str(step))


def _parse_timestamp(value):
    """Parse an event timestamp: datetime/Timestamp, ISO with +00:00, 'Z', or another offset."""
    if value is None:
        return None
    if isinstance(value, datetime):          # includes pandas.Timestamp
        return value
    s = str(value).strip()
    if not s:
        return None
    if s[-1] in ("Z", "z"):                  # '...Z' -> explicit UTC offset (works on 3.9/3.10 too)
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    try:                                     # tolerant fallback for odd-but-real formats
        parsed = pd.to_datetime(s)
        return parsed.to_pydatetime() if hasattr(parsed, "to_pydatetime") else parsed
    except Exception:
        return None


def format_local_datetime(value, *, include_date: bool = True, include_time: bool = True,
                          include_timezone: bool = True) -> str:
    """Render a UTC/offset-aware event timestamp in Asia/Karachi, e.g. '29 Jul 2026 · 11:03 AM PKT'.

    * timezone-AWARE input (``+00:00``, ``Z``, any offset) is converted to Asia/Karachi;
    * NAIVE input is shown with its clock time untouched — never silently shifted by +5h —
      because legacy values (e.g. pilot_manifest.generated_at) may already be local;
    * null / blank / unparseable input returns ``"—"``.

    Use this only for EVENT timestamps. Plain business dates (as_of_date, selection_cutoff,
    forecast dates, historical sales dates) are not UTC instants and must not be converted.
    """
    dt = _parse_timestamp(value)
    if dt is None:
        return "—"
    if dt.tzinfo is not None:
        dt = dt.astimezone(DISPLAY_TIMEZONE)
    parts = []
    if include_date:
        parts.append(dt.strftime("%d %b %Y"))
    if include_time:
        stamp = dt.strftime("%I:%M %p")
        if include_timezone:
            stamp = f"{stamp} {dt.tzname() or 'PKT'}"
        parts.append(stamp)
    elif include_timezone and include_date:
        pass                                  # a bare date carries no timezone
    return " · ".join(parts) if parts else "—"


STATUS_SYMBOLS = {"completed": "✓", "completed_with_warnings": "⚠", "failed": "✕"}
# Marks a Top-N chosen by stockout risk. Units-ranked runs stay unmarked so the hundreds of
# pre-existing runs read exactly as they always did.
RISK_RANKED_SYMBOL = "⚡"


def ranking_metric_label(metric) -> str:
    """Human label for a ranking metric; unknown/missing reads as the units default."""
    return RANKING_METRIC_LABELS.get(str(metric or METRIC_UNITS), str(metric))


def is_risk_ranked(record: dict) -> bool:
    return str((record or {}).get("ranking_metric") or METRIC_UNITS) == METRIC_STOCKOUT_RISK


def format_run_label_full(record: dict) -> str:
    """Complete label for tooltips / detail panels:
    '29 Jul 2026 · 11:03 AM PKT · Groceries & Pets · Top 10 by stockout risk · completed'."""
    when = format_local_datetime(record.get("created_at"))
    if when == "—":
        when = str(record.get("created_at") or "?")
    cat = record.get("category") or "?"
    top = f"Top {record.get('top_n')}"
    if is_risk_ranked(record):
        top += " by stockout risk"
    return f"{when} · {cat} · {top} · {record.get('status')}"


def format_run_label_short(record: dict, *, disambiguate: bool = False) -> str:
    """Compact selectbox label that fits the sidebar: '31 Jul · Groceries & Pets · Top 10 · ✓'.

    Deliberately omits the full timestamp, 'PKT', the operational model, the complete run id
    and the long status word — those live in the tooltip and the Run details expander.
    ``disambiguate`` appends the run id's last 6 characters when two runs would otherwise
    collide (same day + category + Top N).
    """
    when = format_local_datetime(record.get("created_at"), include_time=False,
                                 include_timezone=False)
    day = "?" if when == "—" else when[:6].strip()          # '29 Jul 2026' -> '29 Jul'
    cat = record.get("category") or "?"
    top = f"Top {record.get('top_n')}"
    if is_risk_ranked(record):
        top = f"{top} {RISK_RANKED_SYMBOL}"
    parts = [day, str(cat), top]
    sym = STATUS_SYMBOLS.get(str(record.get("status")))
    if sym:
        parts.append(sym)
    if disambiguate:
        parts.append(str(record.get("run_id") or "")[-6:])
    return " · ".join(p for p in parts if p)


def build_short_labels(records: "list[dict]") -> dict:
    """run_id -> UNIQUE short label. Collisions get the run-id suffix appended so two runs
    from the same day/category/Top-N stay distinguishable in the selectbox."""
    counts: dict[str, int] = {}
    for r in records:
        counts[format_run_label_short(r)] = counts.get(format_run_label_short(r), 0) + 1
    out: dict = {}
    for r in records:
        base = format_run_label_short(r)
        label = format_run_label_short(r, disambiguate=True) if counts.get(base, 0) > 1 else base
        while label in out.values():                        # last-resort uniqueness guard
            label = f"{label}·"
        out[r.get("run_id")] = label
    return out


def format_run_label(record: dict) -> str:
    """Backwards-compatible alias for the full label (kept for existing call sites)."""
    return format_run_label_full(record)


# ── Phase B stockout-risk: pure dashboard helpers (no Streamlit) ─────────────────────────
# Severity ordering for the priority queue. `unknown` is LAST and is deliberately kept
# separate from healthy/low — an un-assessable SKU is not "safe".
# "medium" is the engine's emitted tier; "watch" is the display synonym — both rank together.
RISK_TIER_SEVERITY = {"critical": 0, "high": 1, "watch": 2, "medium": 2,
                      "low": 3, "healthy": 3, "unknown": 4}
# Display tone per tier (maps to the dashboard TONES / semantic colors). unknown -> slate.
RISK_TIER_TONE = {"critical": "red", "high": "amber", "watch": "blue", "medium": "blue",
                  "low": "success", "healthy": "success", "unknown": "slate"}


def risk_severity_rank(tier) -> int:
    """Lower = more urgent. Unknown ranks last (4), never as healthy."""
    return RISK_TIER_SEVERITY.get(str(tier).strip().lower(), 4)


def risk_tier_tone(tier) -> str:
    return RISK_TIER_TONE.get(str(tier).strip().lower(), "slate")


def full_product_label(name, sku) -> str:
    """Complete, never-truncated 'Name (SKU)' label for tooltips / deep-dive headings.
    Falls back to the SKU alone when no distinct name is available."""
    nm = None if (name is None or (isinstance(name, float) and pd.isna(name))) else str(name).strip()
    if nm and nm != str(sku):
        return f"{nm} ({sku})"
    return str(sku)


def sort_risk_queue(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic priority order: severity, then P(stockout) desc, projected date asc
    (nulls last), revenue-at-risk desc, product name asc. Never mutates the input."""
    if df is None or df.empty:
        return df.copy() if df is not None else df
    d = df.copy()
    d["_sev"] = d["overall_risk_tier"].map(risk_severity_rank)
    d["_prob"] = pd.to_numeric(d.get("stockout_probability"), errors="coerce").fillna(-1.0)
    proj = pd.to_datetime(d.get("projected_stockout_date"), errors="coerce")
    # method="min" keeps equal dates tied so the later keys (revenue desc, name asc) break
    # the tie deterministically; na_option="bottom" ranks null projected dates last.
    d["_projkey"] = proj.rank(method="min", na_option="bottom")            # ascending, nulls last
    d["_rev"] = pd.to_numeric(d.get("estimated_revenue_at_risk"), errors="coerce").fillna(-1.0)
    d["_name"] = (d["sku_name"].astype(str) if "sku_name" in d.columns else d["sku"].astype(str))
    d = d.sort_values(by=["_sev", "_prob", "_projkey", "_rev", "_name"],
                      ascending=[True, False, True, False, True], kind="mergesort")
    return d.drop(columns=["_sev", "_prob", "_projkey", "_rev", "_name"]).reset_index(drop=True)


def filter_risk_queue(df: pd.DataFrame, *, tier=None, query=None,
                      projected_only: bool = False, review_only: bool = False) -> pd.DataFrame:
    """Apply the compact queue filters. `tier` None/'all' keeps all tiers; `query` matches
    SKU or product name (case-insensitive substring). Never mutates the input."""
    if df is None or df.empty:
        return df.copy() if df is not None else df
    d = df
    if tier and str(tier).strip().lower() not in ("all", ""):
        d = d[d["overall_risk_tier"].astype(str).str.lower() == str(tier).strip().lower()]
    if query and str(query).strip():
        q = str(query).strip().lower()
        mask = d["sku"].astype(str).str.lower().str.contains(q, regex=False)
        if "sku_name" in d.columns:
            mask = mask | d["sku_name"].astype(str).str.lower().str.contains(q, regex=False, na=False)
        d = d[mask]
    if projected_only:
        d = d[d["projected_stockout_date"].notna()]
    if review_only:
        d = d[d["manual_review_required"].astype(bool)]
    return d.reset_index(drop=True)


def risk_revenue_at_risk_total(df: pd.DataFrame) -> tuple[float, int]:
    """(sum of non-null estimated_revenue_at_risk, count of SKUs with a null value).
    Null revenue is NEVER treated as zero in the sum."""
    if df is None or df.empty or "estimated_revenue_at_risk" not in df.columns:
        return 0.0, 0
    s = pd.to_numeric(df["estimated_revenue_at_risk"], errors="coerce")
    return float(s.dropna().sum()), int(s.isna().sum())


def trajectory_for_sku(traj: pd.DataFrame, sku: str, horizon: "int | None" = None
                       ) -> tuple[pd.DataFrame, list[str]]:
    """Filter a trajectory frame to ONE sku and (optionally) the active horizon, sorted by
    date. De-duplicates any repeated (sku, date) rows with an explicit warning. Returns
    (frame, warnings)."""
    warnings: list[str] = []
    if traj is None or traj.empty:
        return (traj.copy() if traj is not None else pd.DataFrame()), warnings
    d = traj[traj["sku"].astype(str) == str(sku)].copy()
    d["date"] = pd.to_datetime(d["date"])
    if horizon is not None and "forecast_horizon_day" in d.columns:
        d = d[pd.to_numeric(d["forecast_horizon_day"], errors="coerce") <= int(horizon)]
    if d.duplicated(["sku", "date"]).any():
        warnings.append(f"duplicate (sku, date) trajectory rows for {sku} were de-duplicated")
        d = d.drop_duplicates(["sku", "date"], keep="first")
    return d.sort_values("date").reset_index(drop=True), warnings


# ── Phase C: reorder recommendations (pure, framework-free display helpers) ───────────────
# Buyer-facing queue priority (order_now first). Matches reorder_recommendations.ACTION_PRIORITY.
REORDER_ACTION_RANK = {"order_now": 0, "vendor_follow_up": 1, "manual_review": 2,
                       "monitor": 3, "no_order": 4}
REORDER_ACTION_TONE = {"order_now": "red", "vendor_follow_up": "blue", "manual_review": "amber",
                       "monitor": "slate", "no_order": "success"}
REORDER_ACTION_LABEL = {"order_now": "Order now", "vendor_follow_up": "Vendor follow-up",
                        "manual_review": "Manual review", "monitor": "Monitor", "no_order": "No order"}


def reorder_action_rank(action) -> int:
    """Lower = higher on the buyer's queue. Unknown actions rank last."""
    return REORDER_ACTION_RANK.get(str(action).strip().lower(), 5)


def reorder_action_tone(action) -> str:
    return REORDER_ACTION_TONE.get(str(action).strip().lower(), "slate")


def reorder_action_label(action) -> str:
    a = str(action).strip().lower()
    return REORDER_ACTION_LABEL.get(a, a.replace("_", " ").capitalize() or "—")


def sort_reorder_queue(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic buyer priority: action priority → risk severity → P(stockout) desc →
    projected date asc (nulls last) → proposed purchase value desc → product name asc → sku asc.
    Never mutates the input."""
    if df is None or df.empty:
        return df.copy() if df is not None else df
    d = df.copy()
    d["_arank"] = d["action"].map(reorder_action_rank)
    d["_sev"] = d["overall_risk_tier"].map(risk_severity_rank)
    d["_prob"] = pd.to_numeric(d.get("stockout_probability"), errors="coerce").fillna(-1.0)
    proj = pd.to_datetime(d.get("projected_stockout_date"), errors="coerce")
    d["_projkey"] = proj.rank(method="min", na_option="bottom")            # ascending, nulls last
    d["_pv"] = pd.to_numeric(d.get("recommended_purchase_value"), errors="coerce").fillna(-1.0)
    d["_name"] = (d["sku_name"].astype(str) if "sku_name" in d.columns else d["sku"].astype(str))
    d["_sku"] = d["sku"].astype(str)
    d = d.sort_values(by=["_arank", "_sev", "_prob", "_projkey", "_pv", "_name", "_sku"],
                      ascending=[True, True, False, True, False, True, True], kind="mergesort")
    return d.drop(columns=["_arank", "_sev", "_prob", "_projkey", "_pv", "_name", "_sku"]).reset_index(drop=True)


def filter_reorder_queue(df: pd.DataFrame, *, action=None, tier=None, query=None,
                         approval_only: bool = False, assumed_only: bool = False) -> pd.DataFrame:
    """Apply the page-local reorder filters. `action`/`tier` None or 'all' keep everything;
    `query` matches SKU or product name (case-insensitive substring). Never mutates the input."""
    if df is None or df.empty:
        return df.copy() if df is not None else df
    d = df
    if action and str(action).strip().lower() not in ("all", ""):
        d = d[d["action"].astype(str).str.lower() == str(action).strip().lower()]
    if tier and str(tier).strip().lower() not in ("all", ""):
        d = d[d["overall_risk_tier"].astype(str).str.lower() == str(tier).strip().lower()]
    if query and str(query).strip():
        q = str(query).strip().lower()
        mask = d["sku"].astype(str).str.lower().str.contains(q, regex=False)
        if "sku_name" in d.columns:
            mask = mask | d["sku_name"].astype(str).str.lower().str.contains(q, regex=False, na=False)
        d = d[mask]
    if approval_only:
        d = d[d["approval_required"].astype(bool)]
    if assumed_only:
        flags = d["assumption_flags"].astype(str).str.lower()
        d = d[flags.str.contains("assumed", regex=False) | flags.str.contains("imputed", regex=False)
              | flags.str.contains("synthetic", regex=False)]
    return d.reset_index(drop=True)


def reorder_purchase_value_total(df: pd.DataFrame) -> tuple[float, int]:
    """(sum of non-null recommended_purchase_value, count of order_now rows with a null value).
    Null purchase value is NEVER treated as zero in the sum."""
    if df is None or df.empty or "recommended_purchase_value" not in df.columns:
        return 0.0, 0
    s = pd.to_numeric(df["recommended_purchase_value"], errors="coerce")
    missing = 0
    if "action" in df.columns:
        order_now = df["action"].astype(str).str.lower() == "order_now"
        missing = int((s.isna() & order_now).sum())
    return float(s.dropna().sum()), missing


def tail_log(path: "str | Path", max_lines: int = 200) -> str:
    """Return at most the final `max_lines` of a log file as PLAIN text (never HTML)."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-int(max_lines):])
