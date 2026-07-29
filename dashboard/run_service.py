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

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
TERMINAL_STATES = {"completed", "completed_with_warnings", "failed"}
RUNNING_STATES = {"created", "selecting_skus", "preparing_data", "running_baseline",
                  "running_holtwinters", "running_lightgbm", "validating_outputs", "ranking_models"}
STEP_LABELS = {
    "created": "Creating run", "selecting_skus": "Selecting products",
    "preparing_data": "Preparing forecasting data", "running_baseline": "Running baseline models",
    "running_holtwinters": "Running Holt-Winters", "running_lightgbm": "Running LightGBM",
    "validating_outputs": "Validating model outputs", "ranking_models": "Ranking models",
    "completed": "Completed", "completed_with_warnings": "Completed with warnings",
    "failed": "Failed",
}
PROGRESS_PCT = {
    "created": 0, "selecting_skus": 10, "preparing_data": 25, "running_baseline": 40,
    "running_holtwinters": 55, "running_lightgbm": 70, "validating_outputs": 85,
    "ranking_models": 92, "completed": 100, "completed_with_warnings": 100, "failed": 100,
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
    }
    # availability flags let the UI show a graceful "unavailable" notice for older runs
    ctx["has_stockout_risk"] = ctx["stockout_risk"].exists()
    ctx["has_stockout_trajectory"] = ctx["stockout_trajectory"].exists()
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
        # Phase B is run-scoped; the legacy fixed-pilot context never has decision artifacts
        "decisions_dir": None, "stockout_risk": None, "stockout_trajectory": None,
        "has_stockout_risk": False, "has_stockout_trajectory": False, "decisioning_status": None,
    }


# ── warehouse reads (read-only) ─────────────────────────────────────────────────────────
def get_latest_sales_date(db_path: "str | Path" = DEFAULT_DB_PATH) -> "date | None":
    """Maximum sales_transactions.transaction_date, read-only. None on any problem."""
    db_path = Path(db_path)
    if not db_path.exists():
        return None
    con = None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        row = con.execute("SELECT MAX(transaction_date) FROM sales_transactions").fetchone()
        if not row or not row[0]:
            return None
        return datetime.strptime(str(row[0])[:10], "%Y-%m-%d").date()
    except Exception:
        return None
    finally:
        if con is not None:
            con.close()


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
                               allow_partial_success: bool = False) -> list[str]:
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
    ]
    if allow_partial_success:
        cmd.append("--allow-partial-success")
    return cmd


def _validate_launch_inputs(category, top_n, as_of_date, selection_cutoff, min_history_days, horizons):
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category must be a non-blank string")
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
    return a.isoformat(), c.isoformat(), tuple(sorted(set(hz)))


def launch_forecast_run(*, category, top_n, as_of_date, selection_cutoff=None,
                        min_history_days=28, horizons=(7, 14), runs_dir=DEFAULT_RUNS_DIR,
                        run_id=None, db_path=DEFAULT_DB_PATH,
                        allow_partial_success: bool = False) -> dict:
    """Validate inputs and launch the orchestrator as a non-blocking subprocess (shell=False).
    Returns run_id/pid/command/launched_at/expected_run_dir. Does NOT create the run dir."""
    as_of, cutoff, hz = _validate_launch_inputs(
        category, top_n, as_of_date, selection_cutoff, min_history_days, horizons)
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
        db_path=db_path, allow_partial_success=allow_partial_success)
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


def format_run_label(record: dict) -> str:
    """e.g. '29 Jul 2026 · 11:03 AM PKT · Groceries & Pets · Top 10 · completed' (Pakistan time)."""
    when = format_local_datetime(record.get("created_at"))
    if when == "—":
        when = str(record.get("created_at") or "?")
    cat = record.get("category") or "?"
    topn = record.get("top_n")
    return f"{when} · {cat} · Top {topn} · {record.get('status')}"


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
