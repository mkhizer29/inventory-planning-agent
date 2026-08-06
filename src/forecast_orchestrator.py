"""forecast_orchestrator.py — Phase 4: the run-aware forecasting orchestrator.

ONE backend command that, for a chosen category + Top-N, produces a fully isolated,
reproducible forecasting run under ``runs/<run_id>/``:

  1. validate the request           6. run Baselines / Holt-Winters / LightGBM
  2. create an isolated run dir      7. validate each model's outputs
  3. record the request             8. verify all dataset fingerprints match
  4. dynamically select SKUs        9. combine scorecards + rank per horizon
  5. prepare run-specific data      10. write selected forecasts + full manifest

It NEVER modifies the warehouse, the fixed pilot files, global data/processed, or
global outputs — every write lands inside the run directory. Selection, preparation
and the three models are all invoked through their PYTHON APIs (no subprocesses).

Failure policy (documented, single):
  * Request validation errors are raised BEFORE any run directory is created.
  * Once the run directory exists, the pipeline NEVER raises to the caller: it records
    the failure, writes status="failed" + a failed run_manifest.json, and RETURNS that
    manifest. The CLI inspects the returned status and exits non-zero when failed.

Function-based; no framework classes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import model_contract as mc          # noqa: E402
import dynamic_selection as dsel     # noqa: E402
import prepare_pilot_data as prep    # noqa: E402
import baselines                     # noqa: E402
import holtwinters                   # noqa: E402
import lgbm_global                   # noqa: E402
import stockout_risk                 # noqa: E402  (Phase B — forecast-driven stockout risk)
import reorder_recommendations       # noqa: E402  (Phase C — forecast-driven reorder recommendations)
import decision_contract as dc       # noqa: E402

DEFAULT_DB = REPO_ROOT / "inventory_etl" / "output" / "inventory.db"
MODEL_ORDER = ("baseline", "holtwinters", "lightgbm")
VALID_HORIZONS = (7, 14)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")     # a real lowercase SHA-256, not any 64-char string
BASELINE_METHODS = ("last_day_naive", "seasonal_naive_7", "moving_average_7", "moving_average_14")
MODEL_IDENTITY = {"holtwinters": "holtwinters", "lightgbm": "lightgbm"}   # single-identity models
LOCKED = "locked_holdout"

# per-model artifact filenames (written into <run>/outputs/)
MODEL_ARTIFACTS: dict[str, dict[str, str]] = {
    "baseline": {"summary": "baseline_run_summary.json", "scorecard": "baseline_scorecard.csv",
                 "future": "future_forecast_baseline.parquet",
                 "backtest": "baseline_backtest_predictions.parquet"},
    "holtwinters": {"summary": "holtwinters_run_summary.json", "scorecard": "holtwinters_scorecard.csv",
                    "future": "future_forecast_holtwinters.parquet",
                    "backtest": "holtwinters_backtest_predictions.parquet",
                    "selection": "holtwinters_model_selection.json"},
    "lightgbm": {"summary": "lightgbm_run_summary.json", "scorecard": "lightgbm_scorecard.csv",
                 "future": "future_forecast_lightgbm.parquet",
                 "backtest": "lightgbm_backtest_predictions.parquet"},
}

STEP_PROGRESS = {
    "created": 0, "selecting_skus": 10, "preparing_data": 25,
    "running_baseline": 40, "running_holtwinters": 55, "running_lightgbm": 70,
    "validating_outputs": 85, "ranking_models": 92, "calculating_stockout_risk": 96,
    "calculating_reorder_recommendations": 98,
    "completed": 100, "completed_with_warnings": 100, "failed": 100,
}


class RequestError(ValueError):
    """Raised for an invalid request BEFORE any run directory is created."""


# ── small utilities ──────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s or "run"


def _generate_run_id(category: str, top_n: int) -> str:
    import secrets
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{_slugify(category)}_top{top_n}_{secrets.token_hex(3)}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _software_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": sys.version.split()[0]}
    for pkg in ("pandas", "numpy", "lightgbm", "statsmodels", "scikit-learn", "pyarrow"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


# ── request validation (runs BEFORE the run directory is created) ─────────────────────
def _valid_iso_date(s: str) -> str:
    datetime.strptime(str(s), "%Y-%m-%d")     # raises ValueError on a bad date
    return str(s)


def validate_request(*, category, top_n, as_of_date, selection_cutoff, min_history_days,
                     horizons, runs_dir, run_id, db_path, allow_partial_success,
                     skip_models, start_date=None) -> dict:
    """Normalize + validate every input. Raises RequestError on any problem. No I/O side effects."""
    if not isinstance(category, str) or not category.strip():
        raise RequestError("category must be a non-blank string")
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise RequestError("top_n must be an integer")
    if not (1 <= top_n <= 100):
        raise RequestError("top_n must be between 1 and 100")
    try:
        as_of = _valid_iso_date(as_of_date)
    except ValueError:
        raise RequestError(f"as_of_date is not an ISO date: {as_of_date!r}")
    cutoff = as_of if selection_cutoff in (None, "") else str(selection_cutoff)
    try:
        cutoff = _valid_iso_date(cutoff)
    except ValueError:
        raise RequestError(f"selection_cutoff is not an ISO date: {selection_cutoff!r}")
    if datetime.strptime(cutoff, "%Y-%m-%d") > datetime.strptime(as_of, "%Y-%m-%d"):
        raise RequestError("selection_cutoff must not be after as_of_date")
    # Optional hard lower bound on history. None = use everything the warehouse holds
    # (the previous, unchanged behaviour). Passed straight to prepare_pilot_data's
    # --start-date, which filters before any feature engineering.
    start = None
    if start_date not in (None, ""):
        try:
            start = _valid_iso_date(str(start_date))
        except ValueError:
            raise RequestError(f"start_date is not an ISO date: {start_date!r}")
        if datetime.strptime(start, "%Y-%m-%d") > datetime.strptime(as_of, "%Y-%m-%d"):
            raise RequestError("start_date must not be after as_of_date")
    if isinstance(min_history_days, bool) or not isinstance(min_history_days, int) or min_history_days < 1:
        raise RequestError("min_history_days must be an integer >= 1")
    horizons = tuple(int(h) for h in horizons)
    if not horizons or any(h not in VALID_HORIZONS for h in horizons):
        raise RequestError(f"horizons must be a non-empty subset of {VALID_HORIZONS}")
    horizons = tuple(sorted(set(horizons)))
    skip_models = tuple(skip_models or ())
    bad_skip = [m for m in skip_models if m not in MODEL_ORDER]
    if bad_skip:
        raise RequestError(f"invalid skip-model value(s): {bad_skip}")
    requested_models = tuple(m for m in MODEL_ORDER if m not in skip_models)
    if not requested_models:
        raise RequestError("no models requested (all models skipped)")

    db = Path(db_path) if db_path else DEFAULT_DB
    if not db.exists():
        raise RequestError(f"database not found: {db}")
    if not db.is_file():
        raise RequestError(f"database path is not a regular file: {db}")

    runs_dir = Path(runs_dir)
    rid = run_id or _generate_run_id(category, top_n)
    if not RUN_ID_RE.match(rid) or rid in (".", "..") or any(c in rid for c in "/\\") \
            or any(ord(c) < 32 for c in rid):
        raise RequestError(f"unsafe run_id: {run_id!r}")

    return {
        "category": category.strip(), "top_n": top_n, "as_of_date": as_of,
        "start_date": start,
        "selection_cutoff": cutoff, "min_history_days": min_history_days,
        "horizons": horizons, "runs_dir": runs_dir, "run_id": rid,
        "db_path": db, "allow_partial_success": bool(allow_partial_success),
        "skip_models": skip_models, "requested_models": requested_models,
    }


def _resolve_run_dir(runs_dir: Path, run_id: str, explicit: bool) -> Path:
    """Path-safe run directory strictly under runs_dir; never a symlink; explicit id must be new."""
    runs_dir = Path(runs_dir)
    run_dir = runs_dir / run_id
    base = runs_dir.resolve()
    resolved = (base / run_id)
    if resolved.parent != base:                       # slash/traversal already blocked, belt-and-suspenders
        raise RequestError("run_id escapes runs_dir")
    if run_dir.is_symlink() or runs_dir.is_symlink():
        raise RequestError("refusing to use a symlinked run/runs directory")
    if explicit and run_dir.exists():
        raise RequestError(f"run_id already exists: {run_dir}")
    if run_dir.exists():                              # generated id collision (extremely unlikely)
        raise RequestError(f"generated run_id already exists: {run_dir}")
    return run_dir


# ── status management (atomic) ─────────────────────────────────────────────────────────
def _init_status(run_id: str, created_at: str) -> dict:
    return {
        "run_id": run_id, "status": "created", "progress_pct": 0, "current_step": "created",
        "created_at": created_at, "started_at": None, "updated_at": created_at,
        "completed_at": None, "failed_at": None, "error_type": None, "error_message": None,
        "model_status": {},
    }


def _write_status(run_dir: Path, status: dict) -> None:
    status["updated_at"] = _now()
    status["progress_pct"] = STEP_PROGRESS.get(status.get("current_step"), status.get("progress_pct", 0))
    mc.write_json_atomic(status, run_dir / "status.json")


def _set_step(run_dir: Path, status: dict, step: str, logger: logging.Logger) -> None:
    # Record the ACTUAL lifecycle state (selecting_skus / preparing_data / running_<model> /
    # validating_outputs / ranking_models / completed / failed / completed_with_warnings) — not
    # a generic "running". current_step mirrors it.
    status["current_step"] = step
    status["status"] = step
    if status["started_at"] is None and step != "created":
        status["started_at"] = _now()
    logger.info("step: %s", step)
    _write_status(run_dir, status)


# ── logging ────────────────────────────────────────────────────────────────────────────
def _make_logger(run_id: str, log_path: Path) -> tuple[logging.Logger, logging.Handler]:
    logger = logging.getLogger(f"orchestrator.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, handler


# ── selection / preparation ────────────────────────────────────────────────────────────
def _select_skus(req: dict, run_dir: Path, logger: logging.Logger) -> tuple[pd.DataFrame, list[str]]:
    df, warnings = dsel.select_top_skus(
        db_path=req["db_path"], category=req["category"], top_n=req["top_n"],
        selection_cutoff=req["selection_cutoff"], min_history_days=req["min_history_days"])
    if df is None or len(df) == 0:
        raise RuntimeError("dynamic selection returned no SKUs")
    skus = df["sku"].astype(str).tolist()
    if len(set(skus)) != len(skus):
        raise RuntimeError("dynamic selection returned duplicate SKUs")
    if list(df["rank"]) != list(range(1, len(df) + 1)):
        raise RuntimeError("dynamic selection rank is not sequential from 1")
    cats = set(df["category"].astype(str).str.strip().unique())
    if cats != {req["category"]}:
        raise RuntimeError(f"selection category mismatch: {cats} != {req['category']!r}")
    sku_csv = run_dir / "selected_skus.csv"
    mc.write_dataframe_atomic(df, sku_csv, "csv")
    if not sku_csv.exists():
        raise RuntimeError("selected_skus.csv was not written")
    logger.info("selected %d SKUs (requested %d): %s", len(df), req["top_n"], ", ".join(skus))
    for w in warnings:
        logger.info("selection warning: %s", w)
    return df, list(warnings)


def _prepare_data(req: dict, run_dir: Path, logger: logging.Logger) -> Path:
    proc = run_dir / "processed"
    argv = ["--db-path", str(req["db_path"]),
            "--pilot-file", str(run_dir / "selected_skus.csv"),
            "--output-dir", str(proc),
            "--as-of-date", req["as_of_date"],
            "--selection-cutoff", req["selection_cutoff"], "--strict"]
    if req.get("start_date"):
        argv += ["--start-date", req["start_date"]]
    rc = prep.main(argv)
    if rc != 0:
        raise RuntimeError(f"prepare_pilot_data failed (exit {rc})")
    files = {n: proc / n for n in ("model_panel.parquet", "forecast_frame.parquet",
                                   "inventory_context.parquet", "pilot_manifest.json")}
    for name, p in files.items():
        if not p.exists():
            raise RuntimeError(f"prepared file missing: {name}")
    manifest = json.loads(files["pilot_manifest.json"].read_text(encoding="utf-8"))
    if manifest.get("validation_status") != "passed":
        raise RuntimeError(f"prepared manifest validation_status != passed: {manifest.get('validation_status')}")
    sel = set(pd.read_csv(run_dir / "selected_skus.csv")["sku"].astype(str))
    if set(map(str, manifest.get("selected_skus", []))) != sel:
        raise RuntimeError("prepared manifest SKU set does not match selected_skus.csv")
    mp = pd.read_parquet(files["model_panel.parquet"])
    ff = pd.read_parquet(files["forecast_frame.parquet"])
    if mp.empty or ff.empty:
        raise RuntimeError("model_panel/forecast_frame is empty")
    if (pd.to_datetime(ff["date"]) <= pd.Timestamp(req["as_of_date"])).any():
        raise RuntimeError("forecast_frame contains a date on/before as_of_date")
    logger.info("prepared run data under %s (model_panel %d rows, forecast_frame %d rows)",
                proc, len(mp), len(ff))
    return proc


# ── model execution ────────────────────────────────────────────────────────────────────
def _run_model(name: str, proc: Path, outputs: Path, horizons: tuple[int, ...],
               logger: logging.Logger) -> None:
    mp, ff, man = proc / "model_panel.parquet", proc / "forecast_frame.parquet", proc / "pilot_manifest.json"
    logger.info("model %s: START (inputs under %s -> outputs %s)", name, proc, outputs)
    if name == "baseline":
        baselines.run(model_panel=mp, forecast_frame=ff, manifest=man, output_dir=outputs, horizons=horizons)
    elif name == "holtwinters":
        holtwinters.run_pipeline(model_panel=mp, forecast_frame=ff, manifest=man,
                                 output_dir=outputs, horizons=horizons)
    elif name == "lightgbm":
        lgbm_global.run(model_panel=mp, forecast_frame=ff, manifest=man, output_dir=outputs, horizons=horizons)
    else:
        raise RuntimeError(f"unknown model {name!r}")
    for kind, fname in MODEL_ARTIFACTS[name].items():
        if not (outputs / fname).exists():
            raise RuntimeError(f"{name} did not write required output {fname}")
    logger.info("model %s: END (all required outputs present)", name)


def validate_model_outputs(name: str, proc: Path, outputs: Path, req: dict,
                           logger: logging.Logger) -> None:
    """Independently LOAD and validate a completed model's outputs against the shared
    contract (not just existence). Raises RuntimeError on any violation."""
    arts = MODEL_ARTIFACTS[name]
    ff = pd.read_parquet(proc / "forecast_frame.parquet")
    manifest = json.loads((proc / "pilot_manifest.json").read_text(encoding="utf-8"))

    # --- backtest predictions ---
    bt = pd.read_parquet(outputs / arts["backtest"])
    if "y_true" in bt.columns:
        raise RuntimeError(f"{name} backtest predictions contain a y_true column")
    bt_h = set(pd.to_numeric(bt["horizon"], errors="coerce").dropna().astype(int))
    if bt_h - set(req["horizons"]):
        raise RuntimeError(f"{name} backtest has unrequested horizon(s): {sorted(bt_h - set(req['horizons']))}")
    if name == "baseline":
        got = set(bt["model"].astype(str).unique())
        if got != set(BASELINE_METHODS):
            raise RuntimeError(f"baseline backtest models {sorted(got)} != {sorted(BASELINE_METHODS)}")
        for method in BASELINE_METHODS:
            mc.validate_backtest_predictions(bt[bt["model"] == method], method, tuple(req["horizons"]))
    else:
        ident = MODEL_IDENTITY[name]
        if set(bt["model"].astype(str).unique()) != {ident}:
            raise RuntimeError(f"{name} backtest model column must be exactly {{{ident!r}}}")
        mc.validate_backtest_predictions(bt, ident, tuple(req["horizons"]))

    # --- scorecard schema (exact) ---
    sc = pd.read_csv(outputs / arts["scorecard"])
    if list(sc.columns) != mc.SCORECARD_COLUMNS:
        raise RuntimeError(f"{name} scorecard columns {list(sc.columns)} != contract {mc.SCORECARD_COLUMNS}")

    # --- future forecast ---
    fut = pd.read_parquet(outputs / arts["future"])
    if "y_true" in fut.columns:
        raise RuntimeError(f"{name} future forecast contains a y_true column")
    fut_models = set(fut["model"].astype(str).unique())
    if len(fut_models) != 1:
        raise RuntimeError(f"{name} future forecast has non-unique model values {sorted(fut_models)}")
    fut_model = fut_models.pop()
    if name == "baseline":
        if fut_model not in BASELINE_METHODS:
            raise RuntimeError(f"baseline future model {fut_model!r} is not a baseline method")
    elif fut_model != MODEL_IDENTITY[name]:
        raise RuntimeError(f"{name} future model {fut_model!r} != {MODEL_IDENTITY[name]!r}")
    # keys==forecast_frame, dates>as_of, finite & non-negative y_pred, model identity, intervals ordered
    mc.validate_future_predictions(fut, ff, manifest, fut_model)
    logger.info("model %s: outputs independently validated (backtest+scorecard+future)", name)


# ── fingerprint + comparability + ranking ──────────────────────────────────────────────
def _verify_fingerprints(completed: list[str], outputs: Path, req: dict,
                         prepared_n_skus: int, selected_n_skus: int, logger: logging.Logger) -> str:
    fps = {}
    for m in completed:
        s = json.loads((outputs / MODEL_ARTIFACTS[m]["summary"]).read_text(encoding="utf-8"))
        fp = s.get("dataset_fingerprint")
        if not isinstance(fp, str) or not HEX64_RE.match(fp):
            raise RuntimeError(f"{m}: dataset_fingerprint is not a lowercase 64-hex SHA-256: {fp!r}")
        fps[m] = fp
        if s.get("as_of_date") != req["as_of_date"]:
            raise RuntimeError(f"{m}: as_of_date {s.get('as_of_date')} != request {req['as_of_date']}")
        if tuple(s.get("horizons", ())) != req["horizons"]:
            raise RuntimeError(f"{m}: horizons {s.get('horizons')} != request {list(req['horizons'])}")
        if s.get("n_skus") is not None:                       # only where the summary provides it
            n = int(s["n_skus"])
            if n != prepared_n_skus or n != selected_n_skus:
                raise RuntimeError(f"{m}: summary n_skus {n} != prepared {prepared_n_skus} / "
                                   f"selected {selected_n_skus}")
    unique = set(fps.values())
    logger.info("fingerprint comparison: %s", {m: fps[m][:16] for m in fps})
    if len(unique) != 1:
        raise RuntimeError(f"dataset fingerprint mismatch across models: {fps}")
    return unique.pop()


def validate_combined_scorecard(combined: pd.DataFrame,
                                requested_horizons: tuple[int, ...]) -> pd.DataFrame:
    """Production comparability check for the concatenated model scorecards. Raises
    RuntimeError on any violation; returns the frame unchanged on success."""
    requested = tuple(int(h) for h in requested_horizons)
    if list(combined.columns) != mc.SCORECARD_COLUMNS:
        raise RuntimeError(f"combined scorecard columns {list(combined.columns)} != contract")
    if combined.duplicated(["model", "horizon", "evaluation_type"]).any():
        raise RuntimeError("combined scorecard has duplicate model/horizon/evaluation_type rows")
    if set(pd.to_numeric(combined["horizon"], errors="coerce").dropna().astype(int)) - set(requested):
        raise RuntimeError("combined scorecard contains an unrequested horizon")
    locked = combined[combined["evaluation_type"] == LOCKED]
    for h in requested:
        sub = locked[locked["horizon"] == h]
        if sub.empty:
            raise RuntimeError(f"no locked-holdout candidate rows for requested horizon {h}")
        if not np.isfinite(pd.to_numeric(sub["wape"], errors="coerce")).any():
            raise RuntimeError(f"no finite-WAPE ranking candidate for horizon {h}")
        for col in ("cutoff", "n_rows", "n_skus", "n_channels"):
            if sub[col].nunique(dropna=False) != 1:
                raise RuntimeError(f"scorecard comparability mismatch on {col} at horizon {h}: "
                                   f"{sorted(sub[col].unique())}")
    return combined


def _combined_scorecard(completed: list[str], outputs: Path, req: dict) -> pd.DataFrame:
    frames = [pd.read_csv(outputs / MODEL_ARTIFACTS[m]["scorecard"]) for m in completed]
    combined = pd.concat(frames, ignore_index=True)[mc.SCORECARD_COLUMNS]
    return validate_combined_scorecard(combined, req["horizons"])


def rank_models(combined: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Independent per-horizon ranking on locked-holdout rows only:
    finite WAPE -> MASE -> MAE -> |bias| -> model name (all ascending)."""
    rows: list[dict] = []
    locked = combined[combined["evaluation_type"] == LOCKED].copy()
    for h in sorted(horizons):
        sub = locked[(locked["horizon"] == h) & np.isfinite(pd.to_numeric(locked["wape"], errors="coerce"))].copy()
        sub["_abias"] = sub["bias"].abs()
        sub = sub.sort_values(["wape", "mase", "mae", "_abias", "model"],
                              na_position="last").reset_index(drop=True)
        for i, r in sub.iterrows():
            reason = ("winner: lowest WAPE (tie-breaks MASE, MAE, |bias|, name)" if i == 0
                      else f"rank {i + 1} by WAPE, MASE, MAE, |bias|, name")
            rows.append({"horizon": int(h), "rank": i + 1, "model": r["model"],
                         "wape": r["wape"], "mase": r["mase"], "mae": r["mae"],
                         "rmse": r["rmse"], "bias": r["bias"], "selection_reason": reason})
    return pd.DataFrame(rows, columns=["horizon", "rank", "model", "wape", "mase",
                                       "mae", "rmse", "bias", "selection_reason"])


def _select_operational_forecast(ranking: pd.DataFrame, req: dict, proc: Path, outputs: Path,
                                 run_dir: Path, logger: logging.Logger) -> tuple[dict, pd.DataFrame]:
    op_h = max(req["horizons"])
    top = ranking[(ranking["horizon"] == op_h) & (ranking["rank"] == 1)]
    if top.empty:
        raise RuntimeError(f"no rank-1 model for operational horizon {op_h}")
    winner = str(top.iloc[0]["model"])
    reason = str(top.iloc[0]["selection_reason"])

    if winner in BASELINE_METHODS:
        fut_path = outputs / MODEL_ARTIFACTS["baseline"]["future"]
        family = "baseline"
    elif winner in ("holtwinters", "lightgbm"):
        fut_path = outputs / MODEL_ARTIFACTS[winner]["future"]
        family = winner
    else:
        raise RuntimeError(f"unrecognized ranking winner {winner!r}")

    fut = pd.read_parquet(fut_path)
    fut_models = set(fut["model"].astype(str).unique())
    if family == "baseline":
        # the baseline future file must contain exactly the official baseline winner method
        if fut_models != {winner}:
            raise RuntimeError(f"baseline future model {fut_models} != ranking winner {winner!r}")
    else:
        if fut_models != {winner}:
            raise RuntimeError(f"{family} future model {fut_models} != ranking winner {winner!r}")

    ff = pd.read_parquet(proc / "forecast_frame.parquet")
    key = ["sku", "channel", "date"]
    fut2 = fut.copy()
    fut2["date"] = pd.to_datetime(fut2["date"])
    ffk = ff.copy(); ffk["date"] = pd.to_datetime(ffk["date"])
    if set(map(tuple, fut2[key].itertuples(index=False, name=None))) != \
            set(map(tuple, ffk[key].itertuples(index=False, name=None))):
        raise RuntimeError("selected forecast keys do not match forecast_frame")

    fut2["selection_horizon"] = op_h
    fut2["selection_rank"] = 1
    fut2["selection_reason"] = reason
    fut2["selected_at"] = _now()
    mc.write_dataframe_atomic(fut2, run_dir / "selected_forecasts.parquet", "parquet")
    logger.info("operational winner @ horizon %d: %s (%d forecast rows)", op_h, winner, len(fut2))
    winners = {str(int(h)): str(ranking[(ranking["horizon"] == h) & (ranking["rank"] == 1)].iloc[0]["model"])
               for h in req["horizons"]}
    meta = {"operational_horizon": op_h, "operational_model": winner, "winners_by_horizon": winners}
    return meta, fut2


# ── Phase B decision-artifact validation ─────────────────────────────────────────────────
def _validate_decision_artifacts(run_dir: Path, logger: logging.Logger) -> None:
    """Independently re-load and validate the Phase B artifacts against the decision contract.
    Raises RuntimeError on any missing/invalid artifact (fatal — a completed run must have them)."""
    dec = run_dir / "decisions"
    risk_p, traj_p = dec / "stockout_risk.parquet", dec / "stockout_trajectory.parquet"
    for p in (risk_p, traj_p):
        if not p.exists():
            raise RuntimeError(f"Phase B did not write required decision artifact: {p.name}")
    run_id = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))["run_id"]
    sel = pd.read_parquet(run_dir / "selected_forecasts.parquet")
    risk = pd.read_parquet(risk_p)
    traj = pd.read_parquet(traj_p)
    if "y_true" in risk.columns or "units_observed" in risk.columns \
            or "y_true" in traj.columns or "units_observed" in traj.columns:
        raise RuntimeError("Phase B decision artifact contains a truth column")
    dc.validate_stockout_risk(risk, sel[["sku", "channel"]].drop_duplicates(), run_id)
    dc.validate_stockout_trajectory(traj, sel[["sku", "channel", "date"]].drop_duplicates(), run_id)
    logger.info("Phase B artifacts validated: %d risk rows, %d trajectory rows", len(risk), len(traj))


def _validate_reorder_artifacts(run_dir: Path, logger: logging.Logger) -> None:
    """Independently re-load and validate the Phase C artifacts against the decision contract.
    Raises RuntimeError on any missing/invalid artifact (fatal — a completed run must have them)."""
    dec = run_dir / "decisions"
    reco_p, summ_p = dec / "reorder_recommendations.parquet", dec / "reorder_summary.json"
    for p in (reco_p, summ_p):
        if not p.exists():
            raise RuntimeError(f"Phase C did not write required decision artifact: {p.name}")
    run_id = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))["run_id"]
    risk = pd.read_parquet(dec / "stockout_risk.parquet")
    reco = pd.read_parquet(reco_p)
    summary = json.loads(summ_p.read_text(encoding="utf-8"))
    if "y_true" in reco.columns or "units_observed" in reco.columns:
        raise RuntimeError("Phase C reorder artifact contains a truth column")
    dc.validate_reorder_recommendations(reco, risk[["sku", "channel"]].drop_duplicates(), run_id)
    dc.validate_reorder_summary(summary, reco, run_id)
    logger.info("Phase C artifacts validated: %d reorder rows, actions=%s",
                len(reco), summary.get("count_by_action"))


# ── artifact inventory + manifest ──────────────────────────────────────────────────────
_MUTABLE = {"pipeline.log", "status.json", "run_manifest.json"}


def _artifact_inventory(run_dir: Path) -> list[dict]:
    inv = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name not in _MUTABLE:
            inv.append({"path": p.relative_to(run_dir).as_posix(),
                        "size_bytes": p.stat().st_size, "sha256": _sha256(p)})
    return inv


def _model_metrics(combined: pd.DataFrame | None) -> list[dict]:
    if combined is None:
        return []
    locked = combined[combined["evaluation_type"] == LOCKED]
    return locked[["model", "horizon", "wape", "mase", "mae", "rmse", "bias", "n_rows"]].to_dict("records")


# ── the pipeline ────────────────────────────────────────────────────────────────────────
def run_forecast_pipeline(*, category: str, top_n: int, as_of_date: str,
                          selection_cutoff: str | None = None, min_history_days: int = 28,
                          horizons: tuple[int, ...] = (7, 14), runs_dir="runs",
                          run_id: str | None = None, db_path=None,
                          allow_partial_success: bool = False,
                          skip_models: tuple[str, ...] = (),
                          start_date: str | None = None) -> dict:
    """Run the full pipeline. Returns the run manifest dict (status 'completed',
    'completed_with_warnings', or 'failed'). Raises RequestError ONLY for invalid
    requests before the run directory exists."""
    explicit = run_id is not None
    req = validate_request(category=category, top_n=top_n, as_of_date=as_of_date,
                           selection_cutoff=selection_cutoff, min_history_days=min_history_days,
                           horizons=horizons, runs_dir=runs_dir, run_id=run_id, db_path=db_path,
                           allow_partial_success=allow_partial_success, skip_models=skip_models,
                           start_date=start_date)
    run_dir = _resolve_run_dir(req["runs_dir"], req["run_id"], explicit)

    # create the run tree only after validation passed
    created_at = _now()
    (run_dir / "processed").mkdir(parents=True, exist_ok=False)
    outputs = run_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=False)
    logger, handler = _make_logger(req["run_id"], run_dir / "pipeline.log")

    request_json = {
        "run_id": req["run_id"], "category": req["category"], "top_n": req["top_n"],
        "as_of_date": req["as_of_date"], "selection_cutoff": req["selection_cutoff"],
        "start_date": req["start_date"],
        "min_history_days": req["min_history_days"], "horizons": list(req["horizons"]),
        "db_path": str(req["db_path"].resolve()), "requested_models": list(req["requested_models"]),
        "allow_partial_success": req["allow_partial_success"], "created_at": created_at,
    }
    mc.write_json_atomic(request_json, run_dir / "request.json")
    status = _init_status(req["run_id"], created_at)
    status["model_status"] = {m: {"status": "pending", "started_at": None,
                                  "completed_at": None, "error": None} for m in req["requested_models"]}
    _write_status(run_dir, status)
    logger.info("run_id=%s validated request=%s", req["run_id"], request_json)

    errors: list[str] = []
    selection_warning = None
    completed: list[str] = []
    failed: list[str] = []
    combined = ranking = None
    op_meta: dict = {}
    decision_summary = None
    reorder_summary = None
    sel_df = None

    def _fail(err_type: str, msg: str) -> dict:
        logger.error("FAILED (%s): %s", err_type, msg)
        status["status"] = "failed"; status["current_step"] = "failed"
        status["failed_at"] = _now(); status["error_type"] = err_type
        status["error_message"] = msg
        _write_status(run_dir, status)
        errors.append(f"{err_type}: {msg}")
        return _finalize(run_dir, req, request_json, created_at, "failed", status,
                         sel_df, selection_warning, completed, failed, req["skip_models"],
                         combined, ranking, op_meta, errors, logger, handler,
                         decision_summary=decision_summary, reorder_summary=reorder_summary)

    try:
        # 1) selection
        _set_step(run_dir, status, "selecting_skus", logger)
        sel_df, sel_warns = _select_skus(req, run_dir, logger)
        if len(sel_df) < req["top_n"]:
            selection_warning = (f"requested_top_n={req['top_n']} but only "
                                 f"{len(sel_df)} eligible SKUs were selected")
            logger.info(selection_warning)
        if sel_warns:
            selection_warning = "; ".join([w for w in ([selection_warning] if selection_warning else []) + sel_warns])

        # 2) preparation
        _set_step(run_dir, status, "preparing_data", logger)
        proc = _prepare_data(req, run_dir, logger)

        # 3) models
        for m in req["requested_models"]:
            _set_step(run_dir, status, f"running_{m}", logger)
            status["model_status"][m].update(status="running", started_at=_now())
            _write_status(run_dir, status)
            try:
                _run_model(m, proc, outputs, req["horizons"], logger)
                validate_model_outputs(m, proc, outputs, req, logger)   # independent output contract check
                status["model_status"][m].update(status="completed", completed_at=_now())
                completed.append(m)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                logger.error("model %s raised: %s\n%s", m, exc, tb)
                status["model_status"][m].update(status="failed", completed_at=_now(),
                                                 error=f"{type(exc).__name__}: {exc}")
                failed.append(m)
                errors.append(f"model {m}: {type(exc).__name__}: {exc}")
                if not req["allow_partial_success"]:
                    _write_status(run_dir, status)
                    return _fail("model_failed", f"model {m} failed and allow_partial_success=false")
            _write_status(run_dir, status)

        if not completed:
            return _fail("all_models_failed", "no model completed successfully")

        # 4) validation (fingerprints + comparability) — fatal even in partial mode
        _set_step(run_dir, status, "validating_outputs", logger)
        selected_n_skus = int(pd.read_csv(run_dir / "selected_skus.csv")["sku"].astype(str).nunique())
        prepared_n_skus = int(pd.read_parquet(proc / "model_panel.parquet")["sku"].nunique())
        fingerprint = _verify_fingerprints(completed, outputs, req, prepared_n_skus, selected_n_skus, logger)
        combined = _combined_scorecard(completed, outputs, req)
        mc.write_dataframe_atomic(combined, run_dir / "combined_scorecard.csv", "csv")

        # 5) ranking + operational forecast
        _set_step(run_dir, status, "ranking_models", logger)
        ranking = rank_models(combined, req["horizons"])
        mc.write_dataframe_atomic(ranking, run_dir / "model_ranking.csv", "csv")
        logger.info("ranking:\n%s", ranking.to_string(index=False))
        op_meta, _sel_fc = _select_operational_forecast(ranking, req, proc, outputs, run_dir, logger)

        # 6) Phase B — forecast-driven stockout risk. ALWAYS fatal on failure (even under
        # allow_partial_success): a completed run MUST have valid decision artifacts.
        _set_step(run_dir, status, "calculating_stockout_risk", logger)
        try:
            decision_summary = stockout_risk.compute_stockout_risk(
                run_dir, operational_model=op_meta["operational_model"],
                operational_horizon=int(op_meta["operational_horizon"]), logger=logger)
            _validate_decision_artifacts(run_dir, logger)
        except Exception as exc:  # noqa: BLE001
            logger.error("Phase B failed: %s\n%s", exc, traceback.format_exc())
            return _fail("stockout_risk_failed",
                         f"Phase B stockout-risk failed: {type(exc).__name__}: {exc}")

        # 7) Phase C — forecast-driven reorder recommendations. ALWAYS fatal on failure (even
        # under allow_partial_success): allow_partial_success governs forecasting-model
        # availability, NOT downstream decision-contract failures. A completed run MUST have
        # valid reorder recommendations + summary.
        _set_step(run_dir, status, "calculating_reorder_recommendations", logger)
        try:
            reorder_summary = reorder_recommendations.compute_reorder_recommendations(
                run_dir, operational_model=op_meta["operational_model"],
                operational_horizon=int(op_meta["operational_horizon"]), logger=logger)
            _validate_reorder_artifacts(run_dir, logger)
        except Exception as exc:  # noqa: BLE001
            logger.error("Phase C failed: %s\n%s", exc, traceback.format_exc())
            return _fail("reorder_recommendations_failed",
                         f"Phase C reorder recommendations failed: {type(exc).__name__}: {exc}")

        final_status = "completed_with_warnings" if failed else "completed"
        status["status"] = final_status; status["current_step"] = final_status
        status["completed_at"] = _now()
        _write_status(run_dir, status)
        return _finalize(run_dir, req, request_json, created_at, final_status, status,
                         sel_df, selection_warning, completed, failed, req["skip_models"],
                         combined, ranking, op_meta, errors, logger, handler,
                         fingerprint=fingerprint, decision_summary=decision_summary,
                         reorder_summary=reorder_summary)
    except RequestError:
        raise
    except Exception as exc:  # noqa: BLE001 — any post-dir error becomes a failed manifest, never a raise
        return _fail(type(exc).__name__, str(exc))


def _finalize(run_dir, req, request_json, created_at, final_status, status, sel_df,
              selection_warning, completed, failed, skipped, combined, ranking, op_meta,
              errors, logger, handler, fingerprint=None, decision_summary=None,
              reorder_summary=None) -> dict:
    # success/failure timestamp semantics (mirror status.json): completed_at is set only on
    # success and null on failure; failed_at is set only on failure and null on success.
    end_ts = _now()
    if final_status == "failed":
        completed_at, failed_at = None, (status.get("failed_at") or end_ts)
    else:
        completed_at, failed_at = end_ts, None
    started = status.get("started_at") or created_at
    try:
        dur = (datetime.fromisoformat(completed_at or failed_at)
               - datetime.fromisoformat(started)).total_seconds()
    except Exception:
        dur = None
    proc = run_dir / "processed"
    processed_files = [p.name for p in sorted(proc.glob("*"))] if proc.exists() else []
    manifest = {
        "run_id": req["run_id"], "status": final_status, "request": request_json,
        "created_at": created_at, "started_at": started,
        "completed_at": completed_at, "failed_at": failed_at,
        "duration_seconds": dur, "run_directory": str(run_dir.resolve()),
        "selected_sku_count": (0 if sel_df is None else int(len(sel_df))),
        "selected_skus": ([] if sel_df is None else sel_df["sku"].astype(str).tolist()),
        "selection_warning": selection_warning,
        "processed_files": processed_files,
        "completed_models": completed, "failed_models": failed, "skipped_models": list(skipped),
        "dataset_fingerprint": fingerprint,
        "scorecard_file": "combined_scorecard.csv" if combined is not None else None,
        "ranking_file": "model_ranking.csv" if ranking is not None else None,
        "selected_forecast_file": ("selected_forecasts.parquet"
                                   if (run_dir / "selected_forecasts.parquet").exists() else None),
        "winners_by_horizon": op_meta.get("winners_by_horizon"),
        "operational_horizon": op_meta.get("operational_horizon"),
        "operational_model": op_meta.get("operational_model"),
        # Phase B — forecast-driven stockout risk
        "decisioning_status": (decision_summary or {}).get("decisioning_status"),
        "stockout_risk_file": (decision_summary or {}).get("stockout_risk_file"),
        "stockout_trajectory_file": (decision_summary or {}).get("stockout_trajectory_file"),
        "stockout_validation_summary": ({
            "risk_rows": decision_summary.get("risk_rows"),
            "trajectory_rows": decision_summary.get("trajectory_rows"),
            "risk_tier_counts": decision_summary.get("risk_tier_counts"),
            "manual_review_count": decision_summary.get("manual_review_count"),
            "uncertainty_methods": decision_summary.get("uncertainty_methods"),
        } if decision_summary else None),
        "stockout_policy_version": dc.STOCKOUT_POLICY_VERSION,
        # Phase C — forecast-driven reorder recommendations
        "reorder_recommendations_file": (reorder_summary or {}).get("reorder_recommendations_file"),
        "reorder_summary_file": (reorder_summary or {}).get("reorder_summary_file"),
        "reorder_validation_summary": ({
            "reorder_rows": reorder_summary.get("reorder_rows") or reorder_summary.get("selected_series_count"),
            "count_by_action": reorder_summary.get("count_by_action"),
            "total_proposed_order_units": reorder_summary.get("total_proposed_order_units"),
            "total_proposed_purchase_value": reorder_summary.get("total_proposed_purchase_value"),
            "manual_review_count": reorder_summary.get("manual_review_count"),
            "approval_required_count": reorder_summary.get("approval_required_count"),
        } if reorder_summary else None),
        "reorder_policy_version": dc.REORDER_POLICY_VERSION,
        "model_metrics": _model_metrics(combined),
        "validation_summary": {
            "fingerprints_match": fingerprint is not None,
            "requested_models": list(req["requested_models"]),
            "comparability_checked": combined is not None,
        },
        "errors": errors,
        "software_versions": _software_versions(),
    }
    # artifact inventory LAST (after all deliverables written; excludes mutable/log files)
    manifest["artifact_inventory"] = _artifact_inventory(run_dir)
    mc.write_json_atomic(manifest, run_dir / "run_manifest.json")
    logger.info("final status: %s", final_status)
    handler.close()
    logger.removeHandler(handler)
    return manifest


# ── CLI ────────────────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run-aware forecasting orchestrator (Phase 4).")
    ap.add_argument("--category", required=True)
    ap.add_argument("--top-n", required=True, type=int)
    ap.add_argument("--as-of-date", required=True)
    ap.add_argument("--selection-cutoff", default=None)
    ap.add_argument("--start-date", default=None,
                    help="hard lower bound on history (YYYY-MM-DD). Default: all warehouse history.")
    ap.add_argument("--min-history-days", type=int, default=28)
    ap.add_argument("--horizons", nargs="+", type=int, choices=[7, 14], default=[7, 14])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--db-path", default=None)
    ap.add_argument("--allow-partial-success", action="store_true")
    ap.add_argument("--skip-model", action="append", choices=list(MODEL_ORDER), default=[])
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = run_forecast_pipeline(
            category=args.category, top_n=args.top_n, as_of_date=args.as_of_date,
            start_date=args.start_date,
            selection_cutoff=args.selection_cutoff, min_history_days=args.min_history_days,
            horizons=tuple(args.horizons), runs_dir=args.runs_dir, run_id=args.run_id,
            db_path=args.db_path, allow_partial_success=args.allow_partial_success,
            skip_models=tuple(args.skip_model))
    except RequestError as exc:
        print(f"REQUEST INVALID: {exc}", file=sys.stderr)
        return 2
    status = manifest.get("status")
    print(f"run_id={manifest['run_id']} status={status} dir={manifest['run_directory']}")
    if status == "failed":
        print("FAILED:", *manifest.get("errors", []), sep="\n  ", file=sys.stderr)
        return 1
    if status == "completed_with_warnings":
        print("completed with warnings; failed models:", manifest.get("failed_models"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
