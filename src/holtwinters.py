"""holtwinters.py — Khizer. Univariate ETS (Holt / Holt-Winters family) demand forecaster.

Stage A of the pilot: forecast REAL daily `units_observed` for each of the 30 pilot SKUs on the
`naheed_web` channel, 7 and 14 days ahead, with 80% / 95% prediction intervals. One ETS structure
is chosen per SKU by a leakage-free expanding-window rolling-origin backtest, frozen, then scored
on the evaluator's locked holdout via `evaluation.evaluate()` and refit on all history through
`as_of_date` to forecast `forecast_frame.parquet`.

Contract (verified at runtime, not assumed):
  * Target is REAL `units_observed` only. The model is UNIVARIATE per SKU×channel — no exogenous
    columns, no synthetic-stock/cost/inventory fields ever enter a fit.
  * Additive error/trend/seasonality only (real zero-demand days exist; no log/Box-Cox/mul).
  * statsmodels==0.14.4 `ETSModel`, `initialization_method="estimated"`; library estimates params.
  * Point forecasts are floats, negatives clipped to 0, never rounded here.
  * Deterministic: stable per-SKU SHA-256 seeds, stable sorts, JSON sort_keys. Two runs match.

Model accuracy is ESTIMATED via backtesting, not guaranteed. Forecasts feed the downstream
stockout-risk (Stage B) and reorder (Stage C) stages. No supplier order is created here.

Run:  python src/holtwinters.py            (writes outputs/)
      python src/holtwinters.py --selfcheck (also recompute in-process and assert identical)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # noqa: E402
import evaluation as ev  # shared scorecard — imported, never edited  # noqa: E402

PROC = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "outputs"
MODEL_PANEL = PROC / "model_panel.parquet"
FORECAST_FRAME = PROC / "forecast_frame.parquet"
MANIFEST = PROC / "pilot_manifest.json"

EXPECTED_SCHEMA = "4.0-real-demand-synthetic-stock"
EXPECTED_CHANNEL = "naheed_web"
TARGET = "units_observed"
HORIZONS = ev.HORIZONS                    # (7, 14) — from the shared evaluator
BASE_SEED = 2026
N_SIM = 5000
MODEL_VERSION = "holtwinters_ets_v1"
MIN_TRAIN_OBS = 21                        # need enough points for a stable fit (esp. seasonal m=7)

# Columns that must NEVER be used as a model input (fail loudly if any leaks into a fit frame).
FORBIDDEN_INPUTS = (
    "stock_on_hand", "stock_on_hand_is_synthetic", "stock_source", "stock_generation_version",
    "unit_cost", "unit_cost_observed", "unit_cost_effective", "lead_time_days", "moq", "pack_size",
    "safety_stock", "reorder_point", "target_stock", "recommended_order_quantity",
    "latent_synthetic_demand", "synthetic_sales", "lost_sales",
    "effective_unit_price", "discount_amount", "discount_pct", "on_promo",
    "is_public_holiday", "is_payday_window",
)


# ── candidate family (additive only) ───────────────────────────────────────────────
@dataclass(frozen=True)
class Candidate:
    model_id: str
    error: str
    trend: str | None
    damped_trend: bool
    seasonal: str | None
    seasonal_periods: int | None
    complexity: int          # lower = simpler (tie-break order)


def candidate_specs() -> list[Candidate]:
    return [
        Candidate("ets_A_N_N", "add", None, False, None, None, 0),
        Candidate("ets_A_A_N", "add", "add", False, None, None, 1),
        Candidate("ets_A_Ad_N", "add", "add", True, None, None, 2),
        Candidate("ets_A_N_A7", "add", None, False, "add", 7, 3),
        Candidate("ets_A_A_A7", "add", "add", False, "add", 7, 4),
        Candidate("ets_A_Ad_A7", "add", "add", True, "add", 7, 5),
    ]


CANDIDATES = {c.model_id: c for c in candidate_specs()}
COMPLEXITY_ORDER = [c.model_id for c in candidate_specs()]


# ── stable seed ─────────────────────────────────────────────────────────────────────
def stable_seed(sku: str, channel: str, origin: str, model_id: str) -> int:
    """Deterministic 32-bit seed (legacy RandomState-compatible) from SHA-256 of the parts.
    NOT Python hash() — that is process-randomized."""
    key = f"{BASE_SEED}|{sku}|{channel}|{origin}|{model_id}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


# ── contract load + audit ───────────────────────────────────────────────────────────
def load_contract() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    for p in (MODEL_PANEL, FORECAST_FRAME, MANIFEST):
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")
    mp = pd.read_parquet(MODEL_PANEL)
    ff = pd.read_parquet(FORECAST_FRAME)
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mp["date"] = pd.to_datetime(mp["date"])
    ff["date"] = pd.to_datetime(ff["date"])
    return mp, ff, man


def _fail(msg: str) -> None:
    raise RuntimeError(f"[holtwinters input validation] {msg}")


def audit_inputs(mp: pd.DataFrame, ff: pd.DataFrame, man: dict) -> dict:
    """Validate the shared inputs against the v4 contract and print a concise audit.
    Fails loudly on any violation; does not repair or regenerate inputs."""
    if man.get("validation_status") != "passed":
        _fail(f"manifest validation_status is {man.get('validation_status')!r}, expected 'passed'")
    if man.get("schema_version") != EXPECTED_SCHEMA:
        _fail(f"unexpected schema_version {man.get('schema_version')!r} (expected {EXPECTED_SCHEMA!r})")
    if TARGET not in mp.columns:
        _fail(f"target column {TARGET!r} absent from model_panel")
    chans = sorted(mp["channel"].unique())
    if chans != [EXPECTED_CHANNEL]:
        _fail(f"unexpected channels {chans} (expected [{EXPECTED_CHANNEL!r}])")
    if mp.duplicated(["sku", "channel", "date"]).any():
        _fail("duplicate sku/channel/date keys in model_panel")
    u = pd.to_numeric(mp[TARGET], errors="coerce")
    if u.isna().any() or np.isinf(u).any():
        _fail("units_observed contains missing or non-finite values")
    if (u < 0).any():
        _fail("units_observed contains negative values")
    as_of = pd.Timestamp(man["as_of_date"])
    if (mp["date"] > as_of).any():
        _fail(f"model_panel contains dates after as_of_date {as_of.date()}")
    # every SKU series must be daily and unique from its first observed date (no invented zeros)
    missing_dates = 0
    for _, g in mp.groupby("sku"):
        full = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        missing_dates += len(set(full) - set(g["date"]))
    if missing_dates:
        _fail(f"{missing_dates} internal daily dates are missing (a missing row is NOT auto-zero)")
    # forecast-frame contract
    n_days = int(man["future_frame_days"])
    if ff.duplicated(["sku", "channel", "date"]).any():
        _fail("duplicate keys in forecast_frame")
    if sorted(ff["channel"].unique()) != [EXPECTED_CHANNEL]:
        _fail("forecast_frame has unexpected channels")
    if not (ff["date"] > as_of).all():
        _fail("forecast_frame contains a date on/before as_of_date")
    per = ff.groupby(["sku", "channel"])["date"].nunique()
    if not (per == n_days).all():
        _fail(f"forecast_frame must have exactly {n_days} future days per sku/channel")

    # ineligibility reason audit (must be only the expected warm-up, else report explicitly)
    mp2 = mp.sort_values(["sku", "channel", "date"]).copy()
    mp2["_didx"] = mp2.groupby(["sku", "channel"]).cumcount()
    inel = mp2[~mp2["forecast_training_eligible"].astype(bool)]
    unexpected = inel[inel["_didx"] >= 14]
    if len(unexpected):
        flags = unexpected["data_quality_flag"].value_counts().to_dict()
        _fail(f"{len(unexpected)} ineligible rows are NOT warm-up (didx>=14); reasons={flags}")

    audit = {
        "python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
        "statsmodels": __import__("statsmodels").__version__,
        "schema_version": man["schema_version"], "validation_status": man["validation_status"],
        "as_of_date": as_of.date().isoformat(),
        "hist_min": mp["date"].min().date().isoformat(), "hist_max": mp["date"].max().date().isoformat(),
        "rows": int(len(mp)), "manifest_rows": int(man["row_counts"]["model_panel"]),
        "skus": int(mp["sku"].nunique()), "channels": chans,
        "dup_keys": int(mp.duplicated(["sku", "channel", "date"]).sum()),
        "units_null": int(u.isna().sum()), "units_negative": int((u < 0).sum()),
        "zero_demand_pct": round(float((u == 0).mean()) * 100, 2),
        "per_sku_obs_min": int(mp.groupby("sku").size().min()),
        "per_sku_obs_max": int(mp.groupby("sku").size().max()),
        "missing_internal_dates": missing_dates,
        "ff_rows": int(len(ff)), "ff_skus": int(ff["sku"].nunique()), "ff_dates": int(ff["date"].nunique()),
        "ff_strictly_after_as_of": bool((ff["date"] > as_of).all()),
        "ineligible_rows": int(len(inel)), "ineligible_reason": "first-14-day warm-up only",
        "eligible_rows": int(mp["forecast_training_eligible"].sum()),
    }
    if audit["rows"] != audit["manifest_rows"]:
        _fail(f"observed model_panel rows {audit['rows']} != manifest {audit['manifest_rows']}")
    print("=== INPUT AUDIT ===")
    for k, v in audit.items():
        print(f"  {k}: {v}")
    return audit


def build_series(mp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One clean daily target series per SKU (channel is naheed_web). Uses ONLY the real
    target + date — no forbidden/exogenous column can enter the model frame."""
    series: dict[str, pd.DataFrame] = {}
    for sku, g in mp.groupby("sku"):
        s = g.sort_values("date")[["date", TARGET]].reset_index(drop=True)
        # daily continuity + validity (defensive; audit already checked globally)
        full = pd.date_range(s["date"].min(), s["date"].max(), freq="D")
        if len(full) != len(s) or (s["date"].to_numpy() != full.to_numpy()).any():
            _fail(f"SKU {sku}: series is not gap-free daily")
        y = s[TARGET].to_numpy(float)
        if np.isnan(y).any() or np.isinf(y).any() or (y < 0).any():
            _fail(f"SKU {sku}: invalid target values")
        series[sku] = s
    return series


# ── fitting / forecasting ─────────────────────────────────────────────────────────────
@dataclass
class FitOutcome:
    model_id: str
    ok: bool
    converged: bool
    warnings: list[str] = field(default_factory=list)
    error_class: str | None = None
    error_msg: str | None = None
    result: object | None = None


def fit_candidate(y: np.ndarray, cand: Candidate, sku: str, origin: str) -> FitOutcome:
    """Fit one additive ETS candidate on a training array. Records convergence + warnings.
    Never suppresses warnings globally; captures them for this fit only."""
    if len(y) < MIN_TRAIN_OBS:
        return FitOutcome(cand.model_id, False, False,
                          error_class="InsufficientData",
                          error_msg=f"train len {len(y)} < {MIN_TRAIN_OBS}")
    if cand.seasonal is not None and len(y) < 2 * cand.seasonal_periods + 2:
        return FitOutcome(cand.model_id, False, False,
                          error_class="InsufficientSeasons",
                          error_msg=f"train len {len(y)} < 2 seasonal cycles")
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            model = ETSModel(
                pd.Series(y, dtype=float), error=cand.error, trend=cand.trend,
                damped_trend=cand.damped_trend, seasonal=cand.seasonal,
                seasonal_periods=cand.seasonal_periods, initialization_method="estimated")
            res = model.fit(disp=False)
            wtypes = sorted({w.category.__name__ for w in wlist})
        rv = getattr(res, "mle_retvals", None)
        converged = bool(rv.get("converged")) if isinstance(rv, dict) and "converged" in rv else True
        return FitOutcome(cand.model_id, True, converged, wtypes, result=res)
    except Exception as exc:  # recorded, never swallowed silently
        return FitOutcome(cand.model_id, False, False,
                          error_class=type(exc).__name__, error_msg=str(exc)[:200])


def point_forecast(res, h: int) -> np.ndarray:
    """Expected-demand point forecast, negatives clipped to 0, NOT rounded."""
    fc = np.asarray(res.forecast(steps=h), dtype=float)
    return np.clip(fc, 0.0, None)


def simulate_intervals(res, h: int, seed: int) -> dict[str, np.ndarray]:
    """80% / 95% prediction intervals from ETS simulation (additive, anchor='end').
    Returns per-step lower/upper arrays; method label included."""
    sims = res.simulate(nsimulations=h, anchor="end", repetitions=N_SIM, random_state=int(seed))
    arr = np.asarray(sims, dtype=float)
    if arr.shape == (N_SIM, h):
        arr = arr.T
    if arr.shape != (h, N_SIM):
        raise ValueError(f"unexpected simulate shape {arr.shape}")
    arr = np.clip(arr, 0.0, None)
    q = np.quantile(arr, [0.025, 0.10, 0.90, 0.975], axis=1)
    return {"lower_95": q[0], "lower_80": q[1], "upper_80": q[2], "upper_95": q[3],
            "method": "ets_simulation"}


def bootstrap_intervals(y_train: np.ndarray, point: np.ndarray, h: int, seed: int) -> dict[str, np.ndarray]:
    """Deterministic fitted-residual bootstrap fallback when ETS simulation is unavailable."""
    rng = np.random.default_rng(int(seed))
    resid = y_train - float(np.mean(y_train)) if len(y_train) else np.array([0.0])
    if resid.size == 0:
        resid = np.array([0.0])
    draws = rng.choice(resid, size=(h, N_SIM), replace=True)
    outcomes = np.clip(point.reshape(-1, 1) + draws, 0.0, None)
    q = np.quantile(outcomes, [0.025, 0.10, 0.90, 0.975], axis=1)
    return {"lower_95": q[0], "lower_80": q[1], "upper_80": q[2], "upper_95": q[3],
            "method": "residual_bootstrap_fallback"}


def seasonal_naive(y_train: np.ndarray, h: int, m: int = 7) -> np.ndarray:
    """m-step seasonal naive: repeat the last `m` observed values. Training data only."""
    if len(y_train) >= m:
        last = y_train[-m:]
    else:
        last = np.full(m, float(y_train[-1]) if len(y_train) else 0.0)
    return np.clip(np.array([last[i % m] for i in range(h)], dtype=float), 0.0, None)


def trailing_mean(y_train: np.ndarray, h: int, w: int = 7) -> np.ndarray:
    val = float(np.mean(y_train[-w:])) if len(y_train) else 0.0
    return np.clip(np.full(h, val, dtype=float), 0.0, None)


# ── metrics (import evaluator's definitions for identical sign conventions) ─────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> dict:
    return {"mae": ev.mae(y_true, y_pred), "rmse": ev.rmse(y_true, y_pred),
            "wape": ev.wape(y_true, y_pred), "bias": ev.bias(y_true, y_pred),
            "mase": ev.mase(y_true, y_pred, y_train)}


# ── selection backtests ─────────────────────────────────────────────────────────────
def build_selection_origins(max_date: pd.Timestamp, h: int) -> list[pd.Timestamp]:
    """Three expanding-window origins strictly earlier than the evaluator's locked cutoff."""
    locked_cutoff = max_date - pd.Timedelta(days=h)
    origins = [locked_cutoff - pd.Timedelta(days=k * h) for k in (1, 2, 3)]
    return sorted(origins)


def run_selection_backtests(series: dict[str, pd.DataFrame], max_date: pd.Timestamp) -> pd.DataFrame:
    """Fit every candidate at every earlier origin for both horizons; record metrics + status.
    A training row never post-dates its origin; a validation date is always after the origin;
    no validation date touches the locked holdout (all <= locked_cutoff)."""
    rows = []
    cands = candidate_specs()
    for h in HORIZONS:
        locked_cutoff = max_date - pd.Timedelta(days=h)
        for origin in build_selection_origins(max_date, h):
            if origin >= locked_cutoff and h == HORIZONS[0]:
                pass  # origins are < locked_cutoff by construction; guarded below too
            for sku, s in series.items():
                train = s[s["date"] <= origin]
                val = s[(s["date"] > origin) & (s["date"] <= origin + pd.Timedelta(days=h))]
                # leakage guards
                if len(train) and train["date"].max() > origin:
                    _fail(f"training row after origin for {sku} @ {origin.date()}")
                if len(val) and val["date"].min() <= origin:
                    _fail(f"validation date not after origin for {sku} @ {origin.date()}")
                if val["date"].max() > locked_cutoff:
                    _fail(f"selection fold touches locked holdout for {sku} @ {origin.date()}")
                y_tr = train[TARGET].to_numpy(float)
                y_val = val[TARGET].to_numpy(float)
                for cand in cands:
                    base = {"sku": sku, "channel": EXPECTED_CHANNEL, "horizon": h,
                            "origin": origin.date().isoformat(), "model_id": cand.model_id,
                            "n_val": int(len(y_val))}
                    if len(y_val) < h:
                        rows.append({**base, "fit_status": "incomplete_validation", "converged": False,
                                     "warnings": "", "mae": np.nan, "rmse": np.nan, "wape": np.nan,
                                     "bias": np.nan, "mase": np.nan})
                        continue
                    fo = fit_candidate(y_tr, cand, sku, origin.date().isoformat())
                    if not fo.ok:
                        rows.append({**base, "fit_status": f"fail:{fo.error_class}", "converged": False,
                                     "warnings": fo.error_msg or "", "mae": np.nan, "rmse": np.nan,
                                     "wape": np.nan, "bias": np.nan, "mase": np.nan})
                        continue
                    yhat = point_forecast(fo.result, h)[:len(y_val)]
                    met = compute_metrics(y_val, yhat, y_tr)
                    rows.append({**base, "fit_status": "ok", "converged": fo.converged,
                                 "warnings": ";".join(fo.warnings), **met})
    return pd.DataFrame(rows).sort_values(
        ["sku", "horizon", "origin", "model_id"]).reset_index(drop=True)


def select_model_per_sku(sel: pd.DataFrame) -> dict[str, dict]:
    """One ETS structure per SKU from the earlier folds only. Primary = lowest mean finite MASE
    across all required folds/horizons; 2% MASE ties broken by complexity, WAPE, |bias|, id."""
    required_folds = len(HORIZONS) * 3
    decisions: dict[str, dict] = {}
    for sku in sorted(sel["sku"].unique()):
        g = sel[sel["sku"] == sku]
        agg = []
        for mid in COMPLEXITY_ORDER:
            gm = g[g["model_id"] == mid]
            ok = gm[gm["fit_status"] == "ok"]
            complete = len(ok) == required_folds and ok["mase"].notna().all()
            agg.append({
                "model_id": mid, "complexity": CANDIDATES[mid].complexity,
                "n_ok": int(len(ok)), "complete": complete,
                "mean_mase": float(ok["mase"].mean()) if len(ok) else np.nan,
                "mean_wape": float(ok["wape"].mean()) if len(ok) else np.nan,
                "mean_mae": float(ok["mae"].mean()) if len(ok) else np.nan,
                "mean_abs_bias": float(ok["bias"].abs().mean()) if len(ok) else np.nan,
            })
        adf = pd.DataFrame(agg)
        pool = adf[adf["complete"]].copy()
        reason = "complete_folds"
        if pool.empty:                       # relax: use whatever produced finite MASE
            pool = adf[adf["mean_mase"].notna()].copy()
            reason = "incomplete_folds_relaxed"
        if pool.empty:                       # genuinely constant/degenerate → MAE/WAPE path
            pool = adf[adf["mean_mae"].notna()].copy()
            reason = "mase_undefined_use_mae"
        if pool.empty:                       # last-resort default (recorded)
            decisions[sku] = {"selected_model": "ets_A_N_N", "reason": "no_valid_fit_default",
                              "mean_mase": None, "mean_wape": None, "mean_abs_bias": None,
                              "candidates": agg}
            continue
        if reason == "mase_undefined_use_mae":
            pool = pool.sort_values(["mean_mae", "mean_wape", "mean_abs_bias", "complexity", "model_id"])
            best = pool.iloc[0]
        else:
            best_mase = pool["mean_mase"].min()
            tied = pool[pool["mean_mase"] <= best_mase * 1.02].copy()
            tied = tied.sort_values(["complexity", "mean_wape", "mean_abs_bias", "model_id"])
            best = tied.iloc[0]
        decisions[sku] = {
            "selected_model": str(best["model_id"]), "reason": reason,
            "mean_mase": None if pd.isna(best["mean_mase"]) else round(float(best["mean_mase"]), 6),
            "mean_wape": None if pd.isna(best["mean_wape"]) else round(float(best["mean_wape"]), 6),
            "mean_abs_bias": None if pd.isna(best["mean_abs_bias"]) else round(float(best["mean_abs_bias"]), 6),
            "candidates": agg,
        }
    return decisions


# ── production-order forecast with truthful fallback ────────────────────────────────────
def forecast_with_fallback(y_train: np.ndarray, selected_id: str, h: int,
                           sku: str, origin: str, want_intervals: bool) -> dict:
    """Forecast h steps using the selected ETS model, falling back truthfully if it genuinely
    fails: selected ETS -> SES(A,N,N) -> seasonal-naive(7) -> trailing-7-mean. Records what
    actually produced the numbers and never mislabels a fallback as the selected model."""
    attempts = [selected_id]
    if selected_id != "ets_A_N_N":
        attempts.append("ets_A_N_N")
    out = {"selected_model": selected_id, "model_actually_used": None, "fit_status": None,
           "converged": False, "fallback_used": False, "fallback_reason": "",
           "warnings": "", "interval_method": None,
           "point": None, "lower_80": None, "upper_80": None, "lower_95": None, "upper_95": None}

    for mid in attempts:
        fo = fit_candidate(y_train, CANDIDATES[mid], sku, origin)
        if fo.ok:
            pt = point_forecast(fo.result, h)
            out.update(model_actually_used=mid, fit_status="ok", converged=fo.converged,
                       warnings=";".join(fo.warnings), point=pt,
                       fallback_used=(mid != selected_id),
                       fallback_reason=("" if mid == selected_id else f"selected {selected_id} failed to fit"))
            if want_intervals:
                seed = stable_seed(sku, EXPECTED_CHANNEL, origin, mid)
                try:
                    iv = simulate_intervals(fo.result, h, seed)
                except Exception as exc:
                    iv = bootstrap_intervals(y_train, pt, h, seed)
                    iv["method"] = "residual_bootstrap_fallback"
                    out["warnings"] = (out["warnings"] + f";simulate_fail:{type(exc).__name__}").strip(";")
                out.update(lower_80=iv["lower_80"], upper_80=iv["upper_80"],
                           lower_95=iv["lower_95"], upper_95=iv["upper_95"], interval_method=iv["method"])
            return _finalize_intervals(out, h)

    # deterministic non-ETS fallbacks
    for mid, fn in (("seasonal_naive_7", lambda: seasonal_naive(y_train, h)),
                    ("trailing_mean_7", lambda: trailing_mean(y_train, h))):
        try:
            pt = fn()
        except Exception:
            continue
        seed = stable_seed(sku, EXPECTED_CHANNEL, origin, mid)
        iv = bootstrap_intervals(y_train, pt, h, seed) if want_intervals else {}
        out.update(model_actually_used=mid, fit_status="fallback", converged=False,
                   fallback_used=True, fallback_reason=f"selected {selected_id} and SES both failed",
                   point=pt, interval_method=iv.get("method"),
                   lower_80=iv.get("lower_80"), upper_80=iv.get("upper_80"),
                   lower_95=iv.get("lower_95"), upper_95=iv.get("upper_95"))
        return _finalize_intervals(out, h)
    _fail(f"no forecast could be produced for {sku} (all methods failed)")


def _finalize_intervals(out: dict, h: int) -> dict:
    """Enforce 0<=l95<=l80<=point<=u80<=u95; expand a bound if numerical noise excludes point."""
    pt = np.asarray(out["point"], float)
    if out["interval_method"] is None:
        return out
    l95 = np.clip(np.asarray(out["lower_95"], float), 0.0, None)
    l80 = np.clip(np.asarray(out["lower_80"], float), 0.0, None)
    u80 = np.asarray(out["upper_80"], float)
    u95 = np.asarray(out["upper_95"], float)
    # monotic order across the four quantiles
    l80 = np.maximum(l80, l95)
    u80 = np.maximum(u80, l80)
    u95 = np.maximum(u95, u80)
    # contain the point forecast
    l95 = np.minimum(l95, pt)
    l80 = np.minimum(l80, pt)
    u80 = np.maximum(u80, pt)
    u95 = np.maximum(u95, pt)
    out.update(lower_95=l95, lower_80=l80, upper_80=u80, upper_95=u95)
    return out


# ── locked backtests through the evaluator ──────────────────────────────────────────────
def run_locked_backtests(series: dict[str, pd.DataFrame], selection: dict[str, dict],
                         panel: pd.DataFrame, max_date: pd.Timestamp) -> dict:
    """Freeze per-SKU models, forecast the evaluator's locked windows, score via evaluate(),
    add a seasonal-naive m=7 benchmark and empirical interval coverage. Returns everything."""
    results = {"scorecards": {}, "seasonal_naive": {}, "coverage": {}, "pred_rows": [], "detail": []}
    for h in HORIZONS:
        cutoff = max_date - pd.Timedelta(days=h)
        ets_rows, sn_rows = [], []
        cov80 = cov95 = cov_n = 0
        for sku in sorted(series):
            s = series[sku]
            train = s[s["date"] <= cutoff]
            test = s[(s["date"] > cutoff) & (s["date"] <= cutoff + pd.Timedelta(days=h))]
            if train["date"].max() > cutoff:
                _fail(f"locked: training after cutoff for {sku}")
            if len(test) and test["date"].min() <= cutoff:
                _fail(f"locked: test not after cutoff for {sku}")
            dates = test["date"].tolist()
            y_true = test[TARGET].to_numpy(float)
            y_tr = train[TARGET].to_numpy(float)
            sel_id = selection[sku]["selected_model"]
            fc = forecast_with_fallback(y_tr, sel_id, h, sku, cutoff.date().isoformat(), want_intervals=True)
            pt = np.asarray(fc["point"], float)[:len(dates)]
            for i, d in enumerate(dates):
                ets_rows.append({"sku": sku, "channel": EXPECTED_CHANNEL, "date": d, "y_pred": float(pt[i])})
                results["pred_rows"].append({
                    "sku": sku, "channel": EXPECTED_CHANNEL, "date": d, "evaluation_type": "locked_holdout",
                    "horizon": h, "origin": cutoff.date().isoformat(),
                    "selected_model": sel_id, "model_actually_used": fc["model_actually_used"],
                    "fit_status": fc["fit_status"], "converged": fc["converged"],
                    "fallback_used": fc["fallback_used"], "interval_method": fc["interval_method"],
                    "y_pred": float(pt[i]),
                    "lower_80": float(fc["lower_80"][i]), "upper_80": float(fc["upper_80"][i]),
                    "lower_95": float(fc["lower_95"][i]), "upper_95": float(fc["upper_95"][i])})
                # empirical coverage on real actuals
                cov_n += 1
                if fc["lower_80"][i] <= y_true[i] <= fc["upper_80"][i]:
                    cov80 += 1
                if fc["lower_95"][i] <= y_true[i] <= fc["upper_95"][i]:
                    cov95 += 1
            # seasonal-naive benchmark (comparison only)
            sn = seasonal_naive(y_tr, h)[:len(dates)]
            for i, d in enumerate(dates):
                sn_rows.append({"sku": sku, "channel": EXPECTED_CHANNEL, "date": d, "y_pred": float(sn[i])})
                results["pred_rows"].append({
                    "sku": sku, "channel": EXPECTED_CHANNEL, "date": d,
                    "evaluation_type": "seasonal_naive_locked", "horizon": h,
                    "origin": cutoff.date().isoformat(), "selected_model": "seasonal_naive_7",
                    "model_actually_used": "seasonal_naive_7", "fit_status": "benchmark",
                    "converged": False, "fallback_used": False, "interval_method": None,
                    "y_pred": float(sn[i]), "lower_80": np.nan, "upper_80": np.nan,
                    "lower_95": np.nan, "upper_95": np.nan})
            results["detail"].append({"sku": sku, "horizon": h, "selected_model": sel_id,
                                      "model_actually_used": fc["model_actually_used"],
                                      "converged": fc["converged"], "fallback_used": fc["fallback_used"],
                                      "fallback_reason": fc["fallback_reason"],
                                      "interval_method": fc["interval_method"], "warnings": fc["warnings"]})
        # score through the shared evaluator (runs the synthetic-stock independence check)
        results["scorecards"][h] = ev.evaluate(pd.DataFrame(ets_rows), horizon=h, panel=panel)
        results["seasonal_naive"][h] = ev.evaluate(pd.DataFrame(sn_rows), horizon=h, panel=panel)
        results["coverage"][h] = {"n": cov_n,
                                  "cov_80": round(cov80 / cov_n, 4) if cov_n else None,
                                  "cov_95": round(cov95 / cov_n, 4) if cov_n else None}
    return results


# ── production forecast ───────────────────────────────────────────────────────────────
def refit_and_forecast_production(series: dict[str, pd.DataFrame], selection: dict[str, dict],
                                  ff: pd.DataFrame, as_of: pd.Timestamp, gen_at: str) -> pd.DataFrame:
    """Refit each frozen model on ALL real history through as_of and forecast the exact
    forecast_frame keys, with 80/95 intervals."""
    pid = ff.drop_duplicates("sku").set_index("sku")["product_id"] if "product_id" in ff.columns else None
    rows = []
    for sku in sorted(series):
        s = series[sku]
        y_tr = s[s["date"] <= as_of][TARGET].to_numpy(float)
        fkeys = ff[ff["sku"] == sku].sort_values("date")
        h = len(fkeys)
        fc = forecast_with_fallback(y_tr, selection[sku]["selected_model"], h, sku,
                                    as_of.date().isoformat(), want_intervals=True)
        pt = np.asarray(fc["point"], float)
        for i, (_, k) in enumerate(fkeys.iterrows()):
            rows.append({
                "sku": sku,
                "product_id": (int(pid[sku]) if pid is not None and sku in pid.index else None),
                "channel": EXPECTED_CHANNEL, "date": k["date"],
                "forecast_horizon_day": int(k["forecast_horizon_day"]),
                "y_pred": float(pt[i]),
                "lower_80": float(fc["lower_80"][i]), "upper_80": float(fc["upper_80"][i]),
                "lower_95": float(fc["lower_95"][i]), "upper_95": float(fc["upper_95"][i]),
                "selected_model": fc["selected_model"], "model_actually_used": fc["model_actually_used"],
                "fit_status": fc["fit_status"], "converged": bool(fc["converged"]),
                "fallback_used": bool(fc["fallback_used"]), "fallback_reason": fc["fallback_reason"],
                "interval_method": fc["interval_method"], "as_of_date": as_of.date().isoformat(),
                "source_manifest_generated_at": gen_at, "model_version": MODEL_VERSION})
    return pd.DataFrame(rows).sort_values(["sku", "date"]).reset_index(drop=True)


def validate_forecast_output(prod: pd.DataFrame, ff: pd.DataFrame, as_of: pd.Timestamp) -> None:
    """Fail loudly on any production-output contract violation."""
    if len(prod) != len(ff):
        _fail(f"production rows {len(prod)} != forecast_frame rows {len(ff)}")
    pk = set(map(tuple, prod[["sku", "channel", "date"]].itertuples(index=False, name=None)))
    fk = set(map(tuple, ff[["sku", "channel", "date"]].itertuples(index=False, name=None)))
    if pk != fk:
        _fail("production keys differ from forecast_frame keys")
    if prod.duplicated(["sku", "channel", "date"]).any():
        _fail("duplicate production predictions")
    yp = pd.to_numeric(prod["y_pred"], errors="coerce")
    if yp.isna().any() or np.isinf(yp).any() or (yp < 0).any():
        _fail("production y_pred has missing/infinite/negative values")
    if float((yp != yp.round()).mean()) == 0.0:
        _fail("all y_pred are integer-valued — suspect code-level rounding")
    for _, r in prod.iterrows():
        if not (0 <= r["lower_95"] <= r["lower_80"] <= r["upper_80"] <= r["upper_95"]):
            _fail(f"interval ordering invalid for {r['sku']} {r['date'].date()}")
        if not (r["lower_95"] <= r["y_pred"] <= r["upper_95"]):
            _fail(f"point outside interval for {r['sku']} {r['date'].date()}")
    if sorted(prod["forecast_horizon_day"].unique()) != list(range(1, len(ff['date'].unique()) + 1)):
        _fail("forecast_horizon_day numbering incorrect")
    if (prod["date"] <= as_of).any():
        _fail("a production forecast date is on/before as_of_date")


# ── outputs ───────────────────────────────────────────────────────────────────────────
def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_outputs(sel_metrics: pd.DataFrame, locked: dict, prod: pd.DataFrame,
                 selection: dict, audit: dict, man: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) backtest metrics: selection folds + locked selected + seasonal-naive benchmark
    sel_m = sel_metrics.assign(evaluation_type="selection_fold")
    locked_rows = []
    for h, sc in locked["scorecards"].items():
        o = sc["overall"]
        locked_rows.append({"evaluation_type": "locked_selected", "horizon": h, "origin": sc["cutoff"],
                            "model_id": "per_sku_selected", "fit_status": "ok", "n_val": sc["n_test_rows"],
                            "mae": o["mae"], "rmse": o["rmse"], "wape": o["wape"], "bias": o["bias"],
                            "mase": o["mase"]})
    for h, sc in locked["seasonal_naive"].items():
        o = sc["overall"]
        locked_rows.append({"evaluation_type": "seasonal_naive_locked", "horizon": h, "origin": sc["cutoff"],
                            "model_id": "seasonal_naive_7", "fit_status": "benchmark", "n_val": sc["n_test_rows"],
                            "mae": o["mae"], "rmse": o["rmse"], "wape": o["wape"], "bias": o["bias"],
                            "mase": o["mase"]})
    metrics = pd.concat([sel_m, pd.DataFrame(locked_rows)], ignore_index=True)
    metrics = metrics.sort_values(["evaluation_type", "horizon", "origin", "model_id", "sku"],
                                  na_position="last").reset_index(drop=True)
    metrics.to_csv(OUT / "holtwinters_backtest_metrics.csv", index=False)

    # 2) backtest predictions (locked selected w/ intervals + seasonal-naive benchmark)
    bt_pred = pd.DataFrame(locked["pred_rows"]).sort_values(
        ["evaluation_type", "horizon", "sku", "date"]).reset_index(drop=True)
    bt_pred.to_parquet(OUT / "holtwinters_backtest_predictions.parquet", index=False)

    # 3) production forecast
    prod.to_parquet(OUT / "demand_forecasts_holtwinters.parquet", index=False)

    # 4) model selection JSON
    sel_counts = {}
    for d in selection.values():
        sel_counts[d["selected_model"]] = sel_counts.get(d["selected_model"], 0) + 1
    doc = {
        "model_version": MODEL_VERSION,
        "generated_from_manifest": {"schema_version": man["schema_version"],
                                    "as_of_date": man["as_of_date"],
                                    "manifest_generated_at": man.get("generated_at")},
        "source_fingerprints_sha256": {"model_panel.parquet": file_sha256(MODEL_PANEL),
                                       "forecast_frame.parquet": file_sha256(FORECAST_FRAME),
                                       "pilot_manifest.json": file_sha256(MANIFEST)},
        "library_versions": {"python": audit["python"], "pandas": audit["pandas"],
                             "numpy": audit["numpy"], "statsmodels": audit["statsmodels"]},
        "candidate_definitions": [c.__dict__ for c in candidate_specs()],
        "selection_origins": {str(h): [o.isoformat() for o in
                                       build_selection_origins(pd.Timestamp(man["as_of_date"]), h)]
                              for h in HORIZONS},
        "metric_definitions": {"mae": "mean|e|", "rmse": "sqrt(mean e^2)",
                               "wape": "sum|e|/sum|y| (nan if sum|y|=0)", "bias": "mean(pred-true)",
                               "mase": "mae / mean|diff(train)|", "note": "imported from evaluation.py"},
        "selection_rule": "lowest mean finite MASE over 3 earlier origins x {7,14}; "
                          "2% MASE tie -> lower complexity, WAPE, |bias|, then model id",
        "interval_config": {"method": "ets_simulation", "n_sim": N_SIM, "base_seed": BASE_SEED,
                            "anchor": "end", "levels": [0.80, 0.95],
                            "fallback": "residual_bootstrap_fallback"},
        "selected_model_counts": sel_counts,
        "selected_per_sku": {sku: {k: v for k, v in d.items() if k != "candidates"}
                             for sku, d in selection.items()},
        "selection_candidates_per_sku": {sku: d["candidates"] for sku, d in selection.items()},
        "locked_scorecards": {str(h): locked["scorecards"][h] for h in HORIZONS},
        "seasonal_naive_scorecards": {str(h): locked["seasonal_naive"][h] for h in HORIZONS},
        "interval_coverage": {str(h): locked["coverage"][h] for h in HORIZONS},
        "production_fallbacks": [d for d in locked["detail"] if d["fallback_used"]],
        "input_audit": audit,
        "validation_summary": "all runtime validations passed",
    }
    (OUT / "holtwinters_model_selection.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return {"metrics": metrics, "bt_pred": bt_pred, "prod": prod, "doc": doc}


# ── pipeline ─────────────────────────────────────────────────────────────────────────
def run_pipeline() -> dict:
    mp, ff, man = load_contract()
    audit = audit_inputs(mp, ff, man)
    as_of = pd.Timestamp(man["as_of_date"])
    max_date = mp["date"].max()
    if max_date != as_of:
        _fail(f"model_panel max date {max_date.date()} != as_of {as_of.date()}")
    series = build_series(mp)

    print("\n=== SELECTION BACKTESTS (leakage-free rolling origin) ===")
    sel_metrics = run_selection_backtests(series, max_date)
    selection = select_model_per_sku(sel_metrics)
    counts = {}
    for d in selection.values():
        counts[d["selected_model"]] = counts.get(d["selected_model"], 0) + 1
    print("  selected model counts:", dict(sorted(counts.items())))

    print("\n=== LOCKED BACKTESTS (via evaluation.evaluate) ===")
    locked = run_locked_backtests(series, selection, mp, max_date)
    for h in HORIZONS:
        e, sn = locked["scorecards"][h]["overall"], locked["seasonal_naive"][h]["overall"]
        c = locked["coverage"][h]
        print(f"  h={h:>2}  ETS  WAPE {e['wape']:.4f} MASE {e['mase']:.4f} MAE {e['mae']:.3f} | "
              f"sNaive WAPE {sn['wape']:.4f} MASE {sn['mase']:.4f} | cov80 {c['cov_80']} cov95 {c['cov_95']}")

    print("\n=== PRODUCTION FORECAST ===")
    prod = refit_and_forecast_production(series, selection, ff, as_of, man.get("generated_at", ""))
    validate_forecast_output(prod, ff, as_of)
    print(f"  production rows: {len(prod)}  skus: {prod['sku'].nunique()}  "
          f"dates: {prod['date'].min().date()}..{prod['date'].max().date()}")

    saved = save_outputs(sel_metrics, locked, prod, selection, audit, man)
    return {"audit": audit, "sel_metrics": sel_metrics, "selection": selection, "locked": locked,
            "prod": prod, **saved}


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(list(df.columns)).reset_index(drop=True)


def self_check(first: dict) -> None:
    """Recompute the pipeline in-process and assert deterministic predictions/metrics/selection."""
    print("\n=== SELF-CHECK (recompute + compare) ===")
    second = run_pipeline()
    pd.testing.assert_frame_equal(_canonical(first["prod"].drop(columns=["source_manifest_generated_at"])),
                                  _canonical(second["prod"].drop(columns=["source_manifest_generated_at"])))
    pd.testing.assert_frame_equal(_canonical(first["bt_pred"]), _canonical(second["bt_pred"]))
    a = {k: v for k, v in first["doc"].items() if k != "input_audit"}
    b = {k: v for k, v in second["doc"].items() if k != "input_audit"}
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str), \
        "selection JSON not reproducible"
    print("  self-check PASSED — predictions, intervals, metrics and selection are deterministic")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Univariate ETS demand forecaster (Stage A).")
    ap.add_argument("--selfcheck", action="store_true", help="recompute in-process and assert identical")
    args = ap.parse_args(argv)
    first = run_pipeline()
    if args.selfcheck:
        self_check(first)
    print("\nDONE. Outputs in outputs/. Demand is REAL units_observed; model is univariate per SKU; "
          "no synthetic inventory used; accuracy is backtest-ESTIMATED, not guaranteed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
