"""Shared, run-aware forecasting contract for the Inventory Planning Agent.

One place that every demand model (Baselines, Holt-Winters, and — later — Aqib's
LightGBM) uses to agree on:

  * the FUTURE prediction schema         (validate_future_predictions)
  * the BACKTEST prediction schema       (validate_backtest_predictions)
  * the SCORECARD schema                 (scorecard_row, from evaluation.evaluate)
  * atomic, partial-write-safe output    (write_dataframe_atomic / write_json_atomic)
  * a deterministic input fingerprint    (dataset_fingerprint)

Design notes
------------
* Column names `date` and `y_pred` are FIXED. They are NOT renamed to
  `forecast_date` / `point_forecast`, because evaluation.py requires `date` and
  `y_pred`, the Forecast Explorer dashboard reads `date` and `y_pred`, and the
  later LightGBM adapter must reuse this exact contract.
* This module is intentionally small and function-based — no classes, no
  frameworks, no I/O beyond the atomic writers, and it never imports a model.

LightGBM adapter (ACTIVE — implemented in ``src/lgbm_global.py``)
----------------------------------------------------------------
Aqib's pooled LightGBM model (``src/lgbm_global.py``) uses this same shared CLI
and output contract, exactly like Baselines and Holt-Winters::

    python src/lgbm_global.py \
        --model-panel   <run>/processed/model_panel.parquet \
        --forecast-frame <run>/processed/forecast_frame.parquet \
        --manifest      <run>/processed/pilot_manifest.json \
        --output-dir    <run>/outputs \
        --horizons 7 14

writing exactly these files into --output-dir::

    lightgbm_backtest_predictions.parquet   (BACKTEST_REQUIRED_COLUMNS, model="lightgbm")
    lightgbm_scorecard.csv                  (SCORECARD_COLUMNS, via evaluation.evaluate)
    future_forecast_lightgbm.parquet        (FUTURE_REQUIRED_COLUMNS, model="lightgbm")
    lightgbm_run_summary.json               (dataset_fingerprint + run metadata)

Its backtest is a fixed-origin RECURSIVE multi-step forecast (predicted demand
feeds future lags) — never teacher-forced with held-out actuals. The model/lgbm
branch's previously generated CSVs are NOT model source and are never copied
into outputs/.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── shared column contracts ─────────────────────────────────────────────────────────
FUTURE_REQUIRED_COLUMNS = [
    "sku", "channel", "date", "y_pred", "model", "model_version",
    "as_of_date", "forecast_horizon_day",
]
FUTURE_OPTIONAL_COLUMNS = [
    "product_id", "sku_name", "lower_80", "upper_80", "lower_95", "upper_95",
    "selected_model", "model_actually_used", "fit_status", "converged",
    "fallback_used", "fallback_reason", "interval_method",
]
BACKTEST_REQUIRED_COLUMNS = [
    "sku", "channel", "date", "y_pred", "model", "horizon", "origin", "evaluation_type",
]
BACKTEST_OPTIONAL_COLUMNS = [
    "lower_80", "upper_80", "lower_95", "upper_95",
    "selected_model", "model_actually_used", "fit_status", "converged",
    "fallback_used", "fallback_reason", "interval_method",
]
SCORECARD_COLUMNS = [
    "model", "horizon", "wape", "mase", "mae", "rmse", "bias",
    "n_rows", "n_skus", "n_channels", "cutoff", "evaluation_type",
]

_KEY = ["sku", "channel", "date"]
_INTERVAL_ORDER = ["lower_95", "lower_80", "y_pred", "upper_80", "upper_95"]


# ── small internal guards ───────────────────────────────────────────────────────────
def _require_columns(df: pd.DataFrame, cols: list[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} missing required columns: {missing}")


def _no_truth_column(df: pd.DataFrame) -> None:
    if "y_true" in df.columns:
        raise ValueError("predictions must not contain a 'y_true' column (truth stays in the evaluator).")


def _finite_nonneg_ypred(df: pd.DataFrame) -> pd.Series:
    yp = pd.to_numeric(df["y_pred"], errors="coerce")
    if yp.isna().any() or np.isinf(yp).any():
        raise ValueError("y_pred contains NaN or infinite values.")
    if (yp < 0).any():
        raise ValueError("y_pred contains negative values.")
    return yp


def _single_model_column(df: pd.DataFrame, model_name: str) -> None:
    vals = df["model"].astype("string").fillna("").str.strip()
    uniq = set(vals.unique())
    if uniq != {str(model_name).strip()} or "" in uniq:
        raise ValueError(
            f"'model' column must be exactly '{model_name}' on every row; found {sorted(uniq)}")


# ── future-prediction contract ──────────────────────────────────────────────────────
def validate_future_predictions(predictions: pd.DataFrame, forecast_frame: pd.DataFrame,
                                manifest: dict, model_name: str) -> pd.DataFrame:
    """Validate a model's FUTURE forecast against the shared contract + forecast_frame.

    Returns a deterministically sorted copy. Raises ValueError on any violation.
    """
    df = predictions.copy()
    _require_columns(df, FUTURE_REQUIRED_COLUMNS, "future predictions")
    _no_truth_column(df)
    _single_model_column(df, model_name)

    df["date"] = pd.to_datetime(df["date"])
    ff = forecast_frame.copy()
    ff["date"] = pd.to_datetime(ff["date"])

    if df.duplicated(_KEY).any():
        raise ValueError("future predictions contain duplicate sku/channel/date keys.")

    pk = set(map(tuple, df[_KEY].itertuples(index=False, name=None)))
    fk = set(map(tuple, ff[_KEY].itertuples(index=False, name=None)))
    if pk - fk:
        raise ValueError(f"future predictions contain {len(pk - fk)} keys not in the forecast_frame.")
    if fk - pk:
        raise ValueError(f"future predictions are missing {len(fk - pk)} forecast_frame keys.")

    # channel and sku sets must be exactly those of the forecast frame
    if set(df["channel"].unique()) - set(ff["channel"].unique()):
        raise ValueError("future predictions contain channels absent from the forecast_frame.")
    if set(df["sku"].astype(str).unique()) - set(ff["sku"].astype(str).unique()):
        raise ValueError("future predictions contain SKUs absent from the forecast_frame.")

    _finite_nonneg_ypred(df)

    # all dates strictly after as_of_date
    as_of = pd.Timestamp(manifest["as_of_date"])
    if (df["date"] <= as_of).any():
        raise ValueError("future predictions contain a date on or before the manifest as_of_date.")

    # forecast_horizon_day must match the forecast_frame per key
    if "forecast_horizon_day" in ff.columns:
        merged = df.merge(ff[_KEY + ["forecast_horizon_day"]], on=_KEY,
                          how="left", suffixes=("", "_ff"))
        if (pd.to_numeric(merged["forecast_horizon_day"], errors="coerce")
                != pd.to_numeric(merged["forecast_horizon_day_ff"], errors="coerce")).any():
            raise ValueError("forecast_horizon_day does not match the forecast_frame.")

    _validate_interval_ordering(df)
    return df.sort_values(_KEY).reset_index(drop=True)


def _validate_interval_ordering(df: pd.DataFrame) -> None:
    """When interval columns are present, require 0 <= l95 <= l80 <= y_pred <= u80 <= u95."""
    have = [c for c in ("lower_80", "upper_80", "lower_95", "upper_95") if c in df.columns]
    if not have:
        return
    if len(have) != 4:
        raise ValueError(f"interval columns must all be present or all absent; found {have}")
    cols = {c: pd.to_numeric(df[c], errors="coerce") for c in _INTERVAL_ORDER}
    for c, s in cols.items():
        if s.isna().any() or np.isinf(s).any():
            raise ValueError(f"interval column '{c}' has NaN/infinite values.")
    ladder = [cols["lower_95"], cols["lower_80"], cols["y_pred"], cols["upper_80"], cols["upper_95"]]
    if (ladder[0] < 0).any():
        raise ValueError("interval lower_95 is negative.")
    for lo, hi, lname, hname in zip(ladder, ladder[1:], _INTERVAL_ORDER, _INTERVAL_ORDER[1:]):
        if (lo > hi).any():
            raise ValueError(f"interval ordering violated: {lname} > {hname} on some rows.")


# ── backtest-prediction contract ────────────────────────────────────────────────────
def validate_backtest_predictions(predictions: pd.DataFrame, model_name: str,
                                  allowed_horizons: tuple[int, ...] = (7, 14)) -> pd.DataFrame:
    """Validate a model's BACKTEST predictions against the shared contract.

    Returns a deterministically sorted copy. Raises ValueError on any violation.
    """
    df = predictions.copy()
    _require_columns(df, BACKTEST_REQUIRED_COLUMNS, "backtest predictions")
    _no_truth_column(df)
    _single_model_column(df, model_name)

    bad = set(pd.to_numeric(df["horizon"], errors="coerce").dropna().astype(int)) - set(allowed_horizons)
    if bad or pd.to_numeric(df["horizon"], errors="coerce").isna().any():
        raise ValueError(f"backtest horizons must be within {allowed_horizons}; found offenders {sorted(bad)}")

    _finite_nonneg_ypred(df)

    df["date"] = pd.to_datetime(df["date"])          # must parse
    key = ["model", "horizon", "evaluation_type", "sku", "channel", "date"]
    if df.duplicated(key).any():
        raise ValueError("backtest predictions contain duplicate model/horizon/evaluation_type/sku/channel/date keys.")

    return df.sort_values(key).reset_index(drop=True)


# ── scorecard row (from evaluation.evaluate output) ─────────────────────────────────
def scorecard_row(model_name: str, horizon: int, evaluation_result: dict,
                  evaluation_type: str = "locked_holdout") -> dict:
    """Serialize one evaluation.evaluate() result into the exact SCORECARD_COLUMNS."""
    overall = evaluation_result["overall"]
    per_sku = evaluation_result.get("per_sku", {})
    skus = {k.rsplit("|", 1)[0] for k in per_sku}
    channels = {k.rsplit("|", 1)[1] for k in per_sku if "|" in k}
    row = {
        "model": model_name,
        "horizon": int(horizon),
        "wape": overall.get("wape"),
        "mase": overall.get("mase"),
        "mae": overall.get("mae"),
        "rmse": overall.get("rmse"),
        "bias": overall.get("bias"),
        "n_rows": int(evaluation_result.get("n_test_rows", overall.get("n", 0))),
        "n_skus": len(skus),
        "n_channels": len(channels),
        "cutoff": evaluation_result.get("cutoff"),
        "evaluation_type": evaluation_type,
    }
    return {c: row[c] for c in SCORECARD_COLUMNS}


# ── atomic writers ──────────────────────────────────────────────────────────────────
def write_dataframe_atomic(dataframe: pd.DataFrame, path: "str | os.PathLike", format: str) -> Path:
    """Write a DataFrame atomically (temp file + os.replace); never leave a partial file."""
    path = Path(path)
    fmt = format.lower()
    if fmt not in ("parquet", "csv"):
        raise ValueError(f"unsupported format {format!r}; use 'parquet' or 'csv'.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        if fmt == "parquet":
            dataframe.to_parquet(tmp, index=False)
        else:
            dataframe.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def write_json_atomic(document: dict, path: "str | os.PathLike") -> Path:
    """Write JSON atomically and deterministically (sorted keys)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(document, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


# ── dataset fingerprint ─────────────────────────────────────────────────────────────
def _file_sha256(path: "str | os.PathLike") -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_fingerprint(model_panel_path: "str | os.PathLike",
                        forecast_frame_path: "str | os.PathLike",
                        manifest_path: "str | os.PathLike") -> str:
    """One deterministic SHA-256 over the three shared input files' CONTENTS.

    Two model runs that read byte-identical model_panel/forecast_frame/manifest
    produce the same fingerprint, which proves they were scored on the same data.
    """
    h = hashlib.sha256()
    for label, p in (("model_panel", model_panel_path),
                     ("forecast_frame", forecast_frame_path),
                     ("manifest", manifest_path)):
        h.update(label.encode("utf-8"))
        h.update(b"\x00")
        h.update(_file_sha256(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
