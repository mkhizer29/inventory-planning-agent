"""evaluation.py — the SHARED scorecard for the daily, ecommerce-only pilot.

All three models import this so every result is measured identically. It:
  * loads data/processed/model_panel.parquet and forecast_features.parquet (repo-root paths),
  * builds a CHRONOLOGICAL backtest holdout for a 7- or 14-day horizon,
  * keeps the test truth INSIDE the evaluator (models never pass y_true),
  * validates submitted predictions (keys: sku, channel, date, y_pred; optional bounds),
  * reports MAE, RMSE, MASE, WAPE, bias — overall, per-SKU and per-channel — plus
    prediction-interval coverage when lower/upper bounds are supplied.

Do not edit this to fit a model. Import it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
HORIZONS = (7, 14)                       # the pilot's forecast horizons (days)
REQUIRED_PRED_COLS = ("sku", "channel", "date", "y_pred")


# ── loaders (repo-root relative) ────────────────────────────────────────────────
def load_model_panel() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "model_panel.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["sku", "channel", "date"]).reset_index(drop=True)


def load_forecast_features() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "forecast_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["sku", "channel", "date"]).reset_index(drop=True)


# ── point metrics ────────────────────────────────────────────────────────────────
def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def wape(y_true, y_pred) -> float:
    yt = np.asarray(y_true, float)
    denom = np.sum(np.abs(yt))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(yt - np.asarray(y_pred, float))) / denom)


def bias(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_pred, float) - np.asarray(y_true, float)))


def mase(y_true, y_pred, y_train) -> float:
    """Test MAE scaled by the in-sample 1-step naive error of the training series."""
    yt = np.asarray(y_train, float)
    denom = np.mean(np.abs(np.diff(yt))) if len(yt) > 1 else np.nan
    if not denom or np.isnan(denom):
        return float("nan")
    return mae(y_true, y_pred) / denom


# ── chronological backtest ─────────────────────────────────────────────────────────
def backtest_split(panel: pd.DataFrame, horizon: int) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    """Single chronological holdout: last `horizon` days = test, everything before = train."""
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon}")
    max_date = panel["date"].max()
    cutoff = max_date - pd.Timedelta(days=horizon)
    train = panel[panel["date"] <= cutoff]
    test = panel[panel["date"] > cutoff]
    return cutoff, train, test


def rolling_origin_cutoffs(panel: pd.DataFrame, horizon: int, n_folds: int = 3) -> list[pd.Timestamp]:
    """A few backtest origins spaced `horizon` days apart (proportionate to a small pilot)."""
    max_date = panel["date"].max()
    return [max_date - pd.Timedelta(days=horizon * k) for k in range(1, n_folds + 1)]


def _truth_for(test: pd.DataFrame) -> pd.DataFrame:
    """Test truth held inside the evaluator. Censored (stockout) days are dropped from scoring."""
    t = test[~test["demand_censored"].astype(bool)].copy()
    return t[["sku", "channel", "date", "units_observed"]].rename(columns={"units_observed": "y_true"})


def _validate_predictions(preds: pd.DataFrame, truth: pd.DataFrame) -> None:
    missing_cols = [c for c in REQUIRED_PRED_COLS if c not in preds.columns]
    if missing_cols:
        raise ValueError(f"predictions missing columns: {missing_cols}")
    p = preds.copy()
    p["date"] = pd.to_datetime(p["date"])
    if p.duplicated(["sku", "channel", "date"]).any():
        raise ValueError("duplicate predictions for a sku+channel+date")
    yp = pd.to_numeric(p["y_pred"], errors="coerce")
    if yp.isna().any() or np.isinf(yp).any():
        raise ValueError("predictions contain NaN or infinite y_pred")
    if (yp < 0).any():
        raise ValueError("predictions contain negative y_pred")
    key = ["sku", "channel", "date"]
    pk = set(map(tuple, p[key].itertuples(index=False, name=None)))
    tk = set(map(tuple, truth[key].itertuples(index=False, name=None)))
    if pk - tk:
        raise ValueError(f"predictions contain {len(pk - tk)} unexpected keys not in the test window")
    if tk - pk:
        raise ValueError(f"predictions are missing {len(tk - pk)} required sku/channel/date keys")


def _grouped(df: pd.DataFrame, by: str | None) -> dict:
    def block(g):
        return {"mae": mae(g.y_true, g.y_pred), "rmse": rmse(g.y_true, g.y_pred),
                "wape": wape(g.y_true, g.y_pred), "bias": bias(g.y_true, g.y_pred),
                "n": int(len(g))}
    if by is None:
        return block(df)
    return {k: block(g) for k, g in df.groupby(by)}


def evaluate(preds: pd.DataFrame, horizon: int = 14, panel: pd.DataFrame | None = None) -> dict:
    """Score a model's predictions against the internal test truth for `horizon`.

    `preds` must have [sku, channel, date, y_pred]; optional [lower_bound, upper_bound].
    Raises ValueError on any invalid/incomplete submission.
    """
    panel = load_model_panel() if panel is None else panel
    cutoff, train, test = backtest_split(panel, horizon)
    truth = _truth_for(test)
    _validate_predictions(preds, truth)

    p = preds.copy()
    p["date"] = pd.to_datetime(p["date"])
    df = truth.merge(p, on=["sku", "channel", "date"], how="left")

    # MASE needs each series' training history
    per_sku = {}
    for (sku, ch), g in df.groupby(["sku", "channel"]):
        tr = train[(train.sku == sku) & (train.channel == ch)]["units_observed"]
        per_sku[f"{sku}|{ch}"] = {"mase": mase(g.y_true, g.y_pred, tr),
                                  "mae": mae(g.y_true, g.y_pred),
                                  "wape": wape(g.y_true, g.y_pred), "n": int(len(g))}
    mase_vals = [v["mase"] for v in per_sku.values() if not np.isnan(v["mase"])]

    result = {
        "horizon": horizon,
        "cutoff": cutoff.date().isoformat(),
        "n_test_rows": int(len(df)),
        "overall": {**_grouped(df, None), "mase": float(np.mean(mase_vals)) if mase_vals else float("nan")},
        "per_channel": _grouped(df, "channel"),
        "per_sku": per_sku,
    }
    if {"lower_bound", "upper_bound"}.issubset(preds.columns):
        pm = truth.merge(p, on=["sku", "channel", "date"], how="left")
        inside = (pm["y_true"] >= pm["lower_bound"]) & (pm["y_true"] <= pm["upper_bound"])
        result["interval_coverage"] = round(float(inside.mean()), 4)
    return result
