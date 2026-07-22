"""evaluation.py — the SHARED demand-forecast scorecard for the daily naheed_web pilot.

DEMAND FORECAST — "Real historical sales backtesting".
  Scored on REAL `units_observed` with CHRONOLOGICAL train/test windows and
  `forecast_training_eligible` (real-data quality + sufficient history) ONLY. It never
  drops rows using synthetic stock. A runtime assertion proves that changing the
  synthetic `stock_on_hand` columns leaves the scored row set byte-identical
  (see `assert_synthetic_independence`).

Stockout-risk and reorder recommendations are computed downstream from these forecasts
plus `inventory_context.parquet`; they are pilot estimates (synthetic stock), NOT validated
against real Naheed stockouts, and are not scored here.

All three demand models import `evaluate`; they never pass y_true (truth stays inside).

Do not edit this to fit a model. Import it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
HORIZONS = (7, 14)
REQUIRED_PRED_COLS = ("sku", "channel", "date", "y_pred")

# synthetic-stock columns that, though present in model_panel, must never influence demand scoring
SYNTHETIC_STOCK_COLS = (
    "stock_on_hand", "stock_on_hand_is_synthetic", "stock_source", "stock_generation_version",
)


# ── loaders (repo-root relative) ────────────────────────────────────────────────
def load_model_panel() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "model_panel.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["sku", "channel", "date"]).reset_index(drop=True)


def load_forecast_frame() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "forecast_frame.parquet")
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


# ── chronological backtest (demand) ─────────────────────────────────────────────
def backtest_split(panel: pd.DataFrame, horizon: int) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    """Single chronological holdout: last `horizon` days = test, everything before = train.
    Time-ordered by construction — never a random split."""
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon}")
    max_date = panel["date"].max()
    cutoff = max_date - pd.Timedelta(days=horizon)
    return cutoff, panel[panel["date"] <= cutoff], panel[panel["date"] > cutoff]


def rolling_origin_cutoffs(panel: pd.DataFrame, horizon: int, n_folds: int = 3) -> list[pd.Timestamp]:
    """A few backtest origins spaced `horizon` days apart (proportionate to a small pilot)."""
    max_date = panel["date"].max()
    return [max_date - pd.Timedelta(days=horizon * k) for k in range(1, n_folds + 1)]


def _truth_for(test: pd.DataFrame) -> pd.DataFrame:
    """Test truth held inside the evaluator. Rows are selected on REAL data quality only
    (`forecast_training_eligible`) — NEVER on synthetic stock/stockout labels."""
    eligible = test["forecast_training_eligible"].astype(bool) if "forecast_training_eligible" in test else pd.Series(True, index=test.index)
    t = test[eligible].copy()
    return t[["sku", "channel", "date", "units_observed"]].rename(columns={"units_observed": "y_true"})


def assert_synthetic_independence(panel: pd.DataFrame, horizon: int = 14) -> None:
    """Prove that changing the synthetic stock columns cannot change the demand-eval rows.

    Corrupts every synthetic-stock column present, rebuilds the test truth, and asserts the
    scored (sku, channel, date) keys and y_true are byte-identical.
    """
    _, _, test = backtest_split(panel, horizon)
    base = _truth_for(test).sort_values(["sku", "channel", "date"]).reset_index(drop=True)

    mutated = panel.copy()
    for c in SYNTHETIC_STOCK_COLS:
        if c in mutated.columns:
            col = mutated[c]
            if col.dtype == bool:
                mutated[c] = ~col
            elif np.issubdtype(col.dtype, np.number):
                mutated[c] = col.fillna(0) + 999
            else:
                mutated[c] = "MUTATED"
    _, _, test2 = backtest_split(mutated, horizon)
    after = _truth_for(test2).sort_values(["sku", "channel", "date"]).reset_index(drop=True)

    if not base.equals(after):
        raise AssertionError(
            "Demand evaluation is NOT independent of synthetic stock — row set changed "
            "when synthetic-stock columns were mutated. This is leakage and must be fixed.")


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
    """Score demand predictions against internal REAL test truth for `horizon`.

    "Real historical sales backtesting". `preds` needs [sku, channel, date, y_pred];
    optional [lower_bound, upper_bound]. Raises ValueError on any invalid submission.
    """
    panel = load_model_panel() if panel is None else panel
    assert_synthetic_independence(panel, horizon)     # runtime leakage guard
    cutoff, train, test = backtest_split(panel, horizon)
    truth = _truth_for(test)
    _validate_predictions(preds, truth)

    p = preds.copy()
    p["date"] = pd.to_datetime(p["date"])
    df = truth.merge(p, on=["sku", "channel", "date"], how="left")

    per_sku = {}
    for (sku, ch), g in df.groupby(["sku", "channel"]):
        tr = train[(train.sku == sku) & (train.channel == ch)]["units_observed"]
        per_sku[f"{sku}|{ch}"] = {"mase": mase(g.y_true, g.y_pred, tr),
                                  "mae": mae(g.y_true, g.y_pred),
                                  "wape": wape(g.y_true, g.y_pred), "n": int(len(g))}
    mase_vals = [v["mase"] for v in per_sku.values() if not np.isnan(v["mase"])]

    result = {
        "evaluation_type": "real_historical_sales_backtesting",
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
