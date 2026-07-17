"""evaluation.py — the SHARED scorecard. All three models import from here so
every result is measured identically. Owned by Aiman."""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

TEST_WEEKS = 5                       # hold out the last 5 weeks of every SKU
PROC = Path("data/processed")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)


def load_weekly_sales() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "weekly_sales.parquet")
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df.sort_values(["sku", "week_start"]).reset_index(drop=True)


def load_signals() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "weekly_signals.parquet")
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df.sort_values("week_start").reset_index(drop=True)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))

def mase(y_true, y_pred, y_train) -> float:
    """Error scaled by the naive one-step error on the TRAINING series.
    < 1 beats naive, > 1 is worse than naive. Cannot be gamed by flat zero."""
    y_train = np.asarray(y_train, dtype=float)
    denom = np.mean(np.abs(np.diff(y_train))) if len(y_train) > 1 else np.nan
    if not denom or np.isnan(denom):     # a flat/degenerate training series
        return np.nan
    return mae(y_true, y_pred) / denom


def score_model(preds: pd.DataFrame, sales: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """preds columns: [sku, week_start, y_true, y_pred]. Returns per-SKU + an
    'ALL' average row; writes outputs/metrics_<model>.csv and preds_<model>.csv."""
    rows = []
    for sku, g in preds.groupby("sku"):
        train = (sales.loc[sales.sku == sku]
                 .sort_values("week_start")["units"].iloc[:-TEST_WEEKS])
        rows.append({"sku": sku,
                     "mae": mae(g.y_true, g.y_pred),
                     "rmse": rmse(g.y_true, g.y_pred),
                     "mase": mase(g.y_true, g.y_pred, train)})
    m = pd.DataFrame(rows)
    overall = {"sku": "ALL", "mae": m.mae.mean(), "rmse": m.rmse.mean(),
               "mase": m.mase.dropna().mean()}
    m = pd.concat([m, pd.DataFrame([overall])], ignore_index=True)
    preds.to_csv(OUT / f"preds_{model_name}.csv", index=False)
    m.to_csv(OUT / f"metrics_{model_name}.csv", index=False)
    return m
