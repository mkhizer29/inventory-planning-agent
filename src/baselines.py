"""Daily baseline forecasting models for the Inventory Planning Agent pilot.

Contract: v4 (real-demand / synthetic-stock), daily / ecommerce-only (see
Doc1_Aiman_Baselines.md). Row eligibility and the train/test split come from
the shared evaluator; the future frame is forecast_frame.parquet. Rows are
selected on `forecast_training_eligible` (real-data quality), never on
synthetic stock. Aiman owns this file. It implements four simple, honest
forecasts -- last_day_naive, seasonal_naive_7, moving_average_7,
moving_average_14 -- for the 7-day and 14-day horizons, and scores them with
the shared evaluator in evaluation.py. Khizer's Holt model and Aqib's
LightGBM model must beat whichever of these has the lowest WAPE.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from evaluation import HORIZONS, backtest_split, evaluate, load_forecast_frame, load_model_panel

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

ForecastFn = Callable[[pd.DataFrame, pd.DatetimeIndex, pd.Timestamp], pd.Series]


def _eligible_history(train: pd.DataFrame) -> pd.DataFrame:
    """Historical rows whose target may feed a baseline forecast, date-sorted.

    v4 policy: every model_panel row is kept, but only rows with
    forecast_training_eligible == True may contribute a target (units_observed)
    value to a forecast. This is a target-usage mask, never a predictive
    feature. Ineligible rows (e.g. lag/rolling warm-up days) are skipped, so
    the "most recent N observations" are the N most recent ELIGIBLE ones and
    need not be contiguous calendar days.
    """
    if "forecast_training_eligible" not in train.columns:
        raise ValueError(
            "training frame is missing 'forecast_training_eligible' (v4 target-usage mask)."
        )
    return train[train["forecast_training_eligible"].astype(bool)].sort_values("date")


# --- the four baseline forecasters -----------------------------------------
# Each takes the training rows for one sku+channel, the exact list of dates
# that must be forecast, and the train/test cutoff date. Each returns a
# pandas Series indexed by date. None of them ever look at test-period data,
# and each restricts its target inputs to forecast_training_eligible rows.

def last_day_naive(train: pd.DataFrame, required_dates: pd.DatetimeIndex, cutoff: pd.Timestamp) -> pd.Series:
    """Repeat the most recent ELIGIBLE historical units_observed for every forecast date."""
    eligible = _eligible_history(train)
    if eligible.empty:
        raise ValueError("last_day_naive requires at least one eligible historical observation; found none.")
    last_value = float(eligible["units_observed"].iloc[-1])
    return pd.Series(last_value, index=required_dates)


def seasonal_naive_7(train: pd.DataFrame, required_dates: pd.DatetimeIndex, cutoff: pd.Timestamp) -> pd.Series:
    """Predict each date from the same-weekday observation one or more weeks earlier.

    Genuinely date-aligned: day 1 of the horizon looks back 7 days into training
    data. For a 14-day horizon, day 8 would naively look back to day 1 -- but
    day 1 is itself a forecast, not an actual. Held-out predictions may never
    feed each other, so day 8 instead looks back 14 days (one more week),
    landing on real training data. This is equivalent to repeating the last
    known week's weekday pattern across both forecast weeks.

    Eligibility + fallback (v4): the source observation must be
    forecast_training_eligible. The first candidate is the required t-7 (or
    t-14) date. If that date is missing or ineligible, step BACKWARD in whole
    7-day intervals -- which preserves the weekday -- until an eligible
    same-weekday observation is found. If no eligible same-weekday observation
    exists at any prior week, fall back to the most recent eligible historical
    value.
    """
    eligible = _eligible_history(train).set_index("date")["units_observed"]
    if eligible.empty:
        raise ValueError("seasonal_naive_7 requires at least one eligible historical observation; found none.")
    earliest = eligible.index.min()
    most_recent_eligible = float(eligible.iloc[-1])

    predictions: dict[pd.Timestamp, float] = {}
    for date in required_dates:
        days_ahead = (date - cutoff).days
        lag_days = 7 * -(-days_ahead // 7)  # ceil(days_ahead / 7) rounded up to a multiple of 7
        source_date = date - pd.Timedelta(days=lag_days)
        # `eligible.index` contains only eligible dates, so stepping back by
        # whole weeks keeps the same weekday and skips missing/ineligible dates.
        while source_date not in eligible.index and source_date >= earliest:
            source_date -= pd.Timedelta(days=7)
        if source_date in eligible.index:
            predictions[date] = float(eligible.loc[source_date])
        else:
            predictions[date] = most_recent_eligible  # no eligible same-weekday obs -> most recent eligible value
    return pd.Series(predictions)


def _moving_average(train: pd.DataFrame, required_dates: pd.DatetimeIndex, window: int) -> pd.Series:
    """Repeat the mean of the most recent `window` ELIGIBLE observations for every forecast date."""
    observations = _eligible_history(train)["units_observed"]
    if len(observations) < window:
        raise ValueError(
            f"moving_average_{window} requires at least {window} eligible observations; got {len(observations)}."
        )
    average = float(observations.iloc[-window:].mean())
    return pd.Series(average, index=required_dates)


def moving_average_7(train: pd.DataFrame, required_dates: pd.DatetimeIndex, cutoff: pd.Timestamp) -> pd.Series:
    return _moving_average(train, required_dates, window=7)


def moving_average_14(train: pd.DataFrame, required_dates: pd.DatetimeIndex, cutoff: pd.Timestamp) -> pd.Series:
    return _moving_average(train, required_dates, window=14)


BASELINES: dict[str, ForecastFn] = {
    "last_day_naive": last_day_naive,
    "seasonal_naive_7": seasonal_naive_7,
    "moving_average_7": moving_average_7,
    "moving_average_14": moving_average_14,
}


# --- turning one baseline into a scoreable predictions frame ----------------
def build_predictions(panel: pd.DataFrame, horizon: int, forecast_fn: ForecastFn) -> pd.DataFrame:
    """Build a [sku, channel, date, y_pred] frame for one model at one horizon.

    Uses evaluation.py's own backtest_split() so the train/test cutoff is
    identical to what evaluate() will use internally. Required dates are taken
    from the test window restricted to `forecast_training_eligible` rows,
    matching how evaluate() builds its private truth -- this keeps prediction
    keys exactly aligned with the evaluator's expected key set. The full train
    slice is passed to the forecaster, which applies the same
    forecast_training_eligible target mask to its historical inputs; test rows
    are never filtered independently beyond selecting those eligible keys.
    """
    cutoff, train, test = backtest_split(panel, horizon)
    scoreable_test = test[test["forecast_training_eligible"].astype(bool)]

    rows: list[dict[str, object]] = []
    for (sku, channel), truth_group in scoreable_test.groupby(["sku", "channel"]):
        sku_train = train[(train["sku"] == sku) & (train["channel"] == channel)]
        required_dates = pd.DatetimeIndex(sorted(truth_group["date"]))

        if sku_train.empty:
            raise ValueError(f"No training history for {sku}/{channel}; cannot forecast.")

        forecast = forecast_fn(sku_train, required_dates, cutoff)

        for date, value in forecast.items():
            rows.append({"sku": sku, "channel": channel, "date": date, "y_pred": value})

    preds = pd.DataFrame(rows, columns=["sku", "channel", "date", "y_pred"])

    if preds.empty:
        raise ValueError(f"No predictions were generated for horizon={horizon}.")

    if preds.duplicated(subset=["sku", "channel", "date"]).any():
        raise ValueError("Generated predictions contain duplicate sku+channel+date keys.")

    # Non-negative, finite predictions (the evaluator's contract). Natural
    # fractional values are preserved -- integers are not required.
    preds["y_pred"] = np.clip(preds["y_pred"].astype(float), 0, None)

    if preds["y_pred"].isna().any() or np.isinf(preds["y_pred"]).any():
        raise ValueError("Generated predictions contain missing or infinite values.")

    return preds


# --- reporting helpers -------------------------------------------------------
def _sku_channel_count(per_sku: dict) -> tuple[int, int]:
    """Count unique SKUs and channels from evaluate()'s 'sku|channel' keyed dict."""
    skus = {key.rsplit("|", 1)[0] for key in per_sku}
    channels = {key.rsplit("|", 1)[1] for key in per_sku}
    return len(skus), len(channels)


def print_official_baseline(comparison: pd.DataFrame, horizon: int) -> pd.Series:
    """Sort one horizon's rows by WAPE (primary) then MASE (tiebreaker) and announce the winner."""
    subset = comparison[comparison["horizon"] == horizon].sort_values(["wape", "mase"])
    winner = subset.iloc[0]
    print(
        f"\nOfficial {horizon}-day baseline to beat: "
        f"{winner['model']} with WAPE {winner['wape']:.3f} and MASE {winner['mase']:.3f}"
    )
    return winner


def print_sku_extremes(per_sku: dict, model_name: str, horizon: int) -> None:
    """Print the 5 best and 5 worst SKU|channel series (by WAPE) for the winning model."""
    rows = [{"sku_channel": key, **values} for key, values in per_sku.items() if not np.isnan(values["wape"])]
    if not rows:
        print(f"\nNo SKUs had a scoreable WAPE for {model_name} @ horizon={horizon}.")
        return
    table = pd.DataFrame(rows).set_index("sku_channel")

    print(f"\nBest 5 SKUs for {model_name} @ horizon={horizon} (lowest WAPE):")
    print(table.nsmallest(5, "wape").round(3).to_string())

    print(f"\nWorst 5 SKUs for {model_name} @ horizon={horizon} (highest WAPE):")
    print(table.nlargest(5, "wape").round(3).to_string())


# --- Phase 5: optional real-future forecast (unscored) -----------------------
def build_future_forecast(panel: pd.DataFrame, forecast_fn: ForecastFn, horizon: int = 14) -> pd.DataFrame:
    """Forecast the real future days in forecast_frame.parquet using the winning
    14-day baseline, trained on all available ELIGIBLE history (the forecasters
    apply the forecast_training_eligible target mask; no holdout -- there is no
    truth to hold out for genuine future dates). Never scored: the actual
    outcomes for these dates do not exist yet.
    """
    future = load_forecast_frame()
    full_history_cutoff = panel["date"].max()

    rows: list[dict[str, object]] = []
    for (sku, channel), future_group in future.groupby(["sku", "channel"]):
        sku_train = panel[(panel["sku"] == sku) & (panel["channel"] == channel)]
        required_dates = pd.DatetimeIndex(sorted(future_group["date"]))
        forecast = forecast_fn(sku_train, required_dates, full_history_cutoff)
        # descriptive keys for the dashboard (carried from forecast_frame; not model inputs)
        name = future_group["sku_name"].iloc[0] if "sku_name" in future_group else None
        pid = future_group["product_id"].iloc[0] if "product_id" in future_group else None
        for date, value in forecast.items():
            rows.append({"sku": sku, "product_id": pid, "sku_name": name,
                         "channel": channel, "date": date, "y_pred": value})

    future_preds = pd.DataFrame(
        rows, columns=["sku", "product_id", "sku_name", "channel", "date", "y_pred"])
    # Preserve natural fractional forecasts; only enforce non-negativity.
    future_preds["y_pred"] = np.clip(future_preds["y_pred"].astype(float), 0, None)
    return future_preds


def run() -> None:
    panel = load_model_panel()

    comparison_rows: list[dict[str, object]] = []
    predictions_by_model: dict[tuple[str, int], pd.DataFrame] = {}
    results_by_model: dict[tuple[str, int], dict] = {}

    for horizon in HORIZONS:
        for model_name, forecast_fn in BASELINES.items():
            preds = build_predictions(panel, horizon, forecast_fn)
            result = evaluate(preds, horizon=horizon, panel=panel)

            predictions_by_model[(model_name, horizon)] = preds
            results_by_model[(model_name, horizon)] = result

            n_skus, n_channels = _sku_channel_count(result["per_sku"])
            overall = result["overall"]
            comparison_rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "wape": overall["wape"],
                    "mae": overall["mae"],
                    "mase": overall["mase"],
                    "rmse": overall["rmse"],
                    "bias": overall["bias"],
                    "n_rows": result["n_test_rows"],
                    "n_skus": n_skus,
                    "n_channels": n_channels,
                }
            )

    comparison = pd.DataFrame(comparison_rows)

    print("\nBASELINE COMPARISON (all models, both horizons)")
    print("------------------------------------------------")
    print(
        comparison.sort_values(["horizon", "wape"])
        .set_index(["horizon", "model"])
        .round(3)
        .to_string()
    )

    winners: dict[int, pd.Series] = {}
    for horizon in HORIZONS:
        winners[horizon] = print_official_baseline(comparison, horizon)

    # --- extra diagnostics for each horizon's winning model -----------------
    for horizon in HORIZONS:
        winner = winners[horizon]
        model_name = winner["model"]
        result = results_by_model[(model_name, horizon)]
        preds = predictions_by_model[(model_name, horizon)]

        nan_mase_count = sum(1 for v in result["per_sku"].values() if np.isnan(v["mase"]))
        print(
            f"\n[{model_name} @ horizon={horizon}] SKU-level series with NaN MASE: "
            f"{nan_mase_count} / {len(result['per_sku'])}"
        )

        print_sku_extremes(result["per_sku"], model_name, horizon)

        negative_count = (preds["y_pred"] < 0).sum()
        nan_count = preds["y_pred"].isna().sum()
        inf_count = np.isinf(preds["y_pred"]).sum()
        print(
            f"\n[{model_name} @ horizon={horizon}] predictions: {len(preds)} rows, "
            f"negative={negative_count}, NaN={nan_count}, infinite={inf_count}"
        )

        bias_value = winner["bias"]
        direction = "overforecasting" if bias_value > 0 else "underforecasting" if bias_value < 0 else "unbiased"
        meets_target = winner["wape"] <= 0.25
        print(
            f"[{model_name} @ horizon={horizon}] bias={bias_value:.3f} ({direction}); "
            f"WAPE target (<=0.25) met: {meets_target}"
        )

    # --- Phase 5: optional real-future forecast, using the 14-day winner ----
    winner_14 = winners[14]
    winner_14_fn = BASELINES[winner_14["model"]]
    future_preds = build_future_forecast(panel, winner_14_fn, horizon=14)

    OUT_DIR.mkdir(exist_ok=True)
    future_path = OUT_DIR / f"future_forecast_14d_{winner_14['model']}.csv"
    future_preds.to_csv(future_path, index=False)

    print(
        f"\nReal-future forecast (unscored -- outcomes unknown yet): "
        f"{len(future_preds)} rows written to {future_path.relative_to(OUT_DIR.parents[0])}."
    )
    print("This is NOT a backtest prediction and must not be passed to evaluate().")


if __name__ == "__main__":
    run()
