"""lgbm_global.py — Aqib. ONE gradient-boosted (LightGBM) model across all 30 pilot SKUs.

Contract (v4, real-demand / synthetic-stock, daily / naheed_web-only — see
Doc3_Aqib_LightGBM.md's "READ THIS FIRST" callout and pilot_manifest.json):
  * Inputs: data/processed/model_panel.parquet (train/backtest) and
    data/processed/forecast_frame.parquet (future, no actuals). Read-only — never
    regenerated or edited here (that's prepare_pilot_data.py, the pipeline owner's job).
  * Target: REAL units_observed. Row eligibility for both training and scoring is
    forecast_training_eligible (real-data quality + >=14 days history) — never based
    on synthetic stock. evaluate() asserts this independence at runtime.
  * Features: the manifest's demand_feature_whitelist, read at runtime (not hardcoded)
    so this script tracks pilot_manifest.json if it ever changes:
      units_lag_1/7/14, units_roll_mean_7/28, units_roll_std_7, effective_unit_price,
      discount_pct, on_promo, is_public_holiday, is_payday_window, day_of_week,
      is_weekend, week_of_year, month, is_ramadan, ramadan_day, ramadan_week
    Plus ONE addition beyond the whitelist: `sku` as a categorical grouping feature.
    This is a POOLED model (one model, all 30 SKUs stacked) — unlike Khizer's
    per-SKU Holt-Winters, it needs a SKU identity feature to tell products apart and
    to let the tree structure borrow shared patterns across SKUs. `channel` is not
    a feature (constant "naheed_web" for the whole pilot, no variation to learn from).
    NEVER fed to the model (checked at runtime): stock_on_hand, unit_cost,
    unit_cost_observed, unit_cost_effective, or any other model_panel column outside
    the whitelist + sku.
  * Split: evaluation.py's locked chronological backtest_split(panel, horizon) for
    horizon in (7, 14) — identical cutoff/train/test as Aiman's baselines and
    Khizer's Holt-Winters. Never a random split.
  * Multi-day horizon: lag/rolling features for backtest test-days come straight from
    model_panel (they're real, causally shifted at construction — see
    prepare_pilot_data.py's `grp.shift(1)` before rolling — so using them for held-out
    real days is not leakage: date t's lag/rolling only reads real values strictly
    before t). The FUTURE forecast (forecast_frame.parquet) has no such precomputed
    lag/rolling columns since no real future demand exists yet, so those 14 days are
    forecast recursively per SKU: predict day 1, fold that prediction into the
    SKU's running series, recompute lag/rolling for day 2 from that series, etc.
  * Scoring: evaluate(preds, horizon=h, panel=panel) from src/evaluation.py — imported,
    never edited. preds carries only [sku, channel, date, y_pred] (never y_true).
    Reports WAPE (primary), MAE, MASE, RMSE, bias per the shared contract.
  * Explanations: the future forecast additionally carries an `explanation` column — a
    2-3 sentence, fully deterministic, template-based justification for each predicted
    quantity, built from LightGBM's own `pred_contrib` (SHAP) breakdown for that exact
    row (see _build_explanation). Not an LLM call — it can only describe features the
    model actually used, never invent a reason. Backtest predictions don't get one
    (they're for scoring, not dashboard display).

Run:  python src/lgbm_global.py            (writes outputs/)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import evaluation as ev  # shared scorecard — imported, never edited  # noqa: E402

PROC = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "outputs"
MANIFEST_PATH = PROC / "pilot_manifest.json"

EXPECTED_SCHEMA = "4.0-real-demand-synthetic-stock"
EXPECTED_CHANNEL = "naheed_web"
TARGET = "units_observed"
HORIZONS = ev.HORIZONS  # (7, 14) — from the shared evaluator
MODEL_VERSION = "lgbm_global_v1"

# Columns that must NEVER reach the model, whitelist or not (fail loudly if any leak in).
FORBIDDEN_INPUTS = (
    "stock_on_hand", "stock_on_hand_is_synthetic", "stock_source", "stock_generation_version",
    "unit_cost", "unit_cost_observed", "unit_cost_effective",
)

# Recursive-future feature name -> forecast_frame.parquet column name
# (only for the exogenous/calendar features; lag/rolling are simulated, see below)
FUTURE_COLUMN_MAP = {
    "effective_unit_price": "latest_known_price",
    "on_promo": "planned_promo",
    "discount_pct": "planned_discount_pct",
    # identical names in both frames:
    "is_public_holiday": "is_public_holiday",
    "is_payday_window": "is_payday_window",
    "day_of_week": "day_of_week",
    "is_weekend": "is_weekend",
    "week_of_year": "week_of_year",
    "month": "month",
    "is_ramadan": "is_ramadan",
    "ramadan_day": "ramadan_day",
    "ramadan_week": "ramadan_week",
}

WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
EXPLANATION_FLAG_FEATURES = ("on_promo", "is_public_holiday", "is_ramadan")


def _feature_phrase(name: str, value: float) -> str:
    """Human-readable phrase for one feature's real (unit-scale) value, used only to
    pick WHICH features to talk about and how to describe them -- never to state an
    exact numeric contribution in units (see _build_explanation's docstring for why)."""
    if name == "units_lag_1":
        return f"yesterday's sales of {value:.0f} units"
    if name == "units_lag_7":
        return f"sales of {value:.0f} units on this day last week"
    if name == "units_lag_14":
        return f"sales of {value:.0f} units two weeks ago"
    if name == "units_roll_mean_7":
        return f"a recent 7-day average of {value:.1f} units"
    if name == "units_roll_mean_28":
        return f"a recent 28-day average of {value:.1f} units"
    if name == "units_roll_std_7":
        return f"how volatile recent daily sales have been (7-day std of {value:.1f})"
    if name == "effective_unit_price":
        return f"the current price of {value:.0f}"
    if name == "discount_pct":
        return f"an active discount of {value * 100:.0f}%" if value else "no active discount"
    if name == "on_promo":
        return "an active promotion" if value else "no active promotion"
    if name == "is_public_holiday":
        return "a public holiday" if value else "no public holiday"
    if name == "is_payday_window":
        return "the payday window" if value else "being outside the payday window"
    if name == "day_of_week":
        return f"typical {WEEKDAY_NAMES[int(value)]} demand patterns"
    if name == "is_weekend":
        return "weekend demand patterns" if value else "weekday demand patterns"
    if name == "week_of_year":
        return f"this point in the year (week {int(value)})"
    if name == "month":
        return "seasonal patterns for this month"
    if name == "is_ramadan":
        return "the Ramadan period" if value else "being outside Ramadan"
    if name in ("ramadan_day", "ramadan_week"):
        unit = "day" if name == "ramadan_day" else "week"
        return f"being {int(value)} {unit}s into Ramadan"
    if name == "sku":
        return "this product's own typical demand level"
    return name


def _rank_drivers(features: list[str], contrib: np.ndarray, k: int = 2, eps: float = 1e-3) -> list[tuple[str, float]]:
    """Top-k features by |SHAP contribution|, in the model's raw (link) space.

    Ranking and sign are valid in link space (tweedie uses a log link: a positive raw
    contribution always increases the final exp()-transformed prediction, negative
    always decreases it, regardless of the nonlinear rescaling) -- so "which features
    mattered most" and "which direction" are both correct. The MAGNITUDE in link space
    is not directly convertible to "N units", which is why _feature_phrase reports a
    feature's real unit-scale VALUE (e.g. "7-day average of 14.2 units") rather than an
    invented unit-scale contribution size.
    """
    pairs = [(f, contrib[i]) for i, f in enumerate(features) if abs(contrib[i]) > eps]
    pairs.sort(key=lambda p: -abs(p[1]))
    return pairs[:k]


def _build_explanation(sku: str, date: pd.Timestamp, y_pred: float, features: list[str],
                        feature_values: dict, contrib: np.ndarray) -> str:
    """2-3 sentence, fully deterministic explanation for one future prediction.

    Template-based from the model's own LightGBM `pred_contrib` (SHAP) breakdown for
    this exact row -- not an LLM call, so it can never state a reason the model didn't
    actually use. See _rank_drivers for why contribution magnitude isn't quoted directly.
    """
    date_str = pd.Timestamp(date).strftime("%b %d, %Y")
    lead = f"Predicted {y_pred:.1f} units for {sku} on {date_str}."

    drivers = _rank_drivers(features, contrib)
    if not drivers:
        return lead + (" This closely matches the model's baseline expectation for this "
                        "product, with no single factor standing out.")

    phrases = [
        f"{_feature_phrase(f, feature_values[f])} ({'pushing it higher' if c > 0 else 'pulling it lower'})"
        for f, c in drivers
    ]
    body = f"This is mainly shaped by {phrases[0]}" + (f" and {phrases[1]}." if len(phrases) > 1 else ".")

    driver_names = {f for f, _ in drivers}
    active_flags = [f for f in EXPLANATION_FLAG_FEATURES if feature_values.get(f)]
    tail = ""
    if not (driver_names & set(EXPLANATION_FLAG_FEATURES)):
        if active_flags:
            named = " and ".join(_feature_phrase(f, 1) for f in active_flags)
            tail = f" {named.capitalize()} also applies to this date."
        else:
            tail = " No promotion, holiday, or Ramadan effect applies to this date."

    return f"{lead} {body}{tail}"


# ── contract load + audit ───────────────────────────────────────────────────────────
def load_contract() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    panel = ev.load_model_panel()
    future = ev.load_forecast_frame()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if manifest.get("schema_version") != EXPECTED_SCHEMA:
        raise RuntimeError(
            f"pilot_manifest.json schema_version={manifest.get('schema_version')!r}, "
            f"expected {EXPECTED_SCHEMA!r} — contract may have changed, re-check before training."
        )
    if manifest.get("channel_scope") != [EXPECTED_CHANNEL]:
        raise RuntimeError(f"unexpected channel_scope: {manifest.get('channel_scope')}")
    if set(panel["channel"].unique()) != {EXPECTED_CHANNEL} or set(future["channel"].unique()) != {EXPECTED_CHANNEL}:
        raise RuntimeError("model_panel/forecast_frame contain a channel other than naheed_web")
    if sorted(panel["sku"].unique()) != sorted(manifest["selected_skus"]):
        raise RuntimeError("model_panel SKU set does not match pilot_manifest.json selected_skus")
    if list(manifest["forecast_horizon_days"]) != list(HORIZONS):
        raise RuntimeError(
            f"manifest horizons {manifest['forecast_horizon_days']} != evaluation.HORIZONS {HORIZONS}"
        )
    as_of = pd.Timestamp(manifest["as_of_date"])
    if panel["date"].max() != as_of:
        raise RuntimeError(f"model_panel max date {panel['date'].max()} != manifest as_of_date {as_of}")

    return panel, future, manifest


def feature_list(manifest: dict) -> list[str]:
    """Manifest's demand_feature_whitelist + `sku` (pooled-model grouping feature, not in
    the whitelist itself — see module docstring for why it's added)."""
    whitelist = list(manifest["demand_feature_whitelist"])
    for col in FORBIDDEN_INPUTS:
        if col in whitelist:
            raise RuntimeError(f"forbidden column {col!r} found inside demand_feature_whitelist")
    return whitelist + ["sku"]


def _assert_no_forbidden_columns(cols: list[str]) -> None:
    leaked = set(cols) & set(FORBIDDEN_INPUTS)
    if leaked:
        raise RuntimeError(f"forbidden columns about to be fed to the model: {leaked}")


# ── LightGBM config (small-data discipline: 30 SKUs, ~6 months) ────────────────────
def _lgbm_params(**extra) -> dict:
    params = dict(
        objective="tweedie",
        tweedie_variance_power=1.3,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    params.update(extra)
    return params


def _prep_X(df: pd.DataFrame, features: list[str], sku_categories: pd.CategoricalDtype) -> pd.DataFrame:
    _assert_no_forbidden_columns(features)
    X = df[features].copy()
    X["sku"] = X["sku"].astype(sku_categories)
    return X


# ── backtest: one horizon ───────────────────────────────────────────────────────────
def run_backtest(panel: pd.DataFrame, horizon: int, features: list[str],
                  sku_categories: pd.CategoricalDtype) -> tuple[pd.DataFrame, dict, lgb.LGBMRegressor]:
    cutoff, train_all, test_all = ev.backtest_split(panel, horizon)
    train_eligible = train_all[train_all["forecast_training_eligible"].astype(bool)].sort_values("date")

    # small internal validation carve-out (last 14 eligible train days) for early
    # stopping only -- never overlaps the test window, doesn't affect comparability.
    val_start = train_eligible["date"].max() - pd.Timedelta(days=14)
    fit_train = train_eligible[train_eligible["date"] < val_start]
    fit_val = train_eligible[train_eligible["date"] >= val_start]

    X_train = _prep_X(fit_train, features, sku_categories)
    y_train = fit_train[TARGET].clip(lower=0)
    X_val = _prep_X(fit_val, features, sku_categories)
    y_val = fit_val[TARGET].clip(lower=0)

    model = lgb.LGBMRegressor(**_lgbm_params())
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        categorical_feature=["sku"],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    scoreable_test = test_all[test_all["forecast_training_eligible"].astype(bool)].sort_values("date")
    X_test = _prep_X(scoreable_test, features, sku_categories)
    y_pred = np.clip(model.predict(X_test), 0, None)

    preds = scoreable_test[["sku", "channel", "date"]].copy()
    preds["y_pred"] = y_pred

    result = ev.evaluate(preds, horizon=horizon, panel=panel)

    # keep y_true alongside for human review (evaluate() only needs preds without it)
    backtest_detail = preds.copy()
    backtest_detail["y_true"] = scoreable_test[TARGET].to_numpy()
    backtest_detail["horizon"] = horizon
    backtest_detail["cutoff"] = cutoff.date().isoformat()
    backtest_detail["model_version"] = MODEL_VERSION

    return backtest_detail, result, model


# ── future forecast: recursive, single model fit on all eligible history ──────────
def _future_row_features(sku: str, date: pd.Timestamp, series: pd.Series,
                          future_row: pd.Series, features: list[str]) -> dict:
    """One feature row for (sku, date) in the future window.

    Lag/rolling come from `series` (real history + predictions-so-far for this SKU),
    causally: only points strictly before `date` are ever read, matching the
    shift(1)-before-rolling construction in prepare_pilot_data.py.
    """
    before = series[series.index < date]
    window7 = before.iloc[-7:] if len(before) else before
    window28 = before.iloc[-28:] if len(before) else before

    row = {
        "sku": sku,
        "units_lag_1": before.iloc[-1] if len(before) >= 1 else np.nan,
        "units_lag_7": series.get(date - pd.Timedelta(days=7), np.nan),
        "units_lag_14": series.get(date - pd.Timedelta(days=14), np.nan),
        "units_roll_mean_7": window7.mean() if len(window7) else np.nan,
        "units_roll_mean_28": window28.mean() if len(window28) else np.nan,
        "units_roll_std_7": window7.std() if len(window7) >= 2 else np.nan,
    }
    for model_col, frame_col in FUTURE_COLUMN_MAP.items():
        val = future_row[frame_col]
        if model_col == "discount_pct" and pd.isna(val):
            val = 0.0  # no planned promo -> no planned discount, matches historical convention
        row[model_col] = val
    return {k: row[k] for k in features}


def run_future_forecast(panel: pd.DataFrame, future: pd.DataFrame, features: list[str],
                         sku_categories: pd.CategoricalDtype, as_of_date: str) -> pd.DataFrame:
    train_eligible = panel[panel["forecast_training_eligible"].astype(bool)].sort_values("date")
    X_train = _prep_X(train_eligible, features, sku_categories)
    y_train = train_eligible[TARGET].clip(lower=0)

    model = lgb.LGBMRegressor(**_lgbm_params(n_estimators=300))  # no val split -> fixed tree count
    model.fit(X_train, y_train, categorical_feature=["sku"])

    rows: list[dict] = []
    for sku, sku_future in future.sort_values(["sku", "date"]).groupby("sku"):
        sku_hist = panel[panel["sku"] == sku].sort_values("date")
        series = pd.Series(sku_hist[TARGET].to_numpy(), index=sku_hist["date"])

        for _, future_row in sku_future.iterrows():
            date = future_row["date"]
            feat = _future_row_features(sku, date, series, future_row, features)
            X = pd.DataFrame([feat])
            X["sku"] = X["sku"].astype(sku_categories)
            y_pred = float(np.clip(model.predict(X[features])[0], 0, None))
            contrib = model.predict(X[features], pred_contrib=True)[0][:-1]  # drop base-value column
            explanation = _build_explanation(sku, date, y_pred, features, feat, contrib)

            series.loc[date] = y_pred  # feed prediction back in for the next day's lags
            rows.append({
                "sku": sku,
                "product_id": future_row.get("product_id"),
                "channel": future_row["channel"],
                "date": date,
                "forecast_horizon_day": future_row["forecast_horizon_day"],
                "y_pred": y_pred,
                "explanation": explanation,
                "as_of_date": as_of_date,
                "model_version": MODEL_VERSION,
            })

    return pd.DataFrame(rows)


def print_feature_importance(model: lgb.LGBMRegressor, features: list[str], horizon: int) -> None:
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\ntop features @ horizon={horizon}:\n{imp.head(8).to_string()}")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    panel, future, manifest = load_contract()
    as_of_date = manifest["as_of_date"]
    features = feature_list(manifest)
    sku_categories = pd.CategoricalDtype(categories=sorted(panel["sku"].unique()))

    print(f"Loaded model_panel: {panel.shape}, forecast_frame: {future.shape}")
    print(f"Features ({len(features)}): {features}")

    OUT.mkdir(exist_ok=True)
    scorecard_rows = []
    all_backtest_detail = []

    for horizon in HORIZONS:
        backtest_detail, result, model = run_backtest(panel, horizon, features, sku_categories)
        all_backtest_detail.append(backtest_detail)

        overall = result["overall"]
        scorecard_rows.append({
            "model": MODEL_VERSION,
            "horizon": horizon,
            "cutoff": result["cutoff"],
            "wape": overall["wape"],
            "mae": overall["mae"],
            "mase": overall["mase"],
            "rmse": overall["rmse"],
            "bias": overall["bias"],
            "n_test_rows": result["n_test_rows"],
        })
        print(f"\n=== horizon={horizon} (cutoff {result['cutoff']}) ===")
        print(f"WAPE={overall['wape']:.4f}  MAE={overall['mae']:.3f}  MASE={overall['mase']:.3f}  "
              f"RMSE={overall['rmse']:.3f}  bias={overall['bias']:.3f}  n={result['n_test_rows']}")
        print_feature_importance(model, features, horizon)

    backtest_all = pd.concat(all_backtest_detail, ignore_index=True)
    scorecard = pd.DataFrame(scorecard_rows)

    print("\n\nSCORECARD (both horizons)")
    print(scorecard.set_index(["model", "horizon"]).round(4).to_string())

    future_preds = run_future_forecast(panel, future, features, sku_categories, as_of_date)

    # --- persist: parquet (repo convention, survives the outputs/*.csv gitignore
    # rule) + CSV convenience copies, force-added explicitly when committing ---
    backtest_all.to_parquet(OUT / "lgbm_backtest_predictions.parquet", index=False)
    backtest_all.to_csv(OUT / "lgbm_backtest_predictions.csv", index=False)
    future_preds.to_parquet(OUT / "demand_forecasts_lgbm.parquet", index=False)
    future_preds.to_csv(OUT / "demand_forecasts_lgbm.csv", index=False)
    scorecard.to_csv(OUT / "lgbm_scorecard.csv", index=False)

    scorecard_json = {
        "model_version": MODEL_VERSION,
        "as_of_date": as_of_date,
        "channel": EXPECTED_CHANNEL,
        "horizons": list(HORIZONS),
        "n_skus": panel["sku"].nunique(),
        "features": features,
        "results": scorecard_rows,
    }
    (OUT / "lgbm_model_selection.json").write_text(json.dumps(scorecard_json, indent=2, default=str))

    print(f"\nWrote outputs/lgbm_backtest_predictions.{{parquet,csv}} ({len(backtest_all)} rows)")
    print(f"Wrote outputs/demand_forecasts_lgbm.{{parquet,csv}} ({len(future_preds)} rows)")
    print("Wrote outputs/lgbm_scorecard.csv and outputs/lgbm_model_selection.json")


if __name__ == "__main__":
    main()
