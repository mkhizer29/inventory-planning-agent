"""
v6: same 30-SKU dataset/architecture as train_lgbm_v5.py, now on
SKUs/lgbm-dataset-5.csv -- adds the 7 category-level leave-one-out rolling
features (cat_lag_1/7/14, cat_rolling_mean/std_7/14; see
SKUs/build_category_rolling_features.py) validated via 24-fold rolling-
origin CV on 2026-07-27 (model/rolling_origin_cv_category_features.py):
close to a wash on typical weeks but meaningfully reduces worst-case/tail
risk (std 9.29 -> 7.10 across folds, worst fold 27.1% -> 48.3%). Adopted as
the new canonical dataset for that reason. Same 8-week test / 4-week val
split and both-objectives comparison as v5, otherwise unchanged.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "../data/lgbm-dataset-5.csv"
MODEL_DIR = "."

TARGET = "net_qty"
CATEGORICAL = ["sku", "category", "brand", "day_of_week"]
VAL_DAYS = 28
TEST_DAYS = 56

# ---- load & split ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df.sort_values(["sku", "date"]).reset_index(drop=True)

n_negative = (df[TARGET] < 0).sum()
df[TARGET] = df[TARGET].clip(lower=0)
print(f"Clipped {n_negative} rows with negative net_qty to 0")

for col in CATEGORICAL:
    df[col] = df[col].astype("category")

max_date = df["date"].max()
test_start = max_date - pd.Timedelta(days=TEST_DAYS - 1)
val_start = test_start - pd.Timedelta(days=VAL_DAYS)

train_df = df[df["date"] < val_start]
val_df = df[(df["date"] >= val_start) & (df["date"] < test_start)]
test_df = df[df["date"] >= test_start]

print(f"Full window: {df['date'].min().date()} -> {max_date.date()} ({len(df)} rows)")
print(f"Train: {train_df['date'].min().date()} -> {train_df['date'].max().date()} ({len(train_df)} rows)")
print(f"Val:   {val_df['date'].min().date()} -> {val_df['date'].max().date()} ({len(val_df)} rows)")
print(f"Test:  {test_df['date'].min().date()} -> {test_df['date'].max().date()} ({len(test_df)} rows, "
      f"{TEST_DAYS // 7} weeks / {TEST_DAYS // 14} biweeks)")

feature_cols = [c for c in df.columns if c not in ("date", TARGET)]

X_train, y_train = train_df[feature_cols], train_df[TARGET]
X_val, y_val = val_df[feature_cols], val_df[TARGET]
X_test, y_test = test_df[feature_cols], test_df[TARGET]


def train_and_evaluate(objective, label, **extra_params):
    params = dict(
        objective=objective,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    params.update(extra_params)
    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    y_pred = np.clip(model.predict(X_test), 0, None)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    wape = np.abs(y_test - y_pred).sum() / y_test.sum() * 100
    accuracy = 100 - wape

    print(f"\n=== [{label}] daily test-set performance "
          f"({test_df['date'].min().date()} -> {test_df['date'].max().date()}) ===")
    print(f"Best iteration: {model.best_iteration_}")
    print(f"MAE   : {mae:.3f} units/SKU/day")
    print(f"WAPE  : {wape:.2f}%   Accuracy: {accuracy:.2f}%")

    preds = test_df.copy()
    preds["pred"] = y_pred

    model.booster_.save_model(f"{MODEL_DIR}/lgbm_model_{label}.txt")
    preds[["date", "sku", "category", "brand", TARGET, "pred"]].to_csv(
        f"{MODEL_DIR}/test_predictions_{label}.csv", index=False
    )

    return {"label": label, "mae": mae, "rmse": rmse, "r2": r2, "wape": wape, "accuracy": accuracy}


results = []
results.append(train_and_evaluate("regression", "baseline_v6"))
results.append(train_and_evaluate("tweedie", "tweedie_v6", tweedie_variance_power=1.3))

comparison = pd.DataFrame([
    {"model": r["label"], "MAE": r["mae"], "RMSE": r["rmse"], "R2": r["r2"],
     "WAPE_%": r["wape"], "Accuracy_%": r["accuracy"]}
    for r in results
]).set_index("model")

print("\n\n=== Baseline vs Tweedie comparison (v5, 30 SKUs, 56-day test) ===")
print(comparison.to_string(float_format=lambda x: f"{x:.3f}"))
comparison.to_csv(f"{MODEL_DIR}/baseline_vs_tweedie_comparison_v6.csv")
print(f"Saved -> {MODEL_DIR}/baseline_vs_tweedie_comparison_v6.csv")
