"""
Approach A (train daily, sum forecasts to weekly) on all 30 SKUs -- weekly
reporting grain only. Biweekly was evaluated earlier (see project memory,
2026-07-27) and scored slightly higher, but the business decided to report
weekly to match Naheed's actual reorder cadence, so biweekly aggregation is
no longer part of this pipeline (removed 2026-07-27, along with
model/biweekly_forecast_vs_actual_30skus.csv,
model/weekly_vs_biweekly_accuracy_30skus.csv, and model/sku_reports/biweekly/).

Buckets tweedie_v6's daily test predictions (model/test_predictions_tweedie_v6.csv,
56-day test window) into 7-day blocks (counted from the test window's own
start, not calendar weeks) and compares each week's total predicted vs total
actual, per SKU and overall.

WAPE/accuracy: each week is treated as ONE data point -- compare the week's
total actual against the week's total predicted (not daily-within-week
errors). This is what "weekly forecast vs weekly actual" means for a
warehouse reorder decision.

Writes: model/weekly_forecast_vs_actual_30skus.csv (per sku x week detail)
"""
import numpy as np
import pandas as pd

PRED_PATH = "test_predictions_tweedie_v6.csv"
OUT_DETAIL = "weekly_forecast_vs_actual_30skus.csv"
OUT_SUMMARY = "weekly_accuracy_30skus.csv"

df = pd.read_csv(PRED_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["sku", "date"]).reset_index(drop=True)

test_start = df["date"].min()
test_end = df["date"].max()
n_days = (test_end - test_start).days + 1
print(f"Test window: {test_start.date()} -> {test_end.date()} ({n_days} days)")

df["day_offset"] = (df["date"] - test_start).dt.days
df["period"] = df["day_offset"] // 7 + 1
n_periods = df["period"].max()

period_bounds = df.groupby("period")["date"].agg(["min", "max"]).rename(
    columns={"min": "period_start", "max": "period_end"})

rows = []
for (sku, p), g in df.groupby(["sku", "period"]):
    actual = g["net_qty"].sum()
    pred = g["pred"].sum()
    wape = (g["net_qty"] - g["pred"]).abs().sum() / actual * 100 if actual > 0 else np.nan
    rows.append({
        "sku": sku, "week": p,
        "week_start": period_bounds.loc[p, "period_start"].date(),
        "week_end": period_bounds.loc[p, "period_end"].date(),
        "n_days": len(g),
        "actual": actual, "predicted": round(pred, 1),
        "accuracy_pct": round(100 - wape, 1) if not np.isnan(wape) else np.nan,
    })
detail = pd.DataFrame(rows).sort_values(["sku", "week"])
detail.to_csv(OUT_DETAIL, index=False)
print(f"\n{n_periods} weekly periods, saved detail -> {OUT_DETAIL}")

# ---- per-sku accuracy treating each WEEK as one data point ------------------
per_sku = (
    detail.groupby("sku")
    .apply(lambda g: pd.Series({
        "actual_total": g["actual"].sum(),
        "pred_total": g["predicted"].sum(),
        "accuracy_pct": 100 - (g["actual"] - g["predicted"]).abs().sum() / g["actual"].sum() * 100
                        if g["actual"].sum() > 0 else np.nan,
    }), include_groups=False)
    .sort_values("accuracy_pct", ascending=False)
)

overall_actual = detail["actual"].sum()
overall_wape = (detail["actual"] - detail["predicted"]).abs().sum() / overall_actual * 100
overall_accuracy = 100 - overall_wape

pd.set_option("display.width", 140)
print(f"\n=== Per-SKU weekly forecast accuracy (test window {test_start.date()} -> {test_end.date()}) ===")
print(per_sku.to_string(float_format=lambda x: f"{x:.1f}"))
print(f"\nOverall weekly accuracy (8 weeks x 30 SKUs): {overall_accuracy:.2f}%")

per_sku.to_csv(OUT_SUMMARY)
print(f"Saved -> {OUT_SUMMARY}")
