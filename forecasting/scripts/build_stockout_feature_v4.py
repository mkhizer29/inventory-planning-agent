"""
2026-07-27: v4 of the model-ready dataset build -- identical to
build_stockout_feature.py except `is_on_sale` is now KEPT as a model feature
instead of being treated as leakage.

Why the reversal (user decision, 2026-07-27): the forward-looking marketing
promo calendar (data_request.md #4) is still coming, and once it exists,
`is_on_sale` for a FUTURE date will be populated from that calendar -- known
in advance, same as is_ramadan/is_eid_fitr, not derived from that day's own
orders. Until that calendar exists, both train and test windows here are
already-elapsed history, so backfilling `is_on_sale` from realized
sales_order_item price-vs-original_price data (see SKUs/build_sale_flag.py)
is a faithful reconstruction of what the calendar would have said, applied
consistently to both splits. This is different from `is_promo_order_present`,
which stays excluded -- that flag depends on whether an order *happened* that
day (order-level discount_amount on realized orders), not on the SKU's own
listed price, so it can't be reconstructed the same way for a day with no
orders.

Produces SKUs/lgbm-dataset-4.csv (same 19 cols as lgbm-dataset-3.csv, plus
`is_on_sale`). Everything else (adaptive is_in_stock derivation, cutoff
handling) is unchanged from build_stockout_feature.py -- see that file for
the full rationale.
"""
import pandas as pd

SRC = "../data/training_dataset_30skus.csv"
OUT = "../data/lgbm-dataset-4.csv"

TEAM_FLAGGED_OOS_SKUS = [
    "IC-1020926", "IC-1032718", "IC-1037570",
    "IC-1042719", "IC-1166218", "IC-1166220",
]

FLOOR_DAYS = 10
MULTIPLIER = 2

# ---- load -------------------------------------------------------------------
df = pd.read_csv(SRC)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df.sort_values(["sku", "date"]).reset_index(drop=True)

print(f"Loaded {len(df)} rows, date range {df['date'].min().date()} -> {df['date'].max().date()}")


# ---- derive per-SKU adaptive stockout flag -----------------------------------
def zero_runs(group):
    is_zero = (group["net_qty"] == 0).to_numpy()
    runs = []
    i = 0
    n = len(is_zero)
    while i < n:
        if is_zero[i]:
            j = i
            while j < n and is_zero[j]:
                j += 1
            runs.append((i, j, j - i))
            i = j
        else:
            i += 1
    return runs


derived_flags = []
for sku, g in df.groupby("sku", sort=False):
    g = g.sort_values("date").reset_index(drop=True)
    runs = zero_runs(g)
    run_lengths = [r[2] for r in runs]
    p90 = pd.Series(run_lengths).quantile(0.9) if run_lengths else 0
    threshold = max(FLOOR_DAYS, MULTIPLIER * p90)

    flag = pd.Series(1, index=g.index)
    for start, end, length in runs:
        if length > threshold:
            flag.iloc[start:end] = 0

    derived_flags.append(pd.DataFrame({"sku": sku, "date": g["date"], "is_in_stock_derived": flag.values}))

derived = pd.concat(derived_flags, ignore_index=True)
df = df.merge(derived, on=["sku", "date"], how="left")

latest_date = df["date"].max()
last_day = df[df["date"] == latest_date]
print(f"\n=== Team-flagged SKUs: derived is_in_stock at cutoff ({latest_date.date()}) ===")
for sku in TEAM_FLAGGED_OOS_SKUS:
    row = last_day[last_day["sku"] == sku]
    if not row.empty:
        print(f"{sku}: snapshot={row['is_in_stock'].iloc[0]}  derived={row['is_in_stock_derived'].iloc[0]}")

print(f"\nis_on_sale coverage: {df['is_on_sale'].sum()} / {len(df)} rows flagged on-sale "
      f"({df.groupby('sku')['is_on_sale'].max().sum()} / {df['sku'].nunique()} SKUs ever on sale)")

# ---- assemble final model-ready dataset --------------------------------------
LEAKAGE_COLS = ["revenue", "order_count", "coupon_orders", "is_promo_order_present",
                "avg_discount_pct"]  # is_on_sale is now kept, see docstring above
ZERO_VARIANCE_COLS = ["visibility", "is_active"]
FULLY_MISSING_COLS = ["special_price"]
DROP_RAW_STOCK_COLS = ["current_stock_qty", "is_in_stock"]
REFERENCE_ONLY_COLS = ["product_name"]

drop_cols = LEAKAGE_COLS + ZERO_VARIANCE_COLS + FULLY_MISSING_COLS + DROP_RAW_STOCK_COLS + REFERENCE_ONLY_COLS
final = df.drop(columns=[c for c in drop_cols if c in df.columns])
final = final.rename(columns={"is_in_stock_derived": "is_in_stock"})

front = ["date", "sku", "category", "brand", "day_of_week", "net_qty"]
rest = [c for c in final.columns if c not in front]
final = final[front + rest]

final.to_csv(OUT, index=False)
print(f"\nSaved {len(final)} rows, {len(final.columns)} cols -> {OUT}")
print(f"Columns: {list(final.columns)}")
