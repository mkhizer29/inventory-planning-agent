"""
2026-07-27: category-level rolling features, per the 2026-07-26 next-steps
list item ("borrow signal across SKUs in the same category, since 28-30
SKUs means per-SKU history is thin for some").

For each (sku, date), computes lag/rolling features on the LEAVE-ONE-OUT
category total -- i.e. the sum of net_qty across every OTHER SKU in the
same category (Groceries & Pets / Health & Beauty) that day, excluding the
SKU's own contribution. Leave-one-out (not the full category total
including this SKU) is deliberate: the point is to give the model an
independent "how is the rest of the category trending" signal, distinct
from -- not redundant with -- the SKU's own lag_1/7/14 features. Including
the SKU's own contribution would make this most informative exactly for
already-high-volume SKUs (which dominate their category's total) and least
informative for the very sparse-history SKUs this feature is meant to help.

Same lag/rolling structure and same leakage discipline as the existing
per-SKU features (SKUs/training_dataset_30skus_column_sources.md section 4):
shift(1) before rolling, so no day's own value (or same-day other-SKU
values, which are just as unknowable at forecast time as this SKU's own
same-day value) leaks into its own feature row.

New columns: cat_lag_1, cat_lag_7, cat_lag_14, cat_rolling_mean_7,
cat_rolling_std_7, cat_rolling_mean_14, cat_rolling_std_14.

Input: SKUs/lgbm-dataset-4.csv (30 SKUs, current canonical daily dataset).
Output: SKUs/lgbm-dataset-5.csv (27 cols = 20 + 7 new).
"""
import pandas as pd

SRC = "../data/lgbm-dataset-4.csv"
OUT = "../data/lgbm-dataset-5.csv"

df = pd.read_csv(SRC)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df.sort_values(["category", "sku", "date"]).reset_index(drop=True)

# ---- category-day totals, then leave-one-out per SKU -------------------------
cat_day_total = df.groupby(["category", "date"])["net_qty"].sum().rename("cat_day_total")
df = df.merge(cat_day_total, on=["category", "date"], how="left")
df["other_sku_total"] = df["cat_day_total"] - df["net_qty"]

print("Category-day total net_qty (sanity check, first category/date):")
print(df.groupby("category")["cat_day_total"].describe().to_string())

# ---- per-SKU lag/rolling features on the leave-one-out series ----------------
df = df.sort_values(["sku", "date"]).reset_index(drop=True)
g = df.groupby("sku")["other_sku_total"]

df["cat_lag_1"] = g.shift(1)
df["cat_lag_7"] = g.shift(7)
df["cat_lag_14"] = g.shift(14)
df["cat_rolling_mean_7"] = g.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
df["cat_rolling_std_7"] = g.shift(1).rolling(7).std().reset_index(level=0, drop=True)
df["cat_rolling_mean_14"] = g.shift(1).rolling(14).mean().reset_index(level=0, drop=True)
df["cat_rolling_std_14"] = g.shift(1).rolling(14).std().reset_index(level=0, drop=True)

df = df.drop(columns=["cat_day_total", "other_sku_total"])

new_cols = ["cat_lag_1", "cat_lag_7", "cat_lag_14", "cat_rolling_mean_7",
            "cat_rolling_std_7", "cat_rolling_mean_14", "cat_rolling_std_14"]
print(f"\nNaN counts in new category features (expected: same warm-up pattern as SKU-level features):")
print(df[new_cols].isna().sum().to_string())

df["date"] = df["date"].dt.strftime("%d/%m/%Y")
df.to_csv(OUT, index=False)
print(f"\nSaved {len(df)} rows, {len(df.columns)} cols -> {OUT}")
print(f"Columns: {list(df.columns)}")
