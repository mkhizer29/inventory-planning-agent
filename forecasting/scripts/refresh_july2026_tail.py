"""
2026-07-26: refresh the tail end of `training_dataset_30skus.csv`.

The existing file's 2026-07-15 -> 2026-07-20 rows were all zeros, believed at
the time to be a genuine empty-data artifact (order feed frozen at 07-15).
Re-checked live against pg_new_1: that's now stale -- real order data exists
for pilot SKUs through 2026-07-23 (sustained volume, ~150-250 units/day
invoiced, DB-wide order counts back to the pre-freeze normal range of
~2,600-3,400/day). 2026-07-24 drops to just 1 pilot order (41 DB-wide) --
treated as another partial/cutoff day, same as how the original 2026-07-15
partial day was excluded, so it's NOT included here.

Action: replace the existing (stale-zero) 07-15->07-20 rows and add
07-21->07-23 as new rows, using the same query/formulas as
extend_training_dataset_backfill.py. Excludes 07-24 onward.
"""
import pymysql
import pandas as pd

MASTER = "../data/training_dataset_30skus.csv"
REFRESH_START = "2026-07-15"
REFRESH_END = "2026-07-23"  # inclusive; 07-24 excluded as a partial/cutoff day


def load_env(path="../../.env"):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


env = load_env()
existing = pd.read_csv(MASTER)
skus = sorted(existing["sku"].unique().tolist())

conn = pymysql.connect(
    host=env["STAGING_HOST"], port=int(env["STAGING_PORT"]),
    user=env["STAGING_USER"], password=env["STAGING_PASSWORD"].strip(),
    database="pg_new_1", connect_timeout=10,
)
cur = conn.cursor()
fmt = ",".join(["%s"] * len(skus))

cur.execute(f"""
    SELECT i.sku,
           DATE(o.created_at) AS d,
           SUM(i.qty_invoiced) - SUM(i.qty_refunded) AS net_qty,
           SUM(i.row_total) AS revenue,
           COUNT(DISTINCT o.entity_id) AS order_count,
           COUNT(DISTINCT CASE WHEN o.coupon_code IS NOT NULL AND o.coupon_code != ''
                                THEN o.entity_id END) AS coupon_orders,
           MAX(CASE WHEN o.discount_amount IS NOT NULL AND o.discount_amount != 0
                     THEN 1 ELSE 0 END) AS is_promo_order_present
    FROM sales_order o
    JOIN sales_order_item i ON i.order_id = o.entity_id
    WHERE i.sku IN ({fmt})
      AND o.created_at >= %s AND o.created_at < %s
    GROUP BY i.sku, d
""", skus + [REFRESH_START, (pd.Timestamp(REFRESH_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")])
rows = cur.fetchall()
conn.close()

sales = pd.DataFrame(rows, columns=[
    "sku", "date", "net_qty", "revenue", "order_count",
    "coupon_orders", "is_promo_order_present",
])
sales["date"] = pd.to_datetime(sales["date"])

full_dates = pd.date_range(REFRESH_START, REFRESH_END, freq="D")
grid = pd.MultiIndex.from_product([skus, full_dates], names=["sku", "date"]).to_frame(index=False)
refresh_df = grid.merge(sales, on=["sku", "date"], how="left")
for c in ["net_qty", "revenue", "order_count", "coupon_orders", "is_promo_order_present"]:
    refresh_df[c] = refresh_df[c].fillna(0)
refresh_df["net_qty"] = refresh_df["net_qty"].astype(int)
refresh_df["order_count"] = refresh_df["order_count"].astype(int)
refresh_df["coupon_orders"] = refresh_df["coupon_orders"].astype(int)
refresh_df["is_promo_order_present"] = refresh_df["is_promo_order_present"].astype(int)

refresh_df["day_of_week"] = refresh_df["date"].dt.day_name()
refresh_df["is_ramadan"] = 0
refresh_df["is_eid_fitr"] = 0
refresh_df["is_eid_adha"] = 0

snapshot_cols = ["sku", "product_name", "category", "brand", "price", "special_price",
                  "cost", "is_in_stock", "current_stock_qty", "is_active", "visibility"]
snapshot = existing[snapshot_cols].drop_duplicates(subset="sku")
refresh_df = refresh_df.merge(snapshot, on="sku", how="left")

existing["date"] = pd.to_datetime(existing["date"], format="%d/%m/%Y")

combined_cols = ["date", "sku", "product_name", "category", "brand", "day_of_week",
                  "net_qty", "revenue", "order_count", "coupon_orders",
                  "is_promo_order_present", "is_ramadan", "is_eid_fitr", "is_eid_adha",
                  "price", "special_price", "cost", "is_in_stock", "current_stock_qty",
                  "is_active", "visibility"]

# drop the stale rows being replaced (07-15 -> whatever the old max date was, all <= REFRESH_END anyway)
kept = existing[existing["date"] < pd.Timestamp(REFRESH_START)]
combined = pd.concat([kept[combined_cols], refresh_df[combined_cols]], ignore_index=True)
combined = combined.sort_values(["sku", "date"]).reset_index(drop=True)

assert combined.duplicated(subset=["sku", "date"]).sum() == 0, "duplicate sku/date rows after merge"

g = combined.groupby("sku")["net_qty"]
combined["lag_1"] = g.shift(1)
combined["lag_7"] = g.shift(7)
combined["lag_14"] = g.shift(14)
combined["rolling_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
combined["rolling_std_7"] = g.transform(lambda s: s.shift(1).rolling(7).std())
combined["rolling_mean_14"] = g.transform(lambda s: s.shift(1).rolling(14).mean())
combined["rolling_std_14"] = g.transform(lambda s: s.shift(1).rolling(14).std())

combined["date"] = combined["date"].dt.strftime("%d/%m/%Y")

parsed = pd.to_datetime(combined["date"], format="%d/%m/%Y")
print(f"Replaced/added rows for {REFRESH_START} -> {REFRESH_END}: {len(refresh_df)}")
print(f"Combined shape: {combined.shape}")
print(f"Parsed date range: {parsed.min()} -> {parsed.max()}, n_days={parsed.nunique()}")

combined.to_csv(MASTER, index=False)
print(f"Saved -> {MASTER}")
