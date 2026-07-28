"""
2026-07-26: extend `training_dataset_30skus.csv` backward using the team's
promised 6-month historical backfill in `pg_new_1`.

Verified before writing this script (see project memory + the conversation
that produced it):
  - Real production-volume data now exists in pg_new_1 from 2025-07-24 onward
    (DB-wide order count jumps from single digits/day to ~1,600-2,200/day
    exactly on 2025-07-24). 2025-07-15 through 2025-07-23 is still old sparse
    "playground" noise (1-19 orders on scattered days, whole DB) -- same
    character as the pre-2026 noise already excluded from the original
    analysis, so it's excluded here too, NOT treated as part of the backfill.
  - No Ramadan/Eid window falls inside 2025-07-24 -> 2026-01-14 (Ramadan 2025
    was Mar 1-29, Eid al-Fitr Mar 30-31, Eid al-Adha Jun 6-7 -- all before
    this window starts), so is_ramadan/is_eid_fitr/is_eid_adha are 0
    throughout the new rows.
  - New window end (2026-01-14) is contiguous with the existing file's start
    (2026-01-15) -- no gap to patch in the middle.

Column formulas reproduced exactly from `training_dataset_30skus_column_sources.md`:
  net_qty   = SUM(qty_invoiced) - SUM(qty_refunded), per sku/day
  revenue   = SUM(row_total), all rows that day (incl. canceled/returned)
  order_count       = COUNT(DISTINCT order_id)
  coupon_orders     = COUNT(DISTINCT order_id) where coupon_code is set
  is_promo_order_present = 1 if any order that sku/day has discount_amount != 0
Catalog snapshot columns (product_name/category/brand/price/special_price/
cost/is_in_stock/current_stock_qty/is_active/visibility) are current-snapshot
and repeat identically across all dates for a sku -- reused directly from the
existing file rather than re-queried, since re-querying would return the same
values (see column_sources.md note on this).

Lag/rolling features (`lag_1/7/14`, `rolling_mean/std_7/14`) are recomputed
over the FULL combined series per sku after concatenation, since the new
early rows change what's available at the 2026-01-15 boundary (e.g. lag_7 on
2026-01-15 can now be filled from 2026-01-08, whereas before it was NaN).
"""
import pymysql
import pandas as pd

MASTER = "../data/training_dataset_30skus.csv"
NEW_START = "2025-07-24"
NEW_END = "2026-01-14"  # inclusive; contiguous with existing file's 2026-01-15 start


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
""", skus + [NEW_START, (pd.Timestamp(NEW_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")])
rows = cur.fetchall()
conn.close()

sales = pd.DataFrame(rows, columns=[
    "sku", "date", "net_qty", "revenue", "order_count",
    "coupon_orders", "is_promo_order_present",
])
sales["date"] = pd.to_datetime(sales["date"])

# zero-fill full sku x day grid for the new window
full_dates = pd.date_range(NEW_START, NEW_END, freq="D")
grid = pd.MultiIndex.from_product([skus, full_dates], names=["sku", "date"]).to_frame(index=False)
new_df = grid.merge(sales, on=["sku", "date"], how="left")
for c in ["net_qty", "revenue", "order_count", "coupon_orders", "is_promo_order_present"]:
    new_df[c] = new_df[c].fillna(0)
new_df["net_qty"] = new_df["net_qty"].astype(int)
new_df["order_count"] = new_df["order_count"].astype(int)
new_df["coupon_orders"] = new_df["coupon_orders"].astype(int)
new_df["is_promo_order_present"] = new_df["is_promo_order_present"].astype(int)

new_df["day_of_week"] = new_df["date"].dt.day_name()
new_df["is_ramadan"] = 0
new_df["is_eid_fitr"] = 0
new_df["is_eid_adha"] = 0

# reuse catalog snapshot columns per sku from the existing file (current-snapshot,
# identical across all dates -- see column_sources.md)
snapshot_cols = ["sku", "product_name", "category", "brand", "price", "special_price",
                  "cost", "is_in_stock", "current_stock_qty", "is_active", "visibility"]
snapshot = existing[snapshot_cols].drop_duplicates(subset="sku")
new_df = new_df.merge(snapshot, on="sku", how="left")

existing_dates = pd.to_datetime(existing["date"], format="%d/%m/%Y")
existing = existing.copy()
existing["date"] = existing_dates

combined_cols = ["date", "sku", "product_name", "category", "brand", "day_of_week",
                  "net_qty", "revenue", "order_count", "coupon_orders",
                  "is_promo_order_present", "is_ramadan", "is_eid_fitr", "is_eid_adha",
                  "price", "special_price", "cost", "is_in_stock", "current_stock_qty",
                  "is_active", "visibility"]

combined = pd.concat([new_df[combined_cols], existing[combined_cols]], ignore_index=True)
combined = combined.sort_values(["sku", "date"]).reset_index(drop=True)

assert combined.duplicated(subset=["sku", "date"]).sum() == 0, "duplicate sku/date rows after merge"

# recompute lag/rolling features over the full combined series per sku
g = combined.groupby("sku")["net_qty"]
combined["lag_1"] = g.shift(1)
combined["lag_7"] = g.shift(7)
combined["lag_14"] = g.shift(14)
combined["rolling_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
combined["rolling_std_7"] = g.transform(lambda s: s.shift(1).rolling(7).std())
combined["rolling_mean_14"] = g.transform(lambda s: s.shift(1).rolling(14).mean())
combined["rolling_std_14"] = g.transform(lambda s: s.shift(1).rolling(14).std())

combined["date"] = combined["date"].dt.strftime("%d/%m/%Y")

print(f"New rows added: {len(new_df)} ({len(full_dates)} days x {len(skus)} skus)")
print(f"Combined shape: {combined.shape}")
print(f"Date range: {combined['date'].iloc[0]} -> not sorted string, checking via parsed dates instead")
parsed = pd.to_datetime(combined["date"], format="%d/%m/%Y")
print(f"Parsed date range: {parsed.min()} -> {parsed.max()}, n_days={parsed.nunique()}")

combined.to_csv(MASTER, index=False)
print(f"Saved -> {MASTER}")
