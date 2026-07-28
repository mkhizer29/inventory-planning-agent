"""
2026-07-27: derive a per-SKU, per-day "on sale" flag from sales_order_item,
per user request, as a candidate stand-in for the still-missing marketing
promo calendar (data_request.md #4).

Method: for each pilot SKU and each day, compare `sales_order_item.price`
(price actually charged on the order line) against `original_price` (catalog
price at order time). If any order line for that SKU/day has price <
original_price, flag that day as "on sale" for that SKU. Also record the
average discount depth for informational purposes.

Exploratory check before building (2026-07-27, live pg_new_1 query):
  - Across all 30 pilot SKUs' full order-item history, price < original_price
    on 15,741 / 74,557 rows (~21%); price was NEVER observed above
    original_price (as expected -- no line item price exceeds "MSRP").
  - Only 9 of the 30 SKUs EVER show a discounted line at all. The other 21
    have original_price == price on literally every order line in their
    entire history -- for those, this column will be constant 0. Most
    discounted: IC-1134493 (~89% of its lines), IC-1088406 (~76%),
    IC-1001018 (~91%), IC-1178594/1178591 (~85%), IC-1185817 (~64%).

IMPORTANT CAVEAT -- same-day leakage, same category as `is_promo_order_present`:
  This flag can only be computed from order lines that already exist for that
  day. A day with zero orders for a SKU has no rows to inspect, so we can't
  distinguish "genuinely not on sale" from "was on sale but nobody ordered it"
  -- we default those to 0 (not on sale), which understates true sale-day
  coverage on slow days. More importantly: like `is_promo_order_present`, this
  is derived from the SAME transactions net_qty is computed from, so it is
  only known in hindsight, once that day's demand has already been realized.
  It is NOT safe to feed to the model as-is for forecasting a future day,
  for the same reason `is_promo_order_present` sits in LEAKAGE_COLS in
  build_stockout_feature.py. It's useful right now as a RETROSPECTIVE feature
  (e.g. to check whether IC-1147930's under-predicted spike lines up with a
  real sale), not yet as a live model input -- that still needs marketing's
  forward-looking promo calendar, or a lagged/smoothed version of this signal.
"""
import pymysql
import pandas as pd

MASTER = "../data/training_dataset_30skus.csv"


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
df = pd.read_csv(MASTER)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
skus = sorted(df["sku"].unique().tolist())

conn = pymysql.connect(
    host=env["STAGING_HOST"], port=int(env["STAGING_PORT"]),
    user=env["STAGING_USER"], password=env["STAGING_PASSWORD"].strip(),
    database="pg_new_1", connect_timeout=10,
)
cur = conn.cursor()
fmt = ",".join(["%s"] * len(skus))
cur.execute(f"""
    SELECT sku,
           DATE(created_at) AS day,
           MAX(CASE WHEN price < original_price THEN 1 ELSE 0 END) AS is_on_sale,
           AVG(CASE WHEN price < original_price
                    THEN (original_price - price) / original_price END) AS avg_discount_pct
    FROM sales_order_item
    WHERE sku IN ({fmt})
    GROUP BY sku, DATE(created_at)
""", skus)
rows = cur.fetchall()
conn.close()

sale = pd.DataFrame(rows, columns=["sku", "date", "is_on_sale", "avg_discount_pct"])
sale["date"] = pd.to_datetime(sale["date"])
sale["avg_discount_pct"] = sale["avg_discount_pct"].astype(float).round(4)

# ---- coverage report ----------------------------------------------------
per_sku = sale.groupby("sku").agg(
    days_with_orders=("is_on_sale", "size"),
    days_on_sale=("is_on_sale", "sum"),
).reset_index()
per_sku["pct_days_on_sale"] = (per_sku["days_on_sale"] / per_sku["days_with_orders"] * 100).round(1)
per_sku = per_sku.sort_values("days_on_sale", ascending=False)
print("=== Per-SKU: days with any discounted order line ===")
print(per_sku.to_string(index=False))
print(f"\nSKUs that never show a discount: {(per_sku['days_on_sale'] == 0).sum()} / {len(per_sku)}")

# ---- merge into master, days with no orders default to not-on-sale ------
before_cols = set(df.columns)
df = df.merge(sale, on=["sku", "date"], how="left")
df["is_on_sale"] = df["is_on_sale"].fillna(0).astype(int)
# avg_discount_pct stays NaN when not on sale -- 0 would misleadingly imply "on sale, 0% off"

new_cols = [c for c in df.columns if c not in before_cols]
print(f"\nAdded columns: {new_cols}")
print(f"Rows where is_on_sale=1: {df['is_on_sale'].sum()} / {len(df)}")

df["date"] = df["date"].dt.strftime("%d/%m/%Y")
df.to_csv(MASTER, index=False)
print(f"\nSaved -> {MASTER}")
