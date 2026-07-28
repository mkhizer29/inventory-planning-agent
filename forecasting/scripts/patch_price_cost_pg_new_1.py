"""
2026-07-24: patch `price`/`cost` in the 30-SKU master training file using
`pg_new_1` instead of `pg_1`, per team decision.

Root cause (confirmed against both schemas, not assumed): the original ETL
pulled `price` from `pg_1` and `cost` from `pg_new_1` -- two different
snapshots, mixed. `pg_1` has cost data for only 63/31,628 catalog products
(none of our 30 pilot SKUs), so `pg_1` cost was NULL for all of them; cost
in the existing file could only have come from `pg_new_1`. That mixing is
what produced the "cost > price" anomaly for 26/30 SKUs.

Checked whether any other snapshot column also needs re-sourcing:
  - special_price: NULL in both schemas for all 30 SKUs -- no change.
  - is_active, visibility: identical in both schemas for all 30 SKUs -- no change.
  - is_in_stock (cataloginventory_stock_item): identical in both schemas -- no change.
  - qty (current_stock_qty): identical for 29/30 SKUs; IC-1185817 drifted by 3
    units (82 -> 85, a few sales between snapshots) -- immaterial, and this
    column is dropped from the model datasets anyway (see build_stockout_feature.py).
So only price and cost actually need patching.

pg_new_1 is adopted as the primary source for these snapshot columns going
forward (better coverage: price 32,867 vs 31,487 rows catalog-wide; cost
32,613 vs 63 rows catalog-wide).

Special case: IC-1055803 has cost=0.000000 in pg_new_1, which -- like the
NULLs everywhere else -- represents "cost not set", not a real free cost.
Forced to NaN here for consistency with how missing cost is represented for
every other product in the catalog.
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
skus = sorted(df["sku"].unique().tolist())

conn = pymysql.connect(
    host=env["STAGING_HOST"], port=int(env["STAGING_PORT"]),
    user=env["STAGING_USER"], password=env["STAGING_PASSWORD"].strip(),
    database="pg_new_1", connect_timeout=10,
)
cur = conn.cursor()
fmt = ",".join(["%s"] * len(skus))
cur.execute(f"""
    SELECT e.sku,
           MAX(CASE WHEN d.attribute_id = 77 THEN d.value END) AS price,
           MAX(CASE WHEN d.attribute_id = 81 THEN d.value END) AS cost
    FROM catalog_product_entity e
    LEFT JOIN catalog_product_entity_decimal d
           ON d.entity_id = e.entity_id AND d.attribute_id IN (77, 81)
    WHERE e.sku IN ({fmt})
    GROUP BY e.sku
""", skus)
rows = cur.fetchall()
conn.close()

new_vals = pd.DataFrame(rows, columns=["sku", "price_new", "cost_new"]).set_index("sku")
new_vals["price_new"] = new_vals["price_new"].astype(float)
new_vals["cost_new"] = new_vals["cost_new"].astype(float)
new_vals.loc["IC-1055803", "cost_new"] = float("nan")  # unset cost, not a real zero

print("=== pg_new_1 values being applied ===")
print(new_vals.to_string())

before = df[["sku", "price", "cost"]].drop_duplicates(subset="sku").set_index("sku")

df = df.merge(new_vals, on="sku", how="left")
n_price_changed = (df["price"] != df["price_new"]).sum()
df["price"] = df["price_new"]
df["cost"] = df["cost_new"]
df = df.drop(columns=["price_new", "cost_new"])

after = df[["sku", "price", "cost"]].drop_duplicates(subset="sku").set_index("sku")
compare = before.join(after, lsuffix="_old", rsuffix="_new")
still_bad = (compare["cost_new"] > compare["price_new"]).sum()

print(f"\nPatched price+cost for {len(new_vals)} SKUs ({n_price_changed} row-level price cells changed)")
print(f"SKUs with cost > price after patch: {still_bad} (was 26/30 before)")

df.to_csv(MASTER, index=False)
print(f"Saved -> {MASTER}")
