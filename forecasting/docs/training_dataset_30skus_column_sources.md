# training_dataset_30skus.csv — Column Source Mapping

Verified 2026-07-24 by connecting directly to the staging DB (`pg_new_1`, Magento
schema) and reproducing a sample row (`sku=IC-1001018`, `date=2026-01-30`) from
raw tables to confirm each formula matches the CSV exactly.

Grain: one row per (`sku`, `date`).

## 1. Product / catalog attributes (EAV snapshot, current values)

| Column | Table(s) | Key / Attribute | Formula |
|---|---|---|---|
| `sku` | `catalog_product_entity` | `.sku` | direct column |
| `product_name` | `catalog_product_entity_varchar` | `attribute_id = 73` ("name") | direct value, `store_id = 0` |
| `brand` | `catalog_product_entity_int` → `eav_attribute_option_value` | `attribute_id = 83` ("manufacturer") | dropdown option_id resolved to label |
| `category` | `catalog_category_product` → `catalog_category_entity` (`path`) → `catalog_category_entity_varchar` | product's leaf category walked up to its **level-2 ancestor** (top-level category under root) | name of the level-2 category, e.g. leaf "Mascaras" → "Health & Beauty" |
| `is_active` | `catalog_product_entity_int` | `attribute_id = 97` ("status") | direct value (1 = enabled) — **not** attribute_id 46, which is unset |
| `visibility` | `catalog_product_entity_int` | `attribute_id = 99` | direct value |
| `price` | `catalog_product_entity_decimal` | `attribute_id = 77` | direct value, sourced from `pg_new_1` (see `patch_price_cost_pg_new_1.py`) |
| `cost` | `catalog_product_entity_decimal` | `attribute_id = 81` | direct value, sourced from `pg_new_1` |
| `special_price` | `catalog_product_entity_decimal` | `attribute_id = 78` | direct value (NULL for all 30 pilot SKUs) |
| `is_in_stock` | `cataloginventory_stock_item` | joined on `product_id = entity_id` | direct value (current snapshot, repeated across all historical dates) |
| `current_stock_qty` | `cataloginventory_stock_item` | joined on `product_id = entity_id` | `.qty` (current snapshot, repeated across all historical dates) |

Note: these are **current-snapshot** values repeated identically across every
historical `date` row for a SKU — there is no historical catalog/stock feed in
the source DB.

## 2. Daily sales aggregation (per sku, per day)

Source: `sales_order_item` joined to `sales_order` on `sales_order.entity_id = sales_order_item.order_id`,
filtered by `sku` and `DATE(created_at)`, grouped by (`sku`, day).

| Column | Formula (per sku, per day) |
|---|---|
| `date` | `DATE(sales_order_item.created_at)` |
| `net_qty` | `SUM(qty_invoiced) − SUM(qty_refunded)` |
| `revenue` | `SUM(row_total)` (all order-item rows that day, including canceled/returned) |
| `order_count` | `COUNT(DISTINCT order_id)` |
| `coupon_orders` | `COUNT(DISTINCT order_id)` where `sales_order.coupon_code IS NOT NULL AND != ''` |
| `is_promo_order_present` | `1` if any order that sku/day has a non-zero `sales_order.discount_amount`, else `0` |

Verified exact match on sample row (IC-1001018, 2026-01-30): net_qty=30,
revenue=53762, order_count=37, coupon_orders=3, is_promo_order_present=1.

## 3. Calendar / holiday flags — not from the database

| Column | Source |
|---|---|
| `day_of_week` | derived from `date` |
| `is_ramadan` | calendar/holiday lookup (Ramadan date range), not a DB field |
| `is_eid_fitr` | calendar/holiday lookup, not a DB field |
| `is_eid_adha` | calendar/holiday lookup, not a DB field |

## 4. Rolling / lag features — computed after extraction

| Column | Computation |
|---|---|
| `lag_1` | `net_qty` shifted 1 day, per sku |
| `lag_7` | `net_qty` shifted 7 days, per sku |
| `lag_14` | `net_qty` shifted 14 days, per sku |
| `rolling_mean_7` | 7-day rolling mean of `net_qty`, per sku |
| `rolling_std_7` | 7-day rolling std of `net_qty`, per sku |
| `rolling_mean_14` | 14-day rolling mean of `net_qty`, per sku |
| `rolling_std_14` | 14-day rolling std of `net_qty`, per sku |

Not pulled from any table — computed from the `net_qty` time series itself.

## What needs to stay live for the next 6 months

For this dataset to keep being buildable, two things in `pg_new_1` need to
keep being populated/unchanged:

1. **Catalog EAV attributes** on `catalog_product_entity` (attribute IDs 73,
   77, 78, 81, 83, 97, 99) + `cataloginventory_stock_item`, plus the category
   tree (`catalog_category_product` / `catalog_category_entity`).
2. **`sales_order_item` joined to `sales_order`**, which drives every
   sales-side column (`net_qty`, `revenue`, `order_count`, `coupon_orders`,
   `is_promo_order_present`).

No separate reporting/aggregation table is involved in this pipeline.
