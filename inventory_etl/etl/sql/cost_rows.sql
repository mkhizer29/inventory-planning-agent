-- cost_rows : per-product cost candidates from staging_margin.
-- GUARDED (staging_margin may be absent). NOTE: this table has MULTIPLE rows
-- per product with conflicting cost values; transform.py collapses them using
-- config.cost.strategy. The Magento EAV `cost` attribute is ~empty, so this is
-- the primary cost source.
SELECT
    product_id,
    sku,
    final_price,
    cost,
    margin_abs,
    margin_pct
FROM staging_margin
WHERE cost IS NOT NULL
;
