-- flat_cost : fallback cost from Magento's flat catalog table.
-- GUARDED (catalog_product_flat_1 may be absent). Used only where
-- staging_margin has no cost for a product.
SELECT
    entity_id AS product_id,
    cost
FROM catalog_product_flat_1
WHERE cost IS NOT NULL
;
