-- box_products : case-pack / carton mapping (nhd_box_products).
-- GUARDED (production-only table). Maps an outer box SKU to its inner lined SKU
-- and units-per-box -> feeds pack-size rounding for reorder recommendations.
SELECT
    box_sku,
    lined_sku,
    box_qty,
    threshold
FROM nhd_box_products
;
