-- related_products : Magento related / up-sell / cross-sell links.
-- Proxy for substitute/complement mapping (safety-stock via substitutability).
-- link_type code resolved from catalog_product_link_type
-- (commonly 1=relation, 4=up_sell, 5=cross_sell).
SELECT
    l.product_id,
    l.linked_product_id,
    l.link_type_id,
    t.code AS link_type
FROM catalog_product_link l
LEFT JOIN catalog_product_link_type t ON t.link_type_id = l.link_type_id
;
