-- products_raw : one row per stock-carrying product with core attributes.
-- Portable across the staging and production `pg_1` databases:
-- every EAV attribute is resolved BY CODE (ids differ per DB), scoped to the
-- catalog_product entity type, at default scope (store_id = 0). Single
-- statement so it runs directly through pandas.read_sql.
--
-- Placeholders substituted by extract.py:
--   {TYPES}          e.g.  'simple'   or   'simple','configurable'
--   {STATUS_FILTER}  e.g.  AND st.value = 1     (enabled only)  or  (empty)

SELECT
    e.entity_id                             AS product_id,
    e.sku,
    e.type_id,
    e.created_at,
    e.updated_at,
    v_name.value                            AS name,
    st.value                                AS status,
    vis.value                               AS visibility,
    brand_ov.value                          AS brand,
    v_barcode.value                         AS barcode,
    CAST(v_price.value   AS DECIMAL(20,4))  AS price,
    CAST(v_special.value AS DECIMAL(20,4))  AS special_price,
    CAST(v_weight.value  AS DECIMAL(20,4))  AS weight,
    CAST(v_cost.value    AS DECIMAL(20,4))  AS cost,
    v_shelf.value                           AS shelf_life_days
FROM catalog_product_entity e

-- name
LEFT JOIN eav_attribute a_name
       ON a_name.attribute_code = 'name'
      AND a_name.entity_type_id = (SELECT entity_type_id FROM eav_entity_type
                                   WHERE entity_type_code = 'catalog_product' LIMIT 1)
LEFT JOIN catalog_product_entity_varchar v_name
       ON v_name.entity_id = e.entity_id AND v_name.store_id = 0
      AND v_name.attribute_id = a_name.attribute_id
-- status
LEFT JOIN eav_attribute a_status
       ON a_status.attribute_code = 'status'
      AND a_status.entity_type_id = (SELECT entity_type_id FROM eav_entity_type
                                     WHERE entity_type_code = 'catalog_product' LIMIT 1)
LEFT JOIN catalog_product_entity_int st
       ON st.entity_id = e.entity_id AND st.store_id = 0
      AND st.attribute_id = a_status.attribute_id
-- visibility
LEFT JOIN eav_attribute a_vis
       ON a_vis.attribute_code = 'visibility'
      AND a_vis.entity_type_id = (SELECT entity_type_id FROM eav_entity_type
                                  WHERE entity_type_code = 'catalog_product' LIMIT 1)
LEFT JOIN catalog_product_entity_int vis
       ON vis.entity_id = e.entity_id AND vis.store_id = 0
      AND vis.attribute_id = a_vis.attribute_id
-- brand (manufacturer option id -> label)
LEFT JOIN eav_attribute a_manu
       ON a_manu.attribute_code = 'manufacturer'
      AND a_manu.entity_type_id = (SELECT entity_type_id FROM eav_entity_type
                                   WHERE entity_type_code = 'catalog_product' LIMIT 1)
LEFT JOIN catalog_product_entity_int v_manu
       ON v_manu.entity_id = e.entity_id AND v_manu.store_id = 0
      AND v_manu.attribute_id = a_manu.attribute_id
LEFT JOIN eav_attribute_option_value brand_ov
       ON brand_ov.option_id = v_manu.value AND brand_ov.store_id = 0
-- barcode
LEFT JOIN eav_attribute a_barcode
       ON a_barcode.attribute_code = 'barcode'
      AND a_barcode.entity_type_id = (SELECT entity_type_id FROM eav_entity_type
                                      WHERE entity_type_code = 'catalog_product' LIMIT 1)
LEFT JOIN catalog_product_entity_varchar v_barcode
       ON v_barcode.entity_id = e.entity_id AND v_barcode.store_id = 0
      AND v_barcode.attribute_id = a_barcode.attribute_id
-- price
LEFT JOIN eav_attribute a_price
       ON a_price.attribute_code = 'price'
      AND a_price.entity_type_id = (SELECT entity_type_id FROM eav_entity_type
                                    WHERE entity_type_code = 'catalog_product' LIMIT 1)
LEFT JOIN catalog_product_entity_decimal v_price
       ON v_price.entity_id = e.entity_id AND v_price.store_id = 0
      AND v_price.attribute_id = a_price.attribute_id
-- special_price
LEFT JOIN eav_attribute a_special
       ON a_special.attribute_code = 'special_price'
      AND a_special.entity_type_id = (SELECT entity_type_id FROM eav_entity_type
                                      WHERE entity_type_code = 'catalog_product' LIMIT 1)
LEFT JOIN catalog_product_entity_decimal v_special
       ON v_special.entity_id = e.entity_id AND v_special.store_id = 0
      AND v_special.attribute_id = a_special.attribute_id
-- weight
LEFT JOIN eav_attribute a_weight
       ON a_weight.attribute_code = 'weight'
      AND a_weight.entity_type_id = (SELECT entity_type_id FROM eav_entity_type
                                     WHERE entity_type_code = 'catalog_product' LIMIT 1)
LEFT JOIN catalog_product_entity_decimal v_weight
       ON v_weight.entity_id = e.entity_id AND v_weight.store_id = 0
      AND v_weight.attribute_id = a_weight.attribute_id
-- cost (Magento `cost` attribute; populated in pg_new_1 — primary unit-cost source)
LEFT JOIN eav_attribute a_cost
       ON a_cost.attribute_code = 'cost'
      AND a_cost.entity_type_id = (SELECT entity_type_id FROM eav_entity_type
                                   WHERE entity_type_code = 'catalog_product' LIMIT 1)
LEFT JOIN catalog_product_entity_decimal v_cost
       ON v_cost.entity_id = e.entity_id AND v_cost.store_id = 0
      AND v_cost.attribute_id = a_cost.attribute_id
-- shelf life (expire_after_day) — may live in int OR decimal backend
LEFT JOIN eav_attribute a_shelf
       ON a_shelf.attribute_code = 'expire_after_day'
      AND a_shelf.entity_type_id = (SELECT entity_type_id FROM eav_entity_type
                                    WHERE entity_type_code = 'catalog_product' LIMIT 1)
LEFT JOIN (
    SELECT entity_id, attribute_id, value FROM catalog_product_entity_int   WHERE store_id = 0
    UNION ALL
    SELECT entity_id, attribute_id, value FROM catalog_product_entity_decimal WHERE store_id = 0
) v_shelf ON v_shelf.entity_id = e.entity_id AND v_shelf.attribute_id = a_shelf.attribute_id

WHERE e.type_id IN ({TYPES})
{STATUS_FILTER}
;
