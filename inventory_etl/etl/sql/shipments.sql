-- shipments : actual fulfilled quantities with dates (sales_shipment + item).
-- The truest "units that physically left" signal; complements ordered qty.
SELECT
    ss.order_id,
    ssi.sku,
    ssi.product_id,
    ssi.qty            AS qty_shipped,
    ss.created_at      AS shipment_date
FROM sales_shipment_item ssi
JOIN sales_shipment ss ON ss.entity_id = ssi.parent_id
;
