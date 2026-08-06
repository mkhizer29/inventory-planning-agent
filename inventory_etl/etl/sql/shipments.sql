-- shipments : actual fulfilled quantities with dates (sales_shipment + item).
-- The truest "units that physically left" signal; complements ordered qty.
--
-- Placeholder substituted by extract.py:
--   {SINCE_FILTER}   e.g.  WHERE ss.created_at >= '2026-01-01'   or (empty)
-- Bounded by the same window as sales.sql: shipments outside the extracted
-- sales window have no order line in the warehouse to attach to. Unfiltered
-- this is 2.5M rows (ss.created_at is indexed, so the filter is cheap).
SELECT
    ss.order_id,
    ssi.sku,
    ssi.product_id,
    ssi.qty            AS qty_shipped,
    ss.created_at      AS shipment_date
FROM sales_shipment_item ssi
JOIN sales_shipment ss ON ss.entity_id = ssi.parent_id
{SINCE_FILTER}
;
