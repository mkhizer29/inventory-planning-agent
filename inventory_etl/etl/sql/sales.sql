-- sales : raw order-line demand joined to the order header (enriched).
-- Channel is derived from sales_order.shipping_method (Magento has ONE website;
-- channel lives in the carrier code = text before the first underscore).
-- Configurable parent lines are excluded so demand isn't double-counted.
-- customer_id is hashed (SHA2-256) so repeat-purchase features are possible
-- without exposing customer identity.
--
-- Placeholders substituted by extract.py:
--   {SINCE_FILTER}   e.g.  AND oi.created_at >= '2023-01-01'   or (empty)
--   {PAGE_FILTER}    e.g.  AND oi.item_id > 100 AND oi.item_id <= 200   or (empty)
--                    Set when the extract is read in primary-key ranges rather
--                    than one streaming pass (see db.read_sql_key_ranges).

SELECT
    oi.item_id                                         AS transaction_id,
    oi.order_id,
    oi.product_id,
    oi.sku,
    oi.name                                            AS product_name,
    oi.product_type,
    so.status                                          AS order_status,
    so.state                                           AS order_state,
    so.shipping_method,
    SUBSTRING_INDEX(so.shipping_method, '_', 1)        AS carrier_code,
    DATE(oi.created_at)                                AS transaction_date,
    oi.created_at                                      AS transaction_ts,
    oi.qty_ordered,
    oi.qty_invoiced,
    oi.qty_shipped,
    oi.qty_canceled,
    oi.qty_refunded,
    oi.price                                           AS unit_price,
    oi.original_price,
    oi.discount_amount,
    oi.base_cost,
    oi.row_total,
    SHA2(so.customer_id, 256)                          AS customer_id_hash
FROM sales_order_item oi
JOIN sales_order so ON so.entity_id = oi.order_id
WHERE (oi.product_type IS NULL OR oi.product_type <> 'configurable')
{SINCE_FILTER}
{PAGE_FILTER}
;
