-- returns : refunds / credit memos (sales_creditmemo + item).
-- Net-demand correction and return-rate features per SKU.
--
-- Placeholder substituted by extract.py:
--   {SINCE_FILTER}   e.g.  WHERE scm.created_at >= '2026-01-01'   or (empty)
-- Bounded by the same window as sales.sql, for the same reason.
SELECT
    scm.order_id,
    scmi.sku,
    scmi.product_id,
    scmi.qty            AS qty_refunded,
    scmi.row_total      AS refund_amount,
    scm.created_at      AS refund_date
FROM sales_creditmemo_item scmi
JOIN sales_creditmemo scm ON scm.entity_id = scmi.parent_id
{SINCE_FILTER}
;
