-- returns : refunds / credit memos (sales_creditmemo + item).
-- Net-demand correction and return-rate features per SKU.
SELECT
    scm.order_id,
    scmi.sku,
    scmi.product_id,
    scmi.qty            AS qty_refunded,
    scmi.row_total      AS refund_amount,
    scm.created_at      AS refund_date
FROM sales_creditmemo_item scmi
JOIN sales_creditmemo scm ON scm.entity_id = scmi.parent_id
;
