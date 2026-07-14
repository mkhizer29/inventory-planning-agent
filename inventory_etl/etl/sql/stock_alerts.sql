-- stock_alerts : "notify me when back in stock" signups (product_alert_stock).
-- Unmet demand during a stockout — a signal raw sales can never show (sales go
-- to zero during an outage). Count of signups in an OOS window ≈ lost demand.
SELECT
    alert_stock_id,
    customer_id,
    product_id,
    website_id,
    store_id,
    add_date,
    send_date,
    send_count,
    status
FROM product_alert_stock
;
