-- product_views : daily product page-view counts (report aggregate).
-- Demand-intent signal; helps confirm genuine spikes (views AND orders rise)
-- and gives a cold-start signal for never-sold SKUs.
SELECT
    period      AS view_date,
    store_id,
    product_id,
    product_name,
    product_price,
    views_num
FROM report_viewed_product_aggregated_daily
;
