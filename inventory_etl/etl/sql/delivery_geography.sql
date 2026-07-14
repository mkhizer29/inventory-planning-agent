-- delivery_geography : shipping destination per order (regional demand).
SELECT
    parent_id AS order_id,
    city,
    region,
    postcode,
    country_id
FROM sales_order_address
WHERE address_type = 'shipping'
;
