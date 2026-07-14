-- product_ops : Naheed's custom ops/inventory flat table (nhd_product_flat).
-- GUARDED: extract.py only runs this if the table exists.
-- In production this is fully populated (per-warehouse qty, stockout history,
-- picking_mode); on staging most columns are empty but the query still works.
SELECT
    product_id,
    sku,
    parent_sku,
    brand,
    category_tag,
    parent_category,
    category_hierarchy,
    barcode,
    quantity,
    qty_sold,
    is_in_stock,
    days_out_of_stock,
    days_in_stock,
    out_of_stock_at,
    in_stock_at,
    picking_mode,
    last_order_date,
    kokon_pharmacy_qty,
    bahadurabad_qty,
    malir_qty,
    korangi_qty
FROM nhd_product_flat
;
