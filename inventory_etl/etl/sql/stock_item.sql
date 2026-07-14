-- stock_item : authoritative single-source inventory record (one row / product).
-- Magento legacy CatalogInventory, stock_id = 1 (this DB has no MSI).
-- min_qty / notify_stock_qty are Naheed's EXISTING reorder-point config — a
-- baseline the planning agent can benchmark against.
SELECT
    si.product_id,
    si.qty,
    si.is_in_stock,
    si.min_qty,
    si.notify_stock_qty,
    si.min_sale_qty,
    si.max_sale_qty,
    si.backorders,
    si.manage_stock,
    si.qty_increments,
    si.low_stock_date
FROM cataloginventory_stock_item si
WHERE si.stock_id = 1
;
