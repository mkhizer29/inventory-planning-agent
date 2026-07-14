# ETL Data-Quality Report
- source profile : **staging**
- run date       : 2026-07-14

## Row counts
| table                      |   rows |
|----------------------------|--------|
| sku_master                 |  30950 |
| sales_transactions         |   6460 |
| inventory_snapshot         |  31628 |
| channel_master             |      4 |
| external_signals           |   2387 |
| shipments                  |    789 |
| returns                    |      0 |
| promotions_catalog         |  22644 |
| promotions_cart            |     11 |
| delivery_geography         |   2429 |
| related_products           |      0 |
| product_views              |      0 |
| stock_alerts               |      0 |
| search_queries             |   1489 |
| inventory_snapshot_history |  31628 |

## sku_master field coverage
| check                | coverage   |
|----------------------|------------|
| unit_cost present    | 46.2%      |
| price present        | 100.0%     |
| brand present        | 100.0%     |
| category present     | 71.4%      |
| is_perishable = True | 0.0%       |
| is_dropship = True   | 0.0%       |
| pack_size > 1        | 0.0%       |

## inventory_snapshot cleansing flags
| flag               |   rows |
|--------------------|--------|
| ok                 |  28566 |
| negative_clamped   |   2661 |
| sentinel_unmanaged |    401 |

### stock rows by location
| location_id   |   rows |
|---------------|--------|
| ALL           |  31628 |

## sales_transactions summary
| channel         |   lines |   net_units |
|-----------------|---------|-------------|
| foodpanda       |       9 |          11 |
| online_delivery |    5920 |        6798 |
| store           |     531 |         548 |

- date range: 2020-09-14 → 2026-07-13
- distinct SKUs sold: 1265

## external_signals
- days: 2387  | holidays: 99  | payday days: 360

## Supporting signal tables (merged)
| table                      |   rows | status                   |
|----------------------------|--------|--------------------------|
| shipments                  |    789 | populated                |
| returns                    |      0 | empty (present, no rows) |
| promotions_catalog         |  22644 | populated                |
| promotions_cart            |     11 | populated                |
| delivery_geography         |   2429 | populated                |
| related_products           |      0 | empty (present, no rows) |
| product_views              |      0 | empty (present, no rows) |
| stock_alerts               |      0 | empty (present, no rows) |
| search_queries             |   1489 | populated                |
| inventory_snapshot_history |  31628 | populated                |

## ⚠ Warnings
- Cost coverage < 50% — reorder/margin math will be sparse. Confirm cost source with the buying team.
- All inventory is single-pool ('ALL') — per-warehouse columns are unpopulated in this source (expected on staging).