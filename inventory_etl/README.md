# Inventory Planning Agent — ETL Pipeline (Layer 2)

Extracts the Naheed Magento `pg_1` database (staging **or** the production backup)
and transforms it into the five **canonical tables** from the Technical Specification §9,
written to a SQLite warehouse that feeds the forecasting / risk / reorder / allocation
models (Layer 3–4) and the dashboard (Layer 6).

```
Magento pg_1 (MySQL)  ──extract──▶  clean/transform  ──load──▶  output/inventory.db (SQLite)
  staging OR local backup                                        + output/data_quality_report.md
```

## Canonical output tables (spec §9)

| Table | Grain | Built from |
|---|---|---|
| `sku_master` | 1 row / SKU | `catalog_product_entity` + EAV + `nhd_product_flat` + `staging_margin` (cost) + `nhd_box_products` (pack) |
| `sales_transactions` | 1 row / order line | `sales_order_item` ⋈ `sales_order`; channel from `shipping_method`; enriched with product_name, qty_invoiced/shipped, discount, order_state, hashed customer id |
| `inventory_snapshot` | 1 row / SKU / location | `cataloginventory_stock_item` + per-warehouse cols of `nhd_product_flat` (cleaned) |
| `channel_master` | 1 row / channel | config seed (store / naheed_web / online_delivery / foodpanda) |
| `external_signals` | 1 row / day | Pakistan holiday calendar + payday windows (weather = follow-up) |

## Supporting signal tables (merged in — demand/promo/fulfillment signals)

All guarded: a table that doesn't exist in the chosen source is skipped, not fatal.

| Table | Grain | Source | Why it matters |
|---|---|---|---|
| `inventory_snapshot_history` | SKU/location/**date** | accumulates each run | real stock time-series (Magento keeps none); re-running a date replaces it |
| `shipments` | shipment line | `sales_shipment(_item)` | true fulfilled qty + date |
| `returns` | credit-memo line | `sales_creditmemo(_item)` | return rate / net-demand correction |
| `promotions_catalog` | rule × product | `catalogrule(_product)` | promo calendar for spike attribution (FR-C3/C4) |
| `promotions_cart` | rule/coupon | `salesrule(_coupon)` | coupon-driven demand lift |
| `delivery_geography` | order | `sales_order_address` | regional demand |
| `related_products` | link | `catalog_product_link` | substitute/complement mapping |
| `product_views` | SKU/day | `report_viewed_product_aggregated_daily` | intent + cold-start + spike confirmation |
| `stock_alerts` | signup | `product_alert_stock` | unmet demand during stockouts |
| `search_queries` | query | `search_query` / `search_query_1` | cold-start signal for never-sold SKUs |

## Setup

Python is installed at `C:\Users\Bilal\Python312emb\python.exe` (embeddable 3.12.8, all deps
installed and the package editable-installed). To reproduce on another machine:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
copy .env.example .env      # then fill STAGING_PASSWORD / LOCAL_PASSWORD
```

## Run

```powershell
$py = "C:\Users\Bilal\Python312emb\python.exe"
cd "C:\Users\Bilal\Documents\Naheed Ai Internship\inventory_etl"

# against the live staging DB (default)
& $py -m etl.run_etl --source staging

# against the production backup once loaded into local MySQL
& $py -m etl.run_etl --source local_backup --sales-since 2023-01-01

# tests
& $py -m pytest -q
```

Outputs: `output/inventory.db` (canonical warehouse), `output/csv/*.csv` (one CSV per
canonical table, Excel-friendly), and `output/data_quality_report.md`.

## Configuration

- **`.env`** — connection profiles (staging / local_backup) and the SQLite target path. Not committed.
- **`config/config.yaml`** — all business rules & assumptions: channel mapping, cost-dedup
  strategy, perishability threshold, and the replenishment inputs **not present in the DB**
  (lead times, MOQ, channel cost/service factors). Everything marked `ASSUMPTION` must be
  reconciled with the Naheed buying team.

## Key data-handling decisions (from the schema analysis)

- **Cost**: the Magento `cost` EAV attribute is ~empty; cost comes from `staging_margin`,
  which has **duplicate rows per product** — collapsed via `config.cost.strategy` (default `max`),
  with `catalog_product_flat_1.cost` as fallback.
- **Stock cleansing**: negative quantities clamped to 0; sentinel values (≥10,000, i.e.
  "unlimited" drop-ship markers) set to NULL/`sentinel_unmanaged`. Both counted in the DQ report.
- **Channel**: derived from `sales_order.shipping_method` carrier code (Magento has one website).
- **Demand**: `quantity_sold = qty_ordered − qty_canceled − qty_refunded` (net, floored at 0);
  configurable parent lines excluded to avoid double counting.
- **EAV portability**: attribute IDs are resolved **by code at query time**, so the same SQL runs
  on both the staging and production databases (whose attribute IDs differ).
- **Warehouse split**: when `nhd_product_flat` warehouse columns are populated (production), stock
  is emitted per location (MLR/BHD/KKN/KRG); otherwise a single `ALL` pool (staging).

## Known gaps (NOT in the source DB — supplied as configurable assumptions)

`supplier_lead_time_days`, `MOQ`, `stock_in_transit` (on-order), and per-channel
cost/service-level factors. See `config/config.yaml` → `replenishment` / `channels.master`.

## Project layout

```
inventory_etl/
├─ etl/
│  ├─ config.py            # .env + yaml loader
│  ├─ db.py                # source(MySQL)/target(SQLite) engines, table guards
│  ├─ extract.py           # runs SQL, guards optional tables
│  ├─ cleanse.py           # pure DQ functions (tested)
│  ├─ transform.py         # builds the 5 canonical frames
│  ├─ external_signals.py  # holiday/payday calendar
│  ├─ load.py              # SQLite writer + indexes
│  ├─ quality_report.py    # DQ report
│  ├─ run_etl.py           # orchestrator CLI
│  └─ sql/*.sql            # portable extract queries
├─ config/config.yaml
├─ tests/test_cleanse.py
├─ output/                 # inventory.db + data_quality_report.md
└─ requirements.txt, pyproject.toml, .env(.example)
```
