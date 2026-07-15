# ETL Pipeline (Layer 2) — technical reference

This is the data-ingestion component of the **Inventory Planning Agent**. It extracts
the Naheed Magento `pg_1` database (live staging **or** a locally-loaded production
backup) and transforms it into the five **canonical tables** from the Technical
Specification §9, plus supporting signal tables — written to a SQLite warehouse (and
CSVs) that feed the forecasting / risk / reorder / allocation models and the dashboard.

```
Magento pg_1 (MySQL)  ──extract──▶  clean / transform  ──load──▶  inventory_etl/output/inventory.db
  staging OR local_backup                                          + output/csv/*.csv
                                                                   + output/data_quality_report.md
```

> Setup and run commands live in the **root `README.md`** (you run everything from the
> repository root). This file documents *what the pipeline produces and how it works*.

## Where things live (after the move into the repo)

| Item | Location |
|---|---|
| `.env` / `.env.example` | repository root |
| `requirements.txt` / `pyproject.toml` | repository root |
| ETL business config | `inventory_etl/config/config.yaml` |
| Python package | `inventory_etl/etl/` (import name: `etl`) |
| SQL extract queries | `inventory_etl/etl/sql/*.sql` |
| Tests | `inventory_etl/tests/` |
| Generated outputs | `inventory_etl/output/` (`inventory.db`, `csv/`, `data_quality_report.md`) |

Path resolution (`etl/config.py`): `ETL_ROOT = inventory_etl/`, `REPO_ROOT = repo root`,
`ENV_PATH = REPO_ROOT/.env`, `CONFIG_PATH = ETL_ROOT/config/config.yaml`. A relative
`TARGET_SQLITE_PATH` in `.env` resolves against `REPO_ROOT`, so the default
`inventory_etl/output/inventory.db` lands inside this project.

## Canonical output tables (spec §9)

| Table | Grain | Built from |
|---|---|---|
| `sku_master` | 1 row / SKU | `catalog_product_entity` + EAV + `nhd_product_flat` + `staging_margin` (cost) + `nhd_box_products` (pack) |
| `sales_transactions` | 1 row / order line | `sales_order_item` ⋈ `sales_order`; channel from `shipping_method`; enriched with product_name, qty_invoiced/shipped, discount, order_state, hashed customer id |
| `inventory_snapshot` | 1 row / SKU / location | `cataloginventory_stock_item` + per-warehouse cols of `nhd_product_flat` (cleaned) |
| `channel_master` | 1 row / channel | config seed (store / naheed_web / online_delivery / foodpanda) |
| `external_signals` | 1 row / day | Pakistan holiday calendar + payday windows (weather = follow-up) |

## Supporting signal tables (demand / promo / fulfillment)

All guarded: a table absent from the chosen source is skipped, not fatal.

| Table | Grain | Source | Why it matters |
|---|---|---|---|
| `inventory_snapshot_history` | SKU/location/**date** | accumulates each run | real stock time-series (Magento keeps none); re-running a date replaces it |
| `shipments` | shipment line | `sales_shipment(_item)` | true fulfilled qty + date |
| `returns` | credit-memo line | `sales_creditmemo(_item)` | return rate / net-demand correction |
| `promotions_catalog` | rule × product | `catalogrule(_product)` | promo calendar for spike attribution |
| `promotions_cart` | rule/coupon | `salesrule(_coupon)` | coupon-driven demand lift |
| `delivery_geography` | order | `sales_order_address` | regional demand |
| `related_products` | link | `catalog_product_link` | substitute/complement mapping |
| `product_views` | SKU/day | `report_viewed_product_aggregated_daily` | intent + cold-start + spike confirmation |
| `stock_alerts` | signup | `product_alert_stock` | unmet demand during stockouts |
| `search_queries` | query | `search_query` / `search_query_1` | cold-start signal for never-sold SKUs |

## Configuration

- **`.env`** (repo root) — connection profiles (`staging` / `local_backup`) and
  `TARGET_SQLITE_PATH`. Never committed. Copy from `.env.example` and fill in the passwords.
- **`config/config.yaml`** — business rules & assumptions: channel mapping, cost-dedup
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

## Package layout

```
inventory_etl/
├─ etl/
│  ├─ config.py            # .env + yaml loader; REPO_ROOT / ETL_ROOT path anchors
│  ├─ db.py                # source(MySQL)/target(SQLite) engines, table guards
│  ├─ extract.py           # runs SQL, guards optional tables
│  ├─ cleanse.py           # pure data-quality functions (unit-tested)
│  ├─ transform.py         # builds the canonical + supporting frames
│  ├─ external_signals.py  # holiday/payday calendar
│  ├─ load.py              # SQLite writer + indexes + CSV export + stock history
│  ├─ quality_report.py    # data-quality report
│  ├─ run_etl.py           # orchestrator CLI (python -m etl.run_etl)
│  └─ sql/*.sql            # portable extract queries (shipped as package data)
├─ config/config.yaml
├─ tests/                  # test_cleanse.py, test_paths.py
├─ output/                 # generated (git-ignored)
└─ run_etl.bat             # double-click / CLI launcher (portable)
```
