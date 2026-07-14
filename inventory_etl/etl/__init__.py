"""Inventory Planning Agent — ETL (Layer 2).

Extracts the Magento `pg_1` schema (staging or local backup) and transforms it
into the five canonical tables defined in the Technical Specification §9:
sku_master, sales_transactions, inventory_snapshot, channel_master, external_signals.
"""
__version__ = "0.1.0"
