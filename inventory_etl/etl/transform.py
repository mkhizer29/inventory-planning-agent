"""Transform raw Magento frames into the canonical spec §9 tables."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import cleanse, config

log = logging.getLogger("etl.transform")


# ── sku_master ────────────────────────────────────────────────────────────────
def build_sku_master(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cfg = config.settings()
    prod = raw["products"].copy()
    ops = raw.get("product_ops", pd.DataFrame())

    # --- cost: prefer the Magento `cost` attribute (populated in pg_new_1),
    #     then staging_margin (deduped), then catalog_product_flat_1 ---
    prod = prod.rename(columns={"cost": "eav_cost"})
    margin_cost = cleanse.resolve_cost(raw.get("cost_rows", pd.DataFrame()),
                                       strategy=cfg["cost"]["strategy"])
    prod = prod.merge(margin_cost, on="product_id", how="left")   # adds unit_cost, cost_row_count
    flat_cost = raw.get("flat_cost", pd.DataFrame())
    if not flat_cost.empty:
        prod = prod.merge(flat_cost.rename(columns={"cost": "flat_cost"}),
                          on="product_id", how="left")
    else:
        prod["flat_cost"] = pd.NA
    prod["unit_cost"] = (pd.to_numeric(prod["eav_cost"], errors="coerce")
                         .fillna(pd.to_numeric(prod.get("unit_cost"), errors="coerce"))
                         .fillna(pd.to_numeric(prod["flat_cost"], errors="coerce")))
    prod.drop(columns=["eav_cost", "flat_cost", "cost_row_count"], inplace=True, errors="ignore")

    # --- ops enrichment (brand/category/picking_mode) ---
    if not ops.empty:
        keep = ops[["product_id", "brand", "category_tag", "parent_category",
                    "picking_mode", "barcode"]].rename(
            columns={"brand": "brand_ops", "barcode": "barcode_ops"})
        prod = prod.merge(keep, on="product_id", how="left")
        prod["brand"] = prod["brand"].fillna(prod["brand_ops"])
        prod["barcode"] = prod["barcode"].fillna(prod["barcode_ops"])
        prod["sub_category"] = prod["category_tag"]
        prod["category"] = prod["parent_category"]
        prod.drop(columns=["brand_ops", "barcode_ops", "category_tag", "parent_category"],
                  inplace=True, errors="ignore")
        prod["picking_mode"] = prod.get("picking_mode")
    else:
        prod["sub_category"] = np.nan
        prod["category"] = np.nan
        prod["picking_mode"] = np.nan

    # --- perishability ---
    is_per, shelf = cleanse.classify_perishable(
        prod["shelf_life_days"], max_days=cfg["perishability"]["perishable_max_days"])
    prod["is_perishable"] = is_per
    prod["shelf_life_days"] = shelf

    # --- replenishment assumptions (NOT in DB) ---
    rep = cfg["replenishment"]
    lt_map = {k: v for k, v in rep.get("lead_time_by_picking_mode", {}).items()}
    prod["supplier_lead_time_days"] = (
        prod["picking_mode"].map(lt_map).fillna(rep["default_supplier_lead_time_days"]))
    prod["moq"] = rep["default_moq"]
    prod["is_dropship"] = prod["picking_mode"].isin(rep.get("dropship_modes", [])).fillna(False)

    # --- pack size from case-pack table ---
    box = raw.get("box_products", pd.DataFrame())
    if not box.empty:
        packs = (box.dropna(subset=["lined_sku"])
                    .drop_duplicates("lined_sku")[["lined_sku", "box_qty"]])
        prod = prod.merge(packs, left_on="sku", right_on="lined_sku", how="left")
        prod["pack_size"] = pd.to_numeric(prod["box_qty"], errors="coerce").fillna(1).astype(int)
        prod.drop(columns=["lined_sku", "box_qty"], inplace=True, errors="ignore")
    else:
        prod["pack_size"] = 1

    prod["supplier_id"] = np.nan  # not reliably in DB

    out = prod.rename(columns={"sku": "sku_id", "name": "sku_name"})[[
        "sku_id", "product_id", "sku_name", "category", "sub_category", "brand",
        "is_perishable", "shelf_life_days", "unit_cost", "price", "special_price",
        "moq", "pack_size", "supplier_id", "supplier_lead_time_days",
        "picking_mode", "is_dropship", "status", "visibility", "weight", "barcode",
        "created_at", "updated_at",
    ]]
    return out


# ── channel_master ────────────────────────────────────────────────────────────
def build_channel_master() -> pd.DataFrame:
    master = config.settings()["channels"]["master"]
    rows = [{"channel_id": name, "channel_name": name,
             "fulfillment_cost_factor": v["fulfillment_cost_factor"],
             "service_level_target": v["service_level_target"]}
            for name, v in master.items()]
    return pd.DataFrame(rows)


# ── sales_transactions ──────────────────────────────────────────────────────────
def build_sales_transactions(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cfg = config.settings()["channels"]
    s = raw["sales"].copy()
    s["channel"] = cleanse.derive_channel(s["carrier_code"], cfg["mapping"], cfg["default"])

    for col in ["qty_ordered", "qty_invoiced", "qty_shipped", "qty_canceled",
                "qty_refunded", "unit_price", "discount_amount"]:
        s[col] = pd.to_numeric(s.get(col), errors="coerce").fillna(0.0)

    # net demand (never below zero); keep gross for reference
    s["quantity_sold"] = (s["qty_ordered"] - s["qty_canceled"] - s["qty_refunded"]).clip(lower=0)
    s = s.rename(columns={"sku": "sku_id"})
    return s[[
        "transaction_id", "order_id", "product_id", "sku_id", "product_name",
        "channel", "order_status", "order_state", "transaction_date", "transaction_ts",
        "qty_ordered", "qty_invoiced", "qty_shipped", "qty_canceled", "qty_refunded",
        "quantity_sold", "unit_price", "discount_amount", "row_total", "customer_id_hash",
    ]]


# ── supporting signal tables (merged from friend's pipeline + extras) ──────────
def build_support_tables(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Light-clean the extra signal tables. Empty inputs pass through as empty
    frames so downstream load/CSV still emit an (empty) table with headers."""
    out: dict[str, pd.DataFrame] = {}

    def _num(df, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    if "shipments" in raw:
        out["shipments"] = _num(raw["shipments"].copy(), ["qty_shipped"])
    if "returns" in raw:
        out["returns"] = _num(raw["returns"].copy(), ["qty_refunded", "refund_amount"])
    if "promotions_catalog" in raw:
        out["promotions_catalog"] = _num(raw["promotions_catalog"].copy(),
                                         ["is_active", "discount_amount", "product_id"])
    if "promotions_cart" in raw:
        out["promotions_cart"] = _num(raw["promotions_cart"].copy(),
                                      ["is_active", "discount_amount"])
    if "delivery_geography" in raw:
        g = raw["delivery_geography"].copy()
        for c in ("city", "region"):
            if c in g.columns:
                g[c] = g[c].astype("string").str.strip().str.title()
        out["delivery_geography"] = g
    if "related_products" in raw:
        out["related_products"] = raw["related_products"].copy()
    if "product_views" in raw:
        out["product_views"] = _num(raw["product_views"].copy(), ["views_num", "product_price"])
    if "stock_alerts" in raw:
        out["stock_alerts"] = raw["stock_alerts"].copy()
    if "search_queries" in raw:
        out["search_queries"] = _num(raw["search_queries"].copy(), ["num_results", "popularity"])
    return out


# ── inventory_snapshot ────────────────────────────────────────────────────────
_WAREHOUSE_COLS = {
    "malir_qty": "MLR", "bahadurabad_qty": "BHD",
    "kokon_pharmacy_qty": "KKN", "korangi_qty": "KRG",
}


def build_inventory_snapshot(raw: dict[str, pd.DataFrame], run_date: str) -> pd.DataFrame:
    cln = config.settings()["cleansing"]
    stock = raw["stock"].copy()
    clean_qty, flag = cleanse.clean_stock_qty(
        stock["qty"], sentinel_threshold=cln["stock_sentinel_threshold"],
        negative_floor=cln["stock_negative_floor"])
    stock["stock_on_hand"] = clean_qty
    stock["stock_flag"] = flag

    ops = raw.get("product_ops", pd.DataFrame())
    wh_present = (not ops.empty) and any(c in ops.columns for c in _WAREHOUSE_COLS)

    rows = []
    if wh_present:
        o = ops.copy()
        for c in _WAREHOUSE_COLS:
            o[c] = pd.to_numeric(o.get(c), errors="coerce").fillna(0)
        o["wh_total"] = o[list(_WAREHOUSE_COLS)].sum(axis=1)
        split_ids = set(o.loc[o["wh_total"] > 0, "product_id"])
        days_oos = o.set_index("product_id").get("days_out_of_stock")
    else:
        split_ids, days_oos = set(), None

    for _, r in stock.iterrows():
        pid = r["product_id"]
        base = {
            "product_id": pid,
            "stock_in_transit": 0,  # ASSUMPTION: no on-order data in DB
            "is_in_stock": int(r["is_in_stock"]) if pd.notna(r["is_in_stock"]) else None,
            "min_qty": r["min_qty"], "notify_stock_qty": r["notify_stock_qty"],
            "backorders": r["backorders"], "stock_flag": r["stock_flag"],
            "snapshot_date": run_date,
            "days_out_of_stock": (int(days_oos.get(pid)) if days_oos is not None
                                  and pd.notna(days_oos.get(pid)) else None),
        }
        if pid in split_ids:
            orow = o[o["product_id"] == pid].iloc[0]
            for col, code in _WAREHOUSE_COLS.items():
                if orow[col] > 0:
                    rows.append({**base, "location_id": code, "stock_on_hand": float(orow[col])})
        else:
            rows.append({**base, "location_id": "ALL", "stock_on_hand": r["stock_on_hand"]})

    snap = pd.DataFrame(rows)
    snap.insert(0, "snapshot_id", [f"{run_date}:{p}:{l}" for p, l in
                                   zip(snap["product_id"], snap["location_id"])])
    return snap
