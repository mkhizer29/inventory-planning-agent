"""Unit tests for the pure cleansing functions (run: python -m pytest)."""
import numpy as np
import pandas as pd

from etl import cleanse


def test_clean_stock_qty_negative_and_sentinel():
    qty = pd.Series([5, -3, 0, 99986, 100])
    clean, flag = cleanse.clean_stock_qty(qty, sentinel_threshold=10000, negative_floor=0)
    assert list(flag) == ["ok", "negative_clamped", "ok", "sentinel_unmanaged", "ok"]
    assert clean[1] == 0            # negative clamped
    assert np.isnan(clean[3])       # sentinel -> NaN
    assert clean[0] == 5 and clean[4] == 100


def test_resolve_cost_max_and_dedup():
    df = pd.DataFrame({
        "product_id": [132, 132, 132, 133],
        "cost": [265.0, 186.0, 0.135, 1050.0],
        "final_price": [186.0, 186.0, 186.0, 1050.0],
    })
    res = cleanse.resolve_cost(df, strategy="max").set_index("product_id")
    assert res.loc[132, "unit_cost"] == 265.0
    assert res.loc[132, "cost_row_count"] == 3
    assert res.loc[133, "unit_cost"] == 1050.0


def test_resolve_cost_nearest_price():
    df = pd.DataFrame({
        "product_id": [132, 132, 132],
        "cost": [265.0, 186.0, 0.135],
        "final_price": [186.0, 186.0, 186.0],
    })
    res = cleanse.resolve_cost(df, strategy="nearest_price").set_index("product_id")
    assert res.loc[132, "unit_cost"] == 186.0  # closest to final_price


def test_resolve_cost_drops_nonpositive():
    df = pd.DataFrame({"product_id": [1, 1], "cost": [0.0, -5.0], "final_price": [10, 10]})
    assert cleanse.resolve_cost(df, strategy="max").empty


def test_derive_channel():
    codes = pd.Series(["storepickup", "foodpanda", "flatrate", "unknownx", None])
    mapping = {"storepickup": "store", "foodpanda": "foodpanda", "flatrate": "online_delivery"}
    out = cleanse.derive_channel(codes, mapping, default="online_delivery")
    assert list(out) == ["store", "foodpanda", "online_delivery", "online_delivery", "online_delivery"]


def test_classify_perishable():
    sl = pd.Series([None, 0, 30, 500])
    is_per, clean = cleanse.classify_perishable(sl, max_days=365)
    assert list(is_per) == [False, False, True, False]
