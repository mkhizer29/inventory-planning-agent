"""Pure data-cleansing functions (FR-A4) — no DB/IO, fully unit-testable.

Implements the fixes surfaced in the schema analysis:
  * negative stock  -> clamped to a floor and logged
  * sentinel stock  (>= threshold, e.g. 99986/99991 "unlimited" drop-ship) -> NULL
  * conflicting cost rows in staging_margin -> collapsed by a configurable strategy
  * channel derivation from a shipping-method carrier code
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── stock ───────────────────────────────────────────────────────────────────
def clean_stock_qty(qty: pd.Series, *, sentinel_threshold: float, negative_floor: float = 0.0
                    ) -> tuple[pd.Series, pd.Series]:
    """Return (clean_qty, flag) where flag ∈ {ok, negative_clamped, sentinel_unmanaged}.

    Sentinels become NaN (unknown / not inventory-managed), negatives clamp to floor.
    """
    q = pd.to_numeric(qty, errors="coerce")
    flag = pd.Series("ok", index=q.index, dtype=object)

    sentinel = q >= sentinel_threshold
    flag[sentinel] = "sentinel_unmanaged"

    negative = q < negative_floor
    flag[negative] = "negative_clamped"

    clean = q.copy()
    clean[sentinel] = np.nan
    clean[negative] = negative_floor
    return clean, flag


# ── cost ────────────────────────────────────────────────────────────────────
_COST_STRATEGIES = {"max", "min", "median", "nearest_price"}


def resolve_cost(cost_rows: pd.DataFrame, *, strategy: str = "max") -> pd.DataFrame:
    """Collapse duplicate staging_margin rows to one cost per product_id.

    Expects columns: product_id, cost, final_price (final_price only needed for
    'nearest_price'). Returns product_id, unit_cost, cost_row_count.
    """
    if strategy not in _COST_STRATEGIES:
        raise ValueError(f"cost strategy must be one of {_COST_STRATEGIES}, got {strategy!r}")
    if cost_rows.empty:
        return pd.DataFrame(columns=["product_id", "unit_cost", "cost_row_count"])

    df = cost_rows.copy()
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df = df.dropna(subset=["cost"])
    df = df[df["cost"] > 0]  # zero/negative costs are not usable

    counts = df.groupby("product_id").size().rename("cost_row_count")

    if strategy == "max":
        chosen = df.groupby("product_id")["cost"].max()
    elif strategy == "min":
        chosen = df.groupby("product_id")["cost"].min()
    elif strategy == "median":
        chosen = df.groupby("product_id")["cost"].median()
    else:  # nearest_price
        df["fp"] = pd.to_numeric(df.get("final_price"), errors="coerce")
        df["dist"] = (df["cost"] - df["fp"]).abs()
        chosen = (df.sort_values("dist")
                    .groupby("product_id")["cost"].first())

    result = chosen.rename("unit_cost").reset_index().merge(
        counts.reset_index(), on="product_id", how="left")
    return result


# ── channel ───────────────────────────────────────────────────────────────────
def derive_channel(carrier_code: pd.Series, mapping: dict, default: str) -> pd.Series:
    """Map a shipping-method carrier code to a canonical channel."""
    c = carrier_code.fillna("").str.strip().str.lower()
    m = {k.lower(): v for k, v in mapping.items()}
    return c.map(m).fillna(default)


# ── perishability ─────────────────────────────────────────────────────────────
def classify_perishable(shelf_life_days: pd.Series, *, max_days: int) -> tuple[pd.Series, pd.Series]:
    """Return (is_perishable, shelf_life_days_clean)."""
    sl = pd.to_numeric(shelf_life_days, errors="coerce")
    is_per = sl.notna() & (sl > 0) & (sl <= max_days)
    return is_per.astype(bool), sl


# ── dedup ─────────────────────────────────────────────────────────────────────
def dedupe_latest(df: pd.DataFrame, key: str, order_col: str) -> pd.DataFrame:
    """Keep the latest row per key (FR: duplicate handling, keep latest write)."""
    if df.empty:
        return df
    return (df.sort_values(order_col)
              .drop_duplicates(subset=[key], keep="last")
              .reset_index(drop=True))
