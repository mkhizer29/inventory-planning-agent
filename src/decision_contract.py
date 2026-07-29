"""decision_contract.py — Phase B (forecast-driven stockout risk) output contract.

The counterpart to ``model_contract.py`` for the DECISION layer. It fixes the exact
schema of the two Phase B artifacts and validates them, so the orchestrator, the
standalone CLI and the dashboard all agree on columns and invariants:

  * ``decisions/stockout_risk.parquet``        one row per selected (sku, channel)
  * ``decisions/stockout_trajectory.parquet``  one row per forecast (sku, channel, date)

Design notes
------------
* These are DECISION artifacts derived from the operational demand forecast + the
  run's inventory context. They never contain ``y_true``/``units_observed`` and never
  create purchase orders. Stockout risk here is a forecast estimate, not an observed
  historical stockout.
* Atomic, partial-write-safe I/O is reused from ``model_contract`` so there is one
  writer pattern across the whole pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import model_contract as mc  # noqa: E402  (reuse atomic writers + pattern)

write_dataframe_atomic = mc.write_dataframe_atomic
write_json_atomic = mc.write_json_atomic

RISK_TIERS = ("critical", "high", "medium", "low", "unknown")

# ── stockout_risk.parquet — exactly one row per selected (sku, channel) ───────────────
STOCKOUT_RISK_COLUMNS = [
    "run_id", "sku", "channel", "sku_name",
    "operational_model", "operational_horizon", "as_of_date",
    "stock_on_hand", "stock_on_hand_is_synthetic", "stock_source",
    "reported_on_order_quantity", "usable_on_order_quantity", "on_order_available",
    "inventory_position_for_risk",
    "lead_time_days", "lead_time_source", "service_level",
    "forecast_horizon_available", "lead_time_horizon_sufficient",
    "lead_time_demand_p50", "lead_time_demand_p80", "lead_time_demand_p95", "lead_time_sigma",
    "safety_stock", "reorder_point",
    "forecast_days_of_cover", "projected_stockout_date", "days_until_projected_stockout",
    "survives_forecast_horizon",
    "stockout_probability", "probability_risk_tier", "cover_risk_tier", "overall_risk_tier",
    "expected_shortage_units", "estimated_revenue_at_risk",
    "uncertainty_method", "confidence_label", "manual_review_required",
    "assumption_flags", "reason_trace",
]

# ── stockout_trajectory.parquet — one row per forecast (sku, channel, date) ───────────
STOCKOUT_TRAJECTORY_COLUMNS = [
    "run_id", "sku", "channel", "date", "forecast_horizon_day",
    "daily_demand_mean", "daily_sigma",
    "cumulative_demand_mean", "cumulative_sigma",
    "demand_p50", "demand_p80", "demand_p95",
    "projected_p50_inventory", "cumulative_stockout_probability",
]

_RISK_KEY = ["sku", "channel"]
_TRAJ_KEY = ["sku", "channel", "date"]


def _require_exact_columns(df: pd.DataFrame, cols: list[str], what: str) -> None:
    if list(df.columns) != cols:
        missing = [c for c in cols if c not in df.columns]
        extra = [c for c in df.columns if c not in cols]
        raise ValueError(f"{what} columns mismatch. missing={missing} extra={extra} "
                         f"(order must match the contract exactly)")


def _prob_in_unit_interval(series: pd.Series, what: str) -> None:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if np.isinf(valid).any():
        raise ValueError(f"{what} contains infinite values")
    if ((valid < 0) | (valid > 1)).any():
        raise ValueError(f"{what} has values outside [0, 1]")


def validate_stockout_risk(df: pd.DataFrame, selected_keys: pd.DataFrame,
                           run_id: str) -> pd.DataFrame:
    """Validate the per-(sku, channel) risk table. Returns a deterministically sorted copy.

    * exact contract columns
    * exactly one row per selected (sku, channel) — no more, no fewer
    * stockout_probability + cumulative-style fields in [0, 1]
    * risk tiers within the allowed vocabulary
    * run_id constant and matching
    """
    out = df.copy()
    _require_exact_columns(out, STOCKOUT_RISK_COLUMNS, "stockout_risk")
    if out.empty:
        raise ValueError("stockout_risk is empty (expected one row per selected sku/channel)")
    if out.duplicated(_RISK_KEY).any():
        raise ValueError("stockout_risk has duplicate (sku, channel) rows")

    want = set(map(tuple, selected_keys[_RISK_KEY].astype(str).itertuples(index=False, name=None)))
    got = set(map(tuple, out[_RISK_KEY].astype(str).itertuples(index=False, name=None)))
    if want != got:
        raise ValueError(f"stockout_risk keys != selected keys "
                         f"(missing={want - got}, extra={got - want})")

    if set(out["run_id"].astype(str).unique()) != {str(run_id)}:
        raise ValueError("stockout_risk run_id is not constant/matching")

    _prob_in_unit_interval(out["stockout_probability"], "stockout_probability")
    for col in ("probability_risk_tier", "cover_risk_tier", "overall_risk_tier"):
        bad = set(out[col].dropna().astype(str)) - set(RISK_TIERS)
        if bad:
            raise ValueError(f"{col} has values outside {RISK_TIERS}: {sorted(bad)}")
    return out.sort_values(_RISK_KEY).reset_index(drop=True)


def validate_stockout_trajectory(df: pd.DataFrame, forecast_keys: pd.DataFrame,
                                 run_id: str) -> pd.DataFrame:
    """Validate the per-(sku, channel, date) trajectory. Returns a sorted copy.

    * exact contract columns
    * one row per forecast (sku, channel, date) key
    * cumulative_stockout_probability in [0, 1]
    """
    out = df.copy()
    _require_exact_columns(out, STOCKOUT_TRAJECTORY_COLUMNS, "stockout_trajectory")
    out["date"] = pd.to_datetime(out["date"])
    if out.duplicated(_TRAJ_KEY).any():
        raise ValueError("stockout_trajectory has duplicate (sku, channel, date) rows")

    fk = forecast_keys.copy()
    fk["date"] = pd.to_datetime(fk["date"])
    want = set(map(tuple, fk[_TRAJ_KEY].astype({"sku": str, "channel": str}).itertuples(index=False, name=None)))
    got = set(map(tuple, out[_TRAJ_KEY].astype({"sku": str, "channel": str}).itertuples(index=False, name=None)))
    if want != got:
        raise ValueError(f"stockout_trajectory keys != forecast keys "
                         f"(missing={len(want - got)}, extra={len(got - want)})")

    if set(out["run_id"].astype(str).unique()) != {str(run_id)}:
        raise ValueError("stockout_trajectory run_id is not constant/matching")
    _prob_in_unit_interval(out["cumulative_stockout_probability"], "cumulative_stockout_probability")
    return out.sort_values(_TRAJ_KEY).reset_index(drop=True)
