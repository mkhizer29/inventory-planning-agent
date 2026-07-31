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


# ══════════════════════════════════════════════════════════════════════════════════════════
# Phase C — forecast-driven reorder recommendations contract
# ══════════════════════════════════════════════════════════════════════════════════════════
# Policy version constants (recorded in the manifest + every output row). Phase B has no
# in-code version, so its label is defined here (decision_contract is the shared contract) to
# avoid touching stockout_risk.py.
STOCKOUT_POLICY_VERSION = "stockout-v1"
REORDER_POLICY_VERSION = "reorder-v1"

VALID_REORDER_ACTIONS = ("order_now", "monitor", "no_order", "vendor_follow_up", "manual_review")
# Actions that must NEVER carry an actionable quantity or an order/arrival date.
NON_ORDER_ACTIONS = ("monitor", "no_order", "vendor_follow_up", "manual_review")

# ── reorder_recommendations.parquet — exactly one row per selected (sku, channel) ────────
# Order is the contract; reorder_recommendations.py builds rows in exactly this order.
REORDER_RECOMMENDATION_COLUMNS = [
    "run_id", "as_of_date", "sku", "product_id", "sku_name", "category", "brand", "channel",
    "operational_model", "operational_horizon",
    "action", "priority_rank", "reorder_triggered", "approval_required",
    "human_follow_up_required", "order_placed", "manual_review_required",
    "manual_review_reason", "review_reason_codes",
    "overall_risk_tier", "stockout_probability", "days_of_cover", "projected_stockout_date",
    "stock_on_hand", "stock_on_hand_is_synthetic", "reported_on_order_quantity",
    "usable_on_order_quantity", "inventory_position_for_risk",
    "lead_time_days", "lead_time_source", "lead_time_demand_mean", "lead_time_safety_stock",
    "forecast_driven_reorder_point",
    "target_cover_days", "planning_horizon_days", "available_forecast_horizon_days",
    "insufficient_horizon",
    "planning_horizon_demand_mean", "planning_horizon_sigma", "service_level_target",
    "service_level_z", "planning_safety_stock", "target_stock",
    "raw_target_gap", "raw_order_quantity", "moq", "moq_source", "moq_adjusted_quantity",
    "pack_size", "pack_size_source", "rounded_order_quantity", "provisional_calculated_quantity",
    "recommended_order_quantity",
    "recommended_order_date", "expected_arrival_date",
    "unit_cost_effective", "cost_source", "cost_is_imputed", "cost_quality_flag",
    "cost_currency", "cost_basis", "recommended_purchase_value",
    "confidence_label", "assumption_flags", "reason_trace",
    "decision_policy_version", "generated_at",
]

# Columns that must be present and non-null in EVERY row (keys + decision spine).
REORDER_REQUIRED_COLUMNS = [
    "run_id", "as_of_date", "sku", "channel", "action", "priority_rank",
    "reorder_triggered", "approval_required", "human_follow_up_required", "order_placed",
    "manual_review_required", "overall_risk_tier", "target_cover_days",
    "available_forecast_horizon_days", "recommended_order_quantity",
    "reason_trace", "decision_policy_version", "generated_at",
]

REORDER_UNIQUE_KEY = ["sku", "channel"]

# Risk tiers accepted for display/support (engine emits RISK_TIERS; watch/healthy are display
# synonyms tolerated gracefully). Unknown is a real, distinct tier (never "safe").
_SUPPORTED_RISK_TIERS = set(RISK_TIERS) | {"watch", "healthy"}

REORDER_SUMMARY_REQUIRED_KEYS = [
    "run_id", "as_of_date", "selected_sku_count", "selected_series_count",
    "count_by_action", "count_by_risk_tier",
    "order_now_count", "monitor_count", "no_order_count", "vendor_follow_up_count",
    "manual_review_count",
    "total_proposed_order_units", "total_proposed_purchase_value", "approval_required_count",
    "synthetic_stock_count", "assumed_lead_time_count", "assumed_moq_count",
    "assumed_pack_size_count", "imputed_cost_count", "insufficient_horizon_count",
    "missing_cost_count",
    "operational_model", "operational_horizon", "target_cover_days",
    "decision_policy_version", "generated_at",
]


def _is_integer_like(x) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(v):
        return False
    return abs(v - round(v)) < 1e-9


def _nn(v) -> bool:
    """True when a scalar is genuinely present (not None / NaN / NaT)."""
    if v is None:
        return False
    try:
        return not pd.isna(v)
    except (TypeError, ValueError):
        return True


def validate_reorder_recommendations(df: pd.DataFrame, selected_keys: pd.DataFrame,
                                     run_id: str) -> pd.DataFrame:
    """Validate the per-(sku, channel) reorder table. Returns a deterministically sorted copy.

    Rejects: wrong/extra/missing columns; missing/extra/duplicate selected keys; invalid action;
    negative or non-integer positive quantities; order_now with zero quantity; non-order action
    with a positive actionable quantity; quantity below MOQ or not pack-divisible; wrong purchase
    value; order_placed True; non-null order/arrival date on a non-order action; missing reason
    trace; invalid risk tier; manual_review without a review reason; missing policy version.
    """
    out = df.copy()
    _require_exact_columns(out, REORDER_RECOMMENDATION_COLUMNS, "reorder_recommendations")
    if out.empty:
        raise ValueError("reorder_recommendations is empty (expected one row per selected sku/channel)")
    if out.duplicated(REORDER_UNIQUE_KEY).any():
        raise ValueError("reorder_recommendations has duplicate (sku, channel) rows")

    want = set(map(tuple, selected_keys[REORDER_UNIQUE_KEY].astype(str).itertuples(index=False, name=None)))
    got = set(map(tuple, out[REORDER_UNIQUE_KEY].astype(str).itertuples(index=False, name=None)))
    if want != got:
        raise ValueError(f"reorder_recommendations keys != selected keys "
                         f"(missing={want - got}, extra={got - want})")

    if set(out["run_id"].astype(str).unique()) != {str(run_id)}:
        raise ValueError("reorder_recommendations run_id is not constant/matching")

    for col in REORDER_REQUIRED_COLUMNS:
        if out[col].map(lambda v: not _nn(v)).any():
            raise ValueError(f"reorder_recommendations has null in required column {col!r}")

    bad_action = set(out["action"].astype(str)) - set(VALID_REORDER_ACTIONS)
    if bad_action:
        raise ValueError(f"reorder_recommendations has invalid action(s): {sorted(bad_action)}")

    bad_tier = set(out["overall_risk_tier"].dropna().astype(str)) - _SUPPORTED_RISK_TIERS
    if bad_tier:
        raise ValueError(f"reorder_recommendations overall_risk_tier outside {sorted(_SUPPORTED_RISK_TIERS)}: "
                         f"{sorted(bad_tier)}")

    if out["order_placed"].astype(bool).any():
        raise ValueError("reorder_recommendations has order_placed=True (Phase C never places orders)")

    for _, r in out.iterrows():
        key = (r["sku"], r["channel"])
        action = str(r["action"])
        q = r["recommended_order_quantity"]
        if not _nn(q):
            raise ValueError(f"{key}: recommended_order_quantity is null")
        q = float(q)
        if q < 0:
            raise ValueError(f"{key}: negative recommended_order_quantity {q}")
        if q > 0 and not _is_integer_like(q):
            raise ValueError(f"{key}: positive recommended_order_quantity {q} is not integer-valued")
        if not str(r["reason_trace"]).strip():
            raise ValueError(f"{key}: empty reason_trace")
        if action == "order_now":
            if not (q > 0):
                raise ValueError(f"{key}: order_now must have a positive recommended_order_quantity")
            moq = r["moq"]
            if _nn(moq) and q < float(moq) - 1e-9:
                raise ValueError(f"{key}: order_now quantity {q} below MOQ {moq}")
            pack = r["pack_size"]
            if _nn(pack) and float(pack) > 0 and abs(q - round(q / float(pack)) * float(pack)) > 1e-9:
                raise ValueError(f"{key}: order_now quantity {q} not divisible by pack_size {pack}")
            pv = r["recommended_purchase_value"]
            cost = r["unit_cost_effective"]
            if not _nn(pv):
                raise ValueError(f"{key}: order_now must have a recommended_purchase_value")
            if _nn(cost) and abs(float(pv) - q * float(cost)) > 1e-6 * max(1.0, abs(q * float(cost))):
                raise ValueError(f"{key}: recommended_purchase_value {pv} != qty*cost ({q}*{cost})")
            for dcol in ("recommended_order_date", "expected_arrival_date"):
                if not _nn(r[dcol]):
                    raise ValueError(f"{key}: order_now missing {dcol}")
        else:  # non-order actions
            if q != 0:
                raise ValueError(f"{key}: non-order action {action!r} has actionable quantity {q}")
            for dcol in ("recommended_order_date", "expected_arrival_date"):
                if _nn(r[dcol]):
                    raise ValueError(f"{key}: non-order action {action!r} must have null {dcol}")
            if action == "manual_review" and not str(r["review_reason_codes"]).strip():
                raise ValueError(f"{key}: manual_review without review_reason_codes")
        if not str(r["decision_policy_version"]).strip():
            raise ValueError(f"{key}: missing decision_policy_version")

    return out.sort_values(REORDER_UNIQUE_KEY).reset_index(drop=True)


def validate_reorder_summary(summary: dict, detail: "pd.DataFrame | None" = None,
                             run_id: "str | None" = None) -> dict:
    """Validate reorder_summary.json structure and (when the detail frame is given) reconcile
    its counts and totals to the per-row recommendations. Returns the summary unchanged."""
    if not isinstance(summary, dict):
        raise ValueError("reorder_summary must be a dict")
    missing = [k for k in REORDER_SUMMARY_REQUIRED_KEYS if k not in summary]
    if missing:
        raise ValueError(f"reorder_summary missing keys: {missing}")
    if not str(summary.get("decision_policy_version", "")).strip():
        raise ValueError("reorder_summary missing decision_policy_version")
    if run_id is not None and str(summary.get("run_id")) != str(run_id):
        raise ValueError("reorder_summary run_id mismatch")

    if detail is not None:
        det = detail
        by_action = det["action"].astype(str).value_counts().to_dict()
        for act in VALID_REORDER_ACTIONS:
            key = f"{act}_count"
            if int(summary.get(key, 0)) != int(by_action.get(act, 0)):
                raise ValueError(f"reorder_summary {key}={summary.get(key)} != detail {by_action.get(act, 0)}")
        if int(summary.get("selected_series_count", -1)) != int(len(det)):
            raise ValueError("reorder_summary selected_series_count != detail row count")
        units = float(pd.to_numeric(det["recommended_order_quantity"], errors="coerce").fillna(0).sum())
        if abs(float(summary.get("total_proposed_order_units", 0)) - units) > 1e-6:
            raise ValueError("reorder_summary total_proposed_order_units does not reconcile")
        pv = pd.to_numeric(det["recommended_purchase_value"], errors="coerce")
        pv_total = float(pv.dropna().sum())
        if abs(float(summary.get("total_proposed_purchase_value", 0)) - pv_total) > 1e-3:
            raise ValueError("reorder_summary total_proposed_purchase_value does not reconcile")
    return summary
