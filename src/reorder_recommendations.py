"""reorder_recommendations.py — Phase C: forecast-driven reorder recommendations.

Turns the Phase B stockout-risk view + the run's operational forecast + inventory context
into exactly ONE proposed buyer action per selected (sku, channel):

    order_now | monitor | no_order | vendor_follow_up | manual_review

It is a downstream DECISION layer. It never trains a model, never generates demand, never
recomputes stockout risk (it READS Phase B), and NEVER creates, submits or places a purchase
order. Every recommendation is a planning proposal that requires buyer approval.

Consumes ONLY run-specific artifacts (never global data/processed or global outputs):
  runs/<run_id>/decisions/stockout_risk.parquet
  runs/<run_id>/decisions/stockout_trajectory.parquet
  runs/<run_id>/selected_forecasts.parquet
  runs/<run_id>/processed/inventory_context.parquet
  runs/<run_id>/request.json
  inventory_etl/config/config.yaml            (decisioning pilot policies)

Writes (atomically) under runs/<run_id>/decisions/:
  reorder_recommendations.parquet   one row per selected (sku, channel)
  reorder_summary.json

Public API:
  run_reorder_recommendations(*, stockout_risk, stockout_trajectory, selected_forecasts,
                              inventory_context, request, config, output_dir, run_id) -> dict
  compute_reorder_recommendations(run_dir, *, operational_model=None,
                                  operational_horizon=None, config_path=None, logger=None) -> dict

Standalone (path-aware) CLI for isolated testing:
  python src/reorder_recommendations.py --run-dir runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import decision_contract as dc  # noqa: E402

CONFIG_PATH = REPO_ROOT / "inventory_etl" / "config" / "config.yaml"
DECISION_POLICY_VERSION = dc.REORDER_POLICY_VERSION
_ND = NormalDist()

# Buyer-facing queue priority (order_now is most actionable). Distinct from the action-
# ASSIGNMENT precedence used internally below (dropship / review win when deciding WHICH action).
ACTION_PRIORITY = {"order_now": 1, "vendor_follow_up": 2, "manual_review": 3,
                   "monitor": 4, "no_order": 5}
# tiers that keep a product on the buyer's radar (engine emits "medium"; "watch" is its synonym)
MONITOR_TIERS = {"critical", "high", "medium", "watch"}


# ── config ─────────────────────────────────────────────────────────────────────────────
def load_reorder_config(config_path: "str | Path | None" = None) -> dict:
    """Read the Phase C policy from the SINGLE existing ``decisioning`` config block."""
    path = Path(config_path) if config_path else CONFIG_PATH
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    d = (cfg.get("decisioning") or {})
    actions = d.get("action_names") or {}
    approval = d.get("human_approval") or {}
    missing = d.get("missing_critical_input_policy") or {}
    rounding = d.get("rounding_policy") or {}
    return {
        "target_cover_days": int(d.get("target_cover_days", 14)),
        "target_cover_days_source": "config.decisioning.target_cover_days",
        "action_names": {a: str(actions.get(a, a)) for a in dc.VALID_REORDER_ACTIONS},
        "approval_required_for": {
            "order_now": bool(approval.get("required_for_order_now", True)),
            "vendor_follow_up": bool(approval.get("required_for_vendor_follow_up", True)),
            "manual_review": bool(approval.get("required_for_manual_review", True)),
        },
        "missing_input_action": str(missing.get("action", "manual_review")),
        "missing_input_quantity": int(missing.get("actionable_quantity", 0)),
        "apply_moq_before_pack": bool(rounding.get("apply_moq_before_pack_rounding", True)),
        "require_integer_like_moq": bool(rounding.get("require_integer_like_moq", True)),
        "require_integer_like_pack_size": bool(rounding.get("require_integer_like_pack_size", True)),
    }


# ── scalar helpers ───────────────────────────────────────────────────────────────────────
def _present(v) -> bool:
    if v is None:
        return False
    try:
        return not pd.isna(v)
    except (TypeError, ValueError):
        return True


def _num(v):
    if not _present(v):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _finite_pos(v) -> bool:
    f = _num(v)
    return f is not None and f > 0


def _int_like_pos(v) -> bool:
    f = _num(v)
    return f is not None and f > 0 and abs(f - round(f)) < 1e-9


def _iso_date(v):
    if not _present(v):
        return None
    try:
        d = pd.to_datetime(v)
        return None if pd.isna(d) else d.date().isoformat()
    except (TypeError, ValueError):
        return None


def _fmt_date_human(iso: "str | None") -> str:
    if not iso:
        return "an unavailable date"
    try:
        return pd.to_datetime(iso).strftime("%-d %b %Y")
    except (TypeError, ValueError):
        try:
            return pd.to_datetime(iso).strftime("%#d %b %Y")   # Windows strftime
        except (TypeError, ValueError):
            return str(iso)


# ── per-(sku, channel) recommendation ────────────────────────────────────────────────────
def _recommend_row(rk: dict, traj_by_hd: dict, ic: "dict | None", *, category, cfg: dict,
                   run_id: str, generated_at: str) -> dict:
    sku = str(rk["sku"])
    channel = str(rk["channel"])
    target_cover_days = int(cfg["target_cover_days"])

    # ── Phase B fields (read, never recomputed) ──────────────────────────────────────────
    inv_pos = _num(rk.get("inventory_position_for_risk"))
    stock_on_hand = _num(rk.get("stock_on_hand"))
    fd_reorder_point = _num(rk.get("reorder_point"))                 # forecast-driven reorder point
    lead_time_days = rk.get("lead_time_days")
    available_h = rk.get("forecast_horizon_available")
    available_h = int(available_h) if _present(available_h) else None
    service_level = _num(rk.get("service_level"))
    service_level_z = (round(_ND.inv_cdf(service_level), 6)
                       if service_level is not None and 0.0 < service_level < 1.0 else None)
    overall_tier = rk.get("overall_risk_tier")
    tier_l = str(overall_tier).strip().lower() if _present(overall_tier) else "unknown"
    days_of_cover = _num(rk.get("forecast_days_of_cover"))
    phaseb_manual = bool(rk.get("manual_review_required"))

    # ── inventory-context critical inputs (join by sku) ───────────────────────────────────
    def _ic(col, default=None):
        if ic is None:
            return default
        v = ic.get(col)
        return v if _present(v) else default

    product_id = _ic("product_id")
    moq = _num(_ic("moq"))
    moq_source = _ic("moq_source")
    pack_size = _num(_ic("pack_size"))
    pack_size_source = _ic("pack_size_source")
    unit_cost = _num(_ic("unit_cost_effective"))
    cost_source = _ic("cost_source")
    cost_is_imputed = bool(_ic("cost_is_imputed", False))
    cost_quality_flag = _ic("cost_quality_flag")
    cost_currency = _ic("cost_currency")
    cost_basis = _ic("cost_basis")
    is_dropship = bool(_ic("is_dropship", False))

    # ── validity of critical inputs ──────────────────────────────────────────────────────
    lead_valid = _int_like_pos(lead_time_days)
    lead_time_days_i = int(round(float(lead_time_days))) if lead_valid else None
    moq_valid = _int_like_pos(moq) if cfg["require_integer_like_moq"] else _finite_pos(moq)
    pack_valid = _int_like_pos(pack_size) if cfg["require_integer_like_pack_size"] else _finite_pos(pack_size)
    cost_valid = _finite_pos(unit_cost)
    stock_valid = (stock_on_hand is not None and stock_on_hand >= 0)
    pos_valid = (inv_pos is not None and inv_pos >= 0)
    tier_valid = tier_l in (set(dc.RISK_TIERS) | {"watch", "healthy"})

    review_codes: list[str] = []
    if lead_valid and available_h is not None:
        planning_horizon_days = max(lead_time_days_i, target_cover_days)
    else:
        planning_horizon_days = None
    insufficient_horizon = bool(
        planning_horizon_days is not None and available_h is not None
        and planning_horizon_days > available_h)

    if not stock_valid:
        review_codes.append("invalid_stock")
    if not pos_valid:
        review_codes.append("invalid_inventory_position")
    if not lead_valid:
        review_codes.append("invalid_lead_time")
    if not moq_valid:
        review_codes.append("invalid_moq")
    if not pack_valid:
        review_codes.append("invalid_pack_size")
    if not cost_valid:
        review_codes.append("invalid_cost")
    if insufficient_horizon:
        review_codes.append("insufficient_forecast_horizon")
    if phaseb_manual:
        review_codes.append("phase_b_manual_review")
    if not tier_valid:
        review_codes.append("unusable_risk_tier")

    # ── planning-horizon demand + target stock (only when the horizon row exists) ─────────
    planning_demand = planning_sigma = planning_safety = target_stock = raw_target_gap = None
    if (planning_horizon_days is not None and not insufficient_horizon
            and planning_horizon_days in traj_by_hd):
        tr = traj_by_hd[planning_horizon_days]
        planning_demand = _num(tr.get("cumulative_demand_mean"))
        planning_sigma = _num(tr.get("cumulative_sigma"))
        if planning_demand is not None:
            if service_level_z is not None and planning_sigma is not None:
                planning_safety = int(math.ceil(service_level_z * planning_sigma))
            else:
                planning_safety = 0
            target_stock = int(math.ceil(planning_demand + planning_safety))
            if pos_valid:
                raw_target_gap = float(max(0.0, target_stock - inv_pos))

    # ── reorder trigger (Phase B forecast-driven reorder point) ───────────────────────────
    reorder_triggered = bool(fd_reorder_point is not None and pos_valid
                             and inv_pos <= fd_reorder_point)

    # ── provisional (safely calculated) quantity: raw gap -> MOQ -> pack, when computable ──
    provisional = None
    if raw_target_gap is not None and moq_valid and pack_valid:
        adj = max(raw_target_gap, float(moq))
        provisional = int(math.ceil(adj / float(pack_size)) * float(pack_size))

    critical_inputs_valid = (stock_valid and pos_valid and lead_valid and moq_valid
                             and pack_valid and cost_valid and tier_valid
                             and not insufficient_horizon)

    # ── deterministic single-action precedence ───────────────────────────────────────────
    raw_order_quantity = 0.0
    moq_adjusted_quantity = None
    rounded_order_quantity = None
    recommended_qty = 0
    recommended_order_date = None
    expected_arrival_date = None

    if is_dropship:
        action = "vendor_follow_up"
        review_codes.append("dropship_vendor_fulfilled")
    elif review_codes or phaseb_manual or not critical_inputs_valid:
        action = "manual_review"
    elif reorder_triggered and raw_target_gap is not None and raw_target_gap > 0 and provisional and provisional > 0:
        action = "order_now"
        raw_order_quantity = float(raw_target_gap)
        moq_adjusted_quantity = float(max(raw_target_gap, float(moq)))
        rounded_order_quantity = int(provisional)
        recommended_qty = int(provisional)
        recommended_order_date = str(rk.get("as_of_date"))
        if lead_time_days_i is not None:
            expected_arrival_date = _iso_date(
                pd.to_datetime(rk.get("as_of_date")) + pd.Timedelta(days=lead_time_days_i))
    elif (days_of_cover is not None and days_of_cover < target_cover_days) or tier_l in MONITOR_TIERS:
        action = "monitor"
    else:
        action = "no_order"

    # ── approval / follow-up flags ────────────────────────────────────────────────────────
    approval_map = cfg["approval_required_for"]
    approval_required = bool(approval_map.get(action, False)) if action in approval_map else False
    if action == "order_now":
        approval_required = bool(approval_map.get("order_now", True))
    human_follow_up_required = bool(action == "vendor_follow_up")
    manual_review_required = bool(action == "manual_review")

    # ── cost / purchase value (only an actionable positive order has a value) ─────────────
    recommended_purchase_value = None
    if action == "order_now" and cost_valid and recommended_qty > 0:
        recommended_purchase_value = float(recommended_qty) * float(unit_cost)

    # ── assumption flags (carry Phase B + add Phase C) ────────────────────────────────────
    flags: list[str] = []
    for f in str(rk.get("assumption_flags") or "").split(";"):
        f = f.strip()
        if f:
            flags.append(f)
    if not cost_valid:
        flags.append("missing_cost")
    if cost_is_imputed:
        flags.append("imputed_cost")
    if moq_source is not None and any(t in str(moq_source).lower() for t in ("assumed", "default")):
        flags.append("assumed_moq")
    if pack_size_source is not None and any(t in str(pack_size_source).lower() for t in ("assumed", "default")):
        flags.append("assumed_pack_size")
    if insufficient_horizon:
        flags.append("insufficient_horizon")
    if is_dropship:
        flags.append("dropship")
    flags = list(dict.fromkeys(flags))                     # dedup, preserve order

    # ── deterministic reason trace + human review reason ──────────────────────────────────
    reason_trace, manual_review_reason = _build_reason(
        action=action, rk=rk, sku=sku, tier=tier_l, inv_pos=inv_pos,
        fd_reorder_point=fd_reorder_point, reorder_triggered=reorder_triggered,
        lead_time_days=lead_time_days_i, target_cover_days=target_cover_days,
        planning_horizon_days=planning_horizon_days, available_h=available_h,
        planning_demand=planning_demand, planning_safety=planning_safety,
        target_stock=target_stock, raw_target_gap=raw_target_gap,
        moq=moq, pack_size=pack_size, moq_adjusted_quantity=moq_adjusted_quantity,
        recommended_qty=recommended_qty, expected_arrival_date=expected_arrival_date,
        provisional=provisional, review_codes=review_codes, unit_cost=unit_cost,
        cost_is_imputed=cost_is_imputed, insufficient_horizon=insufficient_horizon,
        days_of_cover=days_of_cover, flags=flags)

    return {
        "run_id": run_id, "as_of_date": str(rk.get("as_of_date")), "sku": sku,
        "product_id": (int(product_id) if _int_like_pos(product_id) or (product_id == 0) else product_id),
        "sku_name": rk.get("sku_name"), "category": category, "brand": None, "channel": channel,
        "operational_model": rk.get("operational_model"),
        "operational_horizon": (int(rk["operational_horizon"]) if _present(rk.get("operational_horizon")) else None),
        "action": action, "priority_rank": ACTION_PRIORITY[action],
        "reorder_triggered": bool(reorder_triggered), "approval_required": bool(approval_required),
        "human_follow_up_required": bool(human_follow_up_required), "order_placed": False,
        "manual_review_required": bool(manual_review_required),
        "manual_review_reason": manual_review_reason,
        "review_reason_codes": ";".join(review_codes),
        "overall_risk_tier": (tier_l if tier_valid else "unknown"),
        "stockout_probability": _num(rk.get("stockout_probability")),
        "days_of_cover": days_of_cover, "projected_stockout_date": _iso_date(rk.get("projected_stockout_date")),
        "stock_on_hand": stock_on_hand, "stock_on_hand_is_synthetic": bool(rk.get("stock_on_hand_is_synthetic")),
        "reported_on_order_quantity": _num(rk.get("reported_on_order_quantity")),
        "usable_on_order_quantity": _num(rk.get("usable_on_order_quantity")),
        "inventory_position_for_risk": inv_pos,
        "lead_time_days": lead_time_days_i, "lead_time_source": rk.get("lead_time_source"),
        "lead_time_demand_mean": _num(rk.get("lead_time_demand_p50")),
        "lead_time_safety_stock": _num(rk.get("safety_stock")),
        "forecast_driven_reorder_point": fd_reorder_point,
        "target_cover_days": target_cover_days, "planning_horizon_days": planning_horizon_days,
        "available_forecast_horizon_days": available_h, "insufficient_horizon": bool(insufficient_horizon),
        "planning_horizon_demand_mean": planning_demand, "planning_horizon_sigma": planning_sigma,
        "service_level_target": service_level, "service_level_z": service_level_z,
        "planning_safety_stock": planning_safety, "target_stock": target_stock,
        "raw_target_gap": raw_target_gap, "raw_order_quantity": raw_order_quantity,
        "moq": moq, "moq_source": moq_source, "moq_adjusted_quantity": moq_adjusted_quantity,
        "pack_size": pack_size, "pack_size_source": pack_size_source,
        "rounded_order_quantity": rounded_order_quantity,
        "provisional_calculated_quantity": provisional, "recommended_order_quantity": int(recommended_qty),
        "recommended_order_date": recommended_order_date, "expected_arrival_date": expected_arrival_date,
        "unit_cost_effective": unit_cost, "cost_source": cost_source,
        "cost_is_imputed": bool(cost_is_imputed), "cost_quality_flag": cost_quality_flag,
        "cost_currency": cost_currency, "cost_basis": cost_basis,
        "recommended_purchase_value": recommended_purchase_value,
        "confidence_label": (rk.get("confidence_label") if _present(rk.get("confidence_label")) else "unknown"),
        "assumption_flags": ";".join(flags), "reason_trace": reason_trace,
        "decision_policy_version": DECISION_POLICY_VERSION, "generated_at": generated_at,
    }


def _build_reason(*, action, rk, sku, tier, inv_pos, fd_reorder_point, reorder_triggered,
                  lead_time_days, target_cover_days, planning_horizon_days, available_h,
                  planning_demand, planning_safety, target_stock, raw_target_gap, moq, pack_size,
                  moq_adjusted_quantity, recommended_qty, expected_arrival_date, provisional,
                  review_codes, unit_cost, cost_is_imputed, insufficient_horizon, days_of_cover,
                  flags) -> tuple[str, "str | None"]:
    """Deterministic, LLM-free explanation text generated from structured fields only."""
    model = rk.get("operational_model")
    ltd = rk.get("lead_time_demand_p50")
    ltd_s = f"{float(ltd):.0f}" if _present(ltd) else "—"
    pos_s = f"{inv_pos:.0f}" if inv_pos is not None else "—"
    rop_s = f"{fd_reorder_point:.0f}" if fd_reorder_point is not None else "—"
    assume_bits = []
    if "synthetic_stock" in flags:
        assume_bits.append("stock is synthetically reconstructed")
    if "assumed_lead_time" in flags:
        assume_bits.append("lead time uses a pilot assumption")
    if "assumed_moq" in flags:
        assume_bits.append("MOQ is a default assumption")
    if "assumed_pack_size" in flags:
        assume_bits.append("pack size is a default assumption")
    if cost_is_imputed:
        assume_bits.append("unit cost is imputed (estimated)")
    assume = (" Inputs: " + "; ".join(assume_bits) + "." ) if assume_bits else ""

    if action == "order_now":
        gap_s = f"{raw_target_gap:.0f}" if raw_target_gap is not None else "—"
        adj_s = f"{moq_adjusted_quantity:.0f}" if _present(moq_adjusted_quantity) else "—"
        txt = (f"Propose ordering {recommended_qty} units. The selected {model} forecast estimates "
               f"{ltd_s} units during the {lead_time_days}-day lead time. Usable inventory is {pos_s} "
               f"units against a forecast-driven reorder point of {rop_s} units. The "
               f"{target_cover_days}-day target stock is {target_stock} units, giving a raw gap of "
               f"{gap_s} units. MOQ is {moq:.0f} units and pack size is {pack_size:.0f}, so after MOQ "
               f"adjustment ({adj_s}) and pack rounding the final proposed quantity is {recommended_qty} "
               f"units. Expected arrival is {_fmt_date_human(expected_arrival_date)}. This is a planning "
               f"proposal and requires buyer approval — no purchase order has been created or sent."
               + assume)
        return txt, None
    if action == "vendor_follow_up":
        txt = (f"No warehouse replenishment: {sku} is a dropship / vendor-fulfilled product, so "
               f"replenishment is a vendor follow-up rather than a warehouse order. Risk tier is "
               f"{tier}. This requires buyer/vendor follow-up; no purchase order has been created."
               + assume)
        return txt, None
    if action == "manual_review":
        readable = {
            "insufficient_forecast_horizon": (
                f"the {target_cover_days}-day target-cover policy needs a "
                f"{planning_horizon_days}-day planning horizon but only {available_h} forecast days are "
                f"available, so a target order-up-to quantity cannot be computed without extrapolation"),
            "invalid_lead_time": "lead time is missing or not a valid positive whole number of days",
            "invalid_moq": "MOQ is missing or not a valid positive whole number",
            "invalid_pack_size": "pack size is missing or not a valid positive whole number",
            "invalid_cost": "effective unit cost is missing, zero or negative",
            "invalid_stock": "current stock is missing or invalid",
            "invalid_inventory_position": "inventory position is missing or invalid",
            "phase_b_manual_review": "Phase B stockout risk already flagged this SKU for manual review",
            "unusable_risk_tier": "the Phase B risk tier is missing or outside the known vocabulary",
        }
        causes = [readable.get(c, c) for c in review_codes] or ["a critical input could not be validated"]
        prov_s = (f" A provisional (non-actionable) quantity of {provisional} units was safely "
                  f"calculated for reference only." if provisional is not None else
                  " No provisional quantity could be safely calculated.")
        reason_txt = ("Manual review required: " + "; ".join(causes)
                      + ". No actionable quantity was released (recommended quantity is 0)." + prov_s
                      + " Requires buyer review." + assume)
        return reason_txt, "; ".join(causes)
    if action == "monitor":
        why = []
        if days_of_cover is not None and days_of_cover < target_cover_days:
            why.append(f"days of cover ({days_of_cover:.1f}) is below the {target_cover_days}-day target")
        if tier in MONITOR_TIERS:
            why.append(f"risk tier is {tier}")
        gap_note = (f" A positive target gap of {raw_target_gap:.0f} units exists but the forecast-driven "
                    f"reorder point ({rop_s}) has not been reached, so no order is proposed."
                    if (raw_target_gap is not None and raw_target_gap > 0) else "")
        txt = ("Monitor: " + (" and ".join(why) if why else "close to policy thresholds")
               + f". Inventory position is {pos_s} against a reorder point of {rop_s}." + gap_note
               + " No order is proposed yet." + assume)
        return txt, None
    # no_order
    txt = (f"No order: the reorder trigger has not fired (inventory position {pos_s} is above the "
           f"forecast-driven reorder point {rop_s}) and cover/risk are within policy (tier {tier}). "
           f"No action needed." + assume)
    return txt, None


# ── summary ────────────────────────────────────────────────────────────────────────────────
def _build_summary(det: pd.DataFrame, *, run_id, as_of_date, operational_model,
                   operational_horizon, target_cover_days, generated_at) -> dict:
    def _flag_count(substr):
        return int(det["assumption_flags"].astype(str).str.contains(substr, regex=False).sum())
    by_action = det["action"].astype(str).value_counts().to_dict()
    by_tier = det["overall_risk_tier"].astype(str).value_counts().to_dict()
    pv = pd.to_numeric(det["recommended_purchase_value"], errors="coerce")
    units = pd.to_numeric(det["recommended_order_quantity"], errors="coerce").fillna(0)
    return {
        "run_id": str(run_id), "as_of_date": str(as_of_date),
        "selected_sku_count": int(det["sku"].nunique()),
        "selected_series_count": int(len(det)),
        "count_by_action": {a: int(by_action.get(a, 0)) for a in dc.VALID_REORDER_ACTIONS},
        "count_by_risk_tier": {t: int(v) for t, v in by_tier.items()},
        "order_now_count": int(by_action.get("order_now", 0)),
        "monitor_count": int(by_action.get("monitor", 0)),
        "no_order_count": int(by_action.get("no_order", 0)),
        "vendor_follow_up_count": int(by_action.get("vendor_follow_up", 0)),
        "manual_review_count": int(by_action.get("manual_review", 0)),
        "total_proposed_order_units": float(units.sum()),
        "total_proposed_purchase_value": float(pv.dropna().sum()),
        # integrity check: every positive proposed order must carry a purchase value (expected 0)
        "purchase_value_unavailable_count": int(((units > 0) & pv.isna()).sum()),
        "approval_required_count": int(det["approval_required"].astype(bool).sum()),
        "synthetic_stock_count": int(det["stock_on_hand_is_synthetic"].astype(bool).sum()),
        "assumed_lead_time_count": _flag_count("assumed_lead_time"),
        "assumed_moq_count": _flag_count("assumed_moq"),
        "assumed_pack_size_count": _flag_count("assumed_pack_size"),
        "imputed_cost_count": int(det["cost_is_imputed"].astype(bool).sum()),
        "insufficient_horizon_count": int(det["insufficient_horizon"].astype(bool).sum()),
        "missing_cost_count": _flag_count("missing_cost"),
        "operational_model": operational_model,
        "operational_horizon": (int(operational_horizon) if operational_horizon is not None else None),
        "target_cover_days": int(target_cover_days),
        "decision_policy_version": DECISION_POLICY_VERSION,
        "generated_at": generated_at,
    }


# ── public API (pure): build + validate + write ──────────────────────────────────────────
def run_reorder_recommendations(*, stockout_risk: pd.DataFrame, stockout_trajectory: pd.DataFrame,
                                selected_forecasts: pd.DataFrame, inventory_context: pd.DataFrame,
                                request: dict, config: dict, output_dir: "str | Path",
                                run_id: str, logger=None) -> dict:
    """Compute Phase C recommendations from in-memory inputs, validate against the decision
    contract, and write ``reorder_recommendations.parquet`` + ``reorder_summary.json`` atomically
    under ``output_dir``. Returns a summary dict (also written to disk)."""
    output_dir = Path(output_dir)
    generated_at = datetime.now(timezone.utc).isoformat()
    category = request.get("category")
    as_of_date = request.get("as_of_date")

    # operational metadata is carried on every Phase B risk row
    op_model = (str(stockout_risk["operational_model"].dropna().iloc[0])
                if "operational_model" in stockout_risk.columns and stockout_risk["operational_model"].notna().any()
                else None)
    op_h = (int(pd.to_numeric(stockout_risk["operational_horizon"], errors="coerce").dropna().iloc[0])
            if "operational_horizon" in stockout_risk.columns
            and stockout_risk["operational_horizon"].notna().any() else None)

    # inventory context is per-sku (no channel); index by sku for O(1) join
    ic_by_sku: dict[str, dict] = {}
    if inventory_context is not None and "sku" in inventory_context.columns:
        for _, r in inventory_context.iterrows():
            ic_by_sku[str(r["sku"])] = r.to_dict()

    # trajectory rows grouped by (sku, channel) then keyed by forecast_horizon_day
    traj_by_key: dict[tuple, dict] = {}
    if stockout_trajectory is not None and not stockout_trajectory.empty:
        for (s, c), g in stockout_trajectory.groupby(["sku", "channel"], sort=False):
            traj_by_key[(str(s), str(c))] = {
                int(hd): row
                for hd, row in zip(pd.to_numeric(g["forecast_horizon_day"], errors="coerce"),
                                   g.to_dict("records"))
                if pd.notna(hd)}

    rows = []
    for rk in stockout_risk.to_dict("records"):
        key = (str(rk["sku"]), str(rk["channel"]))
        rows.append(_recommend_row(rk, traj_by_key.get(key, {}), ic_by_sku.get(str(rk["sku"])),
                                   category=category, cfg=config, run_id=str(run_id),
                                   generated_at=generated_at))

    det = pd.DataFrame(rows, columns=dc.REORDER_RECOMMENDATION_COLUMNS)
    selected_keys = stockout_risk[["sku", "channel"]].drop_duplicates()
    det = dc.validate_reorder_recommendations(det, selected_keys, str(run_id))

    summary = _build_summary(det, run_id=run_id, as_of_date=as_of_date, operational_model=op_model,
                             operational_horizon=op_h, target_cover_days=config["target_cover_days"],
                             generated_at=generated_at)
    dc.validate_reorder_summary(summary, det, str(run_id))

    dec_dir = output_dir
    dc.write_dataframe_atomic(det, dec_dir / "reorder_recommendations.parquet", "parquet")
    dc.write_json_atomic(summary, dec_dir / "reorder_summary.json")

    result = dict(summary)
    result["reorder_recommendations_file"] = "decisions/reorder_recommendations.parquet"
    result["reorder_summary_file"] = "decisions/reorder_summary.json"
    result["reorder_rows"] = int(len(det))
    if logger is not None:
        logger.info("reorder recommendations: %d rows, actions=%s, units=%s, value=%s",
                    len(det), summary["count_by_action"], summary["total_proposed_order_units"],
                    summary["total_proposed_purchase_value"])
    return result


# ── run-dir wrapper (used by the orchestrator; reads the run's files) ─────────────────────
def compute_reorder_recommendations(run_dir: "str | Path", *, operational_model: str | None = None,
                                    operational_horizon: int | None = None,
                                    config_path: "str | Path | None" = None, logger=None) -> dict:
    """Load a run's Phase B + input artifacts from disk and produce Phase C outputs under
    ``runs/<run_id>/decisions/``. Callable during the pipeline (before run_manifest.json)."""
    run_dir = Path(run_dir)
    cfg = load_reorder_config(config_path)
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    run_id = str(request["run_id"])
    dec = run_dir / "decisions"
    stockout_risk = pd.read_parquet(dec / "stockout_risk.parquet")
    stockout_trajectory = pd.read_parquet(dec / "stockout_trajectory.parquet")
    selected_forecasts = pd.read_parquet(run_dir / "selected_forecasts.parquet")
    inventory_context = pd.read_parquet(run_dir / "processed" / "inventory_context.parquet")
    return run_reorder_recommendations(
        stockout_risk=stockout_risk, stockout_trajectory=stockout_trajectory,
        selected_forecasts=selected_forecasts, inventory_context=inventory_context,
        request=request, config=cfg, output_dir=dec, run_id=run_id, logger=logger)


# ── CLI (path-aware; never touches global outputs) ───────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Phase C forecast-driven reorder recommendations.")
    ap.add_argument("--run-dir", default=None, help="path to runs/<run_id>")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--config", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.run_id:
        run_dir = Path(args.runs_dir) / args.run_id
    else:
        print("provide --run-dir or --run-id", file=sys.stderr)
        return 2
    if not (run_dir / "decisions" / "stockout_risk.parquet").exists():
        print(f"run has no Phase B stockout_risk.parquet: {run_dir}", file=sys.stderr)
        return 2
    summary = compute_reorder_recommendations(run_dir, config_path=args.config)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
