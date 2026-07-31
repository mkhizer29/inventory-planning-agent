"""Phase B — forecast-driven stockout-risk decision tests.

Exercises src/stockout_risk.py directly on tiny, self-contained fake runs (no warehouse,
no model training). One focused file. Orchestrator-integration and dashboard-exposure of
Phase B are covered in test_forecast_orchestrator.py / test_dashboard_runs.py.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import stockout_risk as sr        # noqa: E402
import reorder_recommendations as rr  # noqa: E402  (Phase C)
import decision_contract as dc    # noqa: E402

AS_OF = "2026-06-30"
CHANNEL = "naheed_web"


def _make_run(tmp: Path, run_id="rid", *, skus=("A", "B", "C"), horizon_days=14, y_pred=5.0,
              with_intervals=False, interval_half=8.0, stock=100.0, on_order=0.0,
              lead_time=7, lead_time_source="sku_master_picking_mode", synthetic=False,
              price=100.0, model="holtwinters", op_horizon=14, backtest=True, backtest_obs=14,
              stock_by_sku=None, moq=1, pack_size=1, unit_cost=50.0,
              moq_source="sku_master_moq", pack_size_source="sku_master_pack",
              cost_source="magento_eav", cost_is_imputed=False, cost_quality_flag="OK",
              cost_currency="PKR", cost_basis="unit", dropship_skus=None,
              moq_by_sku=None, pack_by_sku=None, unit_cost_by_sku=None) -> Path:
    rd = tmp / run_id
    (rd / "processed").mkdir(parents=True)
    (rd / "outputs").mkdir(parents=True)
    as_of = pd.Timestamp(AS_OF)
    (rd / "request.json").write_text(json.dumps({
        "run_id": run_id, "category": "Cat", "top_n": len(skus), "as_of_date": AS_OF,
        "selection_cutoff": AS_OF, "horizons": [7, 14]}), encoding="utf-8")

    future = pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    hist = pd.date_range(as_of - pd.Timedelta(days=119), as_of, freq="D")

    sel_rows, ff_rows, inv_rows, mp_rows, bt_rows = [], [], [], [], []
    for s in skus:
        st = float(stock_by_sku[s]) if stock_by_sku and s in stock_by_sku else float(stock)
        for hd, d in enumerate(future, start=1):
            r = {"sku": s, "channel": CHANNEL, "date": d, "y_pred": float(y_pred),
                 "model": model, "model_version": "v1", "as_of_date": AS_OF,
                 "forecast_horizon_day": hd, "selection_horizon": op_horizon,
                 "selection_rank": 1, "selection_reason": "test", "sku_name": f"Prod {s}"}
            if with_intervals:
                r.update(lower_80=max(0.0, y_pred - interval_half), upper_80=y_pred + interval_half,
                         lower_95=max(0.0, y_pred - 1.6 * interval_half), upper_95=y_pred + 1.6 * interval_half)
            sel_rows.append(r)
            ff_rows.append({"sku": s, "channel": CHANNEL, "date": d, "forecast_horizon_day": hd})
        inv_rows.append({"sku": s, "product_id": 1, "location_id": "ALL", "stock_on_hand": st,
                         "stock_on_hand_is_synthetic": bool(synthetic),
                         "stock_source": ("synthetic_reconstruction" if synthetic else "real_snapshot"),
                         "on_order_quantity": float(on_order), "on_order_is_available": False,
                         "lead_time_days": int(lead_time), "lead_time_source": lead_time_source,
                         "price": (None if price is None else float(price)),
                         # Phase C inventory-context inputs (join by sku)
                         "moq": (moq_by_sku or {}).get(s, moq), "moq_source": moq_source,
                         "pack_size": (pack_by_sku or {}).get(s, pack_size), "pack_size_source": pack_size_source,
                         "unit_cost_effective": (unit_cost_by_sku or {}).get(s, unit_cost),
                         "cost_source": cost_source, "cost_is_imputed": bool(cost_is_imputed),
                         "cost_quality_flag": cost_quality_flag, "cost_currency": cost_currency,
                         "cost_basis": cost_basis, "is_dropship": (s in (dropship_skus or set())),
                         # historical pilot placeholders that Phase C MUST ignore:
                         "reorder_point": 999999, "target_stock": 999999,
                         "recommended_order_quantity": 999999})
        for i, d in enumerate(hist):
            mp_rows.append({"sku": s, "channel": CHANNEL, "date": d, "sku_name": f"Prod {s}",
                            "units_observed": float(5 + (i % 5))})     # varies 5..9 -> positive std
        if backtest:
            hdates = list(hist[-int(backtest_obs):])
            for d in hdates:
                bt_rows.append({"sku": s, "channel": CHANNEL, "date": d, "y_pred": 5.0,
                                "model": model, "horizon": op_horizon, "origin": AS_OF,
                                "evaluation_type": "locked_holdout"})

    pd.DataFrame(sel_rows).to_parquet(rd / "selected_forecasts.parquet", index=False)
    pd.DataFrame(ff_rows).to_parquet(rd / "processed" / "forecast_frame.parquet", index=False)
    pd.DataFrame(inv_rows).to_parquet(rd / "processed" / "inventory_context.parquet", index=False)
    pd.DataFrame(mp_rows).to_parquet(rd / "processed" / "model_panel.parquet", index=False)
    if backtest:
        fname = ("baseline_backtest_predictions.parquet" if model in sr.BASELINE_METHODS
                 else f"{model}_backtest_predictions.parquet")
        pd.DataFrame(bt_rows).to_parquet(rd / "outputs" / fname, index=False)
    return rd


def _run(tmp, **kw):
    rd = _make_run(tmp, **kw)
    model = kw.get("model", "holtwinters")
    summary = sr.compute_stockout_risk(rd, operational_model=model,
                                       operational_horizon=kw.get("op_horizon", 14))
    risk = pd.read_parquet(rd / "decisions" / "stockout_risk.parquet")
    traj = pd.read_parquet(rd / "decisions" / "stockout_trajectory.parquet")
    return rd, summary, risk, traj


# 1 + 2 — one risk row per SKU/channel; one trajectory row per forecast key
def test_row_counts(tmp_path):
    rd, _, risk, traj = _run(tmp_path, skus=("A", "B", "C"), horizon_days=14)
    assert len(risk) == 3 and risk.duplicated(["sku", "channel"]).sum() == 0
    assert len(traj) == 3 * 14
    assert list(risk.columns) == dc.STOCKOUT_RISK_COLUMNS
    assert list(traj.columns) == dc.STOCKOUT_TRAJECTORY_COLUMNS


# 3 — probabilities within [0,1]
def test_probabilities_bounded(tmp_path):
    _, _, risk, traj = _run(tmp_path, with_intervals=True)
    p = pd.to_numeric(risk["stockout_probability"], errors="coerce").dropna()
    assert ((p >= 0) & (p <= 1)).all()
    cp = pd.to_numeric(traj["cumulative_stockout_probability"], errors="coerce").dropna()
    assert ((cp >= 0) & (cp <= 1)).all()


# 4 — lower inventory cannot LOWER stockout risk (all else fixed, identical intervals => same sigma)
def test_lower_inventory_higher_risk(tmp_path):
    _, _, risk, _ = _run(tmp_path, skus=("LOW", "HIGH"), with_intervals=True,
                         stock_by_sku={"LOW": 20.0, "HIGH": 500.0})
    p_low = float(risk[risk.sku == "LOW"]["stockout_probability"].iloc[0])
    p_high = float(risk[risk.sku == "HIGH"]["stockout_probability"].iloc[0])
    assert p_low >= p_high


# 5 — no dates after as_of enter residual estimation
def test_residuals_exclude_future(tmp_path):
    rd = _make_run(tmp_path, skus=("A",), model="holtwinters", op_horizon=14)
    mp = pd.read_parquet(rd / "processed" / "model_panel.parquet")
    # inject a backtest row AND a model_panel truth row dated AFTER as_of
    future_day = pd.Timestamp(AS_OF) + pd.Timedelta(days=3)
    bt = pd.read_parquet(rd / "outputs" / "holtwinters_backtest_predictions.parquet")
    bt = pd.concat([bt, pd.DataFrame([{"sku": "A", "channel": CHANNEL, "date": future_day,
                                       "y_pred": 5.0, "model": "holtwinters", "horizon": 14,
                                       "origin": AS_OF, "evaluation_type": "locked_holdout"}])])
    bt.to_parquet(rd / "outputs" / "holtwinters_backtest_predictions.parquet", index=False)
    mp = pd.concat([mp, pd.DataFrame([{"sku": "A", "channel": CHANNEL, "date": future_day,
                                       "sku_name": "Prod A", "units_observed": 999.0}])])
    resid = sr._operational_backtest_residuals(rd, "holtwinters", 14, mp, pd.Timestamp(AS_OF))
    assert not (resid["residual"].abs() > 100).any()      # the 999 truth must be excluded


# 6 — decision artifacts never contain truth columns
def test_no_truth_in_outputs(tmp_path):
    _, _, risk, traj = _run(tmp_path)
    for df in (risk, traj):
        assert "y_true" not in df.columns and "units_observed" not in df.columns


# 7 — per-SKU residual method when enough obs; pooled fallback when too few
def test_residual_per_sku_and_pooled(tmp_path):
    _, _, risk_ok, _ = _run(tmp_path / "a", model="holtwinters", with_intervals=False, backtest_obs=14)
    assert risk_ok["uncertainty_method"].str.startswith("backtest_residual_per_sku").all()
    _, _, risk_few, _ = _run(tmp_path / "b", model="holtwinters", with_intervals=False, backtest_obs=5)
    assert risk_few["uncertainty_method"].str.startswith("backtest_residual_pooled").all()
    assert risk_few["assumption_flags"].str.contains("used_pooled").all()


# 8 — interval-based uncertainty is preferred when intervals are present
def test_interval_uncertainty(tmp_path):
    _, _, risk, _ = _run(tmp_path, with_intervals=True)
    assert (risk["uncertainty_method"] == "forecast_intervals").all()


# 9 — zero-sigma behaviour (collapsed intervals): prob is 1 iff demand exceeds inventory
def test_zero_sigma(tmp_path):
    _, _, risk, _ = _run(tmp_path / "hi", with_intervals=True, interval_half=0.0,
                         y_pred=5.0, stock=1000.0)          # 70u demand < 1000 stock
    r = risk.iloc[0]
    assert float(r["lead_time_sigma"]) == 0.0 and float(r["stockout_probability"]) == 0.0
    assert "zero_sigma" in r["assumption_flags"]
    _, _, risk2, _ = _run(tmp_path / "lo", with_intervals=True, interval_half=0.0,
                          y_pred=5.0, stock=2.0)            # 35u (7d) demand > 2 stock
    assert float(risk2.iloc[0]["stockout_probability"]) == 1.0


# 10 — zero-demand behaviour: null days of cover, survives horizon, flagged
def test_zero_demand(tmp_path):
    _, _, risk, _ = _run(tmp_path, y_pred=0.0, with_intervals=True, interval_half=0.0)
    r = risk.iloc[0]
    assert pd.isna(r["forecast_days_of_cover"]) and bool(r["survives_forecast_horizon"])
    assert "zero_demand" in r["assumption_flags"]


# 11 — insufficient forecast horizon: nulls + manual review + unknown tier
def test_insufficient_horizon(tmp_path):
    _, _, risk, _ = _run(tmp_path, horizon_days=14, lead_time=20)
    r = risk.iloc[0]
    assert not bool(r["lead_time_horizon_sufficient"])
    assert pd.isna(r["stockout_probability"]) and pd.isna(r["safety_stock"]) and pd.isna(r["reorder_point"])
    assert bool(r["manual_review_required"]) and r["overall_risk_tier"] == "unknown"
    assert "insufficient_forecast_horizon" in r["assumption_flags"]


# 12 — synthetic stock and assumed lead time are flagged
def test_assumption_flags(tmp_path):
    _, _, risk, _ = _run(tmp_path, synthetic=True, lead_time_source="assumed_default")
    fl = risk.iloc[0]["assumption_flags"]
    assert "synthetic_stock" in fl and "assumed_lead_time" in fl


# 13 — undated on-order units are NOT treated as immediately available
def test_on_order_not_available(tmp_path):
    _, _, risk, _ = _run(tmp_path, stock=50.0, on_order=200.0)
    r = risk.iloc[0]
    assert float(r["usable_on_order_quantity"]) == 0.0 and not bool(r["on_order_available"])
    assert float(r["inventory_position_for_risk"]) == 50.0
    assert float(r["reported_on_order_quantity"]) == 200.0
    assert "on_order_excluded_no_arrival_date" in r["assumption_flags"]


# 14 — two runs stay isolated
def test_two_runs_isolated(tmp_path):
    rd1, _, risk1, _ = _run(tmp_path, run_id="run1", skus=("A", "B"))
    rd2, _, risk2, _ = _run(tmp_path, run_id="run2", skus=("X", "Y", "Z"))
    assert set(risk1["run_id"]) == {"run1"} and set(risk2["run_id"]) == {"run2"}
    assert set(risk1["sku"]) == {"A", "B"} and set(risk2["sku"]) == {"X", "Y", "Z"}
    assert (rd1 / "decisions" / "stockout_risk.parquet").exists()
    assert (rd2 / "decisions" / "stockout_risk.parquet").exists()


# 15 — historical-demand fallback when no backtest file exists
def test_historical_std_fallback(tmp_path):
    _, _, risk, _ = _run(tmp_path, with_intervals=False, backtest=False)
    assert (risk["uncertainty_method"] == "historical_demand_std").all()
    assert risk["assumption_flags"].str.contains("backtest_residuals_unavailable").all()


# 16 — baseline operational winner resolves its backtest method + CLI metadata resolution
def test_baseline_winner_and_cli_resolve(tmp_path):
    rd = _make_run(tmp_path, skus=("A",), model="moving_average_7", op_horizon=7,
                   with_intervals=False, backtest_obs=7)
    model, horizon = sr.resolve_operational_metadata(rd)
    assert model == "moving_average_7" and horizon == 7
    summary = sr.compute_stockout_risk(rd, operational_model=model, operational_horizon=horizon)
    assert summary["risk_rows"] == 1
    risk = pd.read_parquet(rd / "decisions" / "stockout_risk.parquet")
    assert risk["uncertainty_method"].iloc[0].startswith("backtest_residual")


# ══════════════════════════════════════════════════════════════════════════════════════════
# Phase C — forecast-driven reorder recommendation tests
# ══════════════════════════════════════════════════════════════════════════════════════════
def _reco(tmp: Path, **kw):
    """Build a fake run, run Phase B then Phase C, and return (rd, reco_df, summary, risk_df)."""
    rd = _make_run(tmp, **kw)
    model = kw.get("model", "holtwinters")
    oph = kw.get("op_horizon", 14)
    sr.compute_stockout_risk(rd, operational_model=model, operational_horizon=oph)
    rr.compute_reorder_recommendations(rd, operational_model=model, operational_horizon=oph)
    reco = pd.read_parquet(rd / "decisions" / "reorder_recommendations.parquet")
    summary = json.loads((rd / "decisions" / "reorder_summary.json").read_text(encoding="utf-8"))
    risk = pd.read_parquet(rd / "decisions" / "stockout_risk.parquet")
    return rd, reco, summary, risk


_ORDER_KW = dict(skus=("A",), stock=0.0, moq=24, pack_size=12, unit_cost=50.0,
                 lead_time=7, horizon_days=14, op_horizon=14, with_intervals=True, interval_half=8.0)


# C1 + C3 + C4 — exactly one row per selected (sku, channel); all keys kept; no extras
def test_c_one_row_per_key(tmp_path):
    _, reco, _, risk = _reco(tmp_path, skus=("A", "B", "C"))
    assert len(reco) == 3 and reco.duplicated(["sku", "channel"]).sum() == 0
    assert set(reco["sku"]) == {"A", "B", "C"} == set(risk["sku"])
    assert list(reco.columns) == dc.REORDER_RECOMMENDATION_COLUMNS


# C2 — exactly one valid action per key
def test_c_one_valid_action(tmp_path):
    _, reco, _, _ = _reco(tmp_path, skus=("A", "B", "C"))
    assert set(reco["action"]) <= set(dc.VALID_REORDER_ACTIONS)
    assert (reco.groupby(["sku", "channel"]).size() == 1).all()


# C5 + C6 — quantities non-negative; positive quantities are integer-valued
def test_c_quantities_nonneg_integer(tmp_path):
    _, reco, _, _ = _reco(tmp_path, **_ORDER_KW)
    q = pd.to_numeric(reco["recommended_order_quantity"])
    assert (q >= 0).all()
    pos = q[q > 0]
    assert (pos == pos.round()).all()


# C7 — the reorder trigger uses the Phase B forecast-driven reorder point
def test_c_trigger_uses_phaseb_reorder_point(tmp_path):
    _, reco, _, risk = _reco(tmp_path, skus=("A",), stock=50.0, with_intervals=True)
    rp = float(risk.iloc[0]["reorder_point"])
    r = reco.iloc[0]
    assert float(r["forecast_driven_reorder_point"]) == rp          # sourced from Phase B, not recomputed
    assert bool(r["reorder_triggered"]) == (float(r["inventory_position_for_risk"]) <= rp)


# C8 — historical inventory-context reorder fields are IGNORED (Phase C never reads them)
def test_c_ignores_inventory_context_reorder_fields(tmp_path):
    # _make_run stamps inventory_context.reorder_point/target_stock/recommended_order_quantity = 999999
    _, reco, _, _ = _reco(tmp_path, **_ORDER_KW)
    r = reco.iloc[0]
    assert float(r["forecast_driven_reorder_point"]) != 999999
    assert float(r["target_stock"]) != 999999
    assert int(r["recommended_order_quantity"]) != 999999


# C9 — quantity construction order: raw gap -> MOQ adjust -> pack rounding -> final
def test_c_raw_moq_pack_order(tmp_path):
    _, reco, _, _ = _reco(tmp_path, **_ORDER_KW)
    r = reco[reco["action"] == "order_now"].iloc[0]
    raw, moq, pack = float(r["raw_target_gap"]), float(r["moq"]), float(r["pack_size"])
    assert float(r["moq_adjusted_quantity"]) == max(raw, moq)
    assert float(r["rounded_order_quantity"]) == math.ceil(max(raw, moq) / pack) * pack
    assert float(r["recommended_order_quantity"]) == float(r["rounded_order_quantity"])


# C10 + C11 — a positive recommendation meets MOQ and is divisible by pack size
def test_c_order_now_meets_moq_and_pack(tmp_path):
    _, reco, _, _ = _reco(tmp_path, **_ORDER_KW)
    r = reco[reco["action"] == "order_now"].iloc[0]
    q = float(r["recommended_order_quantity"])
    assert q > 0 and q >= float(r["moq"]) and q % float(r["pack_size"]) == 0


# C12 — reorder trigger false produces zero actionable quantity
def test_c_no_trigger_zero_quantity(tmp_path):
    _, reco, _, _ = _reco(tmp_path, skus=("A",), stock=100000.0, with_intervals=True)
    r = reco.iloc[0]
    assert not bool(r["reorder_triggered"]) and int(r["recommended_order_quantity"]) == 0
    assert r["action"] in ("no_order", "monitor")


# C13 — dropship produces vendor_follow_up with zero warehouse quantity
def test_c_dropship_vendor_follow_up(tmp_path):
    _, reco, _, _ = _reco(tmp_path, skus=("A",), stock=0.0, dropship_skus={"A"}, **{
        k: v for k, v in _ORDER_KW.items() if k not in ("skus", "stock")})
    r = reco.iloc[0]
    assert r["action"] == "vendor_follow_up" and int(r["recommended_order_quantity"]) == 0
    assert bool(r["human_follow_up_required"]) and bool(r["approval_required"])


# C14 — undated inbound (on-order) stock stays unusable and out of the position
def test_c_undated_inbound_unusable(tmp_path):
    _, reco, _, _ = _reco(tmp_path, skus=("A",), stock=50.0, on_order=200.0, with_intervals=True)
    r = reco.iloc[0]
    assert float(r["usable_on_order_quantity"]) == 0.0
    assert float(r["reported_on_order_quantity"]) == 200.0
    assert float(r["inventory_position_for_risk"]) == 50.0


# C15 — insufficient horizon (14-day target on a 7-day forecast) -> manual_review, no quantity
def test_c_insufficient_horizon_manual_review(tmp_path):
    _, reco, _, _ = _reco(tmp_path, skus=("A",), horizon_days=7, op_horizon=7, lead_time=7,
                          stock=0.0, with_intervals=True)
    r = reco.iloc[0]
    assert r["action"] == "manual_review" and bool(r["insufficient_horizon"])
    assert int(r["recommended_order_quantity"]) == 0
    assert "insufficient_forecast_horizon" in r["review_reason_codes"]


# C16 — invalid lead time -> manual_review
def test_c_invalid_lead_time(tmp_path):
    _, reco, _, _ = _reco(tmp_path, skus=("A",), lead_time=0, stock=0.0, with_intervals=True)
    r = reco.iloc[0]
    assert r["action"] == "manual_review" and int(r["recommended_order_quantity"]) == 0


# C17 — invalid MOQ -> manual_review
def test_c_invalid_moq(tmp_path):
    _, reco, _, _ = _reco(tmp_path, skus=("A",), moq=0, stock=0.0, with_intervals=True)
    r = reco.iloc[0]
    assert r["action"] == "manual_review" and "invalid_moq" in r["review_reason_codes"]


# C18 — invalid pack size -> manual_review
def test_c_invalid_pack_size(tmp_path):
    _, reco, _, _ = _reco(tmp_path, skus=("A",), pack_size=0, stock=0.0, with_intervals=True)
    r = reco.iloc[0]
    assert r["action"] == "manual_review" and "invalid_pack_size" in r["review_reason_codes"]


# C19 — invalid cost -> manual_review WITHOUT dropping the row; no purchase value
def test_c_invalid_cost_manual_review(tmp_path):
    _, reco, _, _ = _reco(tmp_path, skus=("A",), unit_cost=0.0, stock=0.0, with_intervals=True)
    assert len(reco) == 1
    r = reco.iloc[0]
    assert r["action"] == "manual_review" and "invalid_cost" in r["review_reason_codes"]
    assert pd.isna(r["recommended_purchase_value"]) and int(r["recommended_order_quantity"]) == 0


# C20 — a positive imputed cost may be used and is labelled as estimated
def test_c_imputed_cost_used_and_labelled(tmp_path):
    _, reco, _, _ = _reco(tmp_path, cost_is_imputed=True, **_ORDER_KW)
    r = reco[reco["action"] == "order_now"].iloc[0]
    assert bool(r["cost_is_imputed"]) and "imputed_cost" in r["assumption_flags"]
    assert pd.notna(r["recommended_purchase_value"]) and float(r["recommended_purchase_value"]) > 0


# C21 — purchase value equals quantity x effective unit cost
def test_c_purchase_value(tmp_path):
    _, reco, _, _ = _reco(tmp_path, **_ORDER_KW)
    for _, r in reco[reco["action"] == "order_now"].iterrows():
        assert abs(float(r["recommended_purchase_value"])
                   - float(r["recommended_order_quantity"]) * float(r["unit_cost_effective"])) < 1e-6


# C22 — order/arrival dates for order_now (arrival = as_of + lead time)
def test_c_order_dates(tmp_path):
    _, reco, _, _ = _reco(tmp_path, **_ORDER_KW)
    r = reco[reco["action"] == "order_now"].iloc[0]
    assert str(r["recommended_order_date"]) == AS_OF
    expected = (pd.Timestamp(AS_OF) + pd.Timedelta(days=7)).date().isoformat()
    assert str(r["expected_arrival_date"]) == expected


# C23 — non-order actions carry null order/arrival dates
def test_c_non_order_null_dates(tmp_path):
    _, reco, _, _ = _reco(tmp_path, skus=("A",), stock=100000.0, with_intervals=True)
    r = reco.iloc[0]
    assert r["action"] != "order_now"
    assert pd.isna(r["recommended_order_date"]) and pd.isna(r["expected_arrival_date"])


# C24 + C29 — order_placed is ALWAYS false; no reason text implies a placed/submitted order
def test_c_order_placed_false_no_external_action(tmp_path):
    _, reco, _, _ = _reco(tmp_path, skus=("A", "B", "C"), stock=0.0, **{
        k: v for k, v in _ORDER_KW.items() if k not in ("skus", "stock")})
    assert not reco["order_placed"].any()
    txt = " ".join(reco["reason_trace"].astype(str)).lower()
    for banned in ("order placed", "purchase order submitted", "supplier notified", "sent to supplier",
                   "inventory reserved"):
        assert banned not in txt


# C25 — order_now reason trace names the important inputs
def test_c_reason_contains_inputs(tmp_path):
    _, reco, _, _ = _reco(tmp_path, model="holtwinters", **{
        k: v for k, v in _ORDER_KW.items() if k != "op_horizon"})
    r = reco[reco["action"] == "order_now"].iloc[0]
    reason = str(r["reason_trace"]).lower()
    for token in ("holtwinters", "reorder point", "target stock", "lead time", "arrival", "approval"):
        assert token in reason


# C26 — manual review carries structured review reason codes
def test_c_manual_review_has_reason_codes(tmp_path):
    _, reco, _, _ = _reco(tmp_path, skus=("A",), unit_cost=0.0, stock=0.0, with_intervals=True)
    r = reco.iloc[0]
    assert r["action"] == "manual_review" and str(r["review_reason_codes"]).strip()
    assert str(r["manual_review_reason"]).strip()


# C27 + C28 — summary counts and totals reconcile to the detail rows
def test_c_summary_reconciles(tmp_path):
    _, reco, summary, _ = _reco(tmp_path, skus=("A", "B", "C"))
    by_action = reco["action"].value_counts().to_dict()
    for act in dc.VALID_REORDER_ACTIONS:
        assert summary[f"{act}_count"] == int(by_action.get(act, 0))
    assert summary["selected_series_count"] == len(reco)
    assert abs(summary["total_proposed_order_units"]
               - float(reco["recommended_order_quantity"].sum())) < 1e-6
    pv = pd.to_numeric(reco["recommended_purchase_value"], errors="coerce")
    assert abs(summary["total_proposed_purchase_value"] - float(pv.dropna().sum())) < 1e-3
    # a null purchase value is never coerced to zero in the total
    dc.validate_reorder_summary(summary, reco, run_id="rid")


# C30 — two run ids stay isolated
def test_c_two_runs_isolated(tmp_path):
    _, reco1, _, _ = _reco(tmp_path, run_id="runA", skus=("A", "B"))
    _, reco2, _, _ = _reco(tmp_path, run_id="runB", skus=("X", "Y", "Z"))
    assert set(reco1["run_id"]) == {"runA"} and set(reco2["run_id"]) == {"runB"}
    assert set(reco1["sku"]) == {"A", "B"} and set(reco2["sku"]) == {"X", "Y", "Z"}


# C31 — decision outputs are written (atomic contract writers) and reloadable
def test_c_outputs_written(tmp_path):
    rd, reco, summary, _ = _reco(tmp_path, skus=("A",), **{
        k: v for k, v in _ORDER_KW.items() if k != "skus"})
    assert (rd / "decisions" / "reorder_recommendations.parquet").exists()
    assert (rd / "decisions" / "reorder_summary.json").exists()
    assert summary["decision_policy_version"] == dc.REORDER_POLICY_VERSION
    assert (reco["decision_policy_version"] == dc.REORDER_POLICY_VERSION).all()
