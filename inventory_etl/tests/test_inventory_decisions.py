"""Phase B — forecast-driven stockout-risk decision tests.

Exercises src/stockout_risk.py directly on tiny, self-contained fake runs (no warehouse,
no model training). One focused file. Orchestrator-integration and dashboard-exposure of
Phase B are covered in test_forecast_orchestrator.py / test_dashboard_runs.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import stockout_risk as sr        # noqa: E402
import decision_contract as dc    # noqa: E402

AS_OF = "2026-06-30"
CHANNEL = "naheed_web"


def _make_run(tmp: Path, run_id="rid", *, skus=("A", "B", "C"), horizon_days=14, y_pred=5.0,
              with_intervals=False, interval_half=8.0, stock=100.0, on_order=0.0,
              lead_time=7, lead_time_source="sku_master_picking_mode", synthetic=False,
              price=100.0, model="holtwinters", op_horizon=14, backtest=True, backtest_obs=14,
              stock_by_sku=None) -> Path:
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
                         "price": (None if price is None else float(price))})
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
