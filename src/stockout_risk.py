"""stockout_risk.py — Phase B: forecast-driven stockout-risk analysis.

Turns a run's OPERATIONAL demand forecast (``runs/<run_id>/selected_forecasts.parquet``)
plus its inventory context into an auditable, per-SKU stockout-risk view. It is a
downstream DECISION layer — it never trains a model, never generates demand, never
creates purchase orders, and never classifies real historical stockouts.

Consumes ONLY run-specific artifacts (never global data/processed or global outputs):
  runs/<run_id>/selected_forecasts.parquet
  runs/<run_id>/processed/{inventory_context,model_panel,forecast_frame}.parquet
  runs/<run_id>/outputs/<operational-model backtest>.parquet   (residuals, joined privately)
  runs/<run_id>/request.json                                    (as_of_date, run_id)
  inventory_etl/config/config.yaml                              (decisioning pilot policies)
  + the operational-model metadata (model + horizon) from the orchestrator, in memory.

Writes (atomically) under runs/<run_id>/decisions/:
  stockout_risk.parquet        one row per selected (sku, channel)
  stockout_trajectory.parquet  one row per forecast (sku, channel, date)

Public API (callable BEFORE the final run_manifest.json exists):
  compute_stockout_risk(run_dir, operational_model=..., operational_horizon=...) -> summary dict

Standalone CLI (resolves metadata from an already-created run):
  python src/stockout_risk.py --run-dir runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
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
BASELINE_METHODS = ("last_day_naive", "seasonal_naive_7", "moving_average_7", "moving_average_14")
LOCKED = "locked_holdout"

# fixed z-constants (documented, not re-derived at runtime)
_Z_80_INTERVAL_DIVISOR = 1.2815515655     # 80% PI half-width -> sigma (z_0.90)
_Z_95_INTERVAL_DIVISOR = 1.9599639845     # 95% PI half-width -> sigma (z_0.975)
_Z_P80 = 0.841621                          # 80th-percentile z
_Z_P95 = 1.644854                          # 95th-percentile z
_ND = NormalDist()


# ── config ─────────────────────────────────────────────────────────────────────────────
def load_decisioning_config(config_path: "str | Path | None" = None) -> dict:
    path = Path(config_path) if config_path else CONFIG_PATH
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    d = cfg.get("decisioning") or {}
    return {
        "default_service_level": float(d.get("default_service_level", 0.95)),
        "prefer_channel_service_level": bool(d.get("prefer_channel_service_level", True)),
        "min_residual_observations": int(d.get("min_residual_observations", 7)),
        "prob": {k: float(v) for k, v in (d.get("probability_thresholds")
                 or {"critical": 0.80, "high": 0.50, "medium": 0.20}).items()},
        "cover": {k: float(v) for k, v in (d.get("cover_thresholds_days")
                  or {"critical": 2, "high": 5, "medium": 10}).items()},
        "channel_service_levels": {c: (v or {}).get("service_level_target")
                                   for c, v in ((cfg.get("channels") or {}).get("master") or {}).items()},
    }


def _service_level(channel: str, cfg: dict) -> float:
    if cfg["prefer_channel_service_level"]:
        lvl = cfg["channel_service_levels"].get(channel)
        if lvl is not None:
            return float(lvl)
    return cfg["default_service_level"]


# ── tiers ────────────────────────────────────────────────────────────────────────────────
_SEVERITY = {"critical": 3, "high": 2, "medium": 1, "low": 0, "unknown": -1}


def _probability_tier(p, t: dict) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "unknown"
    if p >= t["critical"]:
        return "critical"
    if p >= t["high"]:
        return "high"
    if p >= t["medium"]:
        return "medium"
    return "low"


def _cover_tier(days, t: dict) -> str:
    if days is None or (isinstance(days, float) and math.isnan(days)):
        return "low"                    # null cover only arises from zero demand -> survives horizon
    if days < t["critical"]:
        return "critical"
    if days < t["high"]:
        return "high"
    if days < t["medium"]:
        return "medium"
    return "low"


def _more_severe(a: str, b: str) -> str:
    return a if _SEVERITY.get(a, -1) >= _SEVERITY.get(b, -1) else b


# ── uncertainty ─────────────────────────────────────────────────────────────────────────
class _ResidualStore:
    """Per-(sku, channel) residual std from the operational model's locked-holdout backtest,
    joined privately to model_panel for truth. Falls back to pooled same-model residuals."""

    def __init__(self, residuals: pd.DataFrame, min_obs: int):
        self.min_obs = min_obs
        self._by_key: dict[tuple, np.ndarray] = {}
        self._pooled = np.asarray(residuals["residual"], dtype=float) if len(residuals) else np.array([])
        for (s, c), g in residuals.groupby(["sku", "channel"]):
            self._by_key[(str(s), str(c))] = np.asarray(g["residual"], dtype=float)

    @staticmethod
    def _std(a: np.ndarray):
        a = a[np.isfinite(a)]
        if a.size < 2:
            return None
        return float(np.std(a, ddof=1))

    def sigma(self, sku: str, channel: str):
        """Returns (sigma, method_label) or (None, None) if unavailable."""
        vals = self._by_key.get((str(sku), str(channel)), np.array([]))
        if vals.size >= self.min_obs:
            s = self._std(vals)
            if s is not None:
                return s, f"backtest_residual_per_sku(n={vals.size})"
        s = self._std(self._pooled)
        if s is not None:
            return s, f"backtest_residual_pooled(n={int(np.isfinite(self._pooled).sum())})"
        return None, None


def _daily_sigma(fut_rows: pd.DataFrame, sku: str, channel: str,
                 residuals: _ResidualStore, hist_std, ) -> tuple[np.ndarray, str, list[str]]:
    """Deterministic fallback: forecast intervals -> backtest residuals -> historical demand std.
    Returns (per-day sigma array, method label, fallback notes)."""
    H = len(fut_rows)
    iv = ["lower_80", "upper_80", "lower_95", "upper_95"]
    notes: list[str] = []
    have_iv = all(c in fut_rows.columns for c in iv) and fut_rows[iv].notna().to_numpy().all()
    if have_iv:
        yv = fut_rows["y_pred"].to_numpy(float)
        l80 = fut_rows["lower_80"].to_numpy(float); u80 = fut_rows["upper_80"].to_numpy(float)
        l95 = fut_rows["lower_95"].to_numpy(float); u95 = fut_rows["upper_95"].to_numpy(float)
        sig = np.zeros(H)
        for i in range(H):
            e80 = max(abs(yv[i] - l80[i]), abs(u80[i] - yv[i])) / _Z_80_INTERVAL_DIVISOR
            e95 = max(abs(yv[i] - l95[i]), abs(u95[i] - yv[i])) / _Z_95_INTERVAL_DIVISOR
            pos = [e for e in (e80, e95) if np.isfinite(e) and e > 0]
            sig[i] = float(np.median(pos)) if pos else 0.0
        return sig, "forecast_intervals", notes
    # Method 2: backtest residuals
    rs, method = residuals.sigma(sku, channel)
    if rs is not None:
        if "pooled" in method:
            notes.append("per_sku_residuals_insufficient_used_pooled")
        return np.full(H, rs), method, notes
    # Method 3: historical real-demand variability
    notes.append("backtest_residuals_unavailable")
    if hist_std is not None and hist_std >= 0:
        return np.full(H, float(hist_std)), "historical_demand_std", notes
    notes.append("no_uncertainty_estimate")
    return np.zeros(H), "none", notes


# ── residual extraction (private; never leaks truth) ─────────────────────────────────────
def _operational_backtest_residuals(run_dir: Path, operational_model: str,
                                    operational_horizon: int, model_panel: pd.DataFrame,
                                    as_of: pd.Timestamp) -> pd.DataFrame:
    outputs = run_dir / "outputs"
    if operational_model in BASELINE_METHODS:
        bt_path = outputs / "baseline_backtest_predictions.parquet"
        method = operational_model
    elif operational_model in ("holtwinters", "lightgbm"):
        bt_path = outputs / f"{operational_model}_backtest_predictions.parquet"
        method = None
    else:
        raise ValueError(f"unrecognized operational_model {operational_model!r}")
    if not bt_path.exists():
        return pd.DataFrame(columns=["sku", "channel", "residual"])
    bt = pd.read_parquet(bt_path)
    bt = bt[bt["evaluation_type"] == LOCKED].copy()
    bt = bt[pd.to_numeric(bt["horizon"], errors="coerce").astype("Int64") == int(operational_horizon)]
    if method is not None:
        bt = bt[bt["model"].astype(str) == method]
    if bt.empty:
        return pd.DataFrame(columns=["sku", "channel", "residual"])
    bt = bt[["sku", "channel", "date", "y_pred"]].copy()
    bt["date"] = pd.to_datetime(bt["date"])
    truth = model_panel[["sku", "channel", "date", "units_observed"]].copy()
    truth["date"] = pd.to_datetime(truth["date"])
    j = bt.merge(truth, on=["sku", "channel", "date"], how="inner")
    j = j[j["date"] <= as_of]                       # never use a date after as_of for residuals
    j["residual"] = pd.to_numeric(j["units_observed"], errors="coerce") - pd.to_numeric(j["y_pred"], errors="coerce")
    return j[["sku", "channel", "residual"]].dropna(subset=["residual"])   # truth never leaves this function


def _historical_std_map(model_panel: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    mp = model_panel.copy()
    mp["date"] = pd.to_datetime(mp["date"])
    mp = mp[mp["date"] <= as_of]
    out = {}
    for (s, c), g in mp.groupby(["sku", "channel"]):
        vals = pd.to_numeric(g["units_observed"], errors="coerce").dropna().to_numpy()
        out[(str(s), str(c))] = float(np.std(vals, ddof=1)) if vals.size >= 2 else None
    return out


# ── core computation ─────────────────────────────────────────────────────────────────────
def compute_stockout_risk(run_dir: "str | Path", *, operational_model: str,
                          operational_horizon: int, config_path: "str | Path | None" = None,
                          logger=None) -> dict:
    """Compute Phase B artifacts for a run and write them atomically under decisions/.
    Callable during the pipeline (before run_manifest.json exists)."""
    run_dir = Path(run_dir)
    cfg = load_decisioning_config(config_path)

    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    run_id = str(request["run_id"])
    as_of = pd.Timestamp(request["as_of_date"])

    sel = pd.read_parquet(run_dir / "selected_forecasts.parquet")
    sel["date"] = pd.to_datetime(sel["date"])
    proc = run_dir / "processed"
    inv = pd.read_parquet(proc / "inventory_context.parquet")
    mp = pd.read_parquet(proc / "model_panel.parquet")
    ff = pd.read_parquet(proc / "forecast_frame.parquet"); ff["date"] = pd.to_datetime(ff["date"])

    inv_by_sku = inv.set_index(inv["sku"].astype(str)) if "sku" in inv.columns else inv
    resid = _operational_backtest_residuals(run_dir, operational_model, operational_horizon, mp, as_of)
    residuals = _ResidualStore(resid, cfg["min_residual_observations"])
    hist_std = _historical_std_map(mp, as_of)

    # product name: prefer the selected-forecast column, else model_panel
    name_map = {}
    if "sku_name" in sel.columns:
        name_map = dict(sel.dropna(subset=["sku_name"]).astype({"sku": str}).groupby("sku")["sku_name"].first())
    elif "sku_name" in mp.columns:
        name_map = dict(mp.dropna(subset=["sku_name"]).astype({"sku": str}).groupby("sku")["sku_name"].first())

    risk_rows: list[dict] = []
    traj_rows: list[dict] = []
    for (sku, channel), g in sel.groupby(["sku", "channel"], sort=True):
        g = g.sort_values("forecast_horizon_day" if "forecast_horizon_day" in g.columns else "date").reset_index(drop=True)
        row, trows = _risk_for_group(str(sku), str(channel), g, inv_by_sku, residuals,
                                     hist_std.get((str(sku), str(channel))), cfg, run_id,
                                     operational_model, operational_horizon, as_of, name_map)
        risk_rows.append(row)
        traj_rows.extend(trows)

    risk_df = pd.DataFrame(risk_rows, columns=dc.STOCKOUT_RISK_COLUMNS)
    traj_df = pd.DataFrame(traj_rows, columns=dc.STOCKOUT_TRAJECTORY_COLUMNS)

    selected_keys = sel[["sku", "channel"]].drop_duplicates()
    forecast_keys = sel[["sku", "channel", "date"]].drop_duplicates()
    risk_df = dc.validate_stockout_risk(risk_df, selected_keys, run_id)
    traj_df = dc.validate_stockout_trajectory(traj_df, forecast_keys, run_id)

    dec_dir = run_dir / "decisions"
    risk_path = dec_dir / "stockout_risk.parquet"
    traj_path = dec_dir / "stockout_trajectory.parquet"
    dc.write_dataframe_atomic(risk_df, risk_path, "parquet")
    dc.write_dataframe_atomic(traj_df, traj_path, "parquet")

    tier_counts = risk_df["overall_risk_tier"].value_counts().to_dict()
    summary = {
        "decisioning_status": "completed",
        "stockout_risk_file": "decisions/stockout_risk.parquet",
        "stockout_trajectory_file": "decisions/stockout_trajectory.parquet",
        "risk_rows": int(len(risk_df)),
        "trajectory_rows": int(len(traj_df)),
        "operational_model": operational_model,
        "operational_horizon": int(operational_horizon),
        "risk_tier_counts": {t: int(tier_counts.get(t, 0)) for t in dc.RISK_TIERS},
        "manual_review_count": int(risk_df["manual_review_required"].astype(bool).sum()),
        "uncertainty_methods": risk_df["uncertainty_method"].value_counts().to_dict(),
    }
    if logger is not None:
        logger.info("stockout risk: %d SKUs, tiers=%s, methods=%s", len(risk_df),
                    summary["risk_tier_counts"], summary["uncertainty_methods"])
    return summary


def _risk_for_group(sku, channel, g, inv_by_sku, residuals, hist_std, cfg, run_id,
                    operational_model, operational_horizon, as_of, name_map):
    y = pd.to_numeric(g["y_pred"], errors="coerce").fillna(0.0).clip(lower=0).to_numpy(float)
    dates = pd.to_datetime(g["date"]).tolist()
    H = len(y)
    hdays = (pd.to_numeric(g["forecast_horizon_day"], errors="coerce").to_numpy()
             if "forecast_horizon_day" in g.columns else np.arange(1, H + 1))

    # inventory context (per SKU; may be absent for a SKU with no inventory row)
    irow = inv_by_sku.loc[sku] if (hasattr(inv_by_sku, "index") and sku in inv_by_sku.index) else None

    def _get(col, default=None):
        if irow is None or col not in getattr(irow, "index", []):
            return default
        v = irow[col]
        return default if (v is None or (isinstance(v, float) and math.isnan(v))) else v

    stock_on_hand = float(_get("stock_on_hand", 0.0) or 0.0)
    stock_is_synth = bool(_get("stock_on_hand_is_synthetic", False))
    stock_source = _get("stock_source", None)
    reported_on_order = float(_get("on_order_quantity", 0.0) or 0.0)
    lead_time_days = int(_get("lead_time_days", 0) or 0)
    lead_time_source = _get("lead_time_source", None)
    price = _get("price", None)
    price = float(price) if price is not None else None
    sl = _service_level(channel, cfg)

    # on-order is NEVER auto-available: no inbound-arrival-date contract exists in the pilot.
    usable_on_order = 0.0
    on_order_available = False
    inv_pos = stock_on_hand + usable_on_order

    daily_sigma, unc_method, unc_notes = _daily_sigma(g, sku, channel, residuals, hist_std)
    cum_mean = np.cumsum(y)
    cum_sigma = np.sqrt(np.cumsum(daily_sigma ** 2))

    flags: list[str] = []
    if stock_is_synth:
        flags.append("synthetic_stock")
    if lead_time_source is None or "assumed" in str(lead_time_source).lower():
        flags.append("assumed_lead_time")
    if reported_on_order > 0:
        flags.append("on_order_excluded_no_arrival_date")
    flags.append(f"uncertainty_{unc_method}")
    flags.extend(unc_notes)

    # ── days of cover + projected stockout (available horizon only) ──────────────────────
    mean_daily = float(np.mean(y)) if H else 0.0
    zero_demand = mean_daily <= 0
    if zero_demand:
        days_of_cover = None
        flags.append("zero_demand")
    else:
        days_of_cover = float(inv_pos / mean_daily)
    cross = np.where(cum_mean > inv_pos)[0]
    if cross.size:
        i = int(cross[0])
        projected_stockout_date = pd.Timestamp(dates[i]).date().isoformat()
        days_until = int(hdays[i])
        survives = False
    else:
        projected_stockout_date = None
        days_until = None
        survives = True

    horizon_sufficient = lead_time_days <= H and lead_time_days >= 1
    manual_review = False
    if not horizon_sufficient:
        manual_review = True
        flags.append("insufficient_forecast_horizon")
        lt_mean = lt_sigma = p50 = p80 = p95 = None
        safety_stock = reorder_point = None
        prob = None
        exp_short = None
        rev_at_risk = None
        prob_tier = "unknown"
        overall = "unknown"
    else:
        li = lead_time_days - 1
        lt_mean = float(cum_mean[li])
        lt_sigma = float(cum_sigma[li])
        p50 = lt_mean
        p80 = max(0.0, lt_mean + _Z_P80 * lt_sigma)
        p95 = max(0.0, lt_mean + _Z_P95 * lt_sigma)
        if lt_sigma == 0:
            prob = 1.0 if lt_mean > inv_pos else 0.0
            exp_short = max(0.0, lt_mean - inv_pos)
            flags.append("zero_sigma")
        else:
            z = (inv_pos - lt_mean) / lt_sigma
            prob = min(1.0, max(0.0, 1.0 - _ND.cdf(z)))
            exp_short = float(lt_sigma * _ND.pdf(z) + (lt_mean - inv_pos) * (1.0 - _ND.cdf(z)))
            exp_short = max(0.0, exp_short)
        safety_stock = int(math.ceil(_ND.inv_cdf(sl) * lt_sigma))
        reorder_point = int(math.ceil(lt_mean + safety_stock))
        if price is not None:
            rev_at_risk = float(exp_short * price)
        else:
            rev_at_risk = None
            flags.append("price_missing")
        prob_tier = _probability_tier(prob, cfg["prob"])

    cover_tier = _cover_tier(days_of_cover, cfg["cover"])
    if horizon_sufficient:
        overall = _more_severe(prob_tier, cover_tier)
    # confidence label
    if manual_review:
        confidence = "low"
    elif unc_method in ("historical_demand_std", "none") or "pooled" in unc_method:
        confidence = "low"
    elif stock_is_synth or "assumed_lead_time" in flags:
        confidence = "medium"
    else:
        confidence = "high"

    # deterministic human-readable reason trace
    if horizon_sufficient:
        reason = (f"position={inv_pos:.0f} (stock {stock_on_hand:.0f} + usable on-order {usable_on_order:.0f}); "
                  f"lead-time {lead_time_days}d demand P50={p50:.1f} sigma={lt_sigma:.1f}; "
                  f"P(stockout)={prob:.2f} [{prob_tier}]; "
                  f"cover={'n/a' if days_of_cover is None else f'{days_of_cover:.1f}d'} [{cover_tier}]; "
                  f"overall={overall} (max of probability & cover); uncertainty={unc_method}; "
                  f"projected_stockout={projected_stockout_date or 'none within horizon'}")
    else:
        reason = (f"lead time {lead_time_days}d exceeds available forecast horizon {H}d — "
                  f"lead-time risk not computable; MANUAL REVIEW. "
                  f"position={inv_pos:.0f}; cover={'n/a' if days_of_cover is None else f'{days_of_cover:.1f}d'} "
                  f"[{cover_tier}]; uncertainty={unc_method}")

    risk = {
        "run_id": run_id, "sku": sku, "channel": channel, "sku_name": name_map.get(sku),
        "operational_model": operational_model, "operational_horizon": int(operational_horizon),
        "as_of_date": as_of.date().isoformat(),
        "stock_on_hand": stock_on_hand, "stock_on_hand_is_synthetic": stock_is_synth,
        "stock_source": stock_source,
        "reported_on_order_quantity": reported_on_order, "usable_on_order_quantity": usable_on_order,
        "on_order_available": on_order_available, "inventory_position_for_risk": inv_pos,
        "lead_time_days": lead_time_days, "lead_time_source": lead_time_source, "service_level": sl,
        "forecast_horizon_available": int(H), "lead_time_horizon_sufficient": bool(horizon_sufficient),
        "lead_time_demand_p50": p50, "lead_time_demand_p80": p80, "lead_time_demand_p95": p95,
        "lead_time_sigma": (None if not horizon_sufficient else lt_sigma),
        "safety_stock": safety_stock, "reorder_point": reorder_point,
        "forecast_days_of_cover": days_of_cover, "projected_stockout_date": projected_stockout_date,
        "days_until_projected_stockout": days_until, "survives_forecast_horizon": bool(survives),
        "stockout_probability": prob, "probability_risk_tier": prob_tier,
        "cover_risk_tier": cover_tier, "overall_risk_tier": overall,
        "expected_shortage_units": exp_short, "estimated_revenue_at_risk": rev_at_risk,
        "uncertainty_method": unc_method, "confidence_label": confidence,
        "manual_review_required": bool(manual_review), "assumption_flags": ";".join(flags),
        "reason_trace": reason,
    }

    # ── trajectory rows (available horizon) ──────────────────────────────────────────────
    trows = []
    for i in range(H):
        cm, cs = float(cum_mean[i]), float(cum_sigma[i])
        if cs == 0:
            cprob = 1.0 if cm > inv_pos else 0.0
        else:
            cprob = min(1.0, max(0.0, 1.0 - _ND.cdf((inv_pos - cm) / cs)))
        trows.append({
            "run_id": run_id, "sku": sku, "channel": channel,
            "date": pd.Timestamp(dates[i]), "forecast_horizon_day": int(hdays[i]),
            "daily_demand_mean": float(y[i]), "daily_sigma": float(daily_sigma[i]),
            "cumulative_demand_mean": cm, "cumulative_sigma": cs,
            "demand_p50": cm, "demand_p80": max(0.0, cm + _Z_P80 * cs),
            "demand_p95": max(0.0, cm + _Z_P95 * cs),
            "projected_p50_inventory": float(inv_pos - cm),
            "cumulative_stockout_probability": cprob,
        })
    return risk, trows


# ── CLI (resolves operational metadata from an already-created run) ──────────────────────
def resolve_operational_metadata(run_dir: Path) -> tuple[str, int]:
    """Resolve (operational_model, operational_horizon) from run_manifest.json when present,
    else from selected_forecasts.parquet (selection_horizon + model)."""
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        if m.get("operational_model") and m.get("operational_horizon"):
            return str(m["operational_model"]), int(m["operational_horizon"])
    sel = pd.read_parquet(run_dir / "selected_forecasts.parquet")
    model = str(sel["model"].dropna().unique()[0])
    horizon = int(pd.to_numeric(sel["selection_horizon"], errors="coerce").dropna().max())
    return model, horizon


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Phase B forecast-driven stockout risk.")
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
    if not (run_dir / "selected_forecasts.parquet").exists():
        print(f"run has no selected_forecasts.parquet: {run_dir}", file=sys.stderr)
        return 2
    model, horizon = resolve_operational_metadata(run_dir)
    summary = compute_stockout_risk(run_dir, operational_model=model,
                                    operational_horizon=horizon, config_path=args.config)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
