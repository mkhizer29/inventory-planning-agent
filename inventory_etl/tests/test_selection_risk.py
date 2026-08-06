"""Focused unit tests for src/selection_risk.py (pre-forecast stockout-risk proxy).

Everything runs against a throwaway temp SQLite DB mimicking the warehouse schema
(sku_master + sales_transactions + inventory_snapshot) and a temp config.yaml. The real
Magento DB, the real inventory.db and the real project config are never touched.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import dynamic_selection as ds      # noqa: E402
import selection_risk as sr         # noqa: E402

CUTOFF = "2026-06-30"
WINDOW = 28
WINDOW_START = "2026-06-03"         # inclusive 28-day window ending at CUTOFF
ECOM = "online_delivery"
SNAPSHOT = "2026-06-30"             # on/before cutoff unless a test says otherwise
LEAD = 7


# ── fixtures ──────────────────────────────────────────────────────────────────────────
def _write_config(tmp_path: Path, **overrides) -> Path:
    """Minimal project config carrying every block selection_risk reads."""
    sel = {
        "enabled": True,
        "demand_window_days": WINDOW,
        "stock_snapshot_policy": "latest",
        "include_zero_stock": True,
        "exclude_dropship": False,
    }
    sel.update(overrides)
    body = f"""
cleansing:
  stock_sentinel_threshold: 10000
  stock_negative_floor: 0
replenishment:
  default_supplier_lead_time_days: {LEAD}
pilot:
  min_history_days: 28
  ecommerce_channel_map:
    online_delivery: naheed_web
    foodpanda: foodpanda
decisioning:
  probability_thresholds:
    critical: 0.80
    high: 0.50
    medium: 0.20
selection_risk:
  enabled: {str(sel['enabled']).lower()}
  demand_window_days: {sel['demand_window_days']}
  stock_snapshot_policy: {sel['stock_snapshot_policy']}
  include_zero_stock: {str(sel['include_zero_stock']).lower()}
  exclude_dropship: {str(sel['exclude_dropship']).lower()}
"""
    p = tmp_path / "config.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def _make_db(tmp_path: Path, skus: list[dict], sales: list[dict],
             stock: list[dict], snapshot_date: str = SNAPSHOT) -> Path:
    db = tmp_path / "wh.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE sku_master (sku_id TEXT, product_id INTEGER, sku_name TEXT, "
                "category TEXT, sub_category TEXT, brand TEXT, "
                "supplier_lead_time_days REAL, is_dropship INTEGER)")
    con.execute("CREATE TABLE sales_transactions (sku_id TEXT, channel TEXT, "
                "transaction_date DATE, quantity_sold REAL)")
    con.execute("CREATE TABLE inventory_snapshot (product_id INTEGER, snapshot_date DATE, "
                "stock_on_hand REAL, location_id INTEGER)")
    pid = {}
    for i, s in enumerate(skus):
        pid[s["sku_id"]] = 1000 + i
        con.execute("INSERT INTO sku_master VALUES (?,?,?,?,?,?,?,?)",
                    (s["sku_id"], 1000 + i, s.get("sku_name"), s.get("category", "Groceries"),
                     s.get("sub_category"), s.get("brand"),
                     s.get("supplier_lead_time_days", LEAD), s.get("is_dropship", 0)))
    for t in sales:
        con.execute("INSERT INTO sales_transactions VALUES (?,?,?,?)",
                    (t["sku_id"], t.get("channel", ECOM), t["transaction_date"], t["quantity_sold"]))
    for st in stock:
        con.execute("INSERT INTO inventory_snapshot VALUES (?,?,?,?)",
                    (pid[st["sku_id"]], st.get("snapshot_date", snapshot_date),
                     st["stock_on_hand"], st.get("location_id", 1)))
    con.commit()
    con.close()
    return db


def _daily(sku_id: str, start: str, days: int, qty: float, channel: str = ECOM) -> list[dict]:
    d0 = pd.Timestamp(start)
    return [{"sku_id": sku_id, "channel": channel,
             "transaction_date": (d0 + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
             "quantity_sold": qty} for i in range(days)]


def _score(db, cfg, skus):
    return sr.score_stockout_risk(db, skus, selection_cutoff=CUTOFF, config_path=cfg)


def _row(df, sku):
    return df.loc[df["sku"] == sku].iloc[0]


# ── demand statistics ─────────────────────────────────────────────────────────────────
def test_mean_divides_by_window_not_active_days(tmp_path):
    """A SKU selling on half the window averages over the FULL window: days with no sale
    are real zero-demand days, not missing data."""
    skus = [{"sku_id": "S1"}]
    sales = _daily("S1", WINDOW_START, 14, 10)          # 14 of 28 days, 10 units each
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "S1", "stock_on_hand": 100}])
    cfg = _write_config(tmp_path)
    out, _ = _score(db, cfg, ["S1"])
    r = _row(out, "S1")
    assert r["demand_active_days"] == 14
    assert r["demand_mean_daily"] == pytest.approx(140 / 28)          # 5.0, not 10.0
    # sample variance over the zero-filled window: (14*25 + 14*25) / 27
    assert r["demand_sigma_daily"] == pytest.approx(math.sqrt(700 / 27))


def test_window_excludes_sales_outside_it(tmp_path):
    """Only the trailing window feeds the proxy; older and post-cutoff sales are ignored."""
    skus = [{"sku_id": "S1"}]
    sales = (_daily("S1", "2026-01-01", 20, 99)          # long before the window
             + _daily("S1", WINDOW_START, 28, 2)         # the window itself
             + _daily("S1", "2026-07-01", 5, 99))        # after the cutoff
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "S1", "stock_on_hand": 100}])
    cfg = _write_config(tmp_path)
    out, _ = _score(db, cfg, ["S1"])
    r = _row(out, "S1")
    assert r["demand_mean_daily"] == pytest.approx(2.0)
    assert r["demand_sigma_daily"] == pytest.approx(0.0)
    assert r["demand_window_start"] == WINDOW_START


def test_non_ecommerce_channel_ignored(tmp_path):
    skus = [{"sku_id": "S1"}]
    sales = _daily("S1", WINDOW_START, 28, 5, channel="store")
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "S1", "stock_on_hand": 100}])
    cfg = _write_config(tmp_path)
    out, _ = _score(db, cfg, ["S1"])
    assert _row(out, "S1")["demand_mean_daily"] == pytest.approx(0.0)


# ── risk arithmetic ───────────────────────────────────────────────────────────────────
def test_flat_demand_certain_shortfall(tmp_path):
    """Flat demand -> sigma 0 -> the proxy is certain: P is exactly 1.0 and the shortage is
    the plain arithmetic gap."""
    skus = [{"sku_id": "FLAT"}]
    sales = _daily("FLAT", WINDOW_START, 28, 2)          # mean 2/day, sigma 0
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "FLAT", "stock_on_hand": 10}])
    cfg = _write_config(tmp_path)
    out, _ = _score(db, cfg, ["FLAT"])
    r = _row(out, "FLAT")
    assert r["lead_time_demand_mean"] == pytest.approx(14.0)          # 2 * 7
    assert r["lead_time_demand_sigma"] == pytest.approx(0.0)
    assert r["stockout_probability"] == 1.0                          # exact, not approx
    assert r["expected_shortage_units"] == pytest.approx(4.0)        # 14 - 10
    assert r["proxy_risk_tier"] == "critical"
    assert "zero_demand_sigma" in r["risk_assumption_flags"]


def test_flat_demand_amply_stocked(tmp_path):
    skus = [{"sku_id": "SAFE"}]
    sales = _daily("SAFE", WINDOW_START, 28, 2)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "SAFE", "stock_on_hand": 1000}])
    cfg = _write_config(tmp_path)
    out, _ = _score(db, cfg, ["SAFE"])
    r = _row(out, "SAFE")
    assert r["stockout_probability"] == 0.0
    assert r["expected_shortage_units"] == pytest.approx(0.0)
    assert r["proxy_risk_tier"] == "low"
    assert r["proxy_days_of_cover"] == pytest.approx(500.0)          # 1000 / 2


def test_probability_matches_normal_formula(tmp_path):
    """The variable-demand path must equal 1 - Phi((stock - lt_mean)/lt_sigma) exactly."""
    from statistics import NormalDist
    skus = [{"sku_id": "VAR"}]
    sales = _daily("VAR", WINDOW_START, 14, 10)          # mean 5, sigma sqrt(700/27)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "VAR", "stock_on_hand": 40}])
    cfg = _write_config(tmp_path)
    out, _ = _score(db, cfg, ["VAR"])
    r = _row(out, "VAR")
    lt_mean = 5.0 * LEAD
    lt_sigma = math.sqrt(700 / 27) * math.sqrt(LEAD)
    expected = 1.0 - NormalDist().cdf((40 - lt_mean) / lt_sigma)
    assert r["stockout_probability"] == pytest.approx(expected)
    assert 0.0 < r["stockout_probability"] < 1.0


def test_zero_stock_is_scored_and_flagged(tmp_path):
    skus = [{"sku_id": "OUT"}]
    sales = _daily("OUT", WINDOW_START, 28, 3)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "OUT", "stock_on_hand": 0}])
    cfg = _write_config(tmp_path)
    out, meta = _score(db, cfg, ["OUT"])
    r = _row(out, "OUT")
    assert bool(r["risk_scored"]) is True
    assert r["stockout_probability"] == 1.0
    assert "already_out_of_stock" in r["risk_assumption_flags"]
    assert meta["already_out_of_stock"] == 1


def test_zero_stock_excluded_when_configured(tmp_path):
    skus = [{"sku_id": "OUT"}]
    sales = _daily("OUT", WINDOW_START, 28, 3)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "OUT", "stock_on_hand": 0}])
    cfg = _write_config(tmp_path, include_zero_stock=False)
    out, meta = _score(db, cfg, ["OUT"])
    r = _row(out, "OUT")
    assert bool(r["risk_scored"]) is False
    assert r["risk_exclusion_reason"] == "zero_stock_excluded"
    assert pd.isna(r["stockout_probability"])


# ── stock sourcing, cleansing and exclusions ──────────────────────────────────────────
def test_missing_stock_row_is_unscored_never_zero(tmp_path):
    """Missing stock must never be read as zero stock — that would invent maximum risk.

    A snapshot DOES exist here (OTHER is in it); only NOSTOCK is absent from it, which is
    the case that must not be confused with 'out of stock'.
    """
    skus = [{"sku_id": "NOSTOCK"}, {"sku_id": "OTHER"}]
    sales = _daily("NOSTOCK", WINDOW_START, 28, 3) + _daily("OTHER", WINDOW_START, 28, 3)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "OTHER", "stock_on_hand": 40}])
    cfg = _write_config(tmp_path)
    out, meta = _score(db, cfg, ["NOSTOCK", "OTHER"])
    r = _row(out, "NOSTOCK")
    assert bool(r["risk_scored"]) is False
    assert r["risk_exclusion_reason"] == "no_stock_row"
    assert pd.isna(r["stockout_probability"])
    assert pd.isna(r["stock_on_hand"])
    assert meta["exclusion_reasons"]["no_stock_row"] == 1
    assert bool(_row(out, "OTHER")["risk_scored"]) is True


def test_no_snapshot_at_all_is_distinct_from_missing_row(tmp_path):
    """An empty inventory_snapshot is a different failure from a SKU missing from it."""
    skus = [{"sku_id": "S1"}]
    sales = _daily("S1", WINDOW_START, 28, 3)
    db = _make_db(tmp_path, skus, sales, [])
    out, meta = _score(db, _write_config(tmp_path), ["S1"])
    assert _row(out, "S1")["risk_exclusion_reason"] == "no_inventory_snapshot"
    assert meta["stock_snapshot_date"] is None
    assert any("No inventory snapshot" in w for w in meta["warnings"])


def test_sentinel_stock_excluded(tmp_path):
    """Vendor 'unlimited' sentinel values are not real counts."""
    skus = [{"sku_id": "SENT"}]
    sales = _daily("SENT", WINDOW_START, 28, 3)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "SENT", "stock_on_hand": 99991}])
    cfg = _write_config(tmp_path)
    out, _ = _score(db, cfg, ["SENT"])
    r = _row(out, "SENT")
    assert bool(r["risk_scored"]) is False
    assert r["risk_exclusion_reason"] == "stock_sentinel_value"


def test_negative_stock_clamped_to_floor(tmp_path):
    skus = [{"sku_id": "NEG"}]
    sales = _daily("NEG", WINDOW_START, 28, 3)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "NEG", "stock_on_hand": -25}])
    cfg = _write_config(tmp_path)
    out, _ = _score(db, cfg, ["NEG"])
    r = _row(out, "NEG")
    assert r["stock_on_hand"] == 0.0
    assert bool(r["risk_scored"]) is True


def test_stock_summed_across_locations(tmp_path):
    skus = [{"sku_id": "MULTI"}]
    sales = _daily("MULTI", WINDOW_START, 28, 2)
    stock = [{"sku_id": "MULTI", "stock_on_hand": 30, "location_id": 1},
             {"sku_id": "MULTI", "stock_on_hand": 45, "location_id": 2}]
    db = _make_db(tmp_path, skus, sales, stock)
    cfg = _write_config(tmp_path)
    out, _ = _score(db, cfg, ["MULTI"])
    assert _row(out, "MULTI")["stock_on_hand"] == pytest.approx(75.0)


def test_dropship_excluded_when_configured(tmp_path):
    skus = [{"sku_id": "DROP", "is_dropship": 1}]
    sales = _daily("DROP", WINDOW_START, 28, 3)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "DROP", "stock_on_hand": 5}])
    assert bool(_row(_score(db, _write_config(tmp_path), ["DROP"])[0], "DROP")["risk_scored"]) is True
    cfg2 = _write_config(tmp_path, exclude_dropship=True)
    r = _row(_score(db, cfg2, ["DROP"])[0], "DROP")
    assert bool(r["risk_scored"]) is False
    assert r["risk_exclusion_reason"] == "dropship_excluded"


# ── lead time ─────────────────────────────────────────────────────────────────────────
def test_real_lead_time_preferred_and_fallback_flagged(tmp_path):
    skus = [{"sku_id": "REAL", "supplier_lead_time_days": 3},
            {"sku_id": "NOLT", "supplier_lead_time_days": None}]
    sales = _daily("REAL", WINDOW_START, 28, 2) + _daily("NOLT", WINDOW_START, 28, 2)
    stock = [{"sku_id": "REAL", "stock_on_hand": 50}, {"sku_id": "NOLT", "stock_on_hand": 50}]
    db = _make_db(tmp_path, skus, sales, stock)
    cfg = _write_config(tmp_path)
    out, _ = _score(db, cfg, ["REAL", "NOLT"])
    real, nolt = _row(out, "REAL"), _row(out, "NOLT")
    assert real["lead_time_days"] == 3
    assert real["lead_time_source"] == "sku_master.supplier_lead_time_days"
    assert real["lead_time_demand_mean"] == pytest.approx(6.0)       # 2 * 3
    assert nolt["lead_time_days"] == LEAD
    assert nolt["lead_time_source"].startswith("config.")
    assert "assumed_lead_time" in nolt["risk_assumption_flags"]


# ── snapshot policy / post-cutoff disclosure ──────────────────────────────────────────
def test_post_cutoff_snapshot_flagged_not_hidden(tmp_path):
    skus = [{"sku_id": "S1"}]
    sales = _daily("S1", WINDOW_START, 28, 2)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "S1", "stock_on_hand": 5}],
                  snapshot_date="2026-08-05")            # postdates CUTOFF
    cfg = _write_config(tmp_path)
    out, meta = _score(db, cfg, ["S1"])
    assert meta["stock_is_post_cutoff"] is True
    assert meta["stock_snapshot_date"] == "2026-08-05"
    assert any("POSTDATES" in w for w in meta["warnings"])
    assert "stock_snapshot_post_cutoff" in _row(out, "S1")["risk_assumption_flags"]


def test_strict_policy_refuses_post_cutoff_snapshot(tmp_path):
    skus = [{"sku_id": "S1"}]
    sales = _daily("S1", WINDOW_START, 28, 2)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "S1", "stock_on_hand": 5}],
                  snapshot_date="2026-08-05")
    cfg = _write_config(tmp_path, stock_snapshot_policy="on_or_before_cutoff")
    out, meta = _score(db, cfg, ["S1"])
    assert meta["stock_snapshot_date"] is None
    assert meta["scored"] == 0
    assert bool(_row(out, "S1")["risk_scored"]) is False


def test_invalid_policy_rejected(tmp_path):
    cfg = _write_config(tmp_path, stock_snapshot_policy="whenever")
    with pytest.raises(sr.SelectionRiskError, match="stock_snapshot_policy"):
        sr.load_selection_risk_config(cfg)


def test_degenerate_window_rejected(tmp_path):
    cfg = _write_config(tmp_path, demand_window_days=1)
    with pytest.raises(sr.SelectionRiskError, match="demand_window_days"):
        sr.load_selection_risk_config(cfg)


# ── ranking ───────────────────────────────────────────────────────────────────────────
def test_rank_orders_by_probability_then_shortage(tmp_path):
    """Both out-of-stock SKUs have flat demand -> P is EXACTLY 1.0 for both, so the
    expected-shortage tie-break decides: bigger exposure first."""
    skus = [{"sku_id": "TIE-BIG"}, {"sku_id": "TIE-SMALL"}, {"sku_id": "SAFE"}]
    sales = (_daily("TIE-BIG", WINDOW_START, 28, 5)
             + _daily("TIE-SMALL", WINDOW_START, 28, 1)
             + _daily("SAFE", WINDOW_START, 28, 1))
    stock = [{"sku_id": "TIE-BIG", "stock_on_hand": 0},
             {"sku_id": "TIE-SMALL", "stock_on_hand": 0},
             {"sku_id": "SAFE", "stock_on_hand": 900}]
    db = _make_db(tmp_path, skus, sales, stock)
    cfg = _write_config(tmp_path)
    out, _ = _score(db, cfg, ["SAFE", "TIE-SMALL", "TIE-BIG"])
    assert _row(out, "TIE-BIG")["stockout_probability"] == 1.0
    assert _row(out, "TIE-SMALL")["stockout_probability"] == 1.0
    ranked = sr.rank_by_stockout_risk(out)
    assert list(ranked["sku"]) == ["TIE-BIG", "TIE-SMALL", "SAFE"]


def test_saturated_probabilities_rank_by_exposure(tmp_path):
    """The realistic case, and the one that matters most in production.

    Both SKUs are out of stock with VARYING demand, so neither probability is exactly 1.0 —
    they differ only far out in the tail (~1e-13). Those digits are numerical noise, not a
    business distinction: both mean "certain to stock out". STEADY has the higher raw
    probability because its demand is less erratic, but ERRATIC is exposed for ~3x as many
    units. Ranking must put the bigger exposure first.
    """
    skus = [{"sku_id": "STEADY-SMALL"}, {"sku_id": "ERRATIC-BIG"}]
    small = [3, 5, 7, 7, 7, 9, 11] * 4                   # mean 7/day   -> lead-time 49
    big = [8, 14, 20, 20, 20, 26, 32] * 4                # mean 20/day  -> lead-time 140
    sales = []
    for i, (q_s, q_b) in enumerate(zip(small, big)):
        d = (pd.Timestamp(WINDOW_START) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        sales.append({"sku_id": "STEADY-SMALL", "transaction_date": d, "quantity_sold": q_s})
        sales.append({"sku_id": "ERRATIC-BIG", "transaction_date": d, "quantity_sold": q_b})
    stock = [{"sku_id": "STEADY-SMALL", "stock_on_hand": 0},
             {"sku_id": "ERRATIC-BIG", "stock_on_hand": 0}]
    db = _make_db(tmp_path, skus, sales, stock)
    out, _ = _score(db, _write_config(tmp_path), ["STEADY-SMALL", "ERRATIC-BIG"])

    p_small = _row(out, "STEADY-SMALL")["stockout_probability"]
    p_big = _row(out, "ERRATIC-BIG")["stockout_probability"]
    s_small = _row(out, "STEADY-SMALL")["expected_shortage_units"]
    s_big = _row(out, "ERRATIC-BIG")["expected_shortage_units"]

    # preconditions: saturated but NOT exactly equal, and exposure strongly favours ERRATIC
    assert p_small > p_big, "fixture invalid: STEADY should have the higher raw probability"
    assert p_small != 1.0 and p_big != 1.0, "fixture invalid: neither should be an exact tie"
    assert round(p_small, 6) == round(p_big, 6) == 1.0
    assert s_big > 2 * s_small

    ranked = sr.rank_by_stockout_risk(out)
    assert list(ranked["sku"]) == ["ERRATIC-BIG", "STEADY-SMALL"]


def test_meaningful_probability_gap_still_beats_exposure(tmp_path):
    """Guard against over-rounding: when probabilities genuinely differ, probability wins
    even if the lower-risk SKU carries the larger expected shortage."""
    skus = [{"sku_id": "LIKELY"}, {"sku_id": "UNLIKELY"}]
    small = [3, 5, 7, 7, 7, 9, 11] * 4                   # lead-time demand ~49
    big = [8, 14, 20, 20, 20, 26, 32] * 4                # lead-time demand ~140
    sales = []
    for i, (q_s, q_b) in enumerate(zip(small, big)):
        d = (pd.Timestamp(WINDOW_START) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        sales.append({"sku_id": "LIKELY", "transaction_date": d, "quantity_sold": q_s})
        sales.append({"sku_id": "UNLIKELY", "transaction_date": d, "quantity_sold": q_b})
    stock = [{"sku_id": "LIKELY", "stock_on_hand": 45},      # just under its 49
             {"sku_id": "UNLIKELY", "stock_on_hand": 145}]   # comfortably over its 140
    db = _make_db(tmp_path, skus, sales, stock)
    out, _ = _score(db, _write_config(tmp_path), ["LIKELY", "UNLIKELY"])

    p_likely = _row(out, "LIKELY")["stockout_probability"]
    p_unlikely = _row(out, "UNLIKELY")["stockout_probability"]
    s_likely = _row(out, "LIKELY")["expected_shortage_units"]
    s_unlikely = _row(out, "UNLIKELY")["expected_shortage_units"]

    # preconditions: a real probability gap, with exposure pointing the OTHER way
    assert round(p_likely, 6) != round(p_unlikely, 6)
    assert p_likely > p_unlikely
    assert s_unlikely > s_likely, "fixture invalid: exposure should favour UNLIKELY"

    ranked = sr.rank_by_stockout_risk(out)
    assert list(ranked["sku"]) == ["LIKELY", "UNLIKELY"]


def test_unscored_rows_always_rank_last(tmp_path):
    skus = [{"sku_id": "GOOD"}, {"sku_id": "NOSTOCK"}]
    sales = _daily("GOOD", WINDOW_START, 28, 1) + _daily("NOSTOCK", WINDOW_START, 28, 50)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "GOOD", "stock_on_hand": 900}])
    cfg = _write_config(tmp_path)
    out, _ = _score(db, cfg, ["GOOD", "NOSTOCK"])
    ranked = sr.rank_by_stockout_risk(out)
    # NOSTOCK has far higher demand but no stock reading, so it must not outrank GOOD.
    assert list(ranked["sku"]) == ["GOOD", "NOSTOCK"]
    assert bool(ranked.iloc[-1]["risk_scored"]) is False


def test_rank_is_deterministic_regardless_of_input_order(tmp_path):
    skus = [{"sku_id": f"S{i}"} for i in range(6)]
    sales, stock = [], []
    for i in range(6):
        sales += _daily(f"S{i}", WINDOW_START, 28, 2)
        stock.append({"sku_id": f"S{i}", "stock_on_hand": 14})       # identical -> full tie
    db = _make_db(tmp_path, skus, sales, stock)
    cfg = _write_config(tmp_path)
    a = sr.rank_by_stockout_risk(_score(db, cfg, [f"S{i}" for i in range(6)])[0])
    b = sr.rank_by_stockout_risk(_score(db, cfg, [f"S{i}" for i in reversed(range(6))])[0])
    assert list(a["sku"]) == list(b["sku"]) == [f"S{i}" for i in range(6)]   # sku asc


# ── contracts, safety ─────────────────────────────────────────────────────────────────
def test_output_column_contract(tmp_path):
    skus = [{"sku_id": "S1"}]
    sales = _daily("S1", WINDOW_START, 28, 2)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "S1", "stock_on_hand": 10}])
    out, _ = _score(db, _write_config(tmp_path), ["S1"])
    assert list(out.columns) == sr.SELECTION_RISK_COLUMNS


def test_empty_input_returns_empty_contract(tmp_path):
    db = _make_db(tmp_path, [{"sku_id": "S1"}], [], [])
    out, meta = _score(db, _write_config(tmp_path), [])
    assert list(out.columns) == sr.SELECTION_RISK_COLUMNS
    assert out.empty and meta["scored"] == 0 and meta["candidates"] == 0


def test_every_input_sku_gets_exactly_one_row(tmp_path):
    skus = [{"sku_id": "A"}, {"sku_id": "B"}, {"sku_id": "C"}]
    sales = _daily("A", WINDOW_START, 28, 2)             # B and C never sold
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "A", "stock_on_hand": 5},
                                          {"sku_id": "B", "stock_on_hand": 5}])
    out, meta = _score(db, _write_config(tmp_path), ["A", "B", "C"])
    assert len(out) == 3
    assert list(out["sku"]) == ["A", "B", "C"]           # input order preserved pre-rank
    assert meta["candidates"] == 3


def test_scan_never_writes_to_the_warehouse(tmp_path):
    skus = [{"sku_id": "S1"}]
    sales = _daily("S1", WINDOW_START, 28, 2)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "S1", "stock_on_hand": 10}])
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    _score(db, _write_config(tmp_path), ["S1"])
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_missing_warehouse_raises(tmp_path):
    with pytest.raises(sr.SelectionRiskError, match="Warehouse not found"):
        sr.score_stockout_risk(tmp_path / "nope.db", ["S1"], selection_cutoff=CUTOFF,
                               config_path=_write_config(tmp_path))


# ── integration with dynamic_selection ────────────────────────────────────────────────
def _integration_db(tmp_path: Path) -> Path:
    """A category where the risk ranking and the units ranking DISAGREE.

    BESTSELLER moves the most units but is amply stocked; SLEEPER sells modestly and is
    out of stock. Ranking by units picks BESTSELLER first; ranking by risk must not.
    """
    skus = [{"sku_id": "BESTSELLER"}, {"sku_id": "SLEEPER"}, {"sku_id": "MIDDLE"}]
    sales = (_daily("BESTSELLER", WINDOW_START, 28, 50)
             + _daily("SLEEPER", WINDOW_START, 28, 3)
             + _daily("MIDDLE", WINDOW_START, 28, 10))
    stock = [{"sku_id": "BESTSELLER", "stock_on_hand": 9000},   # deep cover, below sentinel
             {"sku_id": "SLEEPER", "stock_on_hand": 0},         # lt demand 21, shortage 21
             {"sku_id": "MIDDLE", "stock_on_hand": 60}]         # lt demand 70, shortage 10
    return _make_db(tmp_path, skus, sales, stock)


def test_units_metric_unchanged_by_the_new_code_path(tmp_path):
    db = _integration_db(tmp_path)
    sel, _ = ds.select_top_skus(db, "Groceries", 3, CUTOFF, 28)
    assert list(sel["sku"]) == ["BESTSELLER", "MIDDLE", "SLEEPER"]
    assert list(sel.columns) == ds.SELECTION_COLUMNS       # no risk columns leak in


def test_risk_metric_reorders_and_adds_columns(tmp_path, monkeypatch):
    db = _integration_db(tmp_path)
    monkeypatch.setattr(sr, "CONFIG_PATH", _write_config(tmp_path))
    sel, warnings, meta = ds.select_top_skus_detailed(
        db, "Groceries", 3, CUTOFF, 28, ds.METRIC_STOCKOUT_RISK)
    # Risk order inverts the units order entirely: the bestseller has the deepest cover.
    assert list(sel["sku"]) == ["SLEEPER", "MIDDLE", "BESTSELLER"]
    assert list(sel.columns) == ds.SELECTION_COLUMNS + ds.RISK_SELECTION_COLUMNS
    assert list(sel["rank"]) == [1, 2, 3]
    assert meta["ranking_metric"] == ds.METRIC_STOCKOUT_RISK
    assert meta["eligible_count"] == 3


def test_risk_metric_scores_full_pool_not_a_units_shortlist(tmp_path, monkeypatch):
    """Top-N by risk must consider every eligible SKU. If the pool were pre-trimmed by
    units, a low-volume at-risk SKU could never surface."""
    skus = [{"sku_id": f"BULK{i:03d}"} for i in range(30)] + [{"sku_id": "QUIET"}]
    sales, stock = [], []
    for i in range(30):
        sales += _daily(f"BULK{i:03d}", WINDOW_START, 28, 100)       # huge volume
        stock.append({"sku_id": f"BULK{i:03d}", "stock_on_hand": 900000})
    sales += _daily("QUIET", WINDOW_START, 28, 1)                    # lowest volume of all
    stock.append({"sku_id": "QUIET", "stock_on_hand": 0})            # ...but out of stock
    db = _make_db(tmp_path, skus, sales, stock)
    monkeypatch.setattr(sr, "CONFIG_PATH", _write_config(tmp_path))
    sel, _, _ = ds.select_top_skus_detailed(db, "Groceries", 1, CUTOFF, 28,
                                            ds.METRIC_STOCKOUT_RISK)
    assert list(sel["sku"]) == ["QUIET"]


def test_unscorable_sku_never_fills_a_top_n_slot(tmp_path, monkeypatch):
    skus = [{"sku_id": "GOOD"}, {"sku_id": "NOSTOCK"}]
    sales = _daily("GOOD", WINDOW_START, 28, 1) + _daily("NOSTOCK", WINDOW_START, 28, 99)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "GOOD", "stock_on_hand": 5}])
    monkeypatch.setattr(sr, "CONFIG_PATH", _write_config(tmp_path))
    sel, warnings, meta = ds.select_top_skus_detailed(
        db, "Groceries", 2, CUTOFF, 28, ds.METRIC_STOCKOUT_RISK)
    assert list(sel["sku"]) == ["GOOD"]                    # 2 requested, only 1 rankable
    assert meta["scored"] == 1
    assert any("risk-scored" in w for w in warnings)


def test_all_unscorable_raises_eligibility_error(tmp_path, monkeypatch):
    skus = [{"sku_id": "A"}, {"sku_id": "B"}]
    sales = _daily("A", WINDOW_START, 28, 2) + _daily("B", WINDOW_START, 28, 2)
    db = _make_db(tmp_path, skus, sales, [])               # no stock rows at all
    monkeypatch.setattr(sr, "CONFIG_PATH", _write_config(tmp_path))
    with pytest.raises(ds.CategoryEligibilityError, match="stockout risk"):
        ds.select_top_skus_detailed(db, "Groceries", 2, CUTOFF, 28, ds.METRIC_STOCKOUT_RISK)


def test_unsupported_metric_still_rejected(tmp_path):
    db = _integration_db(tmp_path)
    with pytest.raises(ds.UnsupportedRankingMetricError):
        ds.select_top_skus(db, "Groceries", 2, CUTOFF, 28, "vibes")


def test_post_cutoff_warning_surfaces_through_selection(tmp_path, monkeypatch):
    skus = [{"sku_id": "S1"}]
    sales = _daily("S1", WINDOW_START, 28, 2)
    db = _make_db(tmp_path, skus, sales, [{"sku_id": "S1", "stock_on_hand": 5}],
                  snapshot_date="2026-08-05")
    monkeypatch.setattr(sr, "CONFIG_PATH", _write_config(tmp_path))
    _sel, warnings, meta = ds.select_top_skus_detailed(
        db, "Groceries", 1, CUTOFF, 28, ds.METRIC_STOCKOUT_RISK)
    assert meta["stock_is_post_cutoff"] is True
    assert any("postdates the selection cutoff" in w for w in warnings)
