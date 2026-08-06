"""Inventory Planning Agent — pilot dashboard prototype.

Naheed AI Explorers 2026 | Daily ecommerce-only pilot | 30 SKUs | naheed_web

Presentation-grade Streamlit dashboard over the pilot's real demand history
and synthetic inventory/stockout artifacts. Built to degrade gracefully:
every optional dataset (synthetic scenarios, model outputs, evaluation
scorecards) is allowed to be absent and shows a clear empty state instead
of crashing.

Run from the repository root:
    streamlit run dashboard/app.py
"""

import base64
import calendar
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from styles import (COLORS, CATEGORICAL, DONUT_COLORS, DONUT_HOVER, STATUS_COLORS, TONES, TONE_CYCLE,
                    CUSTOM_CSS, plotly_layout, style_axes, icon_svg)
import run_service as rs   # Phase 5 run-aware backend service (framework-free, testable)
import export_utils as eu  # one reusable in-memory CSV/Excel/PDF export menu
import deadstock_analysis as da  # standalone ecommerce deadstock (inventory-inactivity) scan

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_SYNTHETIC = BASE_DIR / "data" / "synthetic"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGO_PATH = Path(__file__).resolve().parent / "images" / "Naheed.png"

SYNTHETIC_NOTICE = (
    "Inventory, stockouts and replenishment shown here are simulation-based pilot "
    "results, not observed historical Naheed inventory records."
)
INVENTORY_PAGE_WARNING = (
    "This page uses a synthetic baseline inventory snapshot and assumed "
    "replenishment parameters."
)
PROTOTYPE_FORECAST_LABEL = (
    "Prototype baseline preview — recent 14-day mean, not the final selected model."
)

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SCENARIO_NAMES = [
    "Baseline", "Low Opening Stock", "Demand Spike", "Supplier Delay",
    "Long Lead Time", "High MOQ", "Promotion Peak",
]

# --------------------------------------------------------------------------
# Page config + CSS
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Inventory Planning Agent",
    page_icon="box",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Generic formatting + UI helpers
# --------------------------------------------------------------------------
def format_currency(x, decimals=0):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    try:
        return f"PKR {float(x):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def format_number(x, decimals=0):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    try:
        return f"{float(x):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def format_percentage(fraction, decimals=1):
    """fraction is expected on a 0-1 scale."""
    if fraction is None or (isinstance(fraction, float) and pd.isna(fraction)):
        return "—"
    try:
        return f"{float(fraction) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


# All custom-HTML helpers below emit SINGLE-LINE strings (no leading indentation)
# so Streamlit's Markdown parser never treats a line as a 4-space code block —
# that indentation bug was the source of leaked raw </div> tags.

def render_kpi_card(label, value, icon="dot", sub="", tone="teal", compact=False):
    t = TONES.get(tone, TONES["teal"])
    sub_html = f'<div class="ipa-kpi-sub">{sub}</div>' if sub else ""
    card_cls = "ipa-card ipa-card--compact" if compact else "ipa-card"
    html = (
        f'<div class="{card_cls}" style="--accent:{t["fg"]};">'
        '<div class="ipa-kpi-top">'
        f'<div class="ipa-kpi-icon" style="background:{t["bg"]};color:{t["fg"]};">{icon_svg(icon)}</div>'
        f'<div class="ipa-kpi-label">{label}</div>'
        '</div>'
        f'<div class="ipa-kpi-value">{value}</div>'
        f'{sub_html}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_row(kpis, n_cols=None, compact=False):
    """kpis: list of dicts with keys label, value, icon, sub. Tones auto-cycle unless given."""
    n_cols = n_cols or len(kpis)
    cols = st.columns(n_cols)
    for i, (col, kpi) in enumerate(zip(cols, kpis)):
        kpi = {**kpi}
        kpi.setdefault("tone", TONE_CYCLE[i % len(TONE_CYCLE)])
        kpi.setdefault("compact", compact)
        with col:
            render_kpi_card(**kpi)


def render_chart(fig, title=None, sub=None):
    """Wrap a Plotly figure in a bordered card with an optional native title/subtitle."""
    with st.container(border=True):
        if title:
            st.markdown(f'<div class="ipa-card-title">{title}</div>', unsafe_allow_html=True)
        if sub:
            st.markdown(f'<div class="ipa-card-sub">{sub}</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


_MP_COUNTER = {"n": 0}


def metric_panel(title, rows, sub=None):
    """A bordered card of label -> value rows (compact profile / summary panel).

    Rendered as one flex block so the rows spread to fill the card height — when
    it sits beside a taller chart card, the equal-height CSS stretches this card
    and the divider-separated rows distribute evenly (matches the approved look).
    """
    _MP_COUNTER["n"] += 1
    slug = "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-")
    # counter keeps the container key unique even when the same panel title is
    # rendered several times in one run (e.g. one "Product Profile" per selected SKU).
    slug = f"{slug}-{_MP_COUNTER['n']}"
    sub_html = f'<div class="ipa-card-sub">{sub}</div>' if sub else ""
    body = "".join(
        f'<div class="ipa-mrow"><span class="l">{lab}</span><span class="v">{val}</span></div>'
        for lab, val in rows
    )
    with st.container(border=True, key=f"ipa-mp-{slug}"):
        st.markdown(
            f'<div class="ipa-mpanel"><div class="ipa-card-title">{title}</div>{sub_html}'
            f'<div class="ipa-mbody">{body}</div></div>',
            unsafe_allow_html=True,
        )


def render_insight_row(insights, icons=None):
    """Rule-based insights as a horizontal row of equal-height cards (3 per row)."""
    icons = icons or ["trending-up", "circle-dashed", "tag", "target", "box", "alert-triangle"]
    for start in range(0, len(insights), 3):
        chunk = insights[start:start + 3]
        chunk_icons = icons[start:start + 3] + ["sparkle"] * 3
        cols = st.columns(len(chunk))
        for col, text, icon in zip(cols, chunk, chunk_icons):
            with col:
                st.markdown(
                    f'<div class="ipa-insight"><div class="ico">{icon_svg(icon, 22)}</div><div class="txt">{text}</div></div>',
                    unsafe_allow_html=True,
                )


def section_title(title, sub=None):
    st.markdown(f'<div class="ipa-section-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="ipa-section-sub">{sub}</div>', unsafe_allow_html=True)


def empty_state(title, message, icon="folder"):
    html = (
        '<div class="ipa-banner-empty">'
        f'<div class="ipa-empty-icon">{icon_svg(icon, 30)}</div>'
        f'<div class="ipa-empty-title">{title}</div>'
        f'<div class="ipa-empty-msg">{message}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


_BANNER_ICON = {"info": "info", "success": "check-circle", "synthetic": "flask"}


def info_banner(message, kind="info", icon=None):
    ic = icon_svg(icon or _BANNER_ICON.get(kind, "info"), 15)
    html = (
        f'<div class="ipa-banner ipa-banner-{kind}">'
        f'<span style="display:inline-flex;vertical-align:middle;margin-right:7px;">{ic}</span>{message}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def synthetic_warning(message=None):
    msg = message or SYNTHETIC_NOTICE
    info_banner(f'<strong>Synthetic:</strong> {msg}', kind="synthetic")


def render_page_header(title, subtitle, badges=None):
    """Brand page header: two-tone title (last word red), subtitle, status badge + refresh meta.

    badges: optional list of (icon_key, text) tuples. When given, renders light
    outlined chips instead of the default dark status pill.
    """
    parts = title.rsplit(" ", 1)
    if len(parts) == 2:
        title_html = f'{parts[0]} <span class="accent">{parts[1]}</span>'
    else:
        title_html = f'<span class="accent">{title}</span>'
    if badges:
        chips = "".join(
            f'<span class="ipa-chip">{icon_svg(ic, 14)}{txt}</span>' for ic, txt in badges
        )
        badge_html = f'<div class="ipa-chip-row">{chips}</div>'
    else:
        badge_html = '<div class="ipa-status-badge">Pilot · 30 SKUs · naheed_web</div>'
    html = (
        '<div class="ipa-header">'
        '<div>'
        f'<h1>{title_html}</h1>'
        f'<div class="ipa-sub">{subtitle}</div>'
        f'{badge_html}'
        '</div>'
        '<div class="ipa-meta">'
        f'<div class="refresh-ico">{icon_svg("refresh", 15)}</div>'
        f'<div class="txt"><div class="k">Last refresh</div><div class="v">{refresh_display}</div></div>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _status_cell_css(val):
    for name, (bg, fg) in {
        "Critical": ("#FAE7E9", COLORS["red"]),
        "Reorder Now": ("#FBF0DC", COLORS["amber"]),
        "Watch": ("#E5EDFD", COLORS["blue"]),
        "Healthy": ("#E4F5EC", COLORS["success"]),
    }.items():
        if isinstance(val, str) and name in val:
            return f"background-color:{bg}; color:{fg}; font-weight:700;"
    return ""


# Risk-tier cell colours for the complete datasets (critical / high / medium / low all
# visually distinct — colour always accompanies the tier TEXT, never colour alone).
RISK_TIER_CELL = {
    "critical": ("#FAE7E9", COLORS["red"]),
    "high": ("#FBE3D6", "#C2410C"),
    "medium": ("#FBF0DC", COLORS["amber"]),
    "watch": ("#E5EDFD", COLORS["blue"]),
    "low": ("#E4F5EC", COLORS["success"]),
    "healthy": ("#E3F3F1", COLORS["teal"]),
    "unknown": ("#EAEEF3", COLORS["slate"]),
}
REORDER_ACTION_CELL = {
    "order_now": ("#FAE7E9", COLORS["red"]),
    "manual_review": ("#FBF0DC", COLORS["amber"]),
    "vendor_follow_up": ("#E5EDFD", COLORS["blue"]),
    "monitor": ("#E3F3F1", COLORS["teal"]),
    "no_order": ("#EAEEF3", COLORS["slate"]),
}


def _risk_tier_cell_css(val):
    bg_fg = RISK_TIER_CELL.get(str(val).strip().lower())
    return f"background-color:{bg_fg[0]}; color:{bg_fg[1]}; font-weight:700;" if bg_fg else ""


def _reorder_action_cell_css(val):
    bg_fg = REORDER_ACTION_CELL.get(str(val).strip().lower())
    return f"background-color:{bg_fg[0]}; color:{bg_fg[1]}; font-weight:700;" if bg_fg else ""


# --------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------
def _load_path(path: Path, required_cols=None):
    """Returns (data, status). status is None on success, else a short reason."""
    if not path.exists():
        return None, "not_found"
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), None
        else:
            return None, "unsupported_format"
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
        return None, f"error: {exc}"

    if df is None or df.empty:
        return None, "empty"
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"missing_columns: {missing}"
    return df, None


def load_logo_b64():
    if LOGO_PATH.exists():
        # Use mtime as cache key so replacing the file busts the cache
        _mtime = LOGO_PATH.stat().st_mtime
        return _load_logo_bytes(_mtime)
    return None

@st.cache_data(show_spinner=False)
def _load_logo_bytes(_mtime):
    return base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")


@st.cache_data(show_spinner=False)
def load_manifest():
    return _load_path(DATA_PROCESSED / "pilot_manifest.json")


@st.cache_data(show_spinner=False)
def load_model_panel():
    return _load_path(DATA_PROCESSED / "model_panel.parquet")


@st.cache_data(show_spinner=False)
def load_inventory_context():
    return _load_path(DATA_PROCESSED / "inventory_context.parquet")


@st.cache_data(show_spinner=False)
def load_forecast_features():
    for name in ("forecast_features.parquet", "forecast_frame.parquet"):
        p = DATA_PROCESSED / name
        if p.exists():
            data, status = _load_path(p)
            return data, status, name
    return None, "not_found", "forecast_features.parquet"


@st.cache_data(show_spinner=False)
def load_stockout_scenarios():
    return _load_path(DATA_SYNTHETIC / "stockout_scenarios.parquet")


@st.cache_data(show_spinner=False)
def load_replenishment_events():
    return _load_path(DATA_SYNTHETIC / "replenishment_events.parquet")


@st.cache_data(show_spinner=False)
def load_simulation_parameters():
    return _load_path(DATA_SYNTHETIC / "simulation_parameters.json")


def _classify_model_name(stem: str) -> str:
    low = stem.lower()
    if "lightgbm" in low or "lgbm" in low:
        return "LightGBM"
    if "holt_winters" in low or "holtwinters" in low:
        return "Holt-Winters"
    if "holt" in low or "ses" in low:
        return "SES / Holt"
    if "moving_average" in low or "baseline" in low:
        return "Baseline (Moving Average)"
    return stem.replace("_", " ").title()


EVAL_KEYWORDS = ("wape", "mase", "scorecard", "evaluation", "comparison", "backtest")


@st.cache_data(show_spinner=False)
def discover_outputs(outputs_dir_str=None):
    """Scan an outputs/ dir once and split files into forecast outputs vs evaluation outputs.
    Defaults to the legacy global outputs/; run mode passes the active run's outputs dir."""
    out_dir = Path(outputs_dir_str) if outputs_dir_str else OUTPUTS_DIR
    forecasts, evaluations = {}, {}
    if out_dir.exists():
        for p in sorted(list(out_dir.glob("*.csv")) + list(out_dir.glob("*.parquet"))):
            stem_low = p.stem.lower()
            data, status = _load_path(p)
            if data is None:
                continue
            if any(k in stem_low for k in EVAL_KEYWORDS):
                evaluations[p.stem] = data
            else:
                label = _classify_model_name(p.stem)
                if "date" in data.columns:
                    data = data.copy()
                    data["date"] = pd.to_datetime(data["date"])
                forecasts[label] = {"df": data, "path": str(p.relative_to(BASE_DIR)).replace("\\", "/")}
    return forecasts, evaluations


@st.cache_data(show_spinner=False)
def build_sku_meta(mp_df):
    if mp_df is None:
        return None
    agg = dict(category=("category", "last"), brand=("brand", "last"), product_id=("product_id", "last"))
    if "sku_name" in mp_df.columns:            # real product name when the v4 field is present
        agg["sku_name"] = ("sku_name", "last")
    return mp_df.sort_values("date").groupby("sku", as_index=False).agg(**agg)


def _clean_text(v):
    """Return a trimmed string, or '' for null/blank/'None' placeholder values."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return "" if s == "" or s.lower() == "none" else s


def _sku_display(name, brand, sku):
    """Best display name (no SKU code): 'name — brand', name, brand, then SKU.

    The brand is only appended when it is not already clearly contained in the
    product name, so e.g. 'Nestle Nesvita ...' is not turned into
    'Nestle Nesvita ... — Nestle'.
    """
    if name:
        return f"{name} — {brand}" if brand and brand.lower() not in name.lower() else name
    return brand or str(sku)


def _label_meta(mp_df):
    """Grouped (brand, sku_name) per sku, tolerant of older files without sku_name."""
    has_name = "sku_name" in mp_df.columns
    agg = {"brand": ("brand", "last")}
    if has_name:
        agg["sku_name"] = ("sku_name", "last")
    meta = mp_df.sort_values("date").groupby("sku").agg(**agg)
    return meta, has_name


def build_sku_labels(mp_df):
    """Map each sku -> a searchable display label that ends with the SKU code.

    Uses the real product name (`sku_name`) when the v4 model_panel provides it,
    with this fallback order (SKU code always appended so codes stay searchable):
      a. "<sku_name> — <brand> (SKU)"  when both exist and brand is not already in the name
      b. "<sku_name> (SKU)"            when only the name is usable
      c. "<brand> (SKU)"               when there is no name (older model_panel files)
      d. "SKU"                         when nothing descriptive is available
    Never crashes on null/blank sku_name; SKU remains the stable internal value.
    """
    if mp_df is None:
        return {}
    meta, has_name = _label_meta(mp_df)
    labels = {}
    for sku, row in meta.iterrows():
        name = _clean_text(row["sku_name"]) if has_name else ""
        core = _sku_display(name, _clean_text(row["brand"]), sku)
        labels[sku] = f"{core} ({sku})" if core and core != str(sku) else str(sku)
    return labels


def build_sku_names(mp_df):
    """Map each sku -> the raw product name only (no brand suffix, no code), for
    places that show the SKU/brand separately (profile rows, action-queue table,
    comparison-label modes). Falls back sku_name -> brand -> SKU, and tolerates
    older files lacking sku_name."""
    if mp_df is None:
        return {}
    meta, has_name = _label_meta(mp_df)
    names = {}
    for sku, row in meta.iterrows():
        name = _clean_text(row["sku_name"]) if has_name else ""
        names[sku] = name or _clean_text(row["brand"]) or str(sku)
    return names


CMP_LABEL_MODES = ["Name + SKU", "Product name", "SKU"]


def comparison_display_label(sku, mode, selected_skus=None):
    """Display text for one product under the chosen 'Comparison labels' mode.

    Display-only — SKU is always the stable internal value for filtering, joins,
    grouping and chart data; this affects only legends, hover labels, chart titles
    and selected-product headings.
      - "SKU":          the SKU code only
      - "Product name":  the real product name (falls back brand -> SKU). If two of
                         the SELECTED products share the same name, the SKU is
                         appended to those so the chart stays unambiguous.
      - "Name + SKU":    "<product name> (SKU)"   (default)
    Names come from SKU_NAMES (built once from model_panel: sku_name -> brand -> SKU);
    sku_name is never used as a join/group key.
    """
    if mode == "SKU":
        return str(sku)
    name = SKU_NAMES.get(sku) or str(sku)   # blank/missing name -> SKU
    if mode == "Product name":
        if selected_skus:
            same = [s for s in selected_skus if SKU_NAMES.get(s, str(s)) == name]
            if len(same) > 1:                       # duplicate names among the selection
                return f"{name} ({sku})"
        return name
    # default: "Name + SKU"
    return f"{name} ({sku})" if name and name != str(sku) else str(sku)


@st.cache_data(show_spinner=False)
def build_prototype_forecast(mp_df, horizon_days=14):
    """Trailing 14-day mean per SKU, held flat forward — a display-only baseline."""
    if mp_df is None:
        return pd.DataFrame(columns=["sku", "date", "y_pred"])
    last_date = mp_df["date"].max()
    cutoff = last_date - pd.Timedelta(days=13)
    recent = mp_df[mp_df["date"] >= cutoff]
    means = recent.groupby("sku")["units_observed"].mean()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)]
    rows = [
        {"sku": sku, "date": d, "y_pred": round(mean_val, 2)}
        for sku, mean_val in means.items()
        for d in future_dates
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def apply_filters(df, skus=None, categories=None, brands=None, date_range=None):
    if df is None or df.empty:
        return df
    out = df
    if skus and "sku" in out.columns:
        out = out[out["sku"].isin(skus)]
    if categories and "category" in out.columns:
        out = out[out["category"].isin(categories)]
    if brands and "brand" in out.columns:
        out = out[out["brand"].isin(brands)]
    if date_range and len(date_range) == 2:
        # one production helper for every historical date filter (see run_service)
        out = rs.filter_historical_frame(out, date_from=date_range[0], date_to=date_range[1])
    return out


def classify_reorder_status(row):
    if (row.get("stock_on_hand", 1) <= 0) or (row.get("days_of_cover", 99) <= 2):
        return "Critical"
    if row.get("recommended_order_quantity", 0) > 0:
        return "Reorder Now"
    if row.get("days_of_cover", 99) <= row.get("lead_time_days", 0):
        return "Watch"
    return "Healthy"


# --------------------------------------------------------------------------
# Deterministic executive insights (rule-based, no external LLM)
# --------------------------------------------------------------------------
def generate_insights(mp_f, inv_f, manifest):
    insights = []
    if mp_f is not None and not mp_f.empty:
        cat_totals = mp_f.groupby("category")["units_observed"].sum().sort_values(ascending=False)
        if len(cat_totals) and cat_totals.sum() > 0:
            top_cat = cat_totals.index[0]
            share = cat_totals.iloc[0] / cat_totals.sum() * 100
            insights.append(f"<b>{top_cat}</b> contributes {share:.1f}% of pilot demand in the selected filters.")

        zero_rate = (mp_f["units_observed"] == 0).mean() * 100
        insights.append(f"{zero_rate:.1f}% of demand-history rows in the current view contain zero sales.")

        promo_rate = mp_f["on_promo"].mean() * 100
        insights.append(f"{promo_rate:.1f}% of rows in the current view were on an active promotion.")

        sku_totals = mp_f.groupby("sku")["units_observed"].sum().sort_values(ascending=False)
        if len(sku_totals):
            insights.append(
                f"SKU <b>{sku_totals.index[0]}</b> leads the current view with "
                f"{sku_totals.iloc[0]:,.0f} total units sold."
            )

    if inv_f is not None and not inv_f.empty:
        n_reorder = int((inv_f["recommended_order_quantity"] > 0).sum())
        n_total = inv_f["sku"].nunique()
        insights.append(
            f"{n_reorder} of {n_total} SKUs in view have a reorder recommendation under the baseline simulation."
        )
        n_critical = int((inv_f["days_of_cover"] <= 2).sum())
        if n_critical:
            insights.append(
                f"{n_critical} SKU(s) show 2 days of cover or less under the synthetic baseline snapshot — "
                "flagged Critical on the Inventory & Reorder page."
            )

    return insights


# --------------------------------------------------------------------------
# Phase 5 — resolve the ACTIVE data context (a completed run, or the legacy pilot)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_path_cached(path_string, mtime_ns):
    """Path + mtime keyed loader — a completed run loads correctly and edits bust the cache."""
    return _load_path(Path(path_string))


def load_active(path):
    """Load a context artifact (parquet/csv/json) or return (None, reason)."""
    if not path:
        return None, "not_found"
    p = Path(path)
    if not p.exists():
        return None, "not_found"
    return load_path_cached(str(p), p.stat().st_mtime_ns)


def _discover_runs_fresh():
    try:
        return rs.discover_runs()
    except Exception:            # never let a bad run tree crash the dashboard
        return []


RUNS_ALL = _discover_runs_fresh()
COMPLETED_RUNS = [r for r in RUNS_ALL if r.get("is_completed")]
LEGACY_LABEL = "Legacy fixed pilot"
# Compact, UNIQUE selectbox labels ('31 Jul · Groceries & Pets · Top 10 · ✓'); the full
# label + technical detail live in the tooltip and the "Run details" expander.
_short_by_id = rs.build_short_labels(COMPLETED_RUNS)
_run_labels = {_short_by_id[r["run_id"]]: r for r in COMPLETED_RUNS}

# The sidebar "Data source" selectbox writes this key; read the prior value now (before the
# widget is drawn) so the loaders below pick the right paths on this rerun. A pending value
# set by "Activate" on the Forecast Runs page is applied here, before the widget exists.
_pending_ds = st.session_state.pop("_pending_data_source", None)
if _pending_ds is not None:
    st.session_state["data_source_choice"] = _pending_ds
_ds_choice = st.session_state.get("data_source_choice")
if _ds_choice not in ({LEGACY_LABEL} | set(_run_labels)):
    _ds_choice = next(iter(_run_labels), LEGACY_LABEL)   # default: newest completed, else legacy

ACTIVE_RUN = None
CTX = rs.legacy_context()
DATA_MODE = "legacy"
if _ds_choice != LEGACY_LABEL and _ds_choice in _run_labels:
    try:
        ACTIVE_RUN = _run_labels[_ds_choice]
        CTX = rs.resolve_run_context(ACTIVE_RUN)
        DATA_MODE = "run"
    except rs.RunContextError:
        ACTIVE_RUN, CTX, DATA_MODE = None, rs.legacy_context(), "legacy"

# --------------------------------------------------------------------------
# Load all data once from the active context
# --------------------------------------------------------------------------
manifest, manifest_status = load_active(CTX["pilot_manifest"])
mp_raw, mp_status = load_active(CTX["model_panel"])
inv_raw, inv_status = load_active(CTX["inventory_context"])
if DATA_MODE == "run":
    ff_raw, ff_status = load_active(CTX["forecast_frame"])
    ff_name = "forecast_frame.parquet"
else:
    ff_raw, ff_status, ff_name = load_forecast_features()
# Synthetic scenario data + free-form outputs scanning remain legacy-global (Stockout Lab).
stockout_raw, stockout_status = load_stockout_scenarios()
replen_raw, replen_status = load_replenishment_events()
simparams_raw, simparams_status = load_simulation_parameters()
# Phase B decision artifacts (run-scoped; absent on legacy or pre-Phase-B runs)
if DATA_MODE == "run" and CTX.get("has_stockout_risk"):
    risk_raw, risk_status = load_active(CTX["stockout_risk"])
else:
    risk_raw, risk_status = None, "unavailable"
if DATA_MODE == "run" and CTX.get("has_stockout_trajectory"):
    traj_raw, traj_status = load_active(CTX["stockout_trajectory"])
else:
    traj_raw, traj_status = None, "unavailable"
# Phase C decision artifacts (run-scoped; absent on legacy or pre-Phase-C runs)
if DATA_MODE == "run" and CTX.get("reorder_available"):
    reorder_raw, reorder_status = load_active(CTX["reorder_recommendations"])
    reorder_summary_raw, reorder_summary_status = load_active(CTX["reorder_summary"])
else:
    reorder_raw, reorder_status = None, "unavailable"
    reorder_summary_raw, reorder_summary_status = None, "unavailable"
outputs_forecasts, outputs_evaluations = discover_outputs(
    str(CTX["outputs_dir"]) if DATA_MODE == "run" else None)

if mp_raw is not None:
    mp_raw = mp_raw.copy()
    mp_raw["date"] = pd.to_datetime(mp_raw["date"])

sku_meta = build_sku_meta(mp_raw)

inv_joined = inv_raw
if inv_raw is not None and sku_meta is not None:
    inv_joined = inv_raw.merge(sku_meta[["sku", "category", "brand"]], on="sku", how="left")

# Dynamic page-header badges from the active context (legacy keeps the pilot labels).
def _active_badges():
    if DATA_MODE == "run" and ACTIVE_RUN is not None:
        n = ACTIVE_RUN.get("selected_sku_count") or (mp_raw["sku"].nunique() if mp_raw is not None else "—")
        cat = ACTIVE_RUN.get("category") or "—"
        return [("box", f"{n} selected SKUs"), ("folder", str(cat)), ("globe", "naheed_web")]
    return [("box", "30 pilot SKUs"), ("globe", "naheed_web")]


ACTIVE_BADGES = _active_badges()


def _pretty_date(value, with_time=False):
    if not value:
        return "—"
    try:
        dt = pd.to_datetime(value)
        return dt.strftime("%d %b %Y · %I:%M %p") if with_time else dt.strftime("%d %b %Y")
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------
# Header meta (consumed by render_page_header inside each page)
# --------------------------------------------------------------------------
as_of_display = _pretty_date((manifest or {}).get("as_of_date")) if manifest else "—"
refresh_display = _pretty_date((manifest or {}).get("generated_at"), with_time=True) if manifest else "—"

# --------------------------------------------------------------------------
# Sidebar: brand + navigation + compact filters
# --------------------------------------------------------------------------
_logo_b64 = load_logo_b64()
if _logo_b64:
    st.sidebar.markdown(
        f'<div class="ipa-brand"><div class="logo">'
        f'<img src="data:image/png;base64,{_logo_b64}" alt="Naheed" class="ipa-logo-img" />'
        '</div></div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<div class="ipa-brand"><div class="logo"><span class="n">Naheed<sup>®</sup></span></div></div>',
        unsafe_allow_html=True,
    )

# ---- Data source selector (which run drives every page this rerun) ----
with st.sidebar.container(key="ipa-datasource"):
    st.markdown('<div class="ipa-nav-label">Data source</div>', unsafe_allow_html=True)
    _ds_options = [LEGACY_LABEL] + list(_run_labels)
    _ds_index = _ds_options.index(_ds_choice) if _ds_choice in _ds_options else 0
    _ds_pick = st.selectbox(
        "Data source", options=_ds_options, index=_ds_index,
        key="data_source_choice", label_visibility="collapsed",
        help=(rs.format_run_label_full(ACTIVE_RUN) if (DATA_MODE == "run" and ACTIVE_RUN)
              else "Legacy fixed 30-SKU pilot (data/processed)"))
    if DATA_MODE == "run" and ACTIVE_RUN is not None:
        _op = ACTIVE_RUN.get("operational_model") or "—"
        _sym = rs.STATUS_SYMBOLS.get(str(ACTIVE_RUN.get("status")), "•")
        st.markdown(
            f'<div class="ipa-src"><div class="ipa-src-row">'
            f'<span class="ipa-src-chip">{ACTIVE_RUN.get("selected_sku_count", "—")} SKUs</span>'
            f'<span class="ipa-src-chip">{ACTIVE_RUN.get("category") or "—"}</span></div>'
            f'<div class="ipa-src-row"><span class="ipa-src-meta">As-of {ACTIVE_RUN.get("as_of_date") or "—"}'
            f' · {_op} · {_sym}</span></div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ipa-src"><div class="ipa-src-row">'
                    '<span class="ipa-src-chip">30 SKUs</span>'
                    '<span class="ipa-src-chip">Legacy pilot</span></div>'
                    '<div class="ipa-src-row"><span class="ipa-src-meta">Fixed 30-SKU pilot · naheed_web'
                    '</span></div></div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

NAV_ITEMS = [
    ("Executive Overview", "home"),
    ("Demand Analytics", "bar_chart"),
    ("Forecast Runs", "rocket_launch"),
    ("Forecast Explorer", "auto_graph"),
    ("Inventory & Reorder", "inventory_2"),
    ("Deadstock", "hourglass_empty"),
    ("Stockout Risk", "crisis_alert"),
    ("Data Quality & Assumptions", "fact_check"),
]
PAGES = [name for name, _ in NAV_ITEMS]
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = PAGES[0]

st.sidebar.markdown('<div class="ipa-nav-label">Navigate</div>', unsafe_allow_html=True)
with st.sidebar.container(key="ipa-nav"):
    for name, mat_icon in NAV_ITEMS:
        is_active = st.session_state["nav_page"] == name
        clicked = st.button(name, icon=f":material/{mat_icon}:", key=f"nav_btn_{name}",
                             type="primary" if is_active else "secondary", width="stretch")
        if clicked and not is_active:
            st.session_state["nav_page"] = name
            st.rerun()
page = st.session_state["nav_page"]

# --------------------------------------------------------------------------
# Filter STATE lives here (read from session_state); the filter WIDGETS are rendered
# per page by render_filter_bar() so controls sit next to the data they affect.
# Shadow copies keep values alive on pages that don't render a given control.
# --------------------------------------------------------------------------
FILTER_KEYS = ["flt_skus", "flt_category", "flt_brands", "flt_date_from", "flt_date_to",
               "flt_date_preset", "flt_horizon", "flt_focus_sku", "flt_cmp_labels"]

# The retired single-widget date range: drop any stale value so an old tuple/one-date
# payload can never reach the new From/To controls.
for _legacy in ("flt_daterange", "_shadow_flt_daterange"):
    st.session_state.pop(_legacy, None)

DATE_PRESETS = ("All history", "Last 7 days", "Last 14 days", "Last 30 days")


def _flt(key, default):
    """Current value of a filter: live widget value if present, else the shadow copy."""
    shadow = f"_shadow_{key}"
    if key in st.session_state:
        st.session_state[shadow] = st.session_state[key]
        return st.session_state[key]
    return st.session_state.get(shadow, default)


def preset_range(preset, min_d, max_d):
    """Start/end for a quick preset. 'Last N days' ENDS at the latest available historical
    date and the start is clamped to the earliest available date."""
    if preset in (None, "", "All history") or min_d is None or max_d is None:
        return min_d, max_d
    days = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30}.get(preset)
    if not days:
        return min_d, max_d
    start = max_d - pd.Timedelta(days=days - 1).to_pytimedelta()
    return (max(start, min_d), max_d)


if mp_raw is not None:
    all_skus = sorted(mp_raw["sku"].unique().tolist())
    all_categories = sorted(mp_raw["category"].dropna().unique().tolist())
    all_brands = sorted(mp_raw["brand"].dropna().unique().tolist())
    SKU_LABELS = build_sku_labels(mp_raw)      # "Product name — Brand (SKU)" for search/legends
    SKU_NAMES = build_sku_names(mp_raw)        # name only, for tables/headings that show SKU separately
    min_date, max_date = mp_raw["date"].min().date(), mp_raw["date"].max().date()

    cat_choice = _flt("flt_category", "All categories")
    if cat_choice not in (["All categories"] + all_categories):
        cat_choice = "All categories"
    sel_categories = all_categories if cat_choice == "All categories" else [cat_choice]

    sel_skus = [s for s in _flt("flt_skus", []) if s in all_skus]
    sel_brands = [b for b in _flt("flt_brands", []) if b in all_brands]

    _focus = [s for s in _flt("flt_focus_sku", []) if s in all_skus]
    focus_skus = _focus if _focus else ([all_skus[0]] if all_skus else [])

    # A preset button writes _pending_date_* (a NON-widget key) and reruns; we apply it
    # here, BEFORE the From/To widgets are instantiated — a widget-keyed value cannot be
    # assigned after its widget exists.
    for _side in ("from", "to"):
        _pend = st.session_state.pop(f"_pending_date_{_side}", None)
        if _pend is not None:
            st.session_state[f"flt_date_{_side}"] = _pend
            st.session_state[f"_shadow_flt_date_{_side}"] = _pend

    # Historical display window — two explicit From/To values, clamped to what the
    # active model_panel actually contains. Changing the active run therefore resets a
    # stale out-of-range selection automatically.
    date_from, date_to, DATE_RANGE_ERROR = rs.normalize_history_window(
        min_date, max_date, _flt("flt_date_from", min_date), _flt("flt_date_to", max_date))
    historical_date_range = (date_from, date_to)
    date_range = historical_date_range        # kept: existing call sites use `date_range`

    horizon = _flt("flt_horizon", 14)
    horizon = horizon if horizon in (7, 14) else 14
    cmp_label_mode = _flt("flt_cmp_labels", CMP_LABEL_MODES[0])
    sel_scenario = None      # retired: synthetic Scenario Lab replaced by forecast-driven Stockout Risk
else:
    all_skus, all_categories, all_brands = [], [], []
    sel_categories, sel_brands, sel_skus = [], [], []
    SKU_LABELS = {}
    SKU_NAMES = {}
    cmp_label_mode = CMP_LABEL_MODES[0]
    date_range = None
    min_date = max_date = date_from = date_to = None
    DATE_RANGE_ERROR = None
    focus_skus = []
    horizon = 14
    sel_scenario = SCENARIO_NAMES[0]


def render_history_date_controls(key_prefix="page"):
    """Explicit From / To pickers plus quick presets for the HISTORICAL display window.

    Display-only: it filters the historical rows behind the visible KPIs, charts and
    tables. It never retrains a model, changes the active run, its as-of date or the
    forecast horizons, and it never filters future forecasts or decision artifacts.
    """
    if mp_raw is None or min_date is None or max_date is None:
        return
    with st.container(key=f"ipa-daterange-{key_prefix}"):
        st.markdown('<div class="ipa-daterange-label">Historical display period</div>',
                    unsafe_allow_html=True)
        dc = st.columns([1.1, 1.1, 1.5])
        with dc[0]:
            st.date_input("From", value=date_from or min_date,
                          min_value=min_date, max_value=max_date, key="flt_date_from",
                          format="DD/MM/YYYY")
        with dc[1]:
            st.date_input("To", value=date_to or max_date,
                          min_value=min_date, max_value=max_date, key="flt_date_to",
                          format="DD/MM/YYYY")
        with dc[2]:
            st.write("")
            pcols = st.columns(len(DATE_PRESETS))
            for pc, preset in zip(pcols, DATE_PRESETS):
                with pc:
                    short = preset.replace("Last ", "").replace(" days", "d").replace("All history", "All")
                    if st.button(short, key=f"preset_{key_prefix}_{short}", width="stretch",
                                 help=preset):
                        pf, pt = preset_range(preset, min_date, max_date)
                        # staged as pending: applied at the top of the next run, before the
                        # From/To widgets exist (they cannot be reassigned once instantiated)
                        st.session_state["_pending_date_from"] = pf
                        st.session_state["_pending_date_to"] = pt
                        st.rerun()
        if DATE_RANGE_ERROR:
            st.error(DATE_RANGE_ERROR, icon=":material/event_busy:")
        st.caption(f"Historical display period · available data: "
                   f"{_pretty_date(min_date)} to {_pretty_date(max_date)}")
        # Live result feedback — proves the range actually took effect.
        _n_rows = 0 if mp_f is None or mp_f.empty else len(mp_f)
        st.markdown(
            f'<div class="ipa-daterange-result">Showing <b>{_n_rows:,}</b> historical rows · '
            f'{_pretty_date(date_from)} to {_pretty_date(date_to)}</div>',
            unsafe_allow_html=True)
        if _n_rows == 0:
            st.warning("No historical rows in this period — widen the range or press **All**.",
                       icon=":material/event_busy:")


def render_filter_bar(*, products=True, category=True, dates=False, horizon_ctl=False,
                      compare=False, key_prefix="page"):
    """Compact per-page filter toolbar. Widgets use the SHARED flt_* keys, so a filter set
    on one page still applies everywhere — the control is simply visible where it matters."""
    if mp_raw is None:
        return
    with st.container(key=f"ipa-filterbar-{key_prefix}"):
        cols = st.columns([1.3, 2.2, 1.6, 1.1, 0.8])
        if category:
            with cols[0]:
                st.selectbox("Category", options=["All categories"] + all_categories,
                             key="flt_category")
        if products:
            with cols[1]:
                st.multiselect("Products (blank = all)", options=all_skus,
                               format_func=lambda s: SKU_LABELS.get(s, s),
                               placeholder="Search by product name, brand or SKU…",
                               help="Narrows every page: totals, charts, risk and reorder queues.",
                               key="flt_skus")
        if horizon_ctl:
            with cols[3]:
                st.radio("Horizon", options=[7, 14], format_func=lambda x: f"{x} days",
                         horizontal=True, key="flt_horizon")
        with cols[4]:
            st.write("")
            if st.button("Reset", key=f"reset_{key_prefix}", width="stretch",
                         help="Clear all product/category/date filters"):
                for k in FILTER_KEYS:
                    st.session_state.pop(k, None)
                    st.session_state.pop(f"_shadow_{k}", None)
                st.rerun()
        if dates:
            render_history_date_controls(key_prefix)
        if compare:
            c1, c2 = st.columns([3, 1.4])
            with c1:
                st.multiselect("Compare products", options=all_skus,
                               default=focus_skus if focus_skus else None,
                               format_func=lambda s: SKU_LABELS.get(s, s),
                               placeholder="Search by product name, brand or SKU…",
                               key="flt_focus_sku")
            with c2:
                st.selectbox("Comparison labels", options=CMP_LABEL_MODES, key="flt_cmp_labels")

st.sidebar.markdown("---")
if manifest:
    st.sidebar.caption(
        f"Schema {manifest.get('schema_version', '—')}  ·  Validation: "
        f"{manifest.get('validation_status', '—')}"
    )
else:
    st.sidebar.caption("Pilot manifest not found — metadata unavailable.")

# --------------------------------------------------------------------------
# Apply global filters
# --------------------------------------------------------------------------
mp_f = apply_filters(mp_raw, sel_skus, sel_categories, sel_brands, date_range)
inv_f = apply_filters(inv_joined, sel_skus, sel_categories, sel_brands, None)

filters_active = {
    "skus": sel_skus, "categories": sel_categories, "brands": sel_brands,
    "date_range": date_range, "horizon": horizon, "scenario": sel_scenario,
}

# ---- ONE effective SKU set implied by every sidebar filter (category + brand + SKU list).
# Decision artifacts (stockout risk, reorder recommendations) are keyed by SKU only, so they
# are narrowed through this set — that is what makes a single sidebar change reflect on
# EVERY page instead of only the demand pages.
FILTERED_SKUS = (set(mp_f["sku"].astype(str)) if (mp_f is not None and not mp_f.empty
                                                  and "sku" in mp_f.columns) else None)
FILTER_IS_NARROWED = bool(sel_skus or sel_brands or
                          (sel_categories and all_categories and
                           len(sel_categories) < len(all_categories)))


def narrow_to_filtered_skus(df):
    """Restrict any SKU-keyed frame to the sidebar's effective product selection."""
    if df is None or FILTERED_SKUS is None or getattr(df, "empty", True):
        return df
    if "sku" not in getattr(df, "columns", []):
        return df
    return df[df["sku"].astype(str).isin(FILTERED_SKUS)]


def filter_scope_caption():
    """Short, honest note about what the current selection is showing."""
    if not FILTER_IS_NARROWED or FILTERED_SKUS is None:
        return None
    return f"Filtered to {len(FILTERED_SKUS)} product(s)."


def enrich_product_names(df):
    """Fill a display-only `sku_name` from model_panel.

    The decision artifacts (stockout_risk / reorder_recommendations) ship a `sku_name`
    column that the backend leaves NULL, so every UI surface fell back to the bare SKU
    code. The backend cannot read model_panel — the dashboard can — so names are joined
    here for DISPLAY ONLY. SKU remains the join/identity key everywhere.
    """
    if df is None or getattr(df, "empty", True) or "sku" not in getattr(df, "columns", []):
        return df
    out = df.copy()
    mapped = out["sku"].astype(str).map(SKU_NAMES)
    if "sku_name" in out.columns:
        existing = out["sku_name"]
        blank = existing.isna() | (existing.astype(str).str.strip().isin(["", "nan", "None"]))
        out["sku_name"] = existing.where(~blank, mapped)
    else:
        out["sku_name"] = mapped
    return out


# ==========================================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ==========================================================================
def page_executive_overview():
    render_page_header("Executive Overview", "Daily eCommerce Demand & Inventory Intelligence",
                       badges=ACTIVE_BADGES)
    render_filter_bar(products=True, category=True, dates=True, key_prefix="exec")
    if mp_raw is None:
        empty_state("Historical demand data not found", "data/processed/model_panel.parquet is missing or unreadable.", "mail")
        return
    if mp_f is None or mp_f.empty:
        empty_state("No rows match the current filters", "Try widening the SKU, category, brand or date filters.", "search")
        return

    manifest_row_counts = (manifest or {}).get("row_counts", {})
    zero_rate = (mp_f["units_observed"] == 0).mean()
    promo_rate = mp_f["on_promo"].mean()
    n_eligible = int(mp_f["forecast_training_eligible"].sum())
    cost_valid = (manifest or {}).get("cost_valid_count")
    cost_imputed = (manifest or {}).get("cost_imputed_count")
    sku_count = mp_f["sku"].nunique()
    cost_coverage_str = (
        format_percentage(cost_valid / (cost_valid + cost_imputed))
        if cost_valid is not None and cost_imputed is not None and (cost_valid + cost_imputed) else "—"
    )

    # --- B. Primary KPI row: exactly 4 cards ---
    render_kpi_row([
        dict(label="Pilot SKUs", value=f"{sku_count}", icon="box",
             sub=f"of {(manifest or {}).get('sku_count', 30)} in pilot", tone="red"),
        dict(label="Historical Sales Rows", value=format_number(len(mp_f)), icon="cart",
             sub=f"{format_number(manifest_row_counts.get('model_panel'))} total in pilot", tone="blue"),
        dict(label="Forecast-Eligible Rows", value=format_number(n_eligible), icon="trending-up",
             sub=f"{format_percentage(n_eligible / len(mp_f) if len(mp_f) else None)} of rows", tone="red"),
        dict(label="Valid Cost Coverage", value=cost_coverage_str, icon="shield-check",
             sub=f"{cost_valid if cost_valid is not None else '—'} SKUs validated", tone="success"),
    ], n_cols=4)

    # --- C. Main charts row: exactly 2 ---
    st.write("")
    c1, c2 = st.columns([3, 2])
    with c1:
        with st.container(border=True):
            hdr_left, hdr_right = st.columns([3, 1])
            with hdr_left:
                st.markdown(f'<div class="ipa-card-title">{icon_svg("trending-up", 20)} Daily Units Sold</div>', unsafe_allow_html=True)
                st.markdown('<div class="ipa-card-sub">All SKUs in the current view</div>', unsafe_allow_html=True)
            with hdr_right:
                agg_freq = st.selectbox("Frequency", options=["Daily", "Weekly", "Monthly"], index=0, key="chart_agg_freq", label_visibility="collapsed")
            daily = mp_f.groupby("date", as_index=False)["units_observed"].sum()
            daily["date"] = pd.to_datetime(daily["date"])
            if agg_freq == "Weekly":
                daily = daily.set_index("date").resample("W")["units_observed"].sum().reset_index()
            elif agg_freq == "Monthly":
                daily = daily.set_index("date").resample("ME")["units_observed"].sum().reset_index()
            fig = px.area(daily, x="date", y="units_observed")
            fig.update_traces(line_color=COLORS["red"], line_width=2, fillcolor="rgba(228,35,31,0.10)")
            fig.update_layout(**plotly_layout(legend=False, height=300))
            style_axes(fig)
            fig.update_yaxes(title="")
            fig.update_xaxes(title="")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        cat_totals = mp_f.groupby("category", as_index=False)["units_observed"].sum().sort_values("units_observed", ascending=False)
        total_units = cat_totals["units_observed"].sum()
        n_cats = len(cat_totals)
        d_colors = (DONUT_COLORS * ((n_cats // len(DONUT_COLORS)) + 1))[:n_cats]
        center = f"{format_number(total_units)}<br><span style='font-size:11px;color:{COLORS['subtext']}'>Total Units</span>"
        fig = go.Figure(go.Pie(
            labels=cat_totals["category"], values=cat_totals["units_observed"], hole=0.64,
            marker=dict(colors=d_colors, line=dict(color="#FFFFFF", width=3)),
            hovertemplate="<b>%{label}</b><br>Units: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
            textinfo="percent", textposition="outside",
        ))
        fig.update_layout(**plotly_layout(height=300),
                          annotations=[dict(text=center, x=0.5, y=0.5, showarrow=False, font=dict(size=17, color=COLORS["navy"]))])
        render_chart(fig, "Units Sold by Category")

    # --- D. Secondary row: 2 panels ---
    if inv_f is not None and not inv_f.empty:
        synthetic_warning()
    d1, d2 = st.columns([3, 2])
    with d1:
        top10 = (mp_f.groupby("sku", as_index=False)["units_observed"].sum()
                 .sort_values("units_observed", ascending=False).head(10)
                 .sort_values("units_observed"))
        top10["name"] = top10["sku"].map(lambda s: SKU_NAMES.get(s, s))
        # Categories stay unique SKUs so truncated tick labels can never merge two bars;
        # the real name shows as the tick text (truncated) and in full on hover.
        _short = lambda t: t if len(t) <= 34 else t[:32].rstrip() + "…"
        fig = go.Figure(go.Bar(
            x=top10["units_observed"], y=top10["sku"], orientation="h",
            marker_color=COLORS["teal"], customdata=top10["name"],
            hovertemplate="%{customdata}<br>Units: %{x:,.0f}<extra></extra>",
        ))
        fig.update_layout(**plotly_layout(legend=False, height=320))
        style_axes(fig)
        fig.update_yaxes(categoryorder="array", categoryarray=top10["sku"].tolist(),
                         tickmode="array", tickvals=top10["sku"].tolist(),
                         ticktext=[_short(n) for n in top10["name"]], title="", automargin=True)
        fig.update_xaxes(title="Total units")
        render_chart(fig, "Top 10 Products by Units Sold")
    with d2:
        # Run mode with Phase C → forecast-driven reorder proposals; else legacy synthetic snapshot.
        if reorder_summary_raw and isinstance(reorder_summary_raw, dict):
            s = reorder_summary_raw
            metric_panel("Reorder Recommendations", [
                ("Order Now", f"{int(s.get('order_now_count', 0))} / {int(s.get('selected_series_count', 0))}"),
                ("Proposed Units", format_number(s.get("total_proposed_order_units"), 0)),
                ("Proposed Purchase Value", format_currency(s.get("total_proposed_purchase_value"))),
                ("Manual Review", format_number(s.get("manual_review_count"), 0)),
                ("Vendor Follow-up", format_number(s.get("vendor_follow_up_count"), 0)),
            ], sub="Forecast-driven planning proposals · buyer approval required")
        elif inv_f is not None and not inv_f.empty:
            avg_cover = inv_f["days_of_cover"].mean()
            n_reco = int((inv_f["recommended_order_quantity"] > 0).sum())
            baseline_stockouts = int((inv_f["days_of_cover"] <= 0).sum())
            n_skus = inv_f["sku"].nunique()
            metric_panel("Inventory & Reorder Summary", [
                ("Total Simulated Stock Value", format_currency(inv_f["inventory_value"].sum())),
                ("Recommended Purchase Value", format_currency(inv_f["recommended_purchase_value"].sum())),
                ("SKUs With Reorder", f"{n_reco} / {n_skus}"),
                ("Avg Days of Cover", f"{avg_cover:.1f}"),
                ("Synthetic Stockout SKUs", f"{baseline_stockouts} / {n_skus}"),
            ], sub="Synthetic baseline snapshot")
        else:
            empty_state("Inventory context unavailable", "inventory_context.parquet is missing or unreadable.", "box")

    # --- E. Bottom insight strip: 3 concise insights ---
    section_title("Key Insights")
    insights = generate_insights(mp_f, inv_f, manifest)
    inv_ins = [i for i in insights if "reorder recommendation" in i or "days of cover" in i]
    lead = [insights[0]] if insights else []
    show = (lead + inv_ins + [i for i in insights if i not in lead + inv_ins])[:3]
    if show:
        render_insight_row(show)
    else:
        empty_state("No insights available", "Not enough data in the current filter selection to generate insights.", "sparkle")


# ==========================================================================
# PAGE 2 — DEMAND ANALYTICS
# ==========================================================================
def page_demand_analytics():
    render_page_header("Demand Analytics", "Real historical sales — units observed on naheed_web",
                       badges=ACTIVE_BADGES)
    render_filter_bar(products=True, category=True, dates=True, compare=True, key_prefix="demand")
    if mp_raw is None:
        empty_state("Historical demand data not found", "data/processed/model_panel.parquet is missing or unreadable.", "mail")
        return
    if mp_f is None or mp_f.empty:
        empty_state("No rows match the current filters", "Try widening the SKU, category, brand or date filters.", "search")
        return

    # --- Primary: one main trend + one weekday chart ---
    c1, c2 = st.columns([3, 2])
    with c1:
        daily = mp_f.groupby("date", as_index=False)["units_observed"].sum()
        fig = px.line(daily, x="date", y="units_observed")
        fig.update_traces(line_color=COLORS["red"], line_width=2)
        fig.update_layout(**plotly_layout(legend=False, height=300))
        style_axes(fig)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
        render_chart(fig, "Total Daily Units Observed", "Real historical demand — all SKUs in view")
    with c2:
        dow_avg = mp_f.groupby("day_of_week")["units_observed"].mean().reindex(range(7))
        fig = px.bar(x=DOW_NAMES, y=dow_avg.values)
        fig.update_traces(marker_color=COLORS["slate"])
        fig.update_layout(**plotly_layout(legend=False, height=300))
        style_axes(fig)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Avg units / day")
        render_chart(fig, "Average Demand by Day of Week")

    # --- Multi-product comparison: one overlaid line per selected product ---
    # Readability rule: 1 -> detailed single chart only; 2-8 -> overlaid comparison;
    # >8 -> skip the overlay (too many lines) and warn to reduce the selection.
    MAX_COMPARE_LINES = 8
    present = [s for s in focus_skus if not mp_f[mp_f["sku"] == s].empty]
    if len(present) > MAX_COMPARE_LINES:
        st.warning(
            f"{len(present)} products selected — the comparison chart is capped at "
            f"{MAX_COMPARE_LINES} lines for readability, so it is hidden. Reduce the "
            "selection to see the overlaid comparison (individual deep-dives still show below).",
            icon=":material/warning:",
        )
    elif len(present) > 1:
        section_title("Product Comparison", f"{len(present)} products overlaid — real daily demand")
        cmp_fig = go.Figure()
        for i, fsku in enumerate(present):
            s = mp_f[mp_f["sku"] == fsku].sort_values("date")
            disp = comparison_display_label(fsku, cmp_label_mode, present)
            cmp_fig.add_trace(go.Scatter(
                x=s["date"], y=s["units_observed"], name=disp,
                mode="lines", line=dict(width=2, color=CATEGORICAL[i % len(CATEGORICAL)]),
                hovertemplate=f"<b>{disp}</b><br>%{{x|%d %b %Y}}<br>Units: %{{y:,.0f}}<extra></extra>"))
        cmp_fig.update_layout(**plotly_layout(height=360))
        style_axes(cmp_fig)
        render_chart(cmp_fig, "Daily Units — Product Comparison", "One line per selected product")

    # --- SKU deep-dive: trend + rolling means + profile panel (per product) ---
    for focus_sku in focus_skus:
        focus_label = comparison_display_label(focus_sku, cmp_label_mode, focus_skus)
        section_title(f"Deep-Dive · {focus_label}", "Real demand only — no synthetic inventory on this page.")
        sku_all = mp_f[mp_f["sku"] == focus_sku].sort_values("date")
        if sku_all.empty:
            empty_state("No data for this product in the current view", "Adjust filters to include this product's date range.", "mail")
        else:
            p1, p2 = st.columns([3, 2])
            with p1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sku_all["date"], y=sku_all["units_observed"], name="Daily units",
                                          mode="lines", line=dict(color=COLORS["grid"], width=1)))
                fig.add_trace(go.Scatter(x=sku_all["date"], y=sku_all["units_roll_mean_7"], name="7-day mean",
                                          mode="lines", line=dict(color=COLORS["teal"], width=2)))
                fig.add_trace(go.Scatter(x=sku_all["date"], y=sku_all["units_roll_mean_28"], name="28-day mean",
                                          mode="lines", line=dict(color=COLORS["amber"], width=2)))
                fig.update_layout(**plotly_layout(height=320))
                style_axes(fig)
                render_chart(fig, f"Demand & Rolling Means — {focus_label}")
            with p2:
                meta_row = sku_meta[sku_meta["sku"] == focus_sku].iloc[0] if sku_meta is not None else None
                recent7 = sku_all.tail(7)["units_observed"].mean()
                recent28 = sku_all.tail(28)["units_observed"].mean()
                metric_panel("Product Profile", [
                    ("Product", SKU_NAMES.get(focus_sku, focus_sku)),
                    ("SKU", focus_sku),
                    ("Category", meta_row["category"] if meta_row is not None else "—"),
                    ("Brand", meta_row["brand"] if meta_row is not None else "—"),
                    ("Date Range", f"{sku_all['date'].min():%d %b} – {sku_all['date'].max():%d %b %Y}"),
                    ("Total Units", format_number(sku_all["units_observed"].sum())),
                    ("Avg / Median Daily", f"{sku_all['units_observed'].mean():.2f} / {sku_all['units_observed'].median():.1f}"),
                    ("Zero-Demand %", format_percentage((sku_all['units_observed'] == 0).mean())),
                    ("Promotion-Day %", format_percentage(sku_all['on_promo'].mean())),
                    ("Latest Effective Price", format_currency(sku_all['effective_unit_price'].iloc[-1])),
                    ("Recent 7 / 28-Day Avg", f"{recent7:.2f} / {recent28:.2f}"),
                ])

    # --- Secondary analytics tucked into expanders ---
    with st.expander("Calendar & promotion effects", expanded=False):
        e1, e2, e3, e4 = st.columns(4)
        panels = [
            (e1, "is_weekend", ["Weekday", "Weekend"], "Weekend vs Weekday", [COLORS["teal"], COLORS["amber"]]),
            (e2, "on_promo", ["No promo", "On promo"], "Promotion vs Non-Promotion", [COLORS["slate"], COLORS["success"]]),
            (e3, "is_public_holiday", ["Normal", "Holiday"], "Holiday vs Normal", [COLORS["slate"], COLORS["red"]]),
            (e4, "is_payday_window", ["Normal", "Payday"], "Payday vs Normal", [COLORS["slate"], COLORS["teal"]]),
        ]
        for col, field, labels, title, colors in panels:
            with col:
                avg = mp_f.groupby(field)["units_observed"].mean().reindex([0, 1])
                fig = px.bar(x=labels, y=avg.values)
                fig.update_traces(marker_color=colors)
                fig.update_layout(**plotly_layout(legend=False, height=240))
                style_axes(fig)
                fig.update_yaxes(title="")
                render_chart(fig, title)

    with st.expander("Category, brand & seasonality", expanded=False):
        s1, s2 = st.columns(2)
        with s1:
            monthly = mp_f.groupby("month", as_index=False)["units_observed"].sum().sort_values("month")
            monthly["month_name"] = monthly["month"].apply(lambda m: calendar.month_abbr[int(m)])
            fig = px.bar(monthly, x="month_name", y="units_observed")
            fig.update_traces(marker_color=COLORS["navy"])
            fig.update_layout(**plotly_layout(legend=False, height=300))
            style_axes(fig)
            fig.update_xaxes(title="")
            fig.update_yaxes(title="Total units")
            render_chart(fig, "Monthly Demand")
        with s2:
            brand_totals = mp_f.groupby("brand")["units_observed"].sum().sort_values(ascending=False).head(10)
            fig = px.bar(brand_totals.sort_values(), orientation="h")
            fig.update_traces(marker_color=COLORS["teal"])
            fig.update_layout(**plotly_layout(legend=False, height=300))
            style_axes(fig)
            fig.update_xaxes(title="Total units")
            fig.update_yaxes(title="")
            render_chart(fig, "Brand Contribution (Top 10)")

    with st.expander("SKU-level rankings & heatmap", expanded=False):
        r1, r2 = st.columns(2)
        with r1:
            zero_by_sku = mp_f.groupby("sku")["units_observed"].apply(lambda s: (s == 0).mean() * 100).sort_values(ascending=False)
            fig = px.bar(zero_by_sku.sort_values().tail(15), orientation="h")
            fig.update_traces(marker_color=COLORS["red"])
            fig.update_layout(**plotly_layout(legend=False, height=340))
            style_axes(fig)
            fig.update_xaxes(title="% zero-demand days")
            fig.update_yaxes(title="")
            render_chart(fig, "Zero-Demand Frequency by SKU (Top 15)")
        with r2:
            vol_by_sku = mp_f.groupby("sku")["units_roll_std_7"].mean().dropna().sort_values(ascending=False)
            fig = px.bar(vol_by_sku.sort_values().tail(15), orientation="h")
            fig.update_traces(marker_color=COLORS["amber"])
            fig.update_layout(**plotly_layout(legend=False, height=340))
            style_axes(fig)
            fig.update_xaxes(title="Avg rolling std (7d)")
            fig.update_yaxes(title="")
            render_chart(fig, "Demand Volatility Ranking (Top 15)")
        pivot = mp_f.pivot_table(index="sku", columns="day_of_week", values="units_observed", aggfunc="mean").reindex(columns=range(7))
        pivot.columns = DOW_NAMES
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale=[[0, "#F4F7FA"], [0.5, COLORS["teal"]], [1, COLORS["navy"]]])
        fig.update_layout(**plotly_layout(height=440, legend=False))
        render_chart(fig, "Heatmap — SKU × Weekday (avg units)")


# ==========================================================================
# PAGE 3 — FORECAST EXPLORER
# ==========================================================================
def scroll_into_view(anchor_id: str):
    """Scroll the page to a named anchor after a rerun (used by the Details fallback panel).

    Uses a tiny, self-contained components.html snippet — no external JS, no custom
    frontend. When st.dialog is available the details open in a modal and this is not
    needed; this keeps the fallback path from stranding the user mid-page.
    """
    import streamlit.components.v1 as components
    components.html(
        "<script>"
        "const el = window.parent.document.getElementById('%s');"
        "if (el) { el.scrollIntoView({behavior:'smooth', block:'start'}); }"
        "</script>" % anchor_id, height=0)


def _winner_card(title, row):
    if row is None:
        return f'<div class="ipa-winner-card"><div class="lbl">{title}</div><div class="mdl">—</div></div>'
    return (f'<div class="ipa-winner-card"><div class="lbl">{title}</div>'
            f'<div class="mdl">{row["model"]}</div>'
            f'<div class="met">WAPE {row["wape"]:.3f} · MASE {row["mase"]:.3f} · '
            f'MAE {row["mae"]:.2f} · bias {row["bias"]:+.2f}</div></div>')


def _render_run_ranking_panel():
    """Run-mode: model-ranking winner cards + comparison table + run summary (locked-holdout)."""
    ranking, r_status = load_active(CTX["model_ranking"])
    manifest_run, _ = load_active(CTX["run_manifest"])
    section_title("Model Ranking", "Historical locked-holdout performance — not a guarantee of future accuracy.")
    if ranking is None or getattr(ranking, "empty", True):
        info_banner("Model ranking is unavailable for this run.", kind="synthetic")
    else:
        ranking = ranking.copy()
        ranking["horizon"] = pd.to_numeric(ranking["horizon"], errors="coerce")

        def _rank1(h):
            sub = ranking[(ranking["horizon"] == h) & (ranking["rank"] == 1)]
            return sub.iloc[0] if not sub.empty else None
        op_model = (manifest_run or {}).get("operational_model")
        op_h = (manifest_run or {}).get("operational_horizon")
        op_row = None
        if op_h is not None:
            sub = ranking[(ranking["horizon"] == op_h) & (ranking["model"] == op_model)]
            op_row = sub.iloc[0] if not sub.empty else _rank1(op_h)
        # Only the three winner cards stay on screen — the full comparison table and the
        # run summary live in dropdowns (progressive disclosure).
        c1, c2, c3 = st.columns(3)
        c1.markdown(_winner_card("7-day winner", _rank1(7)), unsafe_allow_html=True)
        c2.markdown(_winner_card("14-day winner", _rank1(14)), unsafe_allow_html=True)
        c3.markdown(_winner_card(f"Operational (h={op_h})", op_row), unsafe_allow_html=True)

        sel_h = horizon if horizon in list(ranking["horizon"].dropna().unique()) else int(ranking["horizon"].max())
        with st.expander(f"Model comparison @ {sel_h}-day horizon (locked-holdout)", expanded=False):
            cmp = ranking[ranking["horizon"] == sel_h][["rank", "model", "wape", "mase", "mae", "rmse", "bias"]]
            cmp = cmp.sort_values("rank")
            st.dataframe(cmp, width="stretch", hide_index=True)
            with st.container(key="ipa-export-1"):
                eu.render_table_export_menu(
                    cmp, filename_stem=f"model_ranking_h{sel_h}",
                    title=f"Model Ranking — {sel_h}-day locked holdout",
                    metadata={"Run": (ACTIVE_RUN or {}).get("run_id", "legacy"),
                              "Horizon": sel_h}, key="exp_ranking")

    if manifest_run:
        with st.expander("Run summary", expanded=False):
            fp = (manifest_run.get("dataset_fingerprint") or "")[:12]
            metric_panel("", [
                ("Run ID", manifest_run.get("run_id", "—")),
                ("Category", (manifest_run.get("request") or {}).get("category", "—")),
                ("Selected SKUs", manifest_run.get("selected_sku_count", "—")),
                ("As-of date", (manifest_run.get("request") or {}).get("as_of_date", "—")),
                ("Dataset fingerprint", f"{fp}…" if fp else "—"),
                ("Completed models", ", ".join(manifest_run.get("completed_models", [])) or "—"),
                ("Failed models", ", ".join(manifest_run.get("failed_models", [])) or "none"),
                ("Operational", f"{manifest_run.get('operational_model','—')} (h={manifest_run.get('operational_horizon','—')})"),
                ("Duration (s)", f"{manifest_run.get('duration_seconds') or 0:.1f}"),
                ("Created (PKT)", rs.format_local_datetime(manifest_run.get("created_at"))),
                ("Finished (PKT)", rs.format_local_datetime(
                    manifest_run.get("completed_at") or manifest_run.get("failed_at"))),
            ])
            with st.expander("Artifact paths (relative)", expanded=False):
                for a in (manifest_run.get("artifact_inventory") or [])[:24]:
                    st.caption(f"{a.get('path')}  ·  {a.get('size_bytes')} bytes")


def page_forecast_explorer():
    render_page_header("Forecast Explorer", "7–14 day demand forecast · historical backtest vs real future",
                       badges=ACTIVE_BADGES)
    render_filter_bar(products=True, category=True, dates=True, horizon_ctl=True,
                      compare=True, key_prefix="forecast")
    if mp_raw is None:
        empty_state("Historical demand data not found", "data/processed/model_panel.parquet is missing or unreadable.", "mail")
        return

    if DATA_MODE == "run" and ACTIVE_RUN is not None:
        _render_run_ranking_panel()

    prototype_df = build_prototype_forecast(mp_raw, horizon_days=14)

    available_models = {}
    for label, payload in outputs_forecasts.items():
        available_models[label] = payload["df"]
    available_models["Prototype Preview (14-day mean)"] = prototype_df

    not_yet = [m for m in ["SES / Holt", "Holt-Winters", "LightGBM"] if m not in available_models]
    model_options = list(available_models.keys()) + [f"{m} — not yet generated" for m in not_yet]

    m1, m2 = st.columns([1, 2])
    with m1:
        model_choice = st.selectbox("Forecast model", options=model_options, key="forecast_model_choice")
    with m2:
        st.write("")
        if model_choice == "Prototype Preview (14-day mean)":
            info_banner("<strong>Prototype baseline preview</strong> — recent 14-day mean, not the final selected model.",
                        kind="synthetic")
        elif not model_choice.endswith("— not yet generated"):
            path = outputs_forecasts.get(model_choice, {}).get("path", "outputs/")
            info_banner(f"<strong>Official model output</strong> from <code>{path}</code>.", kind="success")

    if model_choice.endswith("— not yet generated"):
        empty_state(
            "Model output not generated yet",
            f"{model_choice.replace(' — not yet generated', '')} results will appear here automatically once the "
            "corresponding output file is added to outputs/.",
            "sparkle",
        )
        return

    fc_df = available_models[model_choice].copy()
    fc_df["date"] = pd.to_datetime(fc_df["date"])
    for focus_sku in focus_skus:
        fc_sku = fc_df[fc_df["sku"] == focus_sku].sort_values("date").head(horizon)
        # Historical series honours the From/To display window (display-only — the forecast
        # rows below are NEVER date-filtered). With the full range selected we still cap the
        # view at the most recent 28 days so the chart stays readable next to the forecast.
        hist_sku = mp_f[mp_f["sku"] == focus_sku].sort_values("date")
        _narrowed = bool(date_from and date_to and min_date and max_date
                         and (date_from > min_date or date_to < max_date))
        hist_recent = hist_sku if _narrowed else hist_sku.tail(28)
    
        # --- One large historical + forecast chart ---
        if hist_recent.empty and fc_sku.empty:
            empty_state("No historical or forecast data for this product",
                        comparison_display_label(focus_sku, cmp_label_mode, focus_skus), "mail")
        else:
            fig = go.Figure()
            if not hist_recent.empty:
                fig.add_trace(go.Scatter(x=hist_recent["date"], y=hist_recent["units_observed"], name="Historical (actual)",
                                          mode="lines+markers", line=dict(color=COLORS["navy"], width=2)))
            if not fc_sku.empty:
                fig.add_trace(go.Scatter(x=fc_sku["date"], y=fc_sku["y_pred"], name="Forecast",
                                          mode="lines+markers", line=dict(color=COLORS["teal"], width=2, dash="dash")))
                fig.add_vrect(x0=fc_sku["date"].min(), x1=fc_sku["date"].max(),
                              fillcolor=COLORS["teal"], opacity=0.07, line_width=0,
                              annotation_text="Forecast horizon", annotation_position="top left")
            fig.update_layout(**plotly_layout(height=360))
            style_axes(fig)
            render_chart(fig, f"Historical vs Forecast — {comparison_display_label(focus_sku, cmp_label_mode, focus_skus)}",
                         (f"{_pretty_date(date_from)} – {_pretty_date(date_to)} (solid) + next {horizon} forecast days (dashed)" if _narrowed else
                          f"Last 28 historical days (solid) + next {horizon} forecast days (dashed)"))
    
        # --- 4 forecast summary cards ---
        if not fc_sku.empty:
            r28 = hist_sku.tail(28)["units_observed"].mean()
            f_daily = fc_sku["y_pred"].mean()
            f7 = fc_sku["y_pred"].head(7).sum()
            f14 = fc_sku["y_pred"].head(14).sum()
            render_kpi_row([
                dict(label="Forecasted Daily Avg", value=f"{f_daily:.1f}", icon="sparkle", tone="teal"),
                dict(label="Forecasted 7-Day Demand", value=format_number(f7), icon="calendar", tone="blue"),
                dict(label="Forecasted 14-Day Demand", value=format_number(f14), icon="calendar", tone="amber"),
                dict(label="Recent 28-Day Actual Mean", value=f"{r28:.1f}", icon="trending-up", tone="slate"),
            ], n_cols=4)
    
        # --- Compact forecast table + cumulative ---
        st.write("")
        t1, t2 = st.columns([2, 3])
        with t1:
            if fc_sku.empty:
                empty_state("No forecast rows for this SKU", f"SKU: {focus_sku}", "sparkle")
            else:
                with st.container(border=True):
                    st.markdown('<div class="ipa-card-title">Daily Forecast Table</div>', unsafe_allow_html=True)
                    table = fc_sku[["date", "y_pred"]].rename(columns={"date": "Date", "y_pred": "Predicted"})
                    table["Date"] = table["Date"].dt.strftime("%d %b")
                    table["Cumulative"] = table["Predicted"].cumsum()
                    st.dataframe(table, width="stretch", hide_index=True, height=300)
        with t2:
            if not fc_sku.empty:
                cum = fc_sku[["date", "y_pred"]].copy()
                cum["cumulative"] = cum["y_pred"].cumsum()
                fig = px.area(cum, x="date", y="cumulative")
                fig.update_traces(line_color=COLORS["teal"], fillcolor="rgba(14,124,123,0.16)")
                fig.update_layout(**plotly_layout(legend=False, height=300))
                style_axes(fig)
                fig.update_xaxes(title="")
                fig.update_yaxes(title="Cumulative units")
                render_chart(fig, f"{horizon}-Day Cumulative Predicted Demand ({focus_sku})")
    
        # --- Secondary content in expanders ---
        with st.expander(f"Compare available models for {focus_sku}", expanded=False):
            comp_fig = go.Figure()
            any_series = False
            for label, df_model in available_models.items():
                d = df_model[df_model["sku"] == focus_sku].sort_values("date").head(horizon)
                if not d.empty:
                    comp_fig.add_trace(go.Scatter(x=d["date"], y=d["y_pred"], name=label, mode="lines+markers"))
                    any_series = True
            if any_series:
                comp_fig.update_layout(**plotly_layout(height=320))
                style_axes(comp_fig)
                render_chart(comp_fig, f"Model Comparison — {focus_sku}")
                if not_yet:
                    st.caption(f"Additional models will appear automatically once generated: {', '.join(not_yet)}.")
            else:
                empty_state("No forecasts available to compare", f"SKU: {focus_sku}", "sparkle")

    with st.expander("Backtest accuracy (WAPE / MASE)", expanded=False):
        if outputs_evaluations:
            for name, df_eval in outputs_evaluations.items():
                st.markdown(f"**{name}**")
                st.dataframe(df_eval, width="stretch", hide_index=True)
        else:
            empty_state(
                "Evaluation output not generated yet",
                "No WAPE/MASE backtest scorecard was found in outputs/. Accuracy is only meaningful against historical "
                "backtest windows — never against real future dates, since actual outcomes are not yet known.",
                "target",
            )

    with st.expander("Forecast by SKU & category", expanded=False):
        b1, b2 = st.columns(2)
        with b1:
            all_totals = (
                fc_df.sort_values("date").groupby("sku").head(horizon)
                .groupby("sku", as_index=False)["y_pred"].sum()
                .sort_values("y_pred", ascending=False)
            )
            if all_totals.empty:
                empty_state("No forecast rows", "The selected model has no forecast rows.", "sparkle")
            else:
                fig = px.bar(all_totals, x="sku", y="y_pred")
                fig.update_traces(marker_color=COLORS["navy"])
                fig.update_layout(**plotly_layout(legend=False, height=320))
                style_axes(fig)
                fig.update_xaxes(title="")
                fig.update_yaxes(title="Predicted units")
                render_chart(fig, f"{horizon}-Day Total Predicted Demand by SKU")
        with b2:
            if sku_meta is not None and not fc_df.empty:
                merged = fc_df.merge(sku_meta[["sku", "category"]], on="sku", how="left")
                merged = merged.sort_values("date").groupby("sku").head(horizon)
                cat_fc = merged.groupby(["date", "category"], as_index=False)["y_pred"].sum()
                fig = px.area(cat_fc, x="date", y="y_pred", color="category", color_discrete_sequence=CATEGORICAL)
                fig.update_layout(**plotly_layout(height=320))
                style_axes(fig)
                render_chart(fig, "Aggregated Forecast by Category")
            else:
                empty_state("Category mapping unavailable", "Historical data is required to map SKUs to categories.", "folder")


# ==========================================================================
# PAGE 4 — INVENTORY & REORDER
# ==========================================================================
def _reorder_badges():
    if DATA_MODE == "run" and ACTIVE_RUN is not None:
        n_sel = ACTIVE_RUN.get("selected_sku_count")
        cat = ACTIVE_RUN.get("category") or "All categories"
        asof_raw = ACTIVE_RUN.get("as_of_date")
        asof = rs.format_local_datetime(asof_raw, include_time=False) if asof_raw else "—"
        opm = ACTIVE_RUN.get("operational_model") or "—"
        tcd = 14
        if reorder_summary_raw and isinstance(reorder_summary_raw, dict):
            tcd = reorder_summary_raw.get("target_cover_days", 14)
        return [
            ("box", f"{n_sel if n_sel is not None else '—'} selected SKUs"),
            ("tag", str(cat)),
            ("calendar", f"As-of {asof}"),
            ("layers", f"Model: {opm}"),
            ("target", f"Target cover: {tcd}d"),
        ]
    return ACTIVE_BADGES


REORDER_DISCLOSURE = (
    "These are planning recommendations only. No purchase order has been created or sent. "
    "Inventory stock, lead time, MOQ, pack size and unit cost may use pilot assumptions or imputation.")


def page_inventory_reorder():
    # Run mode with Phase C → the forecast-driven recommendation experience.
    # Run mode without Phase C → a regeneration empty state (never fall back to the historical
    # inventory-context quantity). Legacy fixed pilot → the clearly-labelled legacy page.
    if DATA_MODE == "run":
        if CTX.get("reorder_available") and reorder_raw is not None:
            _page_inventory_reorder_run()
        else:
            render_page_header(
                "Inventory & Reorder",
                "Forecast-driven replenishment proposals · MOQ and pack rounding · buyer approval required",
                badges=_reorder_badges())
            empty_state(
                "Reorder recommendations unavailable for this run",
                "Reorder recommendations are unavailable for this run. Regenerate the forecast using "
                "the current pipeline.", "package-search")
        return
    _page_inventory_reorder_legacy()


def _page_inventory_reorder_run():
    # ------------------------------------------------------------------
    # Display-only helpers. This page ONLY reads the validated Phase C
    # decision artifact — it never recomputes quantities, rounding or value.
    # ------------------------------------------------------------------
    ACTION_HEX = {"order_now": COLORS["red"], "vendor_follow_up": COLORS["blue"],
                  "manual_review": COLORS["amber"], "monitor": COLORS["slate"],
                  "no_order": COLORS["success"]}
    TIER_HEX = {"critical": COLORS["red"], "high": COLORS["amber"], "medium": COLORS["blue"],
                "watch": COLORS["blue"], "low": COLORS["success"], "healthy": COLORS["success"],
                "unknown": COLORS["slate"]}

    def _esc(x):
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _safe_key(x):
        return "".join(c if c.isalnum() else "-" for c in str(x))

    def _help(txt, defn):
        return f'<span title="{_esc(defn)}">{txt}</span>'

    def _meta_val(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        s = str(x).strip()
        return s if s and s.lower() not in ("nan", "none", "nat") else None

    def _date_str(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        try:
            d = pd.to_datetime(x)
            return "—" if pd.isna(d) else d.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            s = str(x).strip()
            return s if s and s.lower() not in ("nat", "nan", "none") else "—"

    def _short(s, n=34):
        s = str(s)
        return s if len(s) <= n else s[:n - 1] + "…"

    def _action_hex(a):
        return ACTION_HEX.get(str(a).strip().lower(), COLORS["slate"])

    def _action_chip(a):
        a = str(a).strip().lower()
        return f'<span class="ipa-action ipa-action-{a}">{rs.reorder_action_label(a)}</span>'

    def _flags_to_list(flags):
        if flags is None:
            return []
        if isinstance(flags, (list, tuple, np.ndarray)):
            return [str(x).strip() for x in list(flags) if str(x).strip()]
        if isinstance(flags, float) and pd.isna(flags):
            return []
        s = str(flags).strip()
        if not s or s.lower() in ("nan", "none", "[]"):
            return []
        for sep in ("|", ";", ","):
            if sep in s:
                return [p.strip() for p in s.split(sep) if p.strip()]
        return [s]

    def _flag_pretty(f):
        return _esc(str(f).replace("_", " ").strip().capitalize())

    # ------------------------------------------------------------------
    # Header + prominent, non-expander disclosure
    # ------------------------------------------------------------------
    render_page_header(
        "Inventory & Reorder",
        "Forecast-driven replenishment proposals · MOQ and pack rounding · buyer approval required",
        badges=_reorder_badges())
    render_filter_bar(products=True, category=True, key_prefix="reorder")
    info_banner(REORDER_DISCLOSURE, kind="synthetic")

    reco = reorder_raw.copy()
    # Shared product filter + display product names (artifact sku_name is null).
    reco = narrow_to_filtered_skus(reco)
    reco = enrich_product_names(reco)
    if reco.empty:
        empty_state("No SKUs in the current filter",
                    "Clear the product filter in the sidebar to see the full recommendation queue.",
                    "search")
        return

    # brand enrichment for display (backend cannot read model_panel; the dashboard can)
    brand_by_sku = {}
    if sku_meta is not None and "sku" in sku_meta.columns and "brand" in sku_meta.columns:
        brand_by_sku = dict(zip(sku_meta["sku"].astype(str), sku_meta["brand"]))

    # ==================================================================
    # SECTION 1 — Reorder overview (exactly five KPIs)
    # ==================================================================
    section_title("Reorder overview",
                  "Aggregated from the validated Phase C recommendations for the current selection.")
    actions = reco["action"].astype(str).str.lower()
    n_order = int((actions == "order_now").sum())
    n_review = int((actions == "manual_review").sum())
    n_vendor = int((actions == "vendor_follow_up").sum())
    proposed_units = float(pd.to_numeric(reco["recommended_order_quantity"], errors="coerce").fillna(0).sum())
    pv_total, pv_missing = rs.reorder_purchase_value_total(reco)

    kpis = [
        {"label": "Order Now", "value": format_number(n_order), "icon": "cart", "tone": "red",
         "sub": _help("Ready to propose", "Count of SKUs whose recommended action is order_now.")},
        {"label": "Proposed Units", "value": format_number(proposed_units), "icon": "box", "tone": "navy",
         "sub": _help("Across order_now SKUs",
                      "Sum of recommended_order_quantity across all rows (only order_now rows carry a positive quantity).")},
        {"label": "Proposed Purchase Value", "value": format_currency(pv_total), "icon": "coin", "tone": "teal",
         "sub": _help((f"{pv_missing} SKU(s) without a value" if pv_missing else "All proposed orders priced"),
                      "Sum of recommended_purchase_value across priced order_now SKUs; unavailable values are excluded, never counted as zero.")},
        {"label": "Manual Review", "value": format_number(n_review), "icon": "shield-check", "tone": "amber",
         "sub": _help("Need a human decision",
                      "Count of SKUs routed to manual_review (invalid input, insufficient horizon, or Phase B review).")},
    ]
    # Four equal-height primary KPIs; Vendor follow-up + Manual review appear as chips
    # beside the queue filters so they stop competing with the headline numbers.
    with st.container(key="ipa-kpirow"):
        render_kpi_row(kpis, n_cols=4)

    # ==================================================================
    # SECTION 2 — Action distribution + purchase exposure
    # ==================================================================
    with st.expander("Action distribution & purchase exposure", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            order = ["order_now", "vendor_follow_up", "manual_review", "monitor", "no_order"]
            vc = actions.value_counts()
            present = [a for a in order if a in vc.index] + [a for a in vc.index if a not in order]
            donut = go.Figure(go.Pie(
                labels=[rs.reorder_action_label(a) for a in present],
                values=[int(vc[a]) for a in present], hole=0.62, sort=False, direction="clockwise",
                marker=dict(colors=[_action_hex(a) for a in present], line=dict(color="white", width=1.5)),
                textinfo="value",
                hovertemplate="%{label}: %{value} SKUs (%{percent})<extra></extra>"))
            donut.update_layout(**plotly_layout(height=330, legend=True))
            donut.update_layout(annotations=[dict(text=f"{len(reco)}<br>SKUs", x=0.5, y=0.5, showarrow=False,
                                                  font=dict(size=15, color=COLORS["navy"]))])
            render_chart(donut, "Recommended Actions",
                         "One proposed buyer action per SKU. Fixed semantic colors; hover for counts and share.")
        with col_b:
            on = reco[actions == "order_now"].copy()
            if on.empty:
                with st.container(border=True):
                    empty_state("No order_now recommendations",
                                "No SKUs in the current selection meet the reorder trigger.", "circle-dashed")
            else:
                on["_pv"] = pd.to_numeric(on["recommended_purchase_value"], errors="coerce").fillna(0.0)
                on = on.sort_values("_pv", ascending=True).tail(15)
                names = [rs.full_product_label(n, s) for n, s in zip(on.get("sku_name"), on["sku"])]
                cust = list(zip(names, on["sku"].astype(str),
                                [format_number(v, 0) for v in on["recommended_order_quantity"]],
                                [format_currency(v, 2) for v in on["unit_cost_effective"]],
                                [(_meta_val(c) or "—") for c in on["cost_source"]],
                                ["imputed" if bool(v) else "observed" for v in on["cost_is_imputed"]]))
                bar = go.Figure(go.Bar(
                    x=on["_pv"], y=on["sku"].astype(str), orientation="h",
                    marker=dict(color=COLORS["teal"]), customdata=cust,
                    hovertemplate=("<b>%{customdata[0]}</b><br>SKU %{customdata[1]}<br>"
                                   "Proposed qty: %{customdata[2]} u<br>Unit cost: %{customdata[3]}<br>"
                                   "Purchase value: %{x:,.0f}<br>Cost source: %{customdata[4]} (%{customdata[5]})"
                                   "<extra></extra>")))
                bar.update_layout(**plotly_layout(legend=False, height=330))
                style_axes(bar)
                bar.update_yaxes(tickmode="array", tickvals=on["sku"].astype(str).tolist(),
                                 ticktext=[_short(n) for n in names], title="", automargin=True)
                bar.update_xaxes(title_text="Proposed purchase value (PKR)")
                render_chart(bar, "Proposed Purchase Value by Product",
                             "order_now SKUs only. Hover for the full product name, quantity, unit cost and cost source.")

    # ==================================================================
    # SECTION 3 — Priority action queue (filters + clickable cards)
    # ==================================================================
    section_title("Priority action queue",
                  "Ranked by action priority, then risk, probability, projected date, purchase value "
                  "and name. Click a card to open its full recommendation below.")
    fc = st.columns([1.2, 1.2, 1.6, 1, 1])
    with fc[0]:
        act_opts = ["All actions"] + [rs.reorder_action_label(a) for a in present]
        act_pick = st.selectbox("Action", options=act_opts, key="reorder_action")
    with fc[1]:
        tiers_present = [t for t in ["critical", "high", "medium", "watch", "low", "healthy", "unknown"]
                        if t in set(reco["overall_risk_tier"].astype(str).str.lower())]
        tier_opts = ["All tiers"] + [t.capitalize() for t in tiers_present]
        tier_pick = st.selectbox("Risk tier", options=tier_opts, key="reorder_tier")
    with fc[2]:
        query = st.text_input("Search product or SKU", key="reorder_query",
                              placeholder="product name or SKU code")
    with fc[3]:
        approval_only = st.checkbox("Approval only", key="reorder_approval_only")
    with fc[4]:
        assumed_only = st.checkbox("Assumed only", key="reorder_assumed_only")

    action_arg = None
    if act_pick != "All actions":
        action_arg = {rs.reorder_action_label(a): a for a in present}.get(act_pick, act_pick)
    tier_arg = None if tier_pick == "All tiers" else tier_pick
    filtered = rs.filter_reorder_queue(reco, action=action_arg, tier=tier_arg, query=query,
                                       approval_only=approval_only, assumed_only=assumed_only)
    ranked = rs.sort_reorder_queue(filtered)
    valid_skus = ranked["sku"].astype(str).tolist() if ranked is not None and not ranked.empty else []

    cur = st.session_state.get("reorder_selected_sku")
    if cur not in valid_skus:
        cur = valid_skus[0] if valid_skus else None
        st.session_state["reorder_selected_sku"] = cur

    if not valid_skus:
        info_banner("No SKUs match these filters. Adjust the action, tier, search or toggles above.", kind="info")
        return

    # Secondary counts as chips (they are no longer primary KPIs) + one export menu.
    qc1, qc2 = st.columns([3, 1])
    with qc1:
        _nrev = int(pd.Series(ranked.get("action", pd.Series(dtype=str))).astype(str)
                    .eq("manual_review").sum()) if not ranked.empty else 0
        _nven = int(pd.Series(ranked.get("action", pd.Series(dtype=str))).astype(str)
                    .eq("vendor_follow_up").sum()) if not ranked.empty else 0
        st.markdown(
            f'<div class="ipa-src-row"><span class="ipa-tier ipa-tier-unknown">{len(ranked)} in queue</span>'
            f'<span class="ipa-tier ipa-tier-{"high" if _nrev else "healthy"}">⚑ {_nrev} manual review</span>'
            f'<span class="ipa-tier ipa-tier-medium">🚚 {_nven} vendor follow-up</span></div>',
            unsafe_allow_html=True)
    with qc2:
        with st.container(key="ipa-export-2"):
            eu.render_table_export_menu(
                ranked, filename_stem="inventory_action_queue",
                title="Inventory & Reorder — Priority Action Queue",
                metadata={"Run": (ACTIVE_RUN or {}).get("run_id", "legacy"),
                          "Category": (ACTIVE_RUN or {}).get("category", "—")},
                key="exp_reco_queue")

    # ---- dense selectable rows (fixed height, 8 per page) ----
    PAGE = 8
    shown = int(st.session_state.get("reorder_queue_shown", PAGE))
    shown = max(PAGE, min(shown, len(ranked)))
    for rec_row in ranked.head(shown).to_dict("records"):
        sku = str(rec_row["sku"])
        safe = _safe_key(sku)
        name = rec_row.get("sku_name")
        nm = None if (name is None or (isinstance(name, float) and pd.isna(name))) else str(name).strip()
        disp = nm if (nm and nm.lower() != "nan") else sku
        full = rs.full_product_label(name, sku)
        tier = str(rec_row.get("overall_risk_tier") or "unknown").lower()
        is_sel = (sku == str(cur))
        rcols = st.columns([3.2, 1.3, 1.0, 1.1, 1.3, 1.0])
        with rcols[0]:
            st.markdown(
                f'<div class="ipa-qrow ipa-q-{tier}" title="{_esc(full)}">'
                f'<div style="min-width:0;"><div class="q-name">{_esc(disp)}</div>'
                f'<div class="q-sub">SKU {_esc(sku)} · risk {_esc(tier)}</div></div></div>',
                unsafe_allow_html=True)
        rcols[1].markdown(f'<div class="ipa-qcell">{_action_chip(rec_row.get("action"))}</div>',
                          unsafe_allow_html=True)
        rcols[2].markdown(f'<div class="ipa-qcell"><b>{format_number(rec_row.get("days_of_cover"), 1)}</b>'
                          f'<div class="q-sub">days cover</div></div>', unsafe_allow_html=True)
        rcols[3].markdown(f'<div class="ipa-qcell"><b>{format_number(rec_row.get("recommended_order_quantity"), 0)}</b>'
                          f'<div class="q-sub">proposed qty</div></div>', unsafe_allow_html=True)
        rcols[4].markdown(f'<div class="ipa-qcell"><b>{format_currency(rec_row.get("recommended_purchase_value"))}</b>'
                          f'<div class="q-sub">value</div></div>', unsafe_allow_html=True)
        with rcols[5]:
            if st.button("Details", key=f"recobtn-{safe}", width="stretch",
                         type="primary" if is_sel else "secondary",
                         help=f"Open full recommendation for {full}"):
                st.session_state["reorder_selected_sku"] = sku
                st.session_state["reorder_open_dialog"] = True
                st.rerun()
    if len(ranked) > shown:
        if st.button(f"Show more ({len(ranked) - shown} remaining)", key="reco_show_more"):
            st.session_state["reorder_queue_shown"] = shown + PAGE
            st.rerun()
    elif shown > PAGE:
        if st.button("Show fewer", key="reco_show_fewer"):
            st.session_state["reorder_queue_shown"] = PAGE
            st.rerun()

    # ==================================================================
    # SECTION 4 — Selected recommendation deep dive
    # ==================================================================
    # ------------------------------------------------------------------
    # Selected recommendation — shown in st.dialog (no long scroll) with an
    # inline-expander fallback. Identical content in both paths.
    # ------------------------------------------------------------------
    def _render_selected_reco(cur):
        sel = reco[reco["sku"].astype(str) == str(cur)]
        if sel.empty:
            return
        r = sel.iloc[0]
        full = rs.full_product_label(r.get("sku_name"), r["sku"])
        cat_val = _meta_val(r.get("category"))
        brand_val = _meta_val(brand_by_sku.get(str(r["sku"])))
        meta_bits = [b for b in (cat_val, brand_val, _meta_val(r.get("channel"))) if b]

        st.markdown('<div class="ipa-dd-sub">'
                    f'{_esc(" · ".join(meta_bits) if meta_bits else "—")} &nbsp;{_action_chip(r.get("action"))}'
                    '</div>', unsafe_allow_html=True)

        # A. decision summary — exactly four cards
        approval_txt = "Buyer approval required" if bool(r.get("approval_required")) else "No approval needed"
        dd = [
            {"label": "Recommended action", "value": rs.reorder_action_label(r.get("action")),
             "icon": "target", "tone": rs.reorder_action_tone(r.get("action")),
             "sub": "One proposed action per SKU"},
            {"label": "Proposed quantity", "value": f'{format_number(r.get("recommended_order_quantity"), 0)} u',
             "icon": "box", "tone": "navy", "sub": "0 for non-order actions"},
            {"label": "Proposed purchase value", "value": format_currency(r.get("recommended_purchase_value")),
             "icon": "coin", "tone": "teal", "sub": "Quantity × effective unit cost"},
            {"label": "Approval status", "value": ("Awaiting review" if bool(r.get("approval_required")) else "—"),
             "icon": "shield-check", "tone": "amber", "sub": approval_txt},
        ]
        render_kpi_row(dd, n_cols=4)

        # Tabs keep the modal short (mirrors the Stockout Risk detail dialog).
        _r_sum, _r_qty, _r_why = st.tabs(["Summary", "Quantity build", "Why & assumptions"])

        # B. trigger & inventory position
        trig = "Yes" if bool(r.get("reorder_triggered")) else "No"
        with _r_sum, st.container(height=460):
            left, right = st.columns([1, 1])
            with left:
                metric_panel("Trigger & inventory position", [
                    ("Current stock", format_number(r.get("stock_on_hand"), 0)
                     + (" (synthetic)" if bool(r.get("stock_on_hand_is_synthetic")) else "")),
                    ("Reported on-order", format_number(r.get("reported_on_order_quantity"), 0)),
                    ("Usable on-order", format_number(r.get("usable_on_order_quantity"), 0)),
                    ("Inventory position", format_number(r.get("inventory_position_for_risk"), 0)),
                    ("Forecast-driven reorder point", format_number(r.get("forecast_driven_reorder_point"), 0)),
                    ("Reorder triggered", trig),
                    ("Days of cover", format_number(r.get("days_of_cover"), 1)),
                    ("Risk tier", str(r.get("overall_risk_tier") or "—").capitalize()),
                    ("Stockout probability", format_percentage(r.get("stockout_probability"))),
                    ("Projected stockout date", _date_str(r.get("projected_stockout_date"))),
                ], sub="“—” means not available (never shown as zero).")
            with right:
                # comparison chart: inventory position vs reorder point vs target stock
                comp_labels = ["Inventory position", "Forecast reorder point", "Target stock"]
                comp_vals = [r.get("inventory_position_for_risk"), r.get("forecast_driven_reorder_point"),
                             r.get("target_stock")]
                comp_num = [float(v) if _meta_val(v) is not None and pd.notna(v) else None for v in comp_vals]
                cmp = go.Figure(go.Bar(
                    x=comp_labels, y=[v if v is not None else 0 for v in comp_num],
                    marker=dict(color=[COLORS["navy"], COLORS["amber"], COLORS["teal"]]),
                    customdata=[format_number(v, 0) for v in comp_num],
                    hovertemplate="%{x}: %{customdata} units<extra></extra>"))
                cmp.update_layout(**plotly_layout(legend=False, height=300))
                style_axes(cmp)
                cmp.update_yaxes(title_text="Units")
                render_chart(cmp, "Position vs Reorder Point vs Target Stock",
                             "Exact units on hover. Target stock is the order-up-to level for the target-cover policy.")

            # C. quantity construction (four-stage flow)
        with _r_qty, st.container(height=460):
            section_title("Quantity construction", "Raw target gap → MOQ adjusted → pack rounded → final proposal.")
            is_order = str(r.get("action")).lower() == "order_now"
            raw_gap = r.get("raw_target_gap")
            moq = r.get("moq")
            pack = r.get("pack_size")
            moq_adj = r.get("moq_adjusted_quantity")
            rounded = r.get("rounded_order_quantity")
            final_q = r.get("recommended_order_quantity")
            prov = r.get("provisional_calculated_quantity")
            if is_order:
                stages = [
                    ("Raw target gap", format_number(raw_gap, 0), "target stock − inventory position"),
                    ("MOQ adjusted", format_number(moq_adj, 0), f"MOQ {format_number(moq, 0)}"),
                    ("Pack rounded", format_number(rounded, 0), f"pack {format_number(pack, 0)}"),
                    ("Final proposal", format_number(final_q, 0), "buyer approval required"),
                ]
                flow = ""
                for i, (lab, val, cap) in enumerate(stages):
                    cls = "ipa-qstage final" if i == len(stages) - 1 else "ipa-qstage"
                    flow += (f'<div class="{cls}"><div class="lab">{_esc(lab)}</div>'
                             f'<div class="val">{_esc(val)}</div><div class="cap">{_esc(cap)}</div></div>')
                    if i < len(stages) - 1:
                        flow += '<div class="ipa-qarrow">→</div>'
                st.markdown(f'<div class="ipa-qflow">{flow}</div>', unsafe_allow_html=True)
            else:
                prov_txt = (f"A provisional (non-actionable) quantity of {format_number(prov, 0)} units was safely "
                            f"calculated for reference." if prov is not None and pd.notna(prov)
                            else "No provisional quantity could be safely calculated.")
                info_banner(f"Final actionable quantity is <strong>0</strong> for a "
                            f"<strong>{_esc(rs.reorder_action_label(r.get('action')))}</strong> recommendation "
                            f"— MOQ and pack rounding are not applied to the actionable quantity. {prov_txt}",
                            kind="info")
                if raw_gap is not None and pd.notna(raw_gap):
                    st.caption(f"Diagnostic raw target gap: {format_number(raw_gap, 0)} units "
                               f"(does not create an order without the reorder trigger).")

            # D. demand & target-stock inputs
            # E. ordering & cost — shown side by side
            d_col, e_col = st.columns(2)
            with d_col:
                metric_panel("Demand & target-stock inputs", [
                    ("Operational model", _esc(str(r.get("operational_model") or "—"))),
                    ("Operational horizon", f'{format_number(r.get("operational_horizon"), 0)} d'),
                    ("Lead time", f'{format_number(r.get("lead_time_days"), 0)} d '
                                  f'({_meta_val(r.get("lead_time_source")) or "—"})'),
                    ("Lead-time demand (mean)", format_number(r.get("lead_time_demand_mean"), 0)),
                    ("Lead-time safety stock", format_number(r.get("lead_time_safety_stock"), 0)),
                    ("Target-cover days", format_number(r.get("target_cover_days"), 0)),
                    ("Planning-horizon days", format_number(r.get("planning_horizon_days"), 0)),
                    ("Planning-horizon demand", format_number(r.get("planning_horizon_demand_mean"), 1)),
                    ("Planning-horizon sigma", format_number(r.get("planning_horizon_sigma"), 1)),
                    ("Service-level target", format_percentage(r.get("service_level_target"))),
                    ("Service-level z", format_number(r.get("service_level_z"), 2)),
                    ("Planning safety stock", format_number(r.get("planning_safety_stock"), 0)),
                    ("Target stock", format_number(r.get("target_stock"), 0)),
                ], sub="Forecast-driven; “—” means not available.")
            with e_col:
                metric_panel("Ordering & cost", [
                    ("Recommended order date", _date_str(r.get("recommended_order_date"))),
                    ("Expected arrival date", _date_str(r.get("expected_arrival_date"))),
                    ("MOQ", f'{format_number(r.get("moq"), 0)} ({_meta_val(r.get("moq_source")) or "—"})'),
                    ("Pack size", f'{format_number(r.get("pack_size"), 0)} ({_meta_val(r.get("pack_size_source")) or "—"})'),
                    ("Unit cost (effective)", format_currency(r.get("unit_cost_effective"), 2)),
                    ("Cost source", _meta_val(r.get("cost_source")) or "—"),
                    ("Cost imputed", "Yes (estimated)" if bool(r.get("cost_is_imputed")) else "No"),
                    ("Cost quality", _meta_val(r.get("cost_quality_flag")) or "—"),
                    ("Currency", _meta_val(r.get("cost_currency")) or "—"),
                    ("Cost basis", _meta_val(r.get("cost_basis")) or "—"),
                    ("Proposed purchase value", format_currency(r.get("recommended_purchase_value"))),
                ], sub="Supplier calendars, weekends and working-day schedules are not available.")

            # ==================================================================
            # SECTION 5 — Why this action was recommended
            # ==================================================================
        with _r_why, st.container(height=460):
            section_title("Why this action was recommended", None)
            reason = r.get("reason_trace")
            reason_txt = "" if (reason is None or (isinstance(reason, float) and pd.isna(reason))) else str(reason)
            st.markdown(
                f'<div class="ipa-reason"><div class="h">{icon_svg("file-text", 16)} Decision rationale</div>'
                f'{_esc(reason_txt) if reason_txt else "No reason trace was recorded for this SKU."}</div>',
                unsafe_allow_html=True)
            codes = _flags_to_list(r.get("review_reason_codes"))
            if codes:
                chips = "".join(f'<span class="ipa-rtier ipa-rtier-unknown">{_flag_pretty(c)}</span> ' for c in codes)
                st.markdown(f'<div style="margin-top:8px;">{chips}</div>', unsafe_allow_html=True)

            # ==================================================================
            # SECTION 6 — Assumptions & approval
            # ==================================================================
            st.markdown(
                '<div class="ipa-approval" style="margin-top:10px;">'
                f'<span class="ico">{icon_svg("clock", 22)}</span>'
                '<div class="txt"><div class="h">Status: Awaiting buyer review</div>'
                '<div class="s">This is a planning proposal. No purchase order has been approved, submitted, '
                'placed or sent to any supplier.</div></div></div>', unsafe_allow_html=True)

            with st.expander("Assumptions, cost quality and decision limitations", expanded=False):
                def _yn(b):
                    return "Yes" if b else "No"
                flag_items = _flags_to_list(r.get("assumption_flags"))
                conds = [
                    f"Synthetic stock: {_yn(bool(r.get('stock_on_hand_is_synthetic')))}",
                    f"Lead-time source: {_meta_val(r.get('lead_time_source')) or '—'}",
                    f"MOQ source: {_meta_val(r.get('moq_source')) or '—'}",
                    f"Pack-size source: {_meta_val(r.get('pack_size_source')) or '—'}",
                    f"Cost source: {_meta_val(r.get('cost_source')) or '—'}",
                    f"Cost imputed (estimated): {_yn(bool(r.get('cost_is_imputed')))}",
                    f"Insufficient forecast horizon: {_yn(bool(r.get('insufficient_horizon')))}",
                    f"Decision policy version: {_meta_val(r.get('decision_policy_version')) or '—'}",
                ]
                st.markdown("**Conditions for this SKU**")
                st.markdown("\n".join(f"- {c}" for c in conds))
                if flag_items:
                    st.markdown("**Assumption flags recorded in the artifact**")
                    st.markdown("\n".join(f"- {_flag_pretty(f)}" for f in flag_items))
                if codes:
                    st.markdown("**Manual-review reasons**")
                    st.markdown("\n".join(f"- {_flag_pretty(c)}" for c in codes))
                st.markdown("**Known limitations**")
                for d in (
                    "Supplier calendars, weekends and working-day schedules are not available, so the expected "
                    "arrival date is a simple lead-time offset.",
                    "Undated inbound (on-order) stock is excluded from the inventory position.",
                    "Lead time, MOQ and pack size may be pilot assumptions; unit cost may be imputed.",
                    "Recommendations are forecast-driven planning proposals — no purchase order is created.",
                ):
                    st.markdown(f"- {d}")
                with st.expander("Advanced: raw decision fields", expanded=False):
                    adv_fields = ["reorder_triggered", "insufficient_horizon", "raw_target_gap",
                                  "raw_order_quantity", "moq_adjusted_quantity", "rounded_order_quantity",
                                  "provisional_calculated_quantity", "planning_horizon_days",
                                  "available_forecast_horizon_days", "service_level_z", "confidence_label",
                                  "order_placed", "human_follow_up_required", "manual_review_required",
                                  "decision_policy_version", "generated_at"]
                    adv_rows = []
                    for f in adv_fields:
                        v = r.get(f)
                        disp = "—" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
                        adv_rows.append({"field": f, "value": disp})
                    st.dataframe(pd.DataFrame(adv_rows), use_container_width=True, hide_index=True)

    # ---- recommendation details: dialog when supported, compact inline panel otherwise ----
    _rsel = reco[reco["sku"].astype(str) == str(cur)]
    _rfull = (rs.full_product_label(_rsel.iloc[0].get("sku_name"), _rsel.iloc[0]["sku"])
              if not _rsel.empty else str(cur))
    if hasattr(st, "dialog"):
        @st.dialog(f"{_rfull}", width="large")
        def _reco_dialog():
            _render_selected_reco(cur)
            if st.button("Close", key="reco_dialog_close", type="primary"):
                st.session_state["reorder_open_dialog"] = False
                st.rerun()

        if st.session_state.get("reorder_open_dialog"):
            st.session_state["reorder_open_dialog"] = False
            _reco_dialog()
    else:
        st.markdown('<div id="reco-details-anchor"></div>', unsafe_allow_html=True)
        with st.expander(f"Recommendation — {_rfull}", expanded=True):
            _render_selected_reco(cur)
        scroll_into_view("reco-details-anchor")

    # ==================================================================
    # SECTION 7 — Complete recommendation table
    # ==================================================================
    with st.expander("View complete reorder recommendations", expanded=False):
        colmap = [
            ("action", "Action"), ("sku_name", "Product"), ("sku", "SKU"), ("channel", "Channel"),
            ("overall_risk_tier", "Risk"), ("stockout_probability", "Stockout Probability"),
            ("days_of_cover", "Days of Cover"), ("inventory_position_for_risk", "Inventory Position"),
            ("forecast_driven_reorder_point", "Reorder Point"), ("target_stock", "Target Stock"),
            ("raw_target_gap", "Raw Gap"), ("moq", "MOQ"), ("pack_size", "Pack Size"),
            ("recommended_order_quantity", "Proposed Quantity"),
            ("recommended_order_date", "Order Date"), ("expected_arrival_date", "Expected Arrival"),
            ("unit_cost_effective", "Unit Cost"), ("recommended_purchase_value", "Purchase Value"),
            ("approval_required", "Approval"), ("manual_review_required", "Manual Review"),
        ]
        src_cols = [c for c, _ in colmap if c in reco.columns]
        disp = reco[src_cols].copy()
        if "action" in disp.columns:
            disp["action"] = disp["action"].map(rs.reorder_action_label)
        if "overall_risk_tier" in disp.columns:
            disp["overall_risk_tier"] = disp["overall_risk_tier"].map(lambda t: str(t).capitalize())
        if "stockout_probability" in disp.columns:
            disp["stockout_probability"] = pd.to_numeric(disp["stockout_probability"], errors="coerce").map(format_percentage)
        for dcol in ("recommended_order_date", "expected_arrival_date"):
            if dcol in disp.columns:
                disp[dcol] = disp[dcol].map(_date_str)
        for ccol in ("unit_cost_effective", "recommended_purchase_value"):
            if ccol in disp.columns:
                disp[ccol] = disp[ccol].map(lambda v: format_currency(v, 2) if ccol == "unit_cost_effective" else format_currency(v))
        disp = disp.rename(columns=dict(colmap))
        _sty = disp.style
        for _c in ("Action", "Risk"):
            if _c in disp.columns:
                _sty = _sty.map(_reorder_action_cell_css if _c == "Action" else _risk_tier_cell_css,
                                subset=[_c])
        st.dataframe(_sty, use_container_width=True, hide_index=True)
        with st.container(key="ipa-export-3"):
            eu.render_table_export_menu(
                disp, filename_stem="reorder_recommendations_complete",
                title="Inventory & Reorder — Complete Recommendations",
                metadata={"Run": (ACTIVE_RUN or {}).get("run_id", "legacy"),
                          "Rows": len(disp)}, key="exp_reco_full")
        st.caption("One row per SKU/channel, read directly from the run's validated reorder artifact.")
        with st.expander("Advanced: technical & contract fields", expanded=False):
            tech = [c for c in reco.columns if c not in src_cols and c != "reason_trace"]
            if "reason_trace" in reco.columns:
                tech = tech + ["reason_trace"]
            st.dataframe(reco[tech], use_container_width=True, hide_index=True)


def _page_inventory_reorder_legacy():
    render_page_header("Inventory & Reorder",
                       "Legacy fixed pilot · synthetic baseline snapshot · prioritised replenishment queue",
                       badges=ACTIVE_BADGES)
    synthetic_warning(INVENTORY_PAGE_WARNING)
    info_banner("Legacy fixed-pilot view: these are historical synthetic-baseline figures from "
                "inventory_context, NOT the forecast-driven Phase C recommendations. Activate a "
                "completed forecast run to see forecast-driven reorder proposals.", kind="info")
    if inv_raw is None:
        empty_state("Synthetic inventory context not generated", "data/processed/inventory_context.parquet is missing or unreadable.", "mail")
        return
    if inv_f is None or inv_f.empty:
        empty_state("No SKUs match the current filters", "Widen the SKU, category or brand filters in the sidebar.", "search")
        return

    inv_status_df = inv_f.copy()
    inv_status_df["status"] = inv_status_df.apply(classify_reorder_status, axis=1)
    n_skus = inv_status_df["sku"].nunique()

    # --- 4 KPI cards only ---
    render_kpi_row([
        dict(label="Simulated Inventory Value", value=format_currency(inv_status_df["inventory_value"].sum()), icon="briefcase", tone="teal"),
        dict(label="Recommended Purchase Value", value=format_currency(inv_status_df["recommended_purchase_value"].sum()), icon="coin", tone="amber"),
        dict(label="SKUs Below Reorder Point", value=f"{(inv_status_df['stock_on_hand'] <= inv_status_df['reorder_point']).sum()} / {n_skus}", icon="trending-down", tone="blue"),
        dict(label="Synthetic Stockout SKUs", value=f"{(inv_status_df['stock_on_hand'] <= 0).sum()} / {n_skus}", icon="alert-triangle", tone="risk"),
    ], n_cols=4)

    # --- Action queue table: the main operational output ---
    section_title("Priority Action Queue",
                  "Critical: stockout or ≤ 2 days cover · Reorder Now: recommended qty > 0 · "
                  "Watch: cover ≤ lead time · Healthy: otherwise.")
    table = inv_status_df.copy()
    table["inventory_position"] = table["stock_on_hand"] + table["on_order_quantity"]
    rank = {"Critical": 0, "Reorder Now": 1, "Watch": 2, "Healthy": 3}
    table["_rank"] = table["status"].map(rank)
    table = table.sort_values(["_rank", "days_of_cover"])
    view = pd.DataFrame({
        "Status": table["status"],
        "Product": table["sku"].map(lambda s: SKU_NAMES.get(s, s)),
        "SKU": table["sku"],
        "Category": table.get("category", "—"),
        "Sim. Stock": table["stock_on_hand"],
        "Days Cover": table["days_of_cover"].round(1),
        "Reorder Pt": table["reorder_point"],
        "Reco. Qty": table["recommended_order_quantity"],
        "Purchase Value": table["recommended_purchase_value"].map(lambda v: format_currency(v)),
        "Unit Cost": table["unit_cost_effective"].map(lambda v: format_currency(v, 2)),
        "Cost Source": table["cost_source"],
    })
    styler = view.style.map(_status_cell_css, subset=["Status"])
    st.dataframe(styler, width="stretch", hide_index=True, height=430)
    with st.container(key="ipa-export-4"):
        eu.render_table_export_menu(view, filename_stem="inventory_action_queue",
                                    title="Inventory & Reorder — Priority Action Queue", metadata={"Run": (ACTIVE_RUN or {}).get("run_id", "legacy"), "Category": (ACTIVE_RUN or {}).get("category", "—"), "As-of": (ACTIVE_RUN or {}).get("as_of_date", "—")},
                                    key="exp_invqueue")

    # --- Two focused charts ---
    c1, c2 = st.columns(2)
    with c1:
        d = inv_status_df.sort_values("days_of_cover")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=d["sku"], y=d["stock_on_hand"], name="Simulated stock", marker_color=COLORS["teal"]))
        fig.add_trace(go.Scatter(x=d["sku"], y=d["reorder_point"], name="Reorder point", mode="markers",
                                  marker=dict(color=COLORS["red"], size=8, symbol="diamond")))
        fig.update_layout(**plotly_layout(height=320))
        style_axes(fig)
        fig.update_xaxes(title="")
        render_chart(fig, "Inventory Position vs Reorder Point")
    with c2:
        d = inv_status_df.sort_values("days_of_cover")
        colors = [COLORS["red"] if v <= 2 else COLORS["amber"] if v <= d["lead_time_days"].iloc[i] else COLORS["success"]
                  for i, v in enumerate(d["days_of_cover"])]
        fig = px.bar(d, x="sku", y="days_of_cover")
        fig.update_traces(marker_color=colors)
        fig.update_layout(**plotly_layout(legend=False, height=320))
        style_axes(fig)
        fig.update_xaxes(title="")
        render_chart(fig, "Days of Cover by SKU")

    # --- Secondary breakdowns in an expander ---
    with st.expander("Cost quality & category breakdowns", expanded=False):
        e1, e2 = st.columns(2)
        with e1:
            if "category" in inv_status_df.columns:
                cat_val = inv_status_df.groupby("category", as_index=False)["inventory_value"].sum()
                n_cats = len(cat_val)
                colors = (DONUT_COLORS * ((n_cats // len(DONUT_COLORS)) + 1))[:n_cats]
                hover_colors = (DONUT_HOVER * ((n_cats // len(DONUT_HOVER)) + 1))[:n_cats]
                fig = go.Figure(go.Pie(
                    labels=cat_val["category"], values=cat_val["inventory_value"], hole=0.5,
                    marker=dict(colors=colors, line=dict(color="#FFFFFF", width=3)),
                    hovertemplate="<b>%{label}</b><br>Value: PKR %{value:,.0f}<br>Share: %{percent}<extra></extra>",
                    textinfo="label+percent", textfont=dict(size=12, color="#FFFFFF"),
                ))
                fig.update_layout(**plotly_layout(height=300))
                # Make donut clickable to filter by category
                event = st.plotly_chart(fig, use_container_width=True, key="donut_cat",
                                        on_select="rerun", selection_mode="points")
                if event and event.selection and event.selection.point_indices:
                    clicked_idx = event.selection.point_indices[0]
                    clicked_cat = cat_val["category"].iloc[clicked_idx]
                    st.session_state["flt_category"] = [clicked_cat]
                    st.rerun()
                st.markdown('<div class="ipa-card-title">Inventory Value by Category</div>'
                            '<div class="ipa-card-sub">Click a segment to filter the dashboard</div>',
                            unsafe_allow_html=True)
            else:
                empty_state("Category mapping unavailable", "model_panel.parquet needed to map categories.", "folder")
        with e2:
            cost_mix = inv_status_df["cost_is_imputed"].map({True: "Imputed", False: "Observed / Valid"}).value_counts()
            fig = go.Figure(go.Pie(labels=cost_mix.index, values=cost_mix.values, hole=0.55,
                                    marker=dict(colors=[COLORS["amber"], COLORS["success"]],
                                                line=dict(color="#FFFFFF", width=3)),
                                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
                                    textinfo="label+percent", textfont=dict(size=12)))
            fig.update_layout(**plotly_layout(height=300))
            render_chart(fig, "Observed vs Imputed Cost Coverage")
        e3, e4 = st.columns(2)
        with e3:
            if "category" in inv_status_df.columns:
                cat_val = inv_status_df.groupby("category", as_index=False)["recommended_purchase_value"].sum()
                fig = px.bar(cat_val, x="category", y="recommended_purchase_value", color="category",
                             color_discrete_sequence=CATEGORICAL)
                fig.update_layout(**plotly_layout(legend=False, height=300))
                style_axes(fig)
                fig.update_xaxes(title="")
                render_chart(fig, "Recommended Purchase Value by Category")
        with e4:
            flags = inv_status_df["cost_quality_flag"].dropna().str.split(";").explode().str.strip()
            flags = flags[flags != ""]
            if flags.empty:
                empty_state("No cost-quality flags", "All SKUs in view passed cost checks cleanly.", "check-circle")
            else:
                flag_counts = flags.value_counts()
                fig = px.bar(flag_counts, orientation="h")
                fig.update_traces(marker_color=COLORS["red"])
                fig.update_layout(**plotly_layout(legend=False, height=300))
                style_axes(fig)
                fig.update_xaxes(title="Occurrences")
                render_chart(fig, "Cost-Quality Flags")


# ==========================================================================
# PAGE 5 — STOCKOUT SCENARIO LAB
# ==========================================================================
def page_stockout_risk():
    # ------------------------------------------------------------------
    # Display-only helpers. This page ONLY reads validated decision
    # artifacts — it never recomputes risk, tiers, probabilities or money.
    # ------------------------------------------------------------------
    TIER_HEX = {"critical": COLORS["red"], "high": COLORS["amber"], "medium": COLORS["blue"],
                "watch": COLORS["blue"], "low": COLORS["success"], "healthy": COLORS["success"],
                "unknown": COLORS["slate"]}

    def _tier_hex(t):
        return TIER_HEX.get(str(t).strip().lower(), COLORS["slate"])

    def _tier_class(t):
        t = str(t).strip().lower()
        if t == "medium":
            return "watch"
        return t if t in ("critical", "high", "watch", "low", "healthy", "unknown") else "unknown"

    def _tier_chip(t):
        label = str(t).strip().capitalize() or "Unknown"
        return f'<span class="ipa-rtier ipa-rtier-{_tier_class(t)}">{label}</span>'

    def _esc(x):
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _safe_key(x):
        return "".join(c if c.isalnum() else "-" for c in str(x))

    def _help(txt, defn):
        return f'<span title="{_esc(defn)}">{txt}</span>'

    def _date_str(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        try:
            d = pd.to_datetime(x)
            return "—" if pd.isna(d) else d.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            s = str(x).strip()
            return s if s and s.lower() not in ("nat", "nan", "none") else "—"

    def _flags_to_list(flags):
        if flags is None:
            return []
        if isinstance(flags, (list, tuple, np.ndarray)):
            return [str(x).strip() for x in list(flags) if str(x).strip()]
        if isinstance(flags, float) and pd.isna(flags):
            return []
        s = str(flags).strip()
        if not s or s.lower() in ("nan", "none", "[]"):
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except (ValueError, TypeError):
            pass
        for sep in ("|", ";", ","):
            if sep in s:
                return [p.strip() for p in s.split(sep) if p.strip()]
        return [s]

    def _flag_pretty(f):
        return _esc(str(f).replace("_", " ").strip().capitalize())

    def _meta_val(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        s = str(x).strip()
        return s if s and s.lower() not in ("nan", "none") else None

    # ------------------------------------------------------------------
    # Header (dynamic badges + a visible, non-expander assumptions notice)
    # ------------------------------------------------------------------
    badges = None
    if DATA_MODE == "run" and ACTIVE_RUN:
        n_sel = ACTIVE_RUN.get("selected_sku_count")
        cat = ACTIVE_RUN.get("category") or "All categories"
        asof_raw = ACTIVE_RUN.get("as_of_date")
        asof = rs.format_local_datetime(asof_raw, include_time=False) if asof_raw else "—"
        opm = ACTIVE_RUN.get("operational_model") or "—"
        dstat = ACTIVE_RUN.get("decisioning_status") or (
            "ready" if CTX.get("has_stockout_risk") else "unavailable")
        badges = [
            ("box", f"{n_sel if n_sel is not None else '—'} selected SKUs"),
            ("tag", str(cat)),
            ("calendar", f"As-of {asof}"),
            ("layers", f"Model: {opm}"),
            ("shield-check", f"Decisioning: {dstat}"),
        ]
    render_page_header(
        "Stockout Risk",
        "Forecast-driven inventory exposure · lead-time demand · projected stockout probability",
        badges=badges)
    render_filter_bar(products=True, category=True, key_prefix="risk")

    # ------------------------------------------------------------------
    # Run-scoped guards → polished empty states (legacy + old runs)
    # ------------------------------------------------------------------
    if DATA_MODE != "run":
        empty_state(
            "Stockout risk is forecast-run scoped",
            "Generate and activate a completed forecast run to view forecast-driven stockout risk. "
            "The legacy fixed pilot does not carry the validated decision artifacts this page reads.",
            "package-search")
        return
    if not CTX.get("has_stockout_risk") or risk_raw is None:
        empty_state(
            "No decision artifacts for this run",
            "Generate and activate a completed forecast run to view forecast-driven stockout risk. "
            "This run was produced before Phase B decisioning, so it carries no stockout-risk artifacts to read.",
            "package-search")
        return

    # Read-only copy of the validated artifact, narrowed to the sidebar product filter
    risk = risk_raw.copy()
    # Narrow through the SHARED product filter, then join display product names (the
    # artifact's sku_name is null — names live in model_panel).
    risk = narrow_to_filtered_skus(risk)
    risk = enrich_product_names(risk)
    if risk.empty:
        empty_state("No SKUs in the current filter",
                    "Clear the product filter in the sidebar to see the full stockout-risk queue.",
                    "search")
        return

    op_model = str(risk["operational_model"].iloc[0]) if "operational_model" in risk.columns else "—"
    try:
        op_h = int(risk["operational_horizon"].iloc[0])
    except (ValueError, TypeError, KeyError):
        op_h = None

    info_banner(
        "Inventory stock may be synthetically reconstructed. Lead time, service level, MOQ and "
        "pack-size values may use pilot assumptions. Risk probabilities are planning estimates, "
        "not guarantees.", kind="synthetic")

    tiers = risk["overall_risk_tier"].astype(str).str.lower()

    # ==================================================================
    # SECTION 1 — Portfolio KPIs (exactly five)
    # ==================================================================
    section_title("Portfolio risk at a glance",
                  "Aggregated from the validated decision artifacts for the current selection.")
    n_crit_high = int(tiers.isin(["critical", "high"]).sum())
    n_projected = int(risk["projected_stockout_date"].notna().sum())
    rev_total, rev_missing = rs.risk_revenue_at_risk_total(risk)
    cover_series = pd.to_numeric(risk["forecast_days_of_cover"], errors="coerce").dropna()
    avg_cover = float(cover_series.mean()) if not cover_series.empty else None
    n_review = int(risk["manual_review_required"].astype(bool).sum())

    kpis = [
        {"label": "Critical / High Risk SKUs", "value": format_number(n_crit_high),
         "icon": "alert-triangle", "tone": "red",
         "sub": _help("Need attention now",
                      "Count of SKUs whose overall risk tier is Critical or High in this run.")},
        {"label": "Projected Stockouts", "value": format_number(n_projected),
         "icon": "trending-down", "tone": "amber",
         "sub": _help("Have a projected stockout date",
                      "SKUs with a non-null projected stockout date within the forecast horizon.")},
        {"label": "Estimated Revenue at Risk", "value": format_currency(rev_total),
         "icon": "coin", "tone": "navy",
         "sub": _help((f"{rev_missing} SKU(s) unpriced" if rev_missing else "All priced SKUs included"),
                      "Sum of estimated revenue at risk across priced SKUs. Unpriced SKUs (null) are "
                      "excluded from the total, never counted as zero.")},
        {"label": "Average Days of Cover",
         "value": (format_number(avg_cover, 1) if avg_cover is not None else "—"),
         "icon": "clock", "tone": "blue",
         "sub": _help("Across SKUs with a cover estimate",
                      "Mean forecast days of cover over SKUs where it is defined; SKUs without an "
                      "estimate are excluded (never treated as zero).")},
    ]
    # Exactly four equal-height primary KPIs. Manual-review count moved to a chip beside
    # the queue filters (see SECTION 3) so it no longer competes with the headline numbers.
    with st.container(key="ipa-kpirow"):
        render_kpi_row(kpis, n_cols=4)

    # ==================================================================
    # SECTION 2 — Risk composition (secondary → progressive disclosure)
    # ==================================================================
    with st.expander("Risk composition charts", expanded=False):
        tier_order = ["critical", "high", "medium", "watch", "low", "healthy", "unknown"]
        vc = tiers.value_counts()
        present = [t for t in tier_order if t in vc.index] + [t for t in vc.index if t not in tier_order]

        col_a, col_b = st.columns(2)
        with col_a:
            donut = go.Figure(go.Pie(
                labels=[t.capitalize() for t in present],
                values=[int(vc[t]) for t in present],
                hole=0.62, sort=False, direction="clockwise",
                marker=dict(colors=[_tier_hex(t) for t in present], line=dict(color="white", width=1.5)),
                textinfo="value",
                hovertemplate="%{label}: %{value} SKUs (%{percent})<extra></extra>"))
            donut.update_layout(**plotly_layout(height=330, legend=True))
            donut.update_layout(annotations=[dict(
                text=f"{len(risk)}<br>SKUs", x=0.5, y=0.5, showarrow=False,
                font=dict(size=15, color=COLORS["navy"]))])
            render_chart(donut, "Risk Tier Distribution",
                         "Fixed semantic colors — Critical (red) through Unknown (grey); Unknown is never green.")
        with col_b:
            rev_all = pd.to_numeric(risk["estimated_revenue_at_risk"], errors="coerce")
            rev_max = float(rev_all.max()) if rev_all.notna().any() and float(rev_all.max()) > 0 else None

            def _msize(v):
                # bubble size ∝ revenue at risk when available; a stable default otherwise
                if rev_max and pd.notna(v) and float(v) > 0:
                    return 11 + 24 * (float(v) / rev_max)
                return 11

            scatter = go.Figure()
            for t in present:
                sub = risk[tiers == t]
                if sub.empty:
                    continue
                names = [rs.full_product_label(n, s) for n, s in zip(sub.get("sku_name"), sub["sku"])]
                cust = list(zip(
                    names,
                    sub["sku"].astype(str),
                    [(_meta_val(c) or "—") for c in sub["channel"]],
                    [str(x).capitalize() for x in sub["overall_risk_tier"]],
                    [format_number(v, 0) for v in sub["stock_on_hand"]],
                    [_date_str(x) for x in sub["projected_stockout_date"]],
                    [format_number(v, 0) for v in sub["expected_shortage_units"]],
                    [format_currency(v) for v in sub["estimated_revenue_at_risk"]]))
                scatter.add_trace(go.Scatter(
                    x=pd.to_numeric(sub["forecast_days_of_cover"], errors="coerce"),
                    y=pd.to_numeric(sub["stockout_probability"], errors="coerce"),
                    mode="markers", name=t.capitalize(),
                    marker=dict(size=[_msize(v) for v in sub["estimated_revenue_at_risk"]],
                                color=_tier_hex(t), line=dict(color="white", width=1)),
                    customdata=cust,
                    hovertemplate=("<b>%{customdata[0]}</b><br>SKU %{customdata[1]} · %{customdata[2]}<br>"
                                   "Tier: %{customdata[3]}<br>P(stockout): %{y:.0%}<br>"
                                   "Days of cover: %{x:.1f}<br>Current stock: %{customdata[4]} u<br>"
                                   "Projected stockout: %{customdata[5]}<br>"
                                   "Expected shortage: %{customdata[6]} u<br>"
                                   "Revenue at risk: %{customdata[7]}<extra></extra>")))
            scatter.update_layout(**plotly_layout(height=330, legend=True))
            style_axes(scatter)
            scatter.update_yaxes(range=[-0.03, 1.03], tickformat=".0%", title_text="P(stockout) in lead time")
            scatter.update_xaxes(title_text="Forecast days of cover")
            render_chart(scatter, "Stockout Probability vs Days of Cover",
                         "Top-left corner = most urgent; bubble size ∝ revenue at risk. Hover for full detail.")

    # ==================================================================
    # SECTION 3 — Priority risk queue (filters + clickable cards)
    # ==================================================================
    section_title("Priority risk queue",
                  "Ranked by urgency: tier, then probability, projected date, revenue and name.")
    with st.container(key="ipa-toolbar"):
        fc = st.columns([1.2, 1.8, 1, 1])
        with fc[0]:
            tier_opts = ["All tiers"] + [t.capitalize() for t in present]
            tier_pick = st.selectbox("Risk tier", options=tier_opts, key="risk_tier")
        with fc[1]:
            query = st.text_input("Search product or SKU", key="risk_query",
                                  placeholder="product name or SKU code")
        with fc[2]:
            proj_only = st.checkbox("Projected only", key="risk_projected_only")
        with fc[3]:
            review_only = st.checkbox("Review only", key="risk_review_only")

    tier_arg = None if tier_pick == "All tiers" else tier_pick
    filtered = rs.filter_risk_queue(risk, tier=tier_arg, query=query,
                                    projected_only=proj_only, review_only=review_only)
    ranked = rs.sort_risk_queue(filtered)
    valid_skus = ranked["sku"].astype(str).tolist() if ranked is not None and not ranked.empty else []

    # Session-state selection with a safe fallback to the top-ranked visible SKU
    cur = st.session_state.get("risk_selected_sku")
    if cur not in valid_skus:
        cur = valid_skus[0] if valid_skus else None
        st.session_state["risk_selected_sku"] = cur

    if not valid_skus:
        info_banner("No SKUs match these filters. Adjust the tier, search or toggles above.", kind="info")
        return

    # Manual-review chip + export sit beside the filters (review is no longer a primary KPI).
    mc1, mc2 = st.columns([3, 1])
    with mc1:
        n_review_v = int(pd.Series(filtered.get("manual_review_required", pd.Series(dtype=bool)))
                         .astype(bool).sum()) if not filtered.empty else 0
        st.markdown(
            f'<div class="ipa-src-row"><span class="ipa-tier ipa-tier-unknown">'
            f'{len(ranked)} in queue</span>'
            f'<span class="ipa-tier ipa-tier-{"high" if n_review_v else "healthy"}">'
            f'⚑ {n_review_v} manual review</span></div>', unsafe_allow_html=True)
    with mc2:
        with st.container(key="ipa-export-5"):
            _q_export = rs.risk_queue_export_frame(ranked) if hasattr(rs, "risk_queue_export_frame") else ranked
            eu.render_table_export_menu(
                _q_export, filename_stem="stockout_priority_queue",
                title="Stockout Risk — Priority Queue",
                metadata={"Run": (ACTIVE_RUN or {}).get("run_id", "legacy"),
                          "Category": (ACTIVE_RUN or {}).get("category", "—"),
                          "As-of": (ACTIVE_RUN or {}).get("as_of_date", "—")},
                key="exp_riskqueue")

    # ---- dense selectable rows (fixed height, 8 per page, no unbounded growth) ----
    PAGE = 8
    shown = int(st.session_state.get("risk_queue_shown", PAGE))
    shown = max(PAGE, min(shown, len(ranked)))
    for rec_row in ranked.head(shown).to_dict("records"):
        sku = str(rec_row["sku"])
        safe = _safe_key(sku)
        tier = str(rec_row.get("overall_risk_tier") or "unknown").lower()
        name = rec_row.get("sku_name")
        nm = None if (name is None or (isinstance(name, float) and pd.isna(name))) else str(name).strip()
        disp = nm if (nm and nm.lower() != "nan") else sku
        full = rs.full_product_label(name, sku)
        is_sel = (sku == str(cur))
        rcols = st.columns([3.2, 1.1, 1.0, 1.2, 1.2, 1.0])
        with rcols[0]:
            st.markdown(
                f'<div class="ipa-qrow ipa-q-{tier}" title="{_esc(full)}">'
                f'<div style="min-width:0;"><div class="q-name">{_esc(disp)}</div>'
                f'<div class="q-sub">SKU {_esc(sku)}'
                f'{" · ⚑ review" if bool(rec_row.get("manual_review_required")) else ""}</div></div>'
                f'</div>', unsafe_allow_html=True)
        rcols[1].markdown(f'<div class="ipa-qcell">{_tier_chip(tier)}</div>', unsafe_allow_html=True)
        rcols[2].markdown(f'<div class="ipa-qcell"><b>{format_percentage(rec_row.get("stockout_probability"))}</b>'
                          f'<div class="q-sub">P(stockout)</div></div>', unsafe_allow_html=True)
        rcols[3].markdown(f'<div class="ipa-qcell"><b>{format_number(rec_row.get("forecast_days_of_cover"), 1)}</b>'
                          f'<div class="q-sub">days cover</div></div>', unsafe_allow_html=True)
        rcols[4].markdown(f'<div class="ipa-qcell"><b>{_date_str(rec_row.get("projected_stockout_date"))}</b>'
                          f'<div class="q-sub">projected</div></div>', unsafe_allow_html=True)
        with rcols[5]:
            if st.button("Details", key=f"riskbtn-{safe}", width="stretch",
                         type="primary" if is_sel else "secondary",
                         help=f"Open full details for {full}"):
                st.session_state["risk_selected_sku"] = sku
                st.session_state["risk_open_dialog"] = True
                st.rerun()
    if len(ranked) > shown:
        if st.button(f"Show more ({len(ranked) - shown} remaining)", key="risk_show_more"):
            st.session_state["risk_queue_shown"] = shown + PAGE
            st.rerun()
    elif shown > PAGE:
        if st.button("Show fewer", key="risk_show_fewer"):
            st.session_state["risk_queue_shown"] = PAGE
            st.rerun()


    # ==================================================================
    # SECTION 4 — Selected SKU deep dive
    # ==================================================================
    # ------------------------------------------------------------------
    # Selected-product details — rendered inside st.dialog when available so the
    # user never scrolls to the page bottom; otherwise a compact panel right
    # under the queue. Content is identical in both paths.
    # ------------------------------------------------------------------
    def _render_selected_detail(cur):
        sel = risk[risk["sku"].astype(str) == str(cur)]
        if sel.empty:
            return
        r = sel.iloc[0]
        full = rs.full_product_label(r.get("sku_name"), r["sku"])

        cat_val = brand_val = None
        if mp_raw is not None and "sku" in mp_raw.columns:
            m = mp_raw[mp_raw["sku"].astype(str) == str(r["sku"])]
            if not m.empty:
                cat_val = _meta_val(m["category"].iloc[0]) if "category" in m.columns else None
                brand_val = _meta_val(m["brand"].iloc[0]) if "brand" in m.columns else None
        meta_bits = [b for b in (cat_val, brand_val, _meta_val(r.get("channel"))) if b]


        st.markdown(
            '<div class="ipa-dd-sub">'
            f'{_esc(" · ".join(meta_bits) if meta_bits else "—")} &nbsp;{_tier_chip(r.get("overall_risk_tier"))}'
            '</div>', unsafe_allow_html=True)

        dd = [
            {"label": "Overall risk tier",
             "value": (str(r.get("overall_risk_tier") or "—").capitalize()),
             "icon": "alert-triangle", "tone": rs.risk_tier_tone(r.get("overall_risk_tier")),
             "sub": "Worse of the probability & cover tiers"},
            {"label": "P(stockout) in lead time", "value": format_percentage(r.get("stockout_probability")),
             "icon": "percent", "tone": "amber",
             "sub": f"Service level {format_percentage(r.get('service_level'))}"},
            {"label": "Forecast days of cover", "value": format_number(r.get("forecast_days_of_cover"), 1),
             "icon": "clock", "tone": "blue",
             "sub": f"Lead time {format_number(r.get('lead_time_days'), 0)} d"},
            {"label": "Projected stockout date", "value": _date_str(r.get("projected_stockout_date")),
             "icon": "calendar", "tone": "amber",
             "sub": (f"In {format_number(r.get('days_until_projected_stockout'), 0)} days"
                     if pd.notna(r.get("days_until_projected_stockout"))
                     else "No stockout projected in horizon")},
        ]
        render_kpi_row(dd, n_cols=4)

        # Tabs keep the modal short: Summary is the answer, the rest is on demand.
        _t_sum, _t_traj, _t_why, _t_data = st.tabs(
            ["Summary", "Trajectory", "Why flagged", "Data"])

        stock_disp = format_number(r.get("stock_on_hand"), 0)
        if bool(r.get("stock_on_hand_is_synthetic")):
            stock_disp += " (synthetic)"
        ltd_p50 = pd.to_numeric(pd.Series([r.get("lead_time_demand_p50")]), errors="coerce").iloc[0]
        lt_days = pd.to_numeric(pd.Series([r.get("lead_time_days")]), errors="coerce").iloc[0]
        mean_daily = (ltd_p50 / lt_days) if (pd.notna(ltd_p50) and pd.notna(lt_days) and lt_days > 0) else None
        short_disp = format_number(r.get("expected_shortage_units"), 0)
        short_disp = f"{short_disp} u" if short_disp != "—" else "—"
        rows = [
            ("Current stock on hand", stock_disp),
            ("Stock source", _meta_val(r.get("stock_source")) or "—"),
            ("Reported on-order qty", format_number(r.get("reported_on_order_quantity"), 0)),
            ("Usable on-order qty", format_number(r.get("usable_on_order_quantity"), 0)),
            ("Inventory position (for risk)", format_number(r.get("inventory_position_for_risk"), 0)),
            ("Mean daily forecast", format_number(mean_daily, 1)),
            ("Lead-time days",
             f"{format_number(r.get('lead_time_days'), 0)} ({_meta_val(r.get('lead_time_source')) or '—'})"),
            ("Lead-time demand (mean / P80 / P95)",
             f"{format_number(r.get('lead_time_demand_p50'), 0)} / "
             f"{format_number(r.get('lead_time_demand_p80'), 0)} / "
             f"{format_number(r.get('lead_time_demand_p95'), 0)}"),
            ("Lead-time demand σ", format_number(r.get("lead_time_sigma"), 1)),
            ("Safety stock", format_number(r.get("safety_stock"), 0)),
            ("Reorder point", format_number(r.get("reorder_point"), 0)),
            ("Expected shortage", short_disp),
            ("Estimated revenue at risk", format_currency(r.get("estimated_revenue_at_risk"))),
            ("Service-level target", format_percentage(r.get("service_level"))),
            ("Operational model", _esc(op_model)),
            ("Uncertainty method", _meta_val(r.get("uncertainty_method")) or "—"),
            ("Confidence", _meta_val(r.get("confidence_label")) or "—"),
        ]

        traj_ok = bool(CTX.get("has_stockout_trajectory")) and traj_raw is not None
        tdf, twarn = (rs.trajectory_for_sku(traj_raw, cur, horizon=op_h)
                      if traj_ok else (pd.DataFrame(), []))
        for w in twarn:
            info_banner(w, kind="info")
        has_traj = traj_ok and tdf is not None and not tdf.empty

        with _t_sum, st.container(height=460):

            metric_panel("Inventory & demand inputs", rows,
                         sub="“—” means the value is not available — it is never shown as zero.")
        with _t_traj, st.container(height=460):
            if has_traj:
                inv = go.Figure()
                # cumulative demand P50–P95 uncertainty band (upper first, then fill down to P50)
                if {"demand_p95", "demand_p50"}.issubset(tdf.columns):
                    inv.add_trace(go.Scatter(
                        x=tdf["date"], y=tdf["demand_p95"], mode="lines", line=dict(width=0),
                        hoverinfo="skip", showlegend=False, name="P95"))
                    inv.add_trace(go.Scatter(
                        x=tdf["date"], y=tdf["demand_p50"], mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor="rgba(14,124,123,0.12)", hoverinfo="skip",
                        name="Demand P50–P95 band"))
                inv.add_trace(go.Scatter(
                    x=tdf["date"], y=tdf["cumulative_demand_mean"], name="Cumulative demand (P50)",
                    mode="lines", line=dict(color=COLORS["teal"], width=2, dash="dash"),
                    hovertemplate="%{x|%Y-%m-%d}<br>Cumulative demand: %{y:.0f} u<extra></extra>"))
                inv.add_trace(go.Scatter(
                    x=tdf["date"], y=tdf["projected_p50_inventory"], name="Projected inventory (P50)",
                    mode="lines+markers", line=dict(color=COLORS["navy"], width=2.5),
                    hovertemplate="%{x|%Y-%m-%d}<br>Projected inventory: %{y:.0f} u<extra></extra>"))
                inv.add_hline(y=0, line_dash="dot", line_color=COLORS["red"], line_width=1.5,
                              annotation_text="Zero stock", annotation_position="bottom right")
                rop = r.get("reorder_point")
                if pd.notna(rop):
                    inv.add_hline(y=float(rop), line_dash="dash", line_color=COLORS["amber"], line_width=1,
                                  annotation_text="Reorder point", annotation_position="top right")
                # lead-time window shading (the decision-relevant horizon)
                if pd.notna(lt_days) and float(lt_days) > 0:
                    start = pd.to_datetime(tdf["date"].iloc[0])
                    # annotations are vertically offset so the two labels never collide
                    inv.add_vrect(x0=start, x1=start + pd.Timedelta(days=int(lt_days)),
                                  fillcolor=COLORS["navy"], opacity=0.05, line_width=0,
                                  annotation_text="Lead-time window",
                                  annotation_position="bottom left",
                                  annotation_font_size=10,
                                  annotation_font_color=COLORS["slate"])
                # projected stockout date marker (label sits above the lead-time label)
                psd = r.get("projected_stockout_date")
                if pd.notna(psd):
                    inv.add_vline(x=pd.to_datetime(psd), line_dash="dash", line_color=COLORS["red"],
                                  line_width=1.2, annotation_text="Projected stockout",
                                  annotation_position="top right",
                                  annotation_font_size=10,
                                  annotation_font_color=COLORS["red"])
                inv.update_layout(**plotly_layout(height=340, legend=True))
                style_axes(inv)
                inv.update_yaxes(title_text="Units")
                render_chart(inv, f"Inventory trajectory — {_esc(full)}",
                             "Forecast-driven; zero-stock line marks the projected stockout, shaded = lead-time "
                             "window, band = cumulative demand P50–P95.")
            else:
                with st.container(border=True):
                    empty_state("Daily trajectory unavailable",
                                "This run did not persist a daily stockout trajectory for this SKU.",
                                "circle-dashed")

        if has_traj:
            pf = go.Figure()
            pf.add_trace(go.Scatter(
                x=tdf["date"], y=tdf["cumulative_stockout_probability"], name="Cumulative P(stockout)",
                mode="lines", fill="tozeroy", line=dict(color=COLORS["red"], width=2.5),
                hovertemplate="%{x|%Y-%m-%d}<br>P(stockout): %{y:.0%}<extra></extra>"))
            for thr, lab in ((0.5, "50%"), (0.8, "80%"), (0.95, "95%")):
                pf.add_hline(y=thr, line_dash="dot", line_color=COLORS["slate"], line_width=1,
                             annotation_text=lab, annotation_position="right")
            pf.update_layout(**plotly_layout(height=300, legend=False))
            style_axes(pf)
            pf.update_yaxes(range=[0, 1.03], tickformat=".0%", title_text="P(stockout)")
            render_chart(pf, "Cumulative stockout probability",
                         "Probability of running out by each day, with 50 / 80 / 95% reference thresholds.")

            tcols = ["date", "daily_demand_mean", "demand_p50", "demand_p80", "demand_p95",
                     "projected_p50_inventory", "cumulative_stockout_probability"]
            tbl = tdf[[c for c in tcols if c in tdf.columns]].copy()
            tbl["date"] = pd.to_datetime(tbl["date"]).dt.strftime("%Y-%m-%d")
            for c in ("daily_demand_mean", "demand_p50", "demand_p80", "demand_p95",
                      "projected_p50_inventory"):
                if c in tbl.columns:
                    tbl[c] = pd.to_numeric(tbl[c], errors="coerce").round(1)
            if "cumulative_stockout_probability" in tbl.columns:
                tbl["cumulative_stockout_probability"] = pd.to_numeric(
                    tbl["cumulative_stockout_probability"], errors="coerce").map(format_percentage)
            tbl = tbl.rename(columns={
                "date": "Date", "daily_demand_mean": "Daily Forecast",
                "demand_p50": "Cumulative Demand P50", "demand_p80": "Cumulative Demand P80",
                "demand_p95": "Cumulative Demand P95", "projected_p50_inventory": "Projected P50 Inventory",
                "cumulative_stockout_probability": "Cumulative Stockout Probability"})
            _daily_tbl = tbl          # exposed to the Data tab below


        # ==================================================================
        # SECTION 5 — Why this SKU is flagged (full reason trace)
        # ==================================================================
        with _t_why:
            reason = r.get("reason_trace")
            reason_txt = "" if (reason is None or (isinstance(reason, float) and pd.isna(reason))) else str(reason)
            st.markdown(
                f'<div class="ipa-reason"><div class="h">{icon_svg("file-text", 16)} Decision rationale</div>'
                f'{_esc(reason_txt) if reason_txt else "No reason trace was recorded for this SKU."}</div>',
                unsafe_allow_html=True)

        # ==================================================================
        # SECTION 6 — Assumptions, uncertainty and data limitations
        # ==================================================================
        with _t_data, st.container(height=460):
            try:
                st.markdown('<div class="ipa-card-title">Daily forecast & inventory</div>',
                            unsafe_allow_html=True)
                st.dataframe(_daily_tbl, use_container_width=True, hide_index=True)
                with st.container(key="ipa-export-riskdaily"):
                    eu.render_table_export_menu(
                        _daily_tbl, filename_stem=f"trajectory_{cur}",
                        title=f"Daily forecast & inventory — {full}",
                        metadata={"SKU": cur}, key=f"exp_traj_{_safe_key(cur)}")
            except NameError:
                st.caption("No daily trajectory is available for this product.")
        with _t_why, st.container(height=460):
            with st.expander("Assumptions, uncertainty and data limitations", expanded=False):
                flag_items = _flags_to_list(r.get("assumption_flags"))
                flags_l = [f.lower() for f in flag_items]

                def _num1(x):
                    return pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]

                def _yn(b):
                    return "Yes" if b else "No"

                lt_src = _meta_val(r.get("lead_time_source"))
                usable = _num1(r.get("usable_on_order_quantity"))
                reported = _num1(r.get("reported_on_order_quantity"))
                conditions = [
                    f"Stock synthetically reconstructed: {_yn(bool(r.get('stock_on_hand_is_synthetic')))}",
                    f"Lead time assumed: {_yn(lt_src is None or lt_src.lower() not in ('actual', 'supplier', 'confirmed'))}"
                    + (f" (source: {lt_src})" if lt_src else ""),
                    f"Service level defaulted: {_yn(any('service' in f for f in flags_l))} "
                    f"(target {format_percentage(r.get('service_level'))})",
                    f"Catalog price missing: {_yn(pd.isna(_num1(r.get('estimated_revenue_at_risk'))) or any('price' in f for f in flags_l))}",
                    f"Forecast horizon shorter than lead time: "
                    f"{_yn(r.get('lead_time_horizon_sufficient') is not None and not bool(r.get('lead_time_horizon_sufficient')))}",
                    f"On-order stock excluded from position: "
                    f"{_yn((not bool(r.get('on_order_available'))) or (pd.notna(usable) and usable == 0 and pd.notna(reported) and reported > 0))}",
                    f"Manual review required: {_yn(bool(r.get('manual_review_required')))}",
                    f"Uncertainty method: {_meta_val(r.get('uncertainty_method')) or '—'}",
                ]
                st.markdown("**Conditions detected for this SKU**")
                st.markdown("\n".join(f"- {c}" for c in conditions))
                if flag_items:
                    st.markdown("**Assumption flags recorded in the artifact**")
                    st.markdown("\n".join(f"- {_flag_pretty(f)}" for f in flag_items))
                st.markdown("**Known limitations**")
                for d in (
                    "Usable on-order quantity may be zero when there are no dated inbound arrivals.",
                    "The stockout probability is a Normal-distribution approximation.",
                    "Independent daily forecast errors are combined via RSS (root-sum-of-squares).",
                    "Lead time, MOQ and pack size may be pilot assumptions, not supplier-confirmed.",
                    "Catalog price may not be the confirmed revenue basis.",
                    "Figures are a forecast-driven estimate, not observed historical stockouts; "
                    "no purchase order or replenishment is created by this dashboard.",
                ):
                    st.markdown(f"- {d}")
                with st.expander("Advanced decision fields", expanded=False):
                    adv_fields = ["uncertainty_method", "confidence_label", "service_level",
                                  "lead_time_source", "stock_source", "on_order_available",
                                  "reported_on_order_quantity", "forecast_horizon_available",
                                  "lead_time_horizon_sufficient", "survives_forecast_horizon",
                                  "probability_risk_tier", "cover_risk_tier", "manual_review_required"]
                    adv_rows = []
                    for f in adv_fields:
                        v = r.get(f)
                        disp = "—" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
                        adv_rows.append({"field": f, "value": disp})
                    st.dataframe(pd.DataFrame(adv_rows), use_container_width=True, hide_index=True)


    # ---- product details: modal dialog when supported, compact inline panel otherwise ----
    _sel_row = risk[risk["sku"].astype(str) == str(cur)]
    _sel_full = (rs.full_product_label(_sel_row.iloc[0].get("sku_name"), _sel_row.iloc[0]["sku"])
                 if not _sel_row.empty else str(cur))
    if hasattr(st, "dialog"):
        @st.dialog(f"{_sel_full}", width="large")
        def _detail_dialog():
            _render_selected_detail(cur)
            if st.button("Close", key="risk_dialog_close", type="primary"):
                st.session_state["risk_open_dialog"] = False
                st.rerun()

        if st.session_state.get("risk_open_dialog"):
            st.session_state["risk_open_dialog"] = False   # one-shot: reopened by a Details click
            _detail_dialog()
    else:
        st.markdown('<div id="risk-details-anchor"></div>', unsafe_allow_html=True)
        with st.expander(f"Product details — {_sel_full}", expanded=True):
            _render_selected_detail(cur)
        scroll_into_view("risk-details-anchor")

    # ==================================================================
    # SECTION 7 — Complete risk dataset
    # ==================================================================
    with st.expander("View complete risk dataset", expanded=False):
        colmap = [
            ("overall_risk_tier", "Risk"), ("sku_name", "Product"), ("sku", "SKU"),
            ("channel", "Channel"), ("stockout_probability", "Stockout Probability"),
            ("forecast_days_of_cover", "Days of Cover"), ("projected_stockout_date", "Projected Stockout"),
            ("stock_on_hand", "Current Stock"), ("lead_time_demand_p50", "Lead-Time Demand"),
            ("safety_stock", "Safety Stock"), ("reorder_point", "Reorder Point"),
            ("expected_shortage_units", "Expected Shortage"),
            ("estimated_revenue_at_risk", "Revenue at Risk"),
            ("manual_review_required", "Manual Review"),
        ]
        src_cols = [c for c, _ in colmap if c in risk.columns]
        disp = risk[src_cols].copy()
        if "stockout_probability" in disp.columns:
            disp["stockout_probability"] = pd.to_numeric(
                disp["stockout_probability"], errors="coerce").map(format_percentage)
        if "projected_stockout_date" in disp.columns:
            disp["projected_stockout_date"] = disp["projected_stockout_date"].map(_date_str)
        if "estimated_revenue_at_risk" in disp.columns:
            disp["estimated_revenue_at_risk"] = disp["estimated_revenue_at_risk"].map(format_currency)
        if "overall_risk_tier" in disp.columns:
            disp["overall_risk_tier"] = disp["overall_risk_tier"].map(lambda t: str(t).capitalize())
        disp = disp.rename(columns=dict(colmap))
        _sty = disp.style
        if "Risk" in disp.columns:
            _sty = _sty.map(_risk_tier_cell_css, subset=["Risk"])
        st.dataframe(_sty, use_container_width=True, hide_index=True)
        with st.container(key="ipa-export-6"):
            eu.render_table_export_menu(
                disp, filename_stem="stockout_risk_complete",
                title="Stockout Risk — Complete Dataset",
                metadata={"Run": (ACTIVE_RUN or {}).get("run_id", "legacy"),
                          "Rows": len(disp)}, key="exp_risk_full")
        st.caption("One row per SKU/channel, read directly from the run's validated decision artifact.")
        with st.expander("Advanced: technical & model contract fields", expanded=False):
            tech = [c for c in risk.columns if c not in src_cols and c != "reason_trace"]
            if "reason_trace" in risk.columns:
                tech = tech + ["reason_trace"]
            st.dataframe(risk[tech], use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Column dictionaries for the Data Quality page (restored — referenced by the
# Feature Dictionary tab in page_data_quality).
# --------------------------------------------------------------------------
MODEL_PANEL_DICT = {
    "sku": "Unique SKU identifier used across all pilot datasets.",
    "product_id": "Internal Naheed product identifier linked to the SKU.",
    "channel": "Sales channel for the row. Pilot scope is naheed_web only.",
    "date": "Calendar date of the observation (daily grain).",
    "category": "Merchandise category (e.g. Health & Beauty, Groceries & Pets).",
    "sub_category": "Merchandise sub-category. Not populated for this pilot slice.",
    "brand": "Product brand.",
    "units_observed": "REAL target variable — units sold that day on naheed_web.",
    "effective_unit_price": "REAL historical selling price per unit after any discount.",
    "discount_amount": "REAL discount amount applied per unit.",
    "discount_pct": "REAL discount as a percentage of list price.",
    "on_promo": "REAL flag — 1 if the SKU was on an active promotion that day.",
    "promo_known_in_advance": "REAL flag — 1 if the promotion was scheduled/known ahead of the date.",
    "is_public_holiday": "REAL calendar flag — 1 if the date is a Pakistan public holiday.",
    "holiday_name": "Name of the public holiday, when applicable.",
    "is_payday_window": "REAL calendar flag — 1 if the date falls in a typical monthly payday window.",
    "day_of_week": "Day of week, 0 = Monday … 6 = Sunday.",
    "is_weekend": "REAL flag — 1 if Saturday or Sunday.",
    "week_of_year": "ISO week number.",
    "month": "Calendar month number.",
    "units_lag_1": "Units observed 1 day earlier — engineered demand feature.",
    "units_lag_7": "Units observed 7 days earlier — engineered demand feature.",
    "units_lag_14": "Units observed 14 days earlier — engineered demand feature.",
    "units_roll_mean_7": "Trailing 7-day rolling mean of units observed.",
    "units_roll_mean_28": "Trailing 28-day rolling mean of units observed.",
    "units_roll_std_7": "Trailing 7-day rolling standard deviation — a proxy for demand volatility.",
    "stock_on_hand": "SYNTHETIC daily stock reconstruction — not an observed inventory record.",
    "stock_on_hand_is_synthetic": "Flag confirming the stock_on_hand value is synthetic.",
    "stock_source": "Source method used to derive stock_on_hand.",
    "stock_generation_version": "Version tag of the synthetic stock reconstruction logic.",
    "product_active": "Whether the SKU was considered active/listed on that date.",
    "forecast_training_eligible": "Whether the row passes minimum-history rules for model training.",
    "data_quality_flag": "Data-quality notes for the row (e.g. insufficient history at activation).",
}


INVENTORY_CONTEXT_DICT = {
    "as_of_date": "Snapshot date the inventory context was generated for.",
    "sku": "Unique SKU identifier.",
    "product_id": "Internal Naheed product identifier.",
    "location_id": "Stock location scope (ALL = network-wide, not store-level).",
    "stock_on_hand": "SYNTHETIC baseline stock quantity for the SKU as of the snapshot date.",
    "stock_on_hand_is_synthetic": "Confirms the stock figure is a synthetic reconstruction.",
    "stock_source": "Method used to derive the synthetic stock figure.",
    "stock_snapshot_date": "Date the synthetic snapshot represents.",
    "stock_generation_method": "Algorithm used for stock reconstruction.",
    "stock_generation_version": "Version tag of the reconstruction logic.",
    "on_order_quantity": "Units already on order (currently unavailable/assumed 0 for this pilot).",
    "on_order_is_available": "Whether real on-order data was available (False = assumed).",
    "expected_daily_demand": "Forecast-derived expected daily demand used to size reorder policy.",
    "lead_time_days": "Assumed supplier lead time in days.",
    "lead_time_source": "Where the lead-time assumption came from.",
    "moq": "Assumed minimum order quantity.",
    "moq_source": "Source of the MOQ assumption.",
    "pack_size": "Assumed order pack/case size.",
    "pack_size_source": "Source of the pack-size assumption.",
    "safety_stock": "Buffer stock computed from demand and lead-time assumptions.",
    "reorder_point": "Stock level that should trigger a new purchase order.",
    "target_stock": "Target stock level after replenishment.",
    "days_of_cover": "Estimated days of stock remaining at current demand.",
    "is_perishable": "Whether the SKU is flagged as perishable.",
    "shelf_life_days": "Shelf life in days, when applicable.",
    "price": "Current selling price.",
    "unit_cost_observed": "Observed unit cost from source systems, when available.",
    "unit_cost_effective": "Unit cost actually used in value calculations (observed or imputed).",
    "cost_source": "Precedence source used to resolve unit cost.",
    "cost_is_valid": "Whether the observed cost passed validity checks.",
    "cost_is_imputed": "Whether the effective cost was imputed (fallback) rather than observed.",
    "cost_quality_flag": "Data-quality flags related to unit cost.",
    "cost_currency": "Currency of all cost/value figures (PKR).",
    "cost_basis": "Cost basis assumption (unit/pack, unconfirmed for this pilot).",
    "recommended_order_quantity": "Simulated purchase recommendation under baseline assumptions.",
    "recommended_purchase_value": "Recommended order quantity × effective unit cost.",
    "inventory_value": "Synthetic stock on hand × effective unit cost.",
    "is_dropship": "Whether the SKU is fulfilled via dropship.",
    "assumption_notes": "Free-text notes on assumptions applied for this SKU.",
}


def page_data_quality():
    render_page_header(
        "Data Quality & Assumptions",
        "Data contract · real vs synthetic · feature dictionary · validation",
        badges=ACTIVE_BADGES,
    )
    tab_contract, tab_split, tab_dict, tab_checks, tab_assume, tab_valid = st.tabs(
        ["Data Contract", "Real vs Synthetic", "Feature Dictionary", "Quality Checks", "Assumptions", "Validation"]
    )

    with tab_contract:
        if manifest:
            render_kpi_row([
                dict(label="Schema Version", value=str(manifest.get("schema_version", "—")), icon="file-text", tone="teal"),
                dict(label="As-Of Date", value=str(manifest.get("as_of_date", "—")), icon="calendar", tone="blue"),
                dict(label="Channel Scope", value=", ".join(manifest.get("channel_scope", [])) or "—", icon="globe", tone="amber"),
                dict(label="Forecast Horizons", value=", ".join(str(h) for h in manifest.get("forecast_horizon_days", [])) + " days", icon="sparkle", tone="slate"),
            ], n_cols=4, compact=True)
            hist_window = manifest.get("historical_window", ["—", "—"])
            cats_str = ", ".join(all_categories) if all_categories else "—"
            st.write("")
            render_kpi_row([
                dict(label="History Range", value=f"{hist_window[0]} → {hist_window[1]}", icon="calendar"),
                dict(label="SKU Count", value=str(manifest.get("sku_count", "—")), icon="tag"),
                dict(label="Categories", value=cats_str, icon="folder"),
                dict(label="Model Panel Rows", value=format_number(manifest.get("row_counts", {}).get("model_panel")), icon="bar-chart"),
            ], n_cols=4, compact=True)
        else:
            empty_state("Pilot manifest not found", "data/processed/pilot_manifest.json is missing or unreadable.", "file-text")

    with tab_split:
        info_banner("Demand history is <strong>real</strong>. Inventory, stockouts and replenishment are "
                    "<strong>synthetic / assumed</strong> pilot reconstructions.", kind="info")
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                title_html = (f'<div class="ipa-card-title" style="color:{COLORS["success"]}; display:flex; '
                               f'align-items:center; gap:8px;">{icon_svg("check-circle", 16)}Real fields</div>')
                st.markdown(title_html, unsafe_allow_html=True)
                real_fields = (manifest or {}).get("real_fields", [])
                if real_fields:
                    st.markdown("\n".join(f"- {f}" for f in real_fields))
                else:
                    st.caption("Not listed in manifest.")
        with c2:
            with st.container(border=True):
                title_html = (f'<div class="ipa-card-title" style="color:{COLORS["amber"]}; display:flex; '
                               f'align-items:center; gap:8px;">{icon_svg("flask", 16)}Synthetic / assumed fields</div>')
                st.markdown(title_html, unsafe_allow_html=True)
                synth_fields = (manifest or {}).get("synthetic_or_assumed_fields", [])
                if synth_fields:
                    st.markdown("\n".join(f"- {f}" for f in synth_fields))
                else:
                    st.caption("Not listed in manifest.")

    with tab_dict:
        with st.expander("model_panel.parquet", expanded=True):
            if mp_raw is not None:
                rows = [{"Column": c, "Type": str(mp_raw[c].dtype), "Description": MODEL_PANEL_DICT.get(c, "—")} for c in mp_raw.columns]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                empty_state("Not available", "model_panel.parquet is missing.", "mail")
        with st.expander("inventory_context.parquet", expanded=False):
            if inv_raw is not None:
                rows = [{"Column": c, "Type": str(inv_raw[c].dtype), "Description": INVENTORY_CONTEXT_DICT.get(c, "—")} for c in inv_raw.columns]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                empty_state("Not available", "inventory_context.parquet is missing.", "mail")
        with st.expander("forecast_features / forecast_frame", expanded=False):
            if ff_raw is not None:
                rows = [{"Column": c, "Type": str(ff_raw[c].dtype)} for c in ff_raw.columns]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                empty_state("Not generated yet", f"{ff_name} was not found under data/processed/.", "mail")
        with st.expander("stockout scenarios / replenishment events", expanded=False):
            if stockout_raw is not None:
                st.dataframe(pd.DataFrame({"Column": stockout_raw.columns, "Type": [str(t) for t in stockout_raw.dtypes]}),
                             width="stretch", hide_index=True)
            else:
                empty_state("Not generated yet", "data/synthetic/stockout_scenarios.parquet has not been generated for this pilot.", "mail")

    with tab_checks:
        if mp_raw is not None:
            eligible = int(mp_raw["forecast_training_eligible"].sum())
            ineligible = len(mp_raw) - eligible
            dup_mp = int(mp_raw.duplicated(subset=["sku", "date"]).sum())
            missing_counts = mp_raw.isna().sum()
            missing_top = missing_counts[missing_counts > 0].sort_values(ascending=False)
            render_kpi_row([
                dict(label="Zero-Sales Rate", value=format_percentage((mp_raw["units_observed"] == 0).mean()), icon="circle-dashed", tone="amber"),
                dict(label="Promotion Coverage", value=format_percentage(mp_raw["on_promo"].mean()), icon="tag", tone="teal"),
                dict(label="Eligible / Ineligible", value=f"{format_number(eligible)} / {format_number(ineligible)}", icon="check-circle", tone="blue"),
                dict(label="Duplicate SKU-Date Keys", value=str(dup_mp), icon="refresh", tone="slate"),
            ], n_cols=4)
            if inv_raw is not None:
                dup_inv = int(inv_raw.duplicated(subset=["sku"]).sum())
                st.write("")
                render_kpi_row([
                    dict(label="Cost-Valid Count", value=str(int(inv_raw["cost_is_valid"].sum())), icon="coin"),
                    dict(label="Cost-Imputed Count", value=str(int(inv_raw["cost_is_imputed"].sum())), icon="calculator"),
                    dict(label="Duplicate SKU Keys (Inv.)", value=str(dup_inv), icon="refresh"),
                    dict(label="Future Promotion Plan", value="Not provided", icon="tag"),
                ], n_cols=4)
            if not missing_top.empty:
                st.markdown("**Missing-value counts (model_panel, non-zero only):**")
                st.dataframe(missing_top.rename("Missing rows").to_frame(), width="stretch")
            else:
                st.caption("No missing values detected in model_panel.parquet.")
        else:
            empty_state("Not available", "model_panel.parquet is missing — data-quality metrics cannot be computed.", "mail")

    with tab_assume:
        if manifest:
            assumptions = manifest.get("assumptions", {})
            if assumptions:
                st.markdown("**Assumptions block (from manifest):**")
                st.dataframe(pd.DataFrame(list(assumptions.items()), columns=["Assumption", "Value"]),
                             width="stretch", hide_index=True)
            confirm_fields = {
                "Cost currency": manifest.get("cost_currency"),
                "Cost unit/pack basis": manifest.get("cost_basis"),
                "Lead time": assumptions.get("default_lead_time_days"),
                "MOQ": assumptions.get("default_moq"),
                "Pack size": assumptions.get("default_pack_size"),
                "Review period": None,
                "Service level": None,
                "Synthetic stock source": manifest.get("synthetic_stock_method"),
                "Frozen SKU selection provenance": manifest.get("sku_selection_cutoff"),
            }
            st.markdown("**Assumptions requiring confirmation:**")
            rows = []
            for k, v in confirm_fields.items():
                value_str = str(v) if v is not None else "Not specified in manifest — confirm with data owner"
                rows.append({"Assumption": k, "Current Value": value_str, "Needs Confirmation": "Yes"})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            empty_state("Not available", "pilot_manifest.json is missing — assumptions cannot be displayed.", "file-text")

    with tab_valid:
        if manifest:
            status = manifest.get("validation_status", "unknown")
            warnings = manifest.get("warnings", [])
            problems = manifest.get("problems", [])
            if status == "passed":
                info_banner("<strong>Validation status: PASSED</strong>", kind="success")
            else:
                info_banner(f"<strong>Validation status:</strong> {status}", kind="synthetic", icon="alert-triangle")
            st.markdown(f"**Warnings:** {'; '.join(warnings) if warnings else 'None recorded.'}")
            st.markdown(f"**Problems:** {'; '.join(problems) if problems else 'None recorded.'}")
        else:
            empty_state("Not available", "pilot_manifest.json is missing — validation status cannot be shown.", "file-text")


# ==========================================================================
# PAGE — FORECAST RUNS (Phase 5): generate a run, watch it, browse history
# ==========================================================================
def _friendly_status(state):
    return rs.step_label(state)


def _render_active_run_status(run_id):
    """Read fresh status.json for `run_id` and render progress + per-model status."""
    rec = next((r for r in rs.discover_runs() if r["run_id"] == run_id), None)
    if rec is None:
        st.info("Waiting for the run to initialise…")
        return None
    status = rec.get("status", "unknown")
    status_json = rs._read_json(Path(rec["run_dir"]) / "status.json")
    stj = status_json.get("model_status", {})
    pct = min(int(rec.get("progress_pct") or 0), 100)
    tone = ("done" if rec.get("is_completed") else "fail" if rec.get("is_failed") else "live")
    icon = {"done": "✓", "fail": "✕", "live": "◐"}[tone]
    updated = rs.format_local_datetime(status_json.get("updated_at") or rec.get("created_at"),
                                       include_date=False)
    sel = rec.get("selected_sku_count")
    sel_txt = f"{sel} selected" if sel is not None else "selecting…"

    # ONE compact status strip: state (left) · context (center) · updated (right).
    # The run id is small text here; the full id lives in Run details.
    with st.container(key="ipa-active-run"):
        st.markdown(
            f'<div class="ipa-runbar ipa-rb-{tone}">'
            f'<div class="rb-state"><span class="rb-ico">{icon}</span>'
            f'<span class="rb-label">{_friendly_status(rec.get("current_step", status))}</span></div>'
            f'<div class="rb-ctx">{rec.get("category") or "—"} · Top {rec.get("top_n") or "—"} · {sel_txt}</div>'
            f'<div class="rb-time">Updated {updated}<span class="rb-run" title="{rec.get("run_id")}">'
            f'Run …{str(rec.get("run_id") or "")[-6:]}</span></div></div>',
            unsafe_allow_html=True)
        st.progress(pct / 100.0, text=f"{pct}%")
        chips = "".join(
            f'<span class="ipa-mchip ipa-ms-{(stj.get(m) or {}).get("status", "pending")}">'
            f'{lbl} · {(stj.get(m) or {}).get("status", "pending")}</span>'
            for m, lbl in (("baseline", "Baseline"), ("holtwinters", "Holt-Winters"),
                           ("lightgbm", "LightGBM")))
        st.markdown(f'<div class="ipa-mchips">{chips}</div>', unsafe_allow_html=True)

        with st.expander("Run details", expanded=False):
            st.caption(f"**Run ID** `{rec.get('run_id')}`")
            ts_rows = [("Created", rec.get("created_at")), ("Started", status_json.get("started_at")),
                       ("Updated", status_json.get("updated_at"))]
            if rec.get("completed_at"):
                ts_rows.append(("Completed", rec.get("completed_at")))
            if rec.get("failed_at"):
                ts_rows.append(("Failed", rec.get("failed_at")))
            for lbl, v in ts_rows:
                st.caption(f"**{lbl}** {rs.format_local_datetime(v)}")
            st.caption(f"Pipeline log: `runs/{rec['run_id']}/pipeline.log`")

    if rec.get("is_failed"):
        st.error(f"Run failed: {rec.get('error_message') or 'see the pipeline log in Run details'}")
    return rec


@st.cache_data(show_spinner=False)
def latest_sales_date_cached(db_mtime_ns):
    """Latest FULL sales day — cached per warehouse mtime (re-queried only after an ETL refresh).

    Not MAX(transaction_date): trailing part-extracted days are discounted, so the default
    as-of date cannot land on a date with no real demand. See rs.get_latest_sales_date.
    """
    return rs.get_latest_sales_date()


@st.cache_data(show_spinner=False)
def sales_date_diagnostics_cached(db_mtime_ns):
    """Usable-window diagnostics so the UI can explain a discounted extract tail."""
    return rs.sales_date_diagnostics()


@st.cache_data(show_spinner="Reading warehouse categories…")
def eligible_categories_cached(db_mtime_ns, cutoff_str, min_history_days):
    """Eligible-category aggregation over the full sales table — expensive, so cached per
    (warehouse mtime, cutoff, min-history). Without this it re-ran on every widget keystroke."""
    return rs.list_categories(rs.DEFAULT_DB_PATH, cutoff_str, int(min_history_days))


def page_forecast_runs():
    render_page_header("Forecast Runs", "Generate a run · watch progress · activate a completed run",
                       badges=ACTIVE_BADGES)
    db_ok = rs.DEFAULT_DB_PATH.exists()
    db_mtime = rs.DEFAULT_DB_PATH.stat().st_mtime_ns if db_ok else 0
    latest = latest_sales_date_cached(db_mtime) if db_ok else None

    # ---- A. Generate Forecast ----------------------------------------------------------
    section_title("Generate Forecast", "Launch the Phase 4 orchestrator over the live warehouse.")
    if db_ok:
        _sd = sales_date_diagnostics_cached(db_mtime)
        _ignored = _sd.get("ignored_dates") or []
        if _ignored and _sd.get("usable_max"):
            # Silently shifting the default as-of would be worse than explaining it: the
            # discounted days are exactly the ones that make a run fail at model ranking.
            _shown = ", ".join(d.strftime("%d %b") for d in _ignored[:6])
            _more = f" (+{len(_ignored) - 6} more)" if len(_ignored) > 6 else ""
            info_banner(
                f"Sales run to <strong>{_sd['raw_max']:%d %b %Y}</strong>, but the last "
                f"{len(_ignored)} day(s) — {_shown}{_more} — hold only a few stray rows against "
                f"a median of <strong>{_sd['median_daily_units']:,.0f}</strong> units/day, so the "
                f"extract looks incomplete there. As-of defaults to the last full day, "
                f"<strong>{_sd['usable_max']:%d %b %Y}</strong>. Choosing a later date leaves the "
                f"holdout window with no real demand, which fails model ranking.",
                kind="warning")
    if not db_ok:
        empty_state("Warehouse unavailable",
                    "inventory_etl/output/inventory.db was not found — run the ETL first.", "database")
    else:
        # Read the user's current cutoff / min-history (if already set) so the eligible counts
        # match what will actually be launched; results are cached per combination.
        _cutoff_pref = st.session_state.get("run_cutoff") or latest
        _mhd_pref = int(st.session_state.get("run_mhd") or 28)
        try:
            cats = eligible_categories_cached(db_mtime, str(_cutoff_pref), _mhd_pref)
        except Exception as exc:  # noqa: BLE001
            cats = pd.DataFrame(columns=["category", "eligible_sku_count", "historical_units"])
            st.warning(f"Could not list categories: {exc}")
        cat_names = cats["category"].astype(str).tolist() if not cats.empty else []
        cat_meta = {r["category"]: r for _, r in cats.iterrows()} if not cats.empty else {}

        # Plain widgets (not st.form): st.form only pushes values to the script on submit, so
        # the preview banner below would show the PREVIOUS submitted values (e.g. "requested 10")
        # while the box already displays what you just typed (e.g. "3"). Live widgets keep the
        # preview and the disabled-state in sync with what's actually in the fields.
        fc1, fc2, fc3 = st.columns([2, 1, 1])
        run_cat = fc1.selectbox("Category", options=cat_names or ["(no eligible categories)"],
                                key="run_category")
        top_n = fc2.number_input("Top N", min_value=1, max_value=100, value=10, step=1, key="run_top_n")
        mhd = fc3.number_input("Min history days", min_value=1, max_value=365, value=28, key="run_mhd")
        fd1, fd2 = st.columns(2)
        as_of = fd1.date_input("As-of date", value=latest or date.today(), key="run_as_of")
        cutoff = fd2.date_input("Selection cutoff", value=latest or date.today(), key="run_cutoff")
        rank_by = st.radio(
            "Rank Top N by",
            options=list(rs.SUPPORTED_RANKING_METRICS),
            format_func=rs.ranking_metric_label,
            horizontal=True, key="run_ranking_metric",
            help="Units sold ranks by historical ecommerce volume using pre-cutoff sales only. "
                 "Stockout risk ranks by a pre-forecast risk estimate — lead-time demand from "
                 "the trailing sales window against current warehouse stock — so the run "
                 "forecasts the products most likely to run out rather than the best sellers.")
        hz = st.multiselect("Forecast horizons", options=[7, 14], default=[7, 14], key="run_horizons")
        allow_partial = st.checkbox("Allow partial success", value=False, key="run_allow_partial")

        meta = cat_meta.get(run_cat)
        elig = int(meta["eligible_sku_count"]) if meta is not None else 0
        info_banner(
            f"<strong>{elig}</strong> eligible products in <strong>{run_cat}</strong> · requested "
            f"<strong>{int(top_n)}</strong> ranked by <strong>{rs.ranking_metric_label(rank_by)}</strong> "
            f"· as-of <strong>{as_of}</strong> · models: Baselines, "
            "Holt-Winters, LightGBM. Fewer than Top N may be selected if eligible history is limited.",
            kind="info")
        if rank_by == rs.METRIC_STOCKOUT_RISK:
            # The proxy only ORDERS candidates; Phase B computes the authoritative risk after
            # the forecast. Saying so here stops the two numbers reading as a contradiction.
            info_banner(
                "Ranking uses a <strong>pre-forecast risk proxy</strong>: a flat demand forecast "
                "from the trailing sales window against the latest warehouse stock snapshot. "
                "Because that snapshot is not tied to the selection cutoff, selection can be "
                "influenced by stock recorded after it — the run records the snapshot date. "
                "Phase B still computes the authoritative per-product risk once the models have "
                "run, so a product's final tier may differ from the tier that selected it.",
                kind="warning")

        session_active = st.session_state.get("active_run_id")
        active_nonterminal = False
        if session_active:
            arec = next((r for r in RUNS_ALL if r["run_id"] == session_active), None)
            active_nonterminal = bool(arec and not arec["is_terminal"])
        disabled = (not db_ok or not cat_names or elig == 0 or cutoff > as_of or not hz or active_nonterminal)
        if active_nonterminal:
            st.warning("A run launched from this session is still in progress — wait for it to finish.")

        if st.button("Generate Forecast", type="primary", disabled=disabled, key="generate_forecast_btn"):
            try:
                info = rs.launch_forecast_run(
                    category=run_cat, top_n=int(top_n), as_of_date=as_of.isoformat(),
                    selection_cutoff=cutoff.isoformat(), min_history_days=int(mhd),
                    horizons=tuple(hz) or (7, 14), allow_partial_success=bool(allow_partial),
                    ranking_metric=rank_by)
                st.session_state["active_run_id"] = info["run_id"]
                st.session_state["active_run_pid"] = info["pid"]
                st.session_state["run_launch_time"] = info["launched_at"]
                st.success(f"Launched run {info['run_id']} (pid {info['pid']}).")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Launch failed: {exc}")

    # ---- B. Active Run Status ----------------------------------------------------------
    active_id = st.session_state.get("active_run_id")
    if active_id:
        section_title("Active Run Status")
        st.caption("Times shown in Pakistan Standard Time (PKT).")
        _launched = st.session_state.get("run_launch_time")
        if _launched:
            st.caption(f"Launched from this session at {rs.format_local_datetime(_launched)}"
                       f" · pid {st.session_state.get('active_run_pid', '—')}")
        if hasattr(st, "fragment"):
            @st.fragment(run_every="2s")
            def _live():
                rec = _render_active_run_status(active_id)
                if rec and rec.get("is_completed"):
                    if st.button("Activate this run as the dashboard data source", key="activate_active"):
                        st.session_state["_pending_data_source"] = _short_by_id.get(rec["run_id"], rs.format_run_label_short(rec))
                        st.rerun()
            _live()
        else:
            _render_active_run_status(active_id)
            if st.button("Refresh status", key="refresh_active"):
                st.rerun()

    # ---- C. Run History ----------------------------------------------------------------
    section_title("Run History")
    st.caption("Times shown in Pakistan Standard Time (PKT).")
    if not RUNS_ALL:
        empty_state("No runs yet", "Generate a forecast above to create your first run.", "rocket_launch")
        return
    rows = []
    for r in RUNS_ALL:
        w = r.get("winners_by_horizon") or {}
        rows.append({
            "Status": r.get("status"), "Created": rs.format_local_datetime(r.get("created_at")),
            "Category": r.get("category"), "Top N": r.get("top_n"),
            "Ranked by": rs.ranking_metric_label(r.get("ranking_metric")),
            "SKUs": r.get("selected_sku_count"), "As-of": r.get("as_of_date"),
            "7-day winner": w.get("7"), "14-day winner": w.get("14"),
            "Operational": r.get("operational_model"),
            "Duration (s)": r.get("duration_seconds"), "Run ID": r.get("run_id"),
        })
    _hist_df = pd.DataFrame(rows)
    st.dataframe(_hist_df, width="stretch", hide_index=True, height=280)
    with st.container(key="ipa-export-7"):
        eu.render_table_export_menu(_hist_df, filename_stem="run_history",
                                    title="Forecast Run History",
                                    metadata={"Runs": len(_hist_df)},
                                    key="exp_runhistory")

    completed_ids = [r["run_id"] for r in RUNS_ALL if r["is_completed"]]
    if completed_ids:
        ac1, ac2 = st.columns([3, 1])
        pick = ac1.selectbox("Activate a completed run", options=completed_ids, key="activate_pick")
        if ac2.button("Activate", key="activate_history"):
            rec = next(r for r in RUNS_ALL if r["run_id"] == pick)
            st.session_state["_pending_data_source"] = _short_by_id.get(rec["run_id"], rs.format_run_label_short(rec))
            st.rerun()

    insp = st.selectbox("Inspect pipeline log (any run)", options=[r["run_id"] for r in RUNS_ALL],
                        key="inspect_run")
    with st.expander("View pipeline log (last 200 lines)", expanded=False):
        log_path = Path(next(r["run_dir"] for r in RUNS_ALL if r["run_id"] == insp)) / "pipeline.log"
        st.text(rs.tail_log(log_path, 200) or "(no log yet)")


def page_deadstock():
    # Standalone ecommerce deadstock (inventory-inactivity) scan. INDEPENDENT of the active
    # Forecast Run: it reads the real warehouse on demand and never launches a run or a model.
    # This page only filters/sorts/formats/displays/exports the backend result — never reclassifies.
    DEAD_HEX = {da.STATUS_CANDIDATE: COLORS["amber"], da.STATUS_NEVER_SOLD: COLORS["slate"],
                da.STATUS_REVIEW: COLORS["blue"], da.STATUS_NOT: COLORS["success"]}
    DEAD_CLS = {da.STATUS_CANDIDATE: "candidate", da.STATUS_NEVER_SOLD: "never",
                da.STATUS_REVIEW: "review", da.STATUS_NOT: "not"}

    def _short(s, n=42):
        s = str(s)
        return s if len(s) <= n else s[:n - 1] + "…"

    def _esc(x):
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _safe_key(x):
        return "".join(c if c.isalnum() else "-" for c in str(x))

    def _dash(v):
        return "—" if (v is None or v is pd.NaT or (isinstance(v, float) and pd.isna(v))) else v

    def _num(v):
        try:
            return None if v is None or pd.isna(v) else float(v)
        except (TypeError, ValueError):
            return None

    def _dead_chip(status):
        s = str(status)
        return f'<span class="ipa-dead ipa-dead-{DEAD_CLS.get(s, "not")}">{_esc(s)}</span>'

    def _full(name, sku):
        return rs.full_product_label(name, sku)

    def _inactivity_text(row):
        s = str(row.get("deadstock_status"))
        d = _num(row.get("days_since_last_sale"))
        if s == da.STATUS_NEVER_SOLD:
            return "Never sold"
        if d is not None:
            return f"{int(d)} days inactive"
        return "—"

    # ── config / warehouse / categories ───────────────────────────────────────────────────
    try:
        dcfg = da._load_config()
    except Exception:  # noqa: BLE001
        dcfg = {"default_inactivity_days": 90, "minimum_interval_days": 1,
                "maximum_interval_days": 365, "sales_scope": "ecommerce"}
    db_path = rs.DEFAULT_DB_PATH

    res = st.session_state.get("deadstock_result")
    completed = bool(res and not res.get("error"))

    # ── header with DYNAMIC deadstock badges from the completed analysis (never the pilot) ──
    if completed:
        s = res["summary"]
        badges = [
            ("box", f"{format_number(s['products_scanned'])} SKUs scanned"),
            ("globe", f"{str(s['sales_scope']).capitalize()} sales scope"),
            ("calendar", f"Snapshot {_pretty_date(s['snapshot_date'])}"),
            ("clock", f"Interval {s['inactivity_interval_days']} days"),
        ]
    else:
        badges = [("database", "Warehouse deadstock scan"), ("globe", "Ecommerce sales scope")]
    render_page_header(
        "Deadstock Analysis",
        "Ecommerce inventory inactivity · configurable interval · latest warehouse snapshot",
        badges=badges)
    info_banner(
        "Deadstock candidates have positive current stock but no recorded ecommerce sale during "
        "the selected interval. This does not prove that the product had no physical-store sales.",
        kind="info")

    if not db_path.exists():
        empty_state("Warehouse unavailable",
                    "The warehouse database (inventory_etl/output/inventory.db) was not found. "
                    "Deadstock analysis needs the real warehouse snapshot.", "database")
        return
    try:
        cats = da.list_deadstock_categories(db_path)
    except Exception as exc:  # noqa: BLE001
        empty_state("Warehouse unavailable", f"Could not read warehouse categories: {exc}", "database")
        return

    # ── inputs (page-local; independent of the sidebar run/product filters) ───────────────
    with st.container(key="ipa-dead-inputs"):
        ic = st.columns([2, 1, 1])
        with ic[0]:
            cat_choice = st.selectbox("Category", options=["All Categories"] + cats, key="deadstock_category")
        with ic[1]:
            interval = st.number_input(
                "Inactivity interval (days)", min_value=int(dcfg["minimum_interval_days"]),
                max_value=int(dcfg["maximum_interval_days"]), value=int(dcfg["default_inactivity_days"]),
                step=1, key="deadstock_interval")
        with ic[2]:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            run_clicked = st.button("Analyse Deadstock", type="primary", width="stretch", key="deadstock_run")

    if run_clicked:
        with st.spinner("Scanning the latest warehouse snapshot…"):
            try:
                cat_arg = None if cat_choice == "All Categories" else cat_choice
                df_all, summ = da.analyse_deadstock(db_path=db_path, inactivity_days=int(interval),
                                                    category=cat_arg, include_not_deadstock=True)
                st.session_state["deadstock_result"] = {
                    "df": df_all, "summary": summ, "analysis_category": cat_choice,
                    "analysis_inactivity_days": int(interval),
                    "analysis_snapshot_date": summ["snapshot_date"]}
                st.session_state["deadstock_queue_shown"] = 8
            except Exception as exc:  # noqa: BLE001
                st.session_state["deadstock_result"] = {"error": str(exc)}
        st.rerun()

    # stale-input warning (retain old result; never relabel it with the new inputs)
    if da.analysis_inputs_changed(res, cat_choice, int(interval)):
        info_banner("Inputs have changed. Click <strong>Analyse Deadstock</strong> to refresh the results.",
                    kind="synthetic")

    if not res:
        empty_state("Run a deadstock scan",
                    "Choose a category and inactivity interval, then click Analyse Deadstock. The scan "
                    "reads the latest warehouse snapshot on demand — it never launches a forecast run "
                    "or a model.", "search")
        return
    if res.get("error"):
        empty_state("Deadstock scan failed", res["error"], "alert-triangle")
        return

    # From here everything uses the COMPLETED analysis metadata (not the current form inputs).
    df_all, summ = res["df"], res["summary"]
    acat = res["analysis_category"]
    is_all = (acat == "All Categories")
    interval_a = int(summ["inactivity_interval_days"])
    scope = summ["sales_scope"]

    if df_all is None or df_all.empty:
        empty_state("No stock-carrying products",
                    f"No active, non-dropship SKU with positive current stock was found "
                    f"{'across All Categories' if is_all else f'in {acat}'}.", "check-circle")
        return

    info_banner(
        f"Interval <strong>{interval_a} days</strong> · snapshot <strong>{summ['snapshot_date'] or '—'}</strong> "
        f"· sales scope <strong>{scope}</strong> · scanned "
        f"<strong>{format_number(summ['products_scanned'])}</strong> stock-carrying SKUs "
        f"{'across All Categories' if is_all else f'in {acat}'}", kind="info")

    # ── Section 1 — five KPI cards (full completed analysis) ──────────────────────────────
    section_title("Deadstock overview", None)
    render_kpi_row([
        {"label": "Deadstock Candidates", "value": format_number(summ["deadstock_candidate_count"]),
         "icon": "alert-triangle", "tone": "amber",
         "sub": f"No ecommerce sale in ≥ {interval_a}d (excl. never-sold)"},
        {"label": "Never-Sold Products", "value": format_number(summ["never_sold_count"]),
         "icon": "circle-dashed", "tone": "slate", "sub": "In stock, no ecommerce sale ever"},
        {"label": "Deadstock Units", "value": format_number(summ["deadstock_units"]),
         "icon": "box", "tone": "navy", "sub": "Candidate + never-sold stock on hand"},
        {"label": "Estimated Deadstock Value", "value": format_currency(summ["estimated_deadstock_value"]),
         "icon": "coin", "tone": "teal", "sub": "Priced deadstock only (nulls excluded)"},
        {"label": "Missing-Cost Products", "value": format_number(summ["missing_cost_count"]),
         "icon": "search", "tone": "blue", "sub": "Deadstock SKUs without a valid cost"},
    ], n_cols=5)

    confirmed = df_all[df_all["deadstock_status"].isin([da.STATUS_CANDIDATE, da.STATUS_NEVER_SOLD])].copy()

    # ── Section 2 — Deadstock Exposure Analysis (status donut + inactivity aging) ─────────
    section_title("Deadstock Exposure Analysis", "Full completed analysis for the selected scope.")
    ca, cb = st.columns(2)
    with ca:
        order = [da.STATUS_CANDIDATE, da.STATUS_NEVER_SOLD, da.STATUS_REVIEW, da.STATUS_NOT]
        present = [s for s in order if (df_all["deadstock_status"] == s).any()]
        counts = [int((df_all["deadstock_status"] == s).sum()) for s in present]
        units = [float(pd.to_numeric(df_all[df_all["deadstock_status"] == s]["stock_on_hand"],
                                     errors="coerce").sum()) for s in present]
        vals = [float(pd.to_numeric(df_all[df_all["deadstock_status"] == s]["estimated_deadstock_value"],
                                    errors="coerce").dropna().sum()) for s in present]
        donut = go.Figure(go.Pie(
            labels=present, values=counts, hole=0.62, sort=False, direction="clockwise",
            marker=dict(colors=[DEAD_HEX[s] for s in present], line=dict(color="white", width=1.5)),
            textinfo="value", customdata=list(zip(units, vals)),
            hovertemplate=("<b>%{label}</b><br>Products: %{value} (%{percent})<br>"
                           "Stock units: %{customdata[0]:,.0f}<br>"
                           "Est. value: PKR %{customdata[1]:,.0f}<extra></extra>")))
        donut.update_layout(**plotly_layout(height=330, legend=True))
        donut.update_layout(annotations=[dict(text=f"{format_number(summ['products_scanned'])}<br>scanned",
                                              x=0.5, y=0.5, showarrow=False,
                                              font=dict(size=13, color=COLORS["navy"]))])
        render_chart(donut, "Status Distribution",
                     "Share of scanned SKUs by status. Status label stays visible in the legend + hover.")
    with cb:
        ag = da.deadstock_aging_summary(df_all, interval_a)
        bar = go.Figure(go.Bar(
            x=ag["bucket"], y=ag["products"],
            marker=dict(color=[COLORS["slate"] if b == "Never Sold" else COLORS["teal"] for b in ag["bucket"]]),
            customdata=np.stack([ag["units"].to_numpy(), ag["value"].to_numpy()], axis=-1),
            hovertemplate=("<b>%{x}</b><br>Products: %{y}<br>Stock units: %{customdata[0]:,.0f}<br>"
                           "Est. value: PKR %{customdata[1]:,.0f}<extra></extra>")))
        bar.update_layout(**plotly_layout(legend=False, height=330))
        style_axes(bar)
        bar.update_yaxes(title_text="Products")
        bar.update_xaxes(title_text="")
        render_chart(bar, "Inactivity Aging",
                     "Mutually exclusive inactivity buckets; Never Sold is separate (no last-sale date).")

    # ── Section 3 — value exposure (context-aware) ─────────────────────────────────────────
    if is_all:
        by_cat = (confirmed.dropna(subset=["estimated_deadstock_value"])
                  .groupby("category", as_index=False)["estimated_deadstock_value"].sum()
                  .sort_values("estimated_deadstock_value", ascending=True).tail(20))
        if by_cat.empty:
            with st.container(border=True):
                empty_state("No priced deadstock value",
                            "No candidate/never-sold SKU has a valid cost, so value cannot be ranked.",
                            "circle-dashed")
        else:
            fig = go.Figure(go.Bar(
                x=by_cat["estimated_deadstock_value"], y=by_cat["category"], orientation="h",
                marker=dict(color=COLORS["teal"]),
                hovertemplate="<b>%{y}</b><br>Estimated deadstock value: PKR %{x:,.0f}<extra></extra>"))
            fig.update_layout(**plotly_layout(legend=False, height=360))
            style_axes(fig)
            fig.update_xaxes(title_text="Estimated deadstock value (PKR)")
            render_chart(fig, "Estimated Deadstock Value by Category",
                         "Candidate + never-sold SKUs with a valid cost (missing-cost SKUs excluded).")
    else:
        top = (confirmed.dropna(subset=["estimated_deadstock_value"])
               .sort_values("estimated_deadstock_value", ascending=True).tail(15))
        if top.empty:
            with st.container(border=True):
                empty_state("No priced deadstock value",
                            "No candidate/never-sold SKU in this category has a valid cost.", "circle-dashed")
        else:
            names = [_full(n, s) for n, s in zip(top["sku_name"], top["sku"])]
            cust = list(zip(names, top["sku"].astype(str),
                            [format_number(v, 0) for v in top["stock_on_hand"]],
                            [str(x) for x in top["deadstock_status"]]))
            fig = go.Figure(go.Bar(
                x=top["estimated_deadstock_value"], y=top["sku"].astype(str), orientation="h",
                marker=dict(color=COLORS["teal"]), customdata=cust,
                hovertemplate=("<b>%{customdata[0]}</b><br>SKU %{customdata[1]}<br>"
                               "Status: %{customdata[3]}<br>Stock: %{customdata[2]} u<br>"
                               "Estimated deadstock value: PKR %{x:,.0f}<extra></extra>")))
            fig.update_layout(**plotly_layout(legend=False, height=360))
            style_axes(fig)
            fig.update_yaxes(tickmode="array", tickvals=top["sku"].astype(str).tolist(),
                             ticktext=[_short(n) for n in names], title="", automargin=True)
            fig.update_xaxes(title_text="Estimated deadstock value (PKR)")
            render_chart(fig, "Top Deadstock Products by Estimated Value",
                         "Highest-value candidate / never-sold SKUs. Hover for the full product name.")

    # ── Section 4 — Deadstock Priority Queue (filters + compact rows) ─────────────────────
    section_title("Deadstock Priority Queue",
                  "Candidate, never-sold and manual-review stock requiring commercial review.")
    status_order = [da.STATUS_CANDIDATE, da.STATUS_NEVER_SOLD, da.STATUS_REVIEW, da.STATUS_NOT]
    status_opts = [s for s in status_order if (df_all["deadstock_status"] == s).any()]
    default_status = [s for s in (da.STATUS_CANDIDATE, da.STATUS_NEVER_SOLD, da.STATUS_REVIEW)
                      if s in status_opts]
    with st.container(key="ipa-dead-toolbar"):
        tc = st.columns([1.8, 1.7, 1.2])
        with tc[0]:
            query = st.text_input("Search product or SKU", key="deadstock_q_search",
                                  placeholder="product name or SKU code")
        with tc[1]:
            statuses = st.multiselect("Status", options=status_opts, default=default_status,
                                      key="deadstock_q_status")
        with tc[2]:
            sort_by = st.selectbox("Sort by", options=list(da.QUEUE_SORT_OPTIONS), key="deadstock_q_sort")

    active_statuses = statuses if statuses else default_status
    filtered = da.filter_deadstock(df_all, query=query, statuses=active_statuses)
    ranked = da.sort_deadstock_queue(filtered, sort_by)
    n_returned = int(df_all["deadstock_status"].isin(da.RETURNED_STATUSES).sum())
    is_filtered = (len(ranked) != n_returned) or bool(query and query.strip())

    mc1, mc2 = st.columns([3, 1])
    with mc1:
        chips = f'<span class="ipa-tier ipa-tier-unknown">{len(ranked)} in queue</span>'
        if is_filtered:
            chips += '<span class="ipa-tier ipa-tier-medium">Filtered</span>'
        st.markdown(f'<div class="ipa-src-row">{chips}</div>', unsafe_allow_html=True)
    with mc2:
        with st.container(key="ipa-export-dead"):
            eu.render_table_export_menu(
                da.deadstock_export_frame(ranked), filename_stem="deadstock_analysis",
                title="Deadstock Analysis",
                metadata={"Category": acat, "Inactivity interval (days)": interval_a,
                          "Snapshot date": summ["snapshot_date"], "Sales scope": scope,
                          "Products scanned": summ["products_scanned"], "Rows (filtered)": len(ranked)},
                key="exp_deadstock")

    if ranked is None or ranked.empty:
        if summ["deadstock_candidate_count"] == 0 and summ["never_sold_count"] == 0 \
                and summ["manual_review_count"] == 0:
            empty_state("No deadstock candidates",
                        f"No ecommerce deadstock candidates were found for the selected {interval_a}-day "
                        f"interval {'across All Categories' if is_all else f'in {acat}'}.", "check-circle")
        else:
            empty_state("No products match", "No products match the current status filter and search. "
                        "Adjust the Status filter or clear the search.", "search")
        return

    # selection state (safe fallback to the top-ranked visible SKU)
    cur = st.session_state.get("deadstock_selected_sku")
    valid = ranked["sku"].astype(str).tolist()
    if cur not in valid:
        cur = valid[0]
        st.session_state["deadstock_selected_sku"] = cur

    PAGE = 8
    shown = int(st.session_state.get("deadstock_queue_shown", PAGE))
    shown = max(PAGE, min(shown, len(ranked)))
    for rec in ranked.head(shown).to_dict("records"):
        sku = str(rec["sku"])
        safe = _safe_key(sku)
        status = str(rec.get("deadstock_status"))
        cls = DEAD_CLS.get(status, "not")
        name = rec.get("sku_name")
        nm = None if (name is None or (isinstance(name, float) and pd.isna(name))) else str(name).strip()
        disp = nm if (nm and nm.lower() != "nan") else sku
        full = _full(name, sku)
        is_sel = (sku == str(cur))
        val = rec.get("estimated_deadstock_value")
        val_txt = format_currency(val) if _num(val) is not None else "Cost unavailable"
        rcols = st.columns([3.0, 1.25, 0.9, 1.3, 1.2, 1.35, 0.95])
        with rcols[0]:
            st.markdown(
                f'<div class="ipa-qrow ipa-q-dead-{cls}" title="{_esc(full)}">'
                f'<div style="min-width:0;"><div class="q-name">{_esc(disp)}</div>'
                f'<div class="q-sub">SKU {_esc(sku)}</div></div></div>', unsafe_allow_html=True)
        rcols[1].markdown(f'<div class="ipa-qcell">{_dead_chip(status)}</div>', unsafe_allow_html=True)
        rcols[2].markdown(f'<div class="ipa-qcell"><b>{format_number(rec.get("stock_on_hand"), 0)}</b>'
                          f'<div class="q-sub">units</div></div>', unsafe_allow_html=True)
        rcols[3].markdown(f'<div class="ipa-qcell"><b>{_esc(_inactivity_text(rec))}</b>'
                          f'<div class="q-sub">inactivity</div></div>', unsafe_allow_html=True)
        rcols[4].markdown(f'<div class="ipa-qcell"><b>{_esc(str(_dash(rec.get("last_sale_date"))))}</b>'
                          f'<div class="q-sub">last sale</div></div>', unsafe_allow_html=True)
        rcols[5].markdown(f'<div class="ipa-qcell"><b>{val_txt}</b>'
                          f'<div class="q-sub">est. value</div></div>', unsafe_allow_html=True)
        with rcols[6]:
            if st.button("Details", key=f"deadbtn-{safe}", width="stretch",
                         type="primary" if is_sel else "secondary",
                         help=f"Open full details for {full}"):
                st.session_state["deadstock_selected_sku"] = sku
                st.session_state["deadstock_open_dialog"] = True
                st.rerun()
    if len(ranked) > shown:
        if st.button(f"Show more ({len(ranked) - shown} remaining)", key="deadstock_show_more"):
            st.session_state["deadstock_queue_shown"] = shown + PAGE
            st.rerun()
    elif shown > PAGE:
        if st.button("Show fewer", key="deadstock_show_fewer"):
            st.session_state["deadstock_queue_shown"] = PAGE
            st.rerun()

    # ── Selected-product details (modal when supported, inline fallback otherwise) ────────
    def _render_deadstock_detail(sku):
        sel = df_all[df_all["sku"].astype(str) == str(sku)]
        if sel.empty:
            return
        r = sel.iloc[0]
        status = str(r.get("deadstock_status"))
        stock = _num(r.get("stock_on_hand"))
        days = _num(r.get("days_since_last_sale"))
        age = _num(r.get("product_age_days"))
        val = r.get("estimated_deadstock_value")
        inactive_val = ("Never sold" if status == da.STATUS_NEVER_SOLD
                        else (f"{int(days)} days" if days is not None else "—"))
        # A. four summary cards
        render_kpi_row([
            {"label": "Deadstock Status", "value": status, "icon": "alert-triangle",
             "tone": {"candidate": "amber", "never": "slate", "review": "blue", "not": "success"}[DEAD_CLS.get(status, "not")],
             "sub": f"{scope} sales scope"},
            {"label": "Current Stock", "value": f"{format_number(stock, 0)} u", "icon": "box",
             "tone": "navy", "sub": f"Snapshot {summ['snapshot_date'] or '—'}"},
            {"label": ("Inactive Days" if status != da.STATUS_NEVER_SOLD else "Sales"),
             "value": inactive_val, "icon": "clock", "tone": "amber",
             "sub": f"Configured interval {interval_a}d"},
            {"label": "Estimated Deadstock Value",
             "value": (format_currency(val) if _num(val) is not None else "—"), "icon": "coin",
             "tone": "teal", "sub": "Stock × unit cost" if _num(val) is not None else "Cost unavailable"},
        ], n_cols=4)

        d1, d2 = st.columns(2)
        with d1:
            metric_panel("Product profile", [
                ("Product name", _esc(str(_dash(r.get("sku_name"))))),
                ("SKU", _esc(str(r.get("sku")))),
                ("Product ID", str(_dash(r.get("product_id")))),
                ("Category", _esc(str(_dash(r.get("category"))))),
                ("Brand", _esc(str(_dash(r.get("brand"))))),
                ("Stock snapshot date", str(_dash(r.get("snapshot_date")))),
                ("Current stock", format_number(stock, 0)),
                ("Selected interval", f"{interval_a} days"),
                ("Sales scope", f"{scope} (ecommerce channels only)"),
            ], sub="Read from the latest warehouse snapshot.")
        with d2:
            if status == da.STATUS_CANDIDATE:
                beyond = (int(days) - interval_a) if days is not None else None
                rows = [("Last sale date", str(_dash(r.get("last_sale_date")))),
                        ("Days since last sale", format_number(days, 0)),
                        ("Configured threshold", f"{interval_a} days"),
                        ("Days beyond threshold", (format_number(beyond, 0) if beyond is not None else "—"))]
            elif status == da.STATUS_NEVER_SOLD:
                rows = [("Last ecommerce sale", "Never recorded"),
                        ("Product created date", str(_dash(r.get("product_created_date")))),
                        ("Product age", (f"{int(age)} days" if age is not None else "—")),
                        ("Configured threshold", f"{interval_a} days")]
            else:  # Manual Review / other
                rows = [("Last sale date", str(_dash(r.get("last_sale_date")))),
                        ("Product created date", str(_dash(r.get("product_created_date")))),
                        ("Product age", (f"{int(age)} days" if age is not None else "—")),
                        ("Configured threshold", f"{interval_a} days")]
            metric_panel("Sales inactivity", rows, sub=f"{scope} sales only — physical-store sales are out of scope.")

        cost = _num(r.get("unit_cost"))
        cost_note = "Valid" if cost is not None else "Missing / invalid — value not computed"
        metric_panel("Cost exposure", [
            ("Unit cost", (format_currency(cost, 2) if cost is not None else "—")),
            ("Cost source", _esc(str(_dash(r.get("cost_source"))))),
            ("Current stock", format_number(stock, 0)),
            ("Estimated deadstock value", (format_currency(val) if _num(val) is not None else "—")),
            ("Cost status", cost_note),
        ], sub="“—” means unavailable — never shown as zero.")

        # E. deterministic "why flagged" (from stored row fields only; ecommerce-scoped wording)
        if status == da.STATUS_CANDIDATE and days is not None:
            beyond = int(days) - interval_a
            why = (f"This SKU has {format_number(stock, 0)} units in the latest warehouse snapshot and its "
                   f"last recorded ecommerce sale was {int(days)} days before the snapshot. The configured "
                   f"deadstock interval is {interval_a} days, so it exceeds the threshold by {beyond} days.")
        elif status == da.STATUS_NEVER_SOLD:
            why = (f"This SKU has {format_number(stock, 0)} units in the latest warehouse snapshot and no "
                   f"positive ecommerce sale is recorded on or before the snapshot date. Its product age"
                   + (f" ({int(age)} days)" if age is not None else "")
                   + f" meets the configured {interval_a}-day interval.")
        elif status == da.STATUS_REVIEW:
            why = (f"This SKU has {format_number(stock, 0)} units and no positive ecommerce sale on or before "
                   f"the snapshot, but its product creation date is unavailable, so its age could not be "
                   f"verified against the {interval_a}-day interval. It is held for manual review rather than "
                   f"classified automatically.")
        else:
            why = (f"This SKU has {format_number(stock, 0)} units and is not flagged as deadstock for the "
                   f"configured {interval_a}-day ecommerce interval.")
        why += " This reflects ecommerce sales only and does not indicate whether the product sold in physical stores."
        st.markdown(
            f'<div class="ipa-reason"><div class="h">{icon_svg("file-text", 16)} Why it was flagged</div>'
            f'{_esc(why)}</div>', unsafe_allow_html=True)

    _sel_row = df_all[df_all["sku"].astype(str) == str(cur)]
    _sel_full = (_full(_sel_row.iloc[0].get("sku_name"), _sel_row.iloc[0]["sku"])
                 if not _sel_row.empty else str(cur))
    if hasattr(st, "dialog"):
        @st.dialog(f"{_sel_full}", width="large")
        def _dead_dialog():
            _render_deadstock_detail(cur)
            if st.button("Close", key="deadstock_dialog_close", type="primary"):
                st.session_state["deadstock_open_dialog"] = False
                st.rerun()

        if st.session_state.get("deadstock_open_dialog"):
            st.session_state["deadstock_open_dialog"] = False   # one-shot: reopened by a Details click
            _dead_dialog()
    else:
        with st.container(border=True, key="deadstock-inline-detail"):
            st.markdown(f'<div class="ipa-dd-head">{_esc(_sel_full)}</div>', unsafe_allow_html=True)
            _render_deadstock_detail(cur)

    # ── Complete dataset (currently filtered rows) ─────────────────────────────────────────
    with st.expander("View Complete Deadstock Dataset", expanded=False):
        disp = da.deadstock_export_frame(ranked)

        def _status_css(val):
            m = {da.STATUS_CANDIDATE: ("#FBF0DC", COLORS["amber"]),
                 da.STATUS_NEVER_SOLD: ("#EAEEF3", COLORS["slate"]),
                 da.STATUS_REVIEW: ("#E5EDFD", COLORS["blue"]),
                 da.STATUS_NOT: ("#E4F5EC", COLORS["success"])}
            if val in m:
                bg, fg = m[val]
                return f"background-color:{bg}; color:{fg}; font-weight:700;"
            return ""

        st.dataframe(disp.style.map(_status_css, subset=["Status"]),
                     use_container_width=True, hide_index=True, height=430)
        st.caption(f"{len(disp)} filtered row(s) · ecommerce sales scope · snapshot {summ['snapshot_date'] or '—'}. "
                   "Null cost/value stay blank (never zero).")
        with st.expander("Advanced: technical fields", expanded=False):
            tech_cols = [c for c in da.OUTPUT_COLUMNS if c in ranked.columns]
            st.dataframe(ranked[tech_cols], use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
PAGE_FUNCS = {
    "Executive Overview": page_executive_overview,
    "Demand Analytics": page_demand_analytics,
    "Forecast Runs": page_forecast_runs,
    "Forecast Explorer": page_forecast_explorer,
    "Inventory & Reorder": page_inventory_reorder,
    "Deadstock": page_deadstock,
    "Stockout Risk": page_stockout_risk,
    "Data Quality & Assumptions": page_data_quality,
}
PAGE_FUNCS[page]()
