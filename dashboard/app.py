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
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from styles import (COLORS, CATEGORICAL, DONUT_COLORS, DONUT_HOVER, STATUS_COLORS, TONES, TONE_CYCLE,
                    CUSTOM_CSS, plotly_layout, style_axes, icon_svg)

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
def discover_outputs():
    """Scan outputs/ once and split files into forecast outputs vs evaluation outputs."""
    forecasts, evaluations = {}, {}
    if OUTPUTS_DIR.exists():
        for p in sorted(list(OUTPUTS_DIR.glob("*.csv")) + list(OUTPUTS_DIR.glob("*.parquet"))):
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
    if date_range and "date" in out.columns and len(date_range) == 2:
        start, end = date_range
        out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
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
            insights.append(f"**{top_cat}** contributes {share:.1f}% of pilot demand in the selected filters.")

        zero_rate = (mp_f["units_observed"] == 0).mean() * 100
        insights.append(f"{zero_rate:.1f}% of demand-history rows in the current view contain zero sales.")

        promo_rate = mp_f["on_promo"].mean() * 100
        insights.append(f"{promo_rate:.1f}% of rows in the current view were on an active promotion.")

        sku_totals = mp_f.groupby("sku")["units_observed"].sum().sort_values(ascending=False)
        if len(sku_totals):
            insights.append(
                f"SKU **{sku_totals.index[0]}** leads the current view with "
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
# Load all data once
# --------------------------------------------------------------------------
manifest, manifest_status = load_manifest()
mp_raw, mp_status = load_model_panel()
inv_raw, inv_status = load_inventory_context()
ff_raw, ff_status, ff_name = load_forecast_features()
stockout_raw, stockout_status = load_stockout_scenarios()
replen_raw, replen_status = load_replenishment_events()
simparams_raw, simparams_status = load_simulation_parameters()
outputs_forecasts, outputs_evaluations = discover_outputs()

if mp_raw is not None:
    mp_raw = mp_raw.copy()
    mp_raw["date"] = pd.to_datetime(mp_raw["date"])

sku_meta = build_sku_meta(mp_raw)

inv_joined = inv_raw
if inv_raw is not None and sku_meta is not None:
    inv_joined = inv_raw.merge(sku_meta[["sku", "category", "brand"]], on="sku", how="left")


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

NAV_ITEMS = [
    ("Executive Overview", "home"),
    ("Demand Analytics", "bar_chart"),
    ("Forecast Explorer", "auto_graph"),
    ("Inventory & Reorder", "inventory_2"),
    ("Stockout Scenario Lab", "science"),
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

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="ipa-nav-label">Filters</div>', unsafe_allow_html=True)

if mp_raw is not None:
    all_skus = sorted(mp_raw["sku"].unique().tolist())
    all_categories = sorted(mp_raw["category"].dropna().unique().tolist())
    all_brands = sorted(mp_raw["brand"].dropna().unique().tolist())
    SKU_LABELS = build_sku_labels(mp_raw)      # "Product name — Brand (SKU)" for search/legends
    SKU_NAMES = build_sku_names(mp_raw)         # name-only, for tables/profile that show SKU separately
    min_date, max_date = mp_raw["date"].min().date(), mp_raw["date"].max().date()

    st.sidebar.selectbox("Channel", options=["naheed_web"], key="flt_channel")

    cat_choice = st.sidebar.selectbox("Category", options=["All categories"] + all_categories, key="flt_category")
    sel_categories = all_categories if cat_choice == "All categories" else [cat_choice]

    # Options stay SKU codes (stable internal keys); the label — real product name,
    # brand, and the SKU code — is shown & searched via format_func, so users find a
    # product by its actual name, its brand, OR its SKU code.
    with st.sidebar.container(key="ipa-sku-search"):
        focus_sku_choice = st.multiselect(
            "Compare products", options=all_skus, default=[all_skus[0]] if all_skus else [],
            format_func=lambda s: SKU_LABELS.get(s, s), placeholder="Search by product name, brand or SKU...",
            help="Each selected product is shown as a separate line on the Demand Analytics "
                 "deep-dive. Search by product name, brand, or SKU code.",
            key="flt_focus_sku"
        )
        cmp_label_mode = st.selectbox(
            "Comparison labels", options=CMP_LABEL_MODES, index=0, key="flt_cmp_labels",
            help="How selected products are labelled in comparison legends, hover text, "
                 "chart titles and headings. Filtering always uses the SKU internally.",
        )
    focus_skus = focus_sku_choice if focus_sku_choice else [all_skus[0]] if all_skus else []

    date_val = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="flt_daterange")
    date_range = date_val if isinstance(date_val, tuple) and len(date_val) == 2 else (min_date, max_date)
    horizon = st.sidebar.radio("Forecast horizon", options=[7, 14], format_func=lambda x: f"{x} days", index=1, horizontal=True, key="flt_horizon")

    with st.sidebar.expander("Advanced filters", expanded=False):
        sel_skus = st.multiselect(
            "Dashboard product filter (blank = all)", options=all_skus, default=[],
            format_func=lambda s: SKU_LABELS.get(s, s),
            help="Filters the WHOLE dashboard: every total and aggregate chart is recalculated "
                 "using only the selected products (they are combined into one total). Blank = all.",
            key="flt_skus")
        sel_brands = st.multiselect("Brand (blank = all)", options=all_brands, default=[], key="flt_brands")
        scenario_available = stockout_raw is not None and "scenario" in (stockout_raw.columns if stockout_raw is not None else [])
        scenario_options = sorted(stockout_raw["scenario"].unique().tolist()) if scenario_available else SCENARIO_NAMES
        sel_scenario = st.selectbox(
            "Scenario", options=scenario_options, key="flt_scenario",
            disabled=not scenario_available,
            help=None if scenario_available else "No synthetic scenario data generated yet for this pilot.",
        )
else:
    all_skus, all_categories, all_brands = [], [], []
    sel_categories, sel_brands, sel_skus = [], [], []
    SKU_LABELS = {}
    SKU_NAMES = {}
    cmp_label_mode = CMP_LABEL_MODES[0]
    date_range = None
    focus_skus = []
    horizon = 14
    sel_scenario = SCENARIO_NAMES[0]

FILTER_KEYS = ["flt_skus", "flt_category", "flt_brands", "flt_daterange", "flt_channel",
               "flt_scenario", "flt_horizon", "flt_focus_sku", "flt_cmp_labels"]
with st.sidebar.container(key="ipa-clear-filters"):
    if st.button("Clear Filters", icon=":material/filter_alt_off:", width="stretch"):
        for k in FILTER_KEYS:
            st.session_state.pop(k, None)
        st.rerun()

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


# ==========================================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ==========================================================================
def page_executive_overview():
    render_page_header("Executive Overview", "Daily eCommerce Demand & Inventory Intelligence",
                       badges=[("box", "30 pilot SKUs"), ("globe", "naheed_web")])
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
        if inv_f is not None and not inv_f.empty:
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
                       badges=[("box", "30 pilot SKUs"), ("globe", "naheed_web")])
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
def page_forecast_explorer():
    render_page_header("Forecast Explorer", "7–14 day demand forecast · historical backtest vs real future",
                       badges=[("box", "30 pilot SKUs"), ("globe", "naheed_web")])
    if mp_raw is None:
        empty_state("Historical demand data not found", "data/processed/model_panel.parquet is missing or unreadable.", "mail")
        return

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
        hist_sku = mp_raw[mp_raw["sku"] == focus_sku].sort_values("date")
        hist_recent = hist_sku.tail(28)
    
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
                         f"Last 28 historical days (solid) + next {horizon} forecast days (dashed)")
    
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
def page_inventory_reorder():
    render_page_header("Inventory & Reorder", "Synthetic baseline snapshot · prioritised replenishment queue",
                       badges=[("box", "30 pilot SKUs"), ("globe", "naheed_web")])
    synthetic_warning(INVENTORY_PAGE_WARNING)
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
def page_stockout_lab():
    render_page_header("Stockout Scenario Lab", "What-if simulation · fully synthetic scenario results",
                       badges=[("box", "30 pilot SKUs"), ("globe", "naheed_web")])
    synthetic_warning(
        "This entire page is synthetic and scenario-based. Rates shown are simulation outputs for pilot planning "
        "— not observed or predicted real-world probabilities."
    )

    if stockout_raw is None or replen_raw is None:
        empty_state(
            "Stockout scenario simulation not generated yet",
            "This page needs data/synthetic/stockout_scenarios.parquet, replenishment_events.parquet and "
            "simulation_parameters.json. None have been generated for this pilot yet — the page will populate "
            "automatically once they are added.",
            "flask",
        )
        section_title("Planned Scenarios", "Names only; no simulated numbers are shown until data is generated.")
        cols = st.columns(4)
        for i, name in enumerate(SCENARIO_NAMES):
            with cols[i % 4]:
                html = (
                    '<div class="ipa-card" style="opacity:0.6; align-items:center; text-align:center; min-height:110px;">'
                    f'<div style="color:{COLORS["subtext"]};">{icon_svg("flask", 22)}</div>'
                    f'<div style="font-weight:700; color:{COLORS["navy"]}; margin-top:6px;">{name}</div>'
                    '<div class="ipa-kpi-sub">Pending data generation</div>'
                    '</div>'
                )
                st.markdown(html, unsafe_allow_html=True)
        return

    d = stockout_raw
    if sel_scenario and "scenario" in d.columns:
        d = d[d["scenario"] == sel_scenario]
    if sel_skus and "sku" in d.columns:
        d = d[d["sku"].isin(sel_skus)]

    section_title("Scenario Outcomes", f"Selected scenario: {sel_scenario}")
    kpi_map = {
        "stockout_rate": "Stockout Rate", "stockout_within_2d": "Stockout Within 2 Days",
        "stockout_within_7d": "Stockout Within 7 Days", "lost_sales_days": "Lost-Sales Days",
        "lost_sales_units": "Lost-Sales Units", "replenishment_order_count": "Replenishment Orders",
    }
    present = {k: v for k, v in kpi_map.items() if k in d.columns}
    if present:
        kpis = []
        for col, label in present.items():
            val = d[col].mean() if d[col].dtype.kind in "fc" and d[col].max() <= 1 else d[col].sum()
            is_rate = "rate" in col or "within" in col
            kpis.append(dict(label=label, value=format_percentage(val) if is_rate else format_number(val), icon="flask"))
        render_kpi_row(kpis, n_cols=min(6, len(kpis)))
    else:
        empty_state("Expected scenario KPI columns not found", "stockout_scenarios.parquet is present but its schema doesn't match the expected columns.", "flask")

    if "scenario" in stockout_raw.columns:
        rate_col = next((c for c in ["stockout_rate"] if c in stockout_raw.columns), None)
        if rate_col:
            agg = stockout_raw.groupby("scenario", as_index=False)[rate_col].mean().sort_values(rate_col, ascending=False)
            fig = px.bar(agg, x="scenario", y=rate_col)
            fig.update_traces(marker_color=COLORS["red"])
            fig.update_layout(**plotly_layout(legend=False, height=320))
            style_axes(fig)
            fig.update_xaxes(title="")
            render_chart(fig, "Stockout-Rate Comparison Across Scenarios")

        with st.expander("More scenario comparisons", expanded=False):
            g1, g2 = st.columns(2)
            with g1:
                two_d, seven_d = "stockout_within_2d", "stockout_within_7d"
                if two_d in stockout_raw.columns and seven_d in stockout_raw.columns:
                    agg = stockout_raw.groupby("scenario", as_index=False)[[two_d, seven_d]].mean()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=agg["scenario"], y=agg[two_d], name="Within 2 days", marker_color=COLORS["amber"]))
                    fig.add_trace(go.Bar(x=agg["scenario"], y=agg[seven_d], name="Within 7 days", marker_color=COLORS["red"]))
                    fig.update_layout(**plotly_layout(height=300), barmode="group")
                    style_axes(fig)
                    render_chart(fig, "2-Day vs 7-Day Stockout Risk")
            with g2:
                if "lost_sales_units" in stockout_raw.columns:
                    agg = stockout_raw.groupby("scenario", as_index=False)["lost_sales_units"].sum().sort_values("lost_sales_units", ascending=False)
                    fig = px.bar(agg, x="scenario", y="lost_sales_units")
                    fig.update_traces(marker_color=COLORS["red"])
                    fig.update_layout(**plotly_layout(legend=False, height=300))
                    style_axes(fig)
                    render_chart(fig, "Lost-Sales Units by Scenario")
            g3, g4 = st.columns(2)
            with g3:
                if replen_raw is not None and "scenario" in replen_raw.columns:
                    agg = replen_raw.groupby("scenario", as_index=False).size().rename(columns={"size": "orders"})
                    fig = px.bar(agg, x="scenario", y="orders")
                    fig.update_traces(marker_color=COLORS["teal"])
                    fig.update_layout(**plotly_layout(legend=False, height=300))
                    style_axes(fig)
                    render_chart(fig, "Replenishment Orders by Scenario")
            with g4:
                radar_cols = [c for c in ["stockout_rate", "stockout_within_2d", "stockout_within_7d", "lost_sales_units"] if c in stockout_raw.columns]
                if radar_cols:
                    agg = stockout_raw.groupby("scenario")[radar_cols].mean()
                    norm = (agg - agg.min()) / (agg.max() - agg.min() + 1e-9)
                    fig = go.Figure()
                    for scen in norm.index:
                        fig.add_trace(go.Scatterpolar(r=norm.loc[scen].values, theta=radar_cols, fill="toself", name=scen))
                    fig.update_layout(**plotly_layout(height=320), polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
                    render_chart(fig, "Scenario Severity")

    for focus_sku in focus_skus:
        section_title(f"Synthetic Inventory Trajectory · {comparison_display_label(focus_sku, cmp_label_mode, focus_skus)}")
        traj = d[d["sku"] == focus_sku].sort_values("date") if "date" in d.columns else pd.DataFrame()
        if traj.empty:
            empty_state("No trajectory data for this product/scenario",
                        comparison_display_label(focus_sku, cmp_label_mode, focus_skus), "trending-down")
        else:
            fig = go.Figure()
            for col, label, color in [
                ("opening_stock", "Opening stock", COLORS["slate"]), ("ending_stock", "Ending stock", COLORS["navy"]),
                ("latent_demand", "Latent demand", COLORS["grid"]), ("synthetic_sales", "Synthetic sales", COLORS["teal"]),
                ("lost_sales", "Lost sales", COLORS["risk"]),
            ]:
                if col in traj.columns:
                    fig.add_trace(go.Scatter(x=traj["date"], y=traj[col], name=label, mode="lines", line=dict(color=color)))
            if "reorder_point" in traj.columns:
                fig.add_trace(go.Scatter(x=traj["date"], y=traj["reorder_point"], name="Reorder point",
                                          mode="lines", line=dict(color=COLORS["amber"], dash="dot")))
            if replen_raw is not None and "sku" in replen_raw.columns:
                events = replen_raw[(replen_raw["sku"] == focus_sku)]
                if "scenario" in events.columns:
                    events = events[events["scenario"] == sel_scenario]
                if "date" in events.columns and not events.empty:
                    fig.add_trace(go.Scatter(x=events["date"], y=[0] * len(events), name="Replenishment event",
                                              mode="markers", marker=dict(symbol="triangle-up", size=12, color=COLORS["success"])))
            fig.update_layout(**plotly_layout(height=340))
            style_axes(fig)
            render_chart(fig, f"Synthetic Inventory Trajectory — {focus_sku} ({sel_scenario})",
                         "Opening/ending stock, latent demand, synthetic sales, lost sales & reorder point")

    section_title("Scenario Assumptions", "What changed vs baseline for this scenario.")
    if simparams_raw and isinstance(simparams_raw, dict):
        scen_params = simparams_raw.get(sel_scenario) or simparams_raw.get("scenarios", {}).get(sel_scenario)
        if scen_params:
            st.json(scen_params)
        else:
            empty_state("No parameters found for this scenario", "simulation_parameters.json does not define this scenario.", "settings")
    else:
        empty_state("Simulation parameters not generated yet", "data/synthetic/simulation_parameters.json is missing.", "settings")


# ==========================================================================
# PAGE 6 — DATA QUALITY & ASSUMPTIONS
# ==========================================================================
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
        badges=[("box", "30 pilot SKUs"), ("globe", "naheed_web")],
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


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
PAGE_FUNCS = {
    "Executive Overview": page_executive_overview,
    "Demand Analytics": page_demand_analytics,
    "Forecast Explorer": page_forecast_explorer,
    "Inventory & Reorder": page_inventory_reorder,
    "Stockout Scenario Lab": page_stockout_lab,
    "Data Quality & Assumptions": page_data_quality,
}
PAGE_FUNCS[page]()
