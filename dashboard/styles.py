"""Visual identity for the Inventory Planning Agent dashboard.

Enterprise-SaaS look: dark navy sidebar, white content on a soft grey page,
subtle borders, one accent at a time. All chart-card chrome comes from native
Streamlit bordered containers styled here — app.py keeps its custom HTML to a
few single-line, tested helpers.
"""

COLORS = {
    "navy": "#0B1F33",
    "red": "#ED1C24",       # Naheed brand red — primary accent
    "teal": "#0E7C7B",
    "blue": "#2563EB",
    "amber": "#D98E04",
    "slate": "#526277",
    "bg": "#F5F7FA",
    "border": "#E7ECF2",
    "card": "#FFFFFF",
    # semantic aliases kept for existing call sites
    "risk": "#ED1C24",
    "success": "#1F9D68",
    "warning": "#D98E04",
    "grid": "#EDF1F5",
    "text": "#0B1F33",
    "subtext": "#6B7A8F",
    "purple": "#7C4DFF",
}

# Donut/multi-series ramp — Naheed red + dark navy lead.
CATEGORICAL = ["#ED1C24", "#0B2942", COLORS["teal"], COLORS["amber"], COLORS["slate"],
               COLORS["success"], COLORS["blue"], COLORS["purple"]]

# Donut-specific hover colors matching the brand palette
DONUT_COLORS = ["#ED1C24", "#0B2942"]
DONUT_HOVER  = ["#C9151D", "#163D5C"]

STATUS_COLORS = {
    "Critical": COLORS["red"],
    "Reorder Now": COLORS["amber"],
    "Watch": COLORS["blue"],
    "Healthy": COLORS["success"],
}

# Soft icon-badge tones (background + foreground) cycled across KPI cards.
TONES = {
    "red": {"bg": "#FCE5E4", "fg": COLORS["red"]},
    "risk": {"bg": "#FCE5E4", "fg": COLORS["red"]},
    "teal": {"bg": "#E3F3F1", "fg": COLORS["teal"]},
    "blue": {"bg": "#E5EDFD", "fg": COLORS["blue"]},
    "amber": {"bg": "#FBF0DC", "fg": COLORS["amber"]},
    "slate": {"bg": "#EAEEF3", "fg": COLORS["slate"]},
    "success": {"bg": "#E4F5EC", "fg": COLORS["success"]},
    "navy": {"bg": "#E7ECF3", "fg": COLORS["navy"]},
    "purple": {"bg": "#F0EAFB", "fg": COLORS["purple"]},
}
TONE_CYCLE = ["red", "slate", "red", "success", "blue", "teal"]

FONT_FAMILY = "'Inter', 'Segoe UI', Arial, sans-serif"

# --------------------------------------------------------------------------
# Minimal line-icon set (Lucide-inspired stroke style) — replaces emoji glyphs
# everywhere in the dashboard. Each value is the INNER svg markup only; wrap
# with icon_svg() to get a full <svg>, sized and coloured via currentColor.
# --------------------------------------------------------------------------
ICONS = {
    "box": '<path d="M21 8 12 3 3 8v8l9 5 9-5V8Z"/><path d="M3 8l9 5 9-5"/><path d="M12 13v8"/>',
    "cart": '<circle cx="9" cy="20" r="1.3"/><circle cx="18" cy="20" r="1.3"/>'
            '<path d="M3 4h2l2.3 12.1a2 2 0 0 0 2 1.7h7.8a2 2 0 0 0 2-1.6L21 8H6"/>',
    "trending-up": '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    "trending-down": '<path d="M3 7l6 6 4-4 8 8"/><path d="M15 17h6v-6"/>',
    "shield-check": '<path d="M12 2 4 5v6c0 5 3.5 8.5 8 11 4.5-2.5 8-6 8-11V5l-8-3Z"/><path d="M9 12l2 2 4-4"/>',
    "calendar": '<rect x="3" y="5" width="18" height="16" rx="2.5"/><path d="M3 10h18"/><path d="M8 3v4"/><path d="M16 3v4"/>',
    "percent": '<circle cx="7.5" cy="7.5" r="2.3"/><circle cx="16.5" cy="16.5" r="2.3"/><path d="M6 18 18 6"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/>'
             '<path d="M12 3c2.4 2.7 3.8 6 3.8 9s-1.4 6.3-3.8 9c-2.4-2.7-3.8-6-3.8-9s1.4-6.3 3.8-9Z"/>',
    "tag": '<path d="M3 11.5V4h7.5L21 14.5l-7.5 7.5L3 11.5Z"/><circle cx="7.3" cy="7.8" r="1.1" fill="currentColor" stroke="none"/>',
    "folder": '<path d="M3 6.5a1 1 0 0 1 1-1h4.6l1.8 2H20a1 1 0 0 1 1 1v9.9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6.5Z"/>',
    "refresh": '<path d="M4 12a8 8 0 0 1 14-5.2L21 9"/><path d="M20 12a8 8 0 0 1-14 5.2L3 15"/>'
               '<path d="M21 4.5V9h-4.5"/><path d="M3 19.5V15h4.5"/>',
    "alert-triangle": '<path d="M12 3 2 20h20L12 3Z"/><path d="M12 9.5v4"/>'
                       '<circle cx="12" cy="16.8" r="0.9" fill="currentColor" stroke="none"/>',
    "coin": '<circle cx="12" cy="12" r="9"/><path d="M12 7.3v9.4"/>'
            '<path d="M9.3 9.6c0-1.3 1.2-2.3 2.7-2.3s2.7 1 2.7 2.1-1.2 1.7-2.7 2-2.7.9-2.7 2.1 1.2 2.1 2.7 2.1 2.7-.9 2.7-2.1"/>',
    "bar-chart": '<path d="M4 21V10"/><path d="M12 21V4"/><path d="M20 21v-8"/>',
    "file-text": '<path d="M7 3h6.5L18 7.5V21H7Z"/><path d="M13.5 3v4.5H18"/><path d="M9.5 12.5h5"/><path d="M9.5 16h5"/>',
    "calculator": '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M8 7.3h8"/>'
                  '<path d="M8.3 11.3h.1"/><path d="M12 11.3h.1"/><path d="M15.7 11.3h.1"/>'
                  '<path d="M8.3 15.3h.1"/><path d="M12 15.3h.1"/><path d="M15.7 15.3h.1"/>',
    "flask": '<path d="M9.5 3h5"/><path d="M10.3 3v6.2l-5.4 9.2A1.4 1.4 0 0 0 6.1 20.5h11.8a1.4 1.4 0 0 0 1.2-2.1l-5.4-9.2V3"/>'
             '<path d="M7.8 15.5h8.4"/>',
    "check-circle": '<circle cx="12" cy="12" r="9"/><path d="M8 12.3l2.6 2.6L16.2 9"/>',
    "briefcase": '<rect x="3" y="8" width="18" height="12" rx="2"/><path d="M8.5 8V6.3a2 2 0 0 1 2-2h3a2 2 0 0 1 2 2V8"/><path d="M3 13.5h18"/>',
    "circle-dashed": '<circle cx="12" cy="12" r="9" stroke-dasharray="3 3.4"/>',
    "search": '<circle cx="10.3" cy="10.3" r="6.8"/><path d="M20.5 20.5l-4.7-4.7"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 8.2h.01"/><path d="M11 11.5h1v5.3h1"/>',
    "sparkle": '<path d="M12 3 13.6 9 20 12 13.6 15 12 21 10.4 15 4 12 10.4 9 12 3Z"/>',
    "layers": '<path d="M12 3 3 8l9 4.5 9-4.5-9-5Z"/><path d="M3 14l9 4.5 9-4.5"/>',
    "database": '<ellipse cx="12" cy="5.5" rx="7.5" ry="2.7"/><path d="M4.5 5.5v13c0 1.5 3.4 2.7 7.5 2.7s7.5-1.2 7.5-2.7v-13"/>'
                '<path d="M4.5 12c0 1.5 3.4 2.7 7.5 2.7s7.5-1.2 7.5-2.7"/>',
    "list": '<path d="M9 6.5h11"/><path d="M9 12h11"/><path d="M9 17.5h11"/>'
            '<circle cx="4.5" cy="6.5" r="1" fill="currentColor" stroke="none"/>'
            '<circle cx="4.5" cy="12" r="1" fill="currentColor" stroke="none"/>'
            '<circle cx="4.5" cy="17.5" r="1" fill="currentColor" stroke="none"/>',
    "mail": '<rect x="3" y="5.5" width="18" height="13" rx="2"/><path d="M3.5 7 12 13l8.5-6"/>',
    "target": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5.3l3.5 2"/>',
    "truck": '<rect x="2.5" y="7" width="12" height="10" rx="1.3"/><path d="M14.5 10.5H18l3 3.3V17h-3"/>'
             '<circle cx="6.5" cy="18.3" r="1.6"/><circle cx="16.5" cy="18.3" r="1.6"/>',
    "dot": '<circle cx="12" cy="12" r="4.5" fill="currentColor" stroke="none"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M12 3v2.4"/><path d="M12 18.6V21"/>'
                '<path d="M3 12h2.4"/><path d="M18.6 12H21"/><path d="M5.6 5.6l1.7 1.7"/>'
                '<path d="M16.7 16.7l1.7 1.7"/><path d="M5.6 18.4l1.7-1.7"/><path d="M16.7 7.3l1.7-1.7"/>',
    "package-search": '<path d="M21 8 12 3 3 8v8l9 5 9-5V8Z"/><path d="M3 8l9 5 9-5"/><path d="M12 13v8"/>',
}


def icon_svg(name, size=18):
    """Return a standalone <svg> for an icon key. Unknown keys fall back to a plain dot."""
    inner = ICONS.get(name, ICONS["dot"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="1.9" stroke-linecap="round" '
        f'stroke-linejoin="round" style="display:block;">{inner}</svg>'
    )


def plotly_layout(height=320, legend=True, **overrides):
    """Base kwargs applied to every chart for a consistent, restrained look."""
    layout = dict(
        height=height,
        margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["text"], size=12.5),
        hoverlabel=dict(bgcolor="white", font_family=FONT_FAMILY, font_size=12,
                         bordercolor=COLORS["border"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                     bgcolor="rgba(0,0,0,0)", font=dict(size=11)) if legend else dict(visible=False),
        colorway=CATEGORICAL,
    )
    layout.update(overrides)
    return layout


AXIS_STYLE = dict(
    showgrid=True, gridcolor=COLORS["grid"], gridwidth=1,
    zeroline=False, showline=False,
    tickfont=dict(color=COLORS["subtext"], size=11),
    title_font=dict(color=COLORS["subtext"], size=11),
)


def style_axes(fig, **kwargs):
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


CUSTOM_CSS = f"""
<style>
    .stApp {{ background-color: {COLORS["bg"]}; }}
    #MainMenu, footer {{ visibility: hidden; }}
    /* Hide the decorative header bar + Deploy/toolbar chrome, keep sidebar toggle accessible.
       NOTE: the sidebar re-expand arrow (stExpandSidebarButton) lives INSIDE stToolbar, so
       stToolbar itself must stay rendered -- only its Deploy/decoration/status children are
       hidden individually below. Hiding stToolbar wholesale removes the only way to reopen
       a collapsed sidebar. */
    header[data-testid="stHeader"] {{ background: transparent !important; }}
    [data-testid="stToolbar"] {{ background: transparent !important; }}
    [data-testid="stAppDeployButton"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}
    header[data-testid="stHeader"] button[data-testid="stBaseButton-headerNoPadding"] {{
        visibility: visible !important;
        opacity: 1 !important;
        color: {COLORS["slate"]} !important;
    }}
    html, body, [class*="css"] {{ font-family: {FONT_FAMILY}; }}
    div.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1360px; }}

    /* ---------- Page header ---------- */
    .ipa-header {{
        display: flex; align-items: flex-start; justify-content: space-between;
        flex-wrap: wrap; gap: 16px; margin-bottom: 22px;
    }}
    .ipa-kicker {{
        color: {COLORS["red"]}; font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 2.5px; margin-bottom: 2px;
    }}
    .ipa-header h1 {{
        color: {COLORS["navy"]}; font-size: 2.05rem; font-weight: 800; margin: 0;
        letter-spacing: -0.4px; line-height: 1.12;
    }}
    .ipa-header h1 .accent {{ color: {COLORS["red"]}; }}
    .ipa-header .ipa-sub {{ color: {COLORS["subtext"]}; font-size: 0.92rem; margin-top: 6px; }}
    .ipa-status-badge {{
        display: inline-flex; align-items: center; gap: 7px; background: {COLORS["navy"]}; color: #FFFFFF;
        border-radius: 999px; padding: 5px 14px; font-size: 0.73rem; font-weight: 600;
        margin-top: 12px; letter-spacing: 0.3px;
    }}
    .ipa-status-badge::before {{
        content: ""; width: 7px; height: 7px; border-radius: 50%; background: {COLORS["red"]};
        box-shadow: 0 0 0 3px rgba(237,28,36,0.25);
    }}
    /* Light outlined chip row (used on Data Quality page) */
    .ipa-chip-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
    .ipa-chip {{
        display: inline-flex; align-items: center; gap: 7px; background: {COLORS["card"]};
        border: 1px solid {COLORS["border"]}; color: {COLORS["navy"]};
        border-radius: 999px; padding: 5px 14px; font-size: 0.76rem; font-weight: 600;
        box-shadow: 0 1px 2px rgba(11,31,51,0.04);
    }}
    .ipa-chip svg {{ color: {COLORS["slate"]}; flex-shrink: 0; }}
    .ipa-meta {{ display: flex; align-items: center; gap: 9px; padding-top: 4px; }}
    .ipa-meta .refresh-ico {{
        width: 30px; height: 30px; border-radius: 50%; border: 1.5px solid {COLORS["border"]};
        display: flex; align-items: center; justify-content: center; color: {COLORS["slate"]}; font-size: 0.85rem;
    }}
    .ipa-meta .txt {{ text-align: left; line-height: 1.3; }}
    .ipa-meta .txt .k {{ color: {COLORS["subtext"]}; font-size: 0.72rem; }}
    .ipa-meta .txt .v {{ color: {COLORS["navy"]}; font-size: 0.8rem; font-weight: 700; white-space: nowrap; }}

    /* ---------- KPI cards ---------- */
    .ipa-card {{
        position: relative; overflow: hidden;
        background: {COLORS["card"]}; border: 1px solid {COLORS["border"]};
        border-radius: 14px; padding: 16px 18px 16px 20px; min-height: 118px;
        box-shadow: 0 1px 3px rgba(11,31,51,0.05);
        display: flex; flex-direction: column; justify-content: flex-start;
        transition: box-shadow 0.22s ease;
        cursor: default;
    }}
    /* Left accent: short segment by default, grows to full height on hover/focus. */
    .ipa-card::before {{
        content: ""; position: absolute; left: 0; width: 4px;
        top: 38%; bottom: 38%;
        background: var(--accent, {COLORS["red"]});
        border-radius: 0 4px 4px 0;
        transition: top 0.28s ease, bottom 0.28s ease, border-radius 0.28s ease;
    }}
    .ipa-card:hover {{ box-shadow: 0 6px 16px rgba(11,31,51,0.09); }}
    .ipa-card:hover::before {{ top: 0; bottom: 0; border-radius: 14px 0 0 14px; }}
    .ipa-kpi-top {{ display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }}
    .ipa-kpi-icon {{
        width: 34px; height: 34px; min-width: 34px; border-radius: 10px; display: flex; align-items: center;
        justify-content: center;
    }}
    .ipa-kpi-label {{
        font-size: 0.68rem; font-weight: 700; color: {COLORS["subtext"]};
        text-transform: uppercase; letter-spacing: 0.6px; margin: 0;
        word-wrap: break-word; word-break: break-word; white-space: normal; flex: 1;
    }}
    .ipa-kpi-value {{
        font-size: 1.65rem; font-weight: 700; color: {COLORS["navy"]}; line-height: 1.2;
        word-wrap: break-word; word-break: break-word; white-space: normal;
    }}
    .ipa-kpi-sub {{ font-size: 0.74rem; color: {COLORS["subtext"]}; margin-top: 5px; line-height: 1.3; }}

    /* Compact uniform variant — Data Quality & Assumptions page only.
       Equal fixed height + wrapping so long contract values never clip. */
    .ipa-card--compact {{
        height: 148px; padding: 16px 18px 16px 20px;
        justify-content: flex-start;
    }}
    .ipa-card--compact .ipa-kpi-top {{ margin-bottom: 10px; }}
    .ipa-card--compact .ipa-kpi-label {{ font-size: 0.64rem; letter-spacing: 0.5px; }}
    .ipa-card--compact .ipa-kpi-value {{ font-size: 1.32rem; font-weight: 700; line-height: 1.28; }}

    /* ---------- Chart / content cards (native bordered containers) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 14px !important; border: 1px solid {COLORS["border"]} !important;
        box-shadow: 0 1px 3px rgba(11,31,51,0.05); background: {COLORS["card"]};
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{ padding: 4px 4px 0 4px; }}
    .ipa-card-title {{ font-size: 0.98rem; font-weight: 700; color: {COLORS["navy"]}; margin: 2px 2px 0 2px;
        display: flex; align-items: center; gap: 8px; }}
    .ipa-card-title svg {{ color: {COLORS["red"]}; flex-shrink: 0; }}
    .ipa-card-sub {{ font-size: 0.76rem; color: {COLORS["subtext"]}; margin: 1px 2px 4px 2px; }}

    /* ---------- Section titles ---------- */
    .ipa-section-title {{
        font-size: 1.3rem; font-weight: 700; color: {COLORS["navy"]}; margin: 26px 0 4px 0;
    }}
    .ipa-section-sub {{ font-size: 0.84rem; color: {COLORS["subtext"]}; margin-bottom: 12px; }}

    /* ---------- Banners ---------- */
    .ipa-banner {{ border-radius: 10px; padding: 10px 15px; margin: 6px 0 16px 0; font-size: 0.84rem; }}
    .ipa-banner-synthetic {{ background: #FBF3E3; border: 1px solid #F0DBAE; color: #7A5804; }}
    .ipa-banner-info {{ background: #E9F4F4; border: 1px solid #C4E3E2; color: #0B4A49; }}
    .ipa-banner-success {{ background: #E4F5EC; border: 1px solid #BEE6CF; color: #10633F; }}
    .ipa-banner-empty {{
        background: {COLORS["card"]}; border: 1px dashed #C7D0DB; border-radius: 14px;
        padding: 30px 22px; margin: 8px 0; text-align: center; color: {COLORS["subtext"]};
    }}
    .ipa-empty-icon {{ font-size: 2rem; }}
    .ipa-empty-title {{ font-weight: 700; color: {COLORS["navy"]}; margin-top: 8px; font-size: 1rem; }}
    .ipa-empty-msg {{ margin-top: 4px; font-size: 0.85rem; line-height: 1.45; }}

    /* ---------- Badges ---------- */
    .ipa-badge {{ display: inline-block; border-radius: 999px; padding: 3px 12px; font-size: 0.73rem; font-weight: 700; }}
    .ipa-badge-critical {{ background: #FAE7E9; color: {COLORS["red"]}; }}
    .ipa-badge-reorder  {{ background: #FBF0DC; color: {COLORS["amber"]}; }}
    .ipa-badge-watch    {{ background: #E5EDFD; color: {COLORS["blue"]}; }}
    .ipa-badge-healthy  {{ background: #E4F5EC; color: {COLORS["success"]}; }}

    /* ---------- Insight cards ---------- */
    .ipa-insight {{
        background: {COLORS["card"]}; border: 1px solid {COLORS["border"]}; border-radius: 12px;
        padding: 16px 18px; min-height: 104px; height: 100%; box-shadow: 0 1px 3px rgba(11,31,51,0.05);
        display: flex; gap: 14px; align-items: center; font-size: 0.9rem; color: {COLORS["text"]}; line-height: 1.45;
    }}
    .ipa-insight .txt {{ flex: 1; }}
    /* Larger icon badge so attention lands on it first. */
    .ipa-insight .ico {{
        width: 44px; height: 44px; min-width: 44px; border-radius: 11px; background: #E3F3F1;
        color: {COLORS["teal"]}; display: flex; align-items: center; justify-content: center; align-self: flex-start;
    }}

    /* ---------- Equal-height side-by-side cards (all pages) ----------
       The horizontal block is flex; make each column a flex item that stretches,
       then let its bordered card fill the column so neighbours match height. */
    [data-testid="stHorizontalBlock"] {{ align-items: stretch; }}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{ display: flex; }}
    [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{ width: 100%; height: 100%; }}
    [data-testid="stColumn"] div[data-testid="stVerticalBlockBorderWrapper"] {{ height: 100%; }}

    /* ---------- Compact metric panel (label/value rows) ---------- */
    /* One flex block that fills the card height so divider-separated rows spread evenly. */
    .ipa-mpanel {{ display: flex; flex-direction: column; height: 100%; }}
    .ipa-mbody {{ display: flex; flex-direction: column; justify-content: space-between;
        flex: 1 1 auto; margin-top: 10px; }}
    .ipa-mrow {{ display: flex; justify-content: space-between; align-items: center;
        padding: 11px 2px; border-bottom: 1px solid {COLORS["grid"]}; }}
    .ipa-mrow:last-child {{ border-bottom: none; }}
    .ipa-mrow .l {{ color: {COLORS["subtext"]}; font-size: 0.83rem; }}
    .ipa-mrow .v {{ color: {COLORS["navy"]}; font-size: 0.94rem; font-weight: 700; }}
    /* Fill chain (scoped to metric panels only): lets .ipa-mpanel reach 100% height
       without stretching chart-card titles. Keyed via st.container(key="ipa-mp-..."). */
    [class*="st-key-ipa-mp-"] [data-testid="stVerticalBlock"],
    [class*="st-key-ipa-mp-"] [data-testid="stElementContainer"],
    [class*="st-key-ipa-mp-"] [data-testid="stMarkdown"],
    [class*="st-key-ipa-mp-"] [data-testid="stMarkdownContainer"] {{ height: 100%; }}

    /* ---------- Sidebar: dark navy with subtle bottom dot-grid texture ---------- */
    section[data-testid="stSidebar"] {{
        position: relative;
        background:
          radial-gradient(circle at 12% 96%, rgba(37,99,235,0.14) 0, rgba(37,99,235,0) 26%),
          radial-gradient(circle at 34% 100%, rgba(237,28,36,0.14) 0, rgba(237,28,36,0) 24%),
          linear-gradient(180deg, #0A1B2D 0%, #0B1F33 60%, #081627 100%);
        border-right: none;
    }}
    section[data-testid="stSidebar"]::after {{
        content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 220px;
        background-image: radial-gradient(rgba(255,255,255,0.16) 1px, transparent 1.4px);
        background-size: 15px 15px;
        -webkit-mask-image: linear-gradient(to top, black, transparent);
        mask-image: linear-gradient(to top, black, transparent);
        pointer-events: none; z-index: 0;
    }}
    section[data-testid="stSidebar"] > div {{ position: relative; z-index: 1; }}
    section[data-testid="stSidebar"] * {{ color: #DCE4EE; }}
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top: 0.8rem !important; }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 0.8rem; }}
    .ipa-brand {{ padding: 0px 0px 14px 4px; margin: -10px -7px -9px 0; }}
    .ipa-brand .logo {{ display: flex; align-items: center; gap: 6px; }}
    .ipa-brand .ipa-logo-img {{ height: 90px; width: auto; display: block; margin-top: -50px; margin-left: -15px; }}
    .ipa-brand .mark {{ font-size: 1.25rem; line-height: 1; }}
    .ipa-brand .n {{ color: {COLORS["red"]}; font-weight: 800; font-size: 2rem; letter-spacing: -0.5px;
        font-family: {FONT_FAMILY}; }}
    .ipa-brand .n sup {{ font-size: 0.6rem; top: -0.3em; }}
    .ipa-nav-label {{ color: #6E829C; font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1.4px; margin: 16px 0 8px 4px; }}

    /* Forecast-horizon toggle (the only remaining sidebar radiogroup) */
    section[data-testid="stSidebar"] div[role="radiogroup"] {{ 
        gap: 8px !important; 
        display: flex !important;
        flex-direction: row !important;
        width: 100%;
        flex-wrap: nowrap !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        border-radius: 12px !important; 
        padding: 12px 18px !important; 
        width: auto !important; 
        flex: 1 !important; 
        transition: background 0.15s ease; 
        cursor: pointer;
        display: flex !important; 
        align-items: center !important; 
        justify-content: center !important;
        flex-direction: row !important;
        background: rgba(255,255,255,0.05);
        margin: 0 !important;
        white-space: nowrap !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: rgba(255,255,255,0.1); }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        background: {COLORS["red"]} !important; box-shadow: 0 4px 12px rgba(237,28,36,0.35); }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div {{ color: #FFFFFF !important; font-weight: 600; }}
    /* Hide the radio circle indicator explicitly */
    section[data-testid="stSidebar"] div[role="radiogroup"] label > *:not(:last-child) {{ display: none !important; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label > *:last-child {{
        display: flex !important; align-items: center !important; justify-content: center !important; width: 100%; white-space: nowrap !important;
    }}
    section[data-testid="stSidebar"] label p {{ font-size: 0.9rem; margin: 0 !important; text-align: center !important; white-space: nowrap !important; line-height: 1.2 !important; }}

    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div,
    section[data-testid="stSidebar"] [data-testid="stMultiSelect"] > div,
    section[data-testid="stSidebar"] [data-testid="stDateInput"] > div {{
        background-color: rgba(255,255,255,0.07) !important; 
        border: 1px solid rgba(255,255,255,0.16) !important; 
        border-radius: 9px !important; 
    }}
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div:focus-within,
    section[data-testid="stSidebar"] [data-testid="stMultiSelect"] > div:focus-within,
    section[data-testid="stSidebar"] [data-testid="stDateInput"] > div:focus-within {{
        border-color: {COLORS["red"]} !important;
        box-shadow: 0 0 0 1px {COLORS["red"]} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] *,
    section[data-testid="stSidebar"] [data-testid="stMultiSelect"] *,
    section[data-testid="stSidebar"] [data-testid="stDateInput"] * {{
        background-color: transparent !important;
        color: #FFFFFF !important;
    }}
    /* Placeholder visibility */
    section[data-testid="stSidebar"] input::placeholder {{
        color: #90A2BC !important;
        opacity: 1 !important;
    }}
    section[data-testid="stSidebar"] input {{ color: #FFFFFF !important; }}
    
    /* Multiselect Tags (Red round boxes) */
    section[data-testid="stSidebar"] [data-baseweb="tag"] {{
        background-color: {COLORS["red"]} !important;
        border-radius: 20px !important;
        padding: 4px 10px !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="tag"] * {{
        color: #FFFFFF !important;
    }}

    /* Icon-based nav buttons */
    .st-key-ipa-nav .stButton {{ margin-bottom: 10px; }}
    .st-key-ipa-nav .stButton:last-child {{ margin-bottom: 0; }}
    
    .st-key-ipa-nav .stButton button {{
        justify-content: flex-start !important; 
        text-align: left !important; 
        border-radius: 12px; 
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        padding-left: 34px !important;
    }}
    .st-key-ipa-nav .stButton button p {{
        margin: 0 !important;
        text-align: left !important;
        font-size: 1rem !important;
    }}
    
    /* Secondary (unselected) buttons */
    .st-key-ipa-nav .stButton button[kind="secondary"],
    .st-key-ipa-nav [data-testid="stBaseButton-secondary"] {{
        position: relative; background: transparent !important; border: none !important; color: #90A2BC !important; font-weight: 500;
        padding: 11px 16px 11px 34px !important;
    }}
    .st-key-ipa-nav .stButton button[kind="secondary"]::before,
    .st-key-ipa-nav [data-testid="stBaseButton-secondary"]::before {{
        content: ""; position: absolute; left: 16px; top: 50%; transform: translateY(-50%);
        width: 5px; height: 5px; border-radius: 50%; background: #55698A;
    }}
    .st-key-ipa-nav .stButton button[kind="secondary"]:hover,
    .st-key-ipa-nav [data-testid="stBaseButton-secondary"]:hover {{
        background: rgba(255,255,255,0.06) !important; color: #FFFFFF !important;
    }}
    
    /* Primary (selected) button — reduced glow per brand spec */
    .st-key-ipa-nav .stButton button[kind="primary"],
    .st-key-ipa-nav [data-testid="stBaseButton-primary"] {{
        background: #ed1c24 !important; border: none !important; color: #FFFFFF !important; font-weight: 700;
        border-radius: 14px; 
        padding: 14px 20px !important;
        box-shadow: 0 8px 22px rgba(237,28,36,0.25) !important;
        transition: background 0.2s ease, box-shadow 0.2s ease;
    }}
    .st-key-ipa-nav .stButton button[kind="primary"]:hover,
    .st-key-ipa-nav [data-testid="stBaseButton-primary"]:hover {{ opacity: 0.95 !important; }}
    
    .st-key-ipa-nav .stButton button > div,
    .st-key-ipa-nav .stButton button span {{
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        text-align: left !important;
    }}

    /* SKU deep-dive field styled as a search box (magnifying-glass icon, no chevron) */
    .st-key-ipa-sku-search div[data-baseweb="select"] > div {{ position: relative; padding-right: 34px; }}
    .st-key-ipa-sku-search div[data-baseweb="select"] svg {{ display: none; }}
    /* Re-enable the cross icon inside tags so users can remove them */
    .st-key-ipa-sku-search span[data-baseweb="tag"] svg {{ display: inline-block !important; }}
    .st-key-ipa-sku-search div[data-baseweb="select"] > div::after {{
        content: ""; position: absolute; right: 12px; top: 50%; transform: translateY(-50%);
        width: 15px; height: 15px; pointer-events: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236B7A8F' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='10.3' cy='10.3' r='6.8'/%3E%3Cpath d='M20.5 20.5l-4.7-4.7'/%3E%3C/svg%3E");
        background-repeat: no-repeat; background-size: contain;
    }}

    /* Clear-filters button keeps a distinct outlined ghost look */
    .st-key-ipa-clear-filters [data-testid="stBaseButton-secondary"] {{
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.22); color: #DCE4EE;
        border-radius: 9px; width: 100%; font-weight: 600;
    }}
    .st-key-ipa-clear-filters [data-testid="stBaseButton-secondary"]:hover {{
        border-color: {COLORS["red"]}; color: #FFFFFF; background: rgba(237,28,36,0.20);
    }}
    section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.09); margin: 14px 0; }}
    
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
        background-color: transparent !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary:focus,
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary:active {{
        background-color: rgba(255,255,255,0.05) !important;
    }}
    
    section[data-testid="stSidebar"] summary, 
    section[data-testid="stSidebar"] summary p {{ 
        color: #B9C6D8 !important; 
        font-size: 0.86rem !important; 
        font-weight: 600 !important;
    }}
    section[data-testid="stSidebar"] summary:hover,
    section[data-testid="stSidebar"] summary:hover p {{
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] summary svg {{
        color: #B9C6D8 !important; 
        fill: #B9C6D8 !important;
    }}
    
    section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] caption {{ color: #7F92AB !important; }}

    /* ---------- Tabs ---------- */
    button[data-baseweb="tab"] {{ font-weight: 600; font-size: 0.9rem; }}
    div[data-baseweb="tab-list"] {{ gap: 4px; }}

    /* ---------- Dataframe ---------- */
    div[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; border: 1px solid {COLORS["border"]}; }}

    /* ---------- Loading skeleton ---------- */
    .stSkeleton, [data-testid="stSkeleton"] {{
        background: linear-gradient(90deg, #EDF1F5 25%, #F5F7FA 50%, #EDF1F5 75%) !important;
        background-size: 200% 100% !important;
        animation: ipa-shimmer 1.5s ease-in-out infinite !important;
        border-radius: 10px !important;
    }}
    @keyframes ipa-shimmer {{
        0% {{ background-position: 200% 0; }}
        100% {{ background-position: -200% 0; }}
    }}

    /* ---------- Smooth chart transitions ---------- */
    .js-plotly-plot, .plotly {{
        transition: opacity 0.3s ease;
    }}

    /* ---------- Sidebar active transition ---------- */
    section[data-testid="stSidebar"] {{
        transition: transform 0.3s ease, width 0.3s ease;
    }}

    /* ---------- Content card hover (charts) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        box-shadow: 0 4px 16px rgba(11,31,51,0.08) !important;
        transition: box-shadow 0.22s ease;
    }}

    /* ---------- Responsive: smaller screens ---------- */
    @media (max-width: 992px) {{
        div.block-container {{ padding-left: 1rem; padding-right: 1rem; max-width: 100%; }}
        .ipa-header h1 {{ font-size: 1.5rem; }}
        .ipa-kpi-value {{ font-size: 1.3rem; }}
    }}
    @media (max-width: 768px) {{
        div.block-container {{ padding-left: 0.5rem; padding-right: 0.5rem; }}
        .ipa-header {{ flex-direction: column; gap: 8px; }}
        .ipa-header h1 {{ font-size: 1.25rem; }}
        .ipa-kpi-value {{ font-size: 1.1rem; }}
        .ipa-card {{ padding: 12px 14px; min-height: 90px; }}
    }}

    /* ---------- Phase 5: data-source chip, run status, model chips, winner cards ---------- */
    .st-key-ipa-datasource {{ margin-bottom: 4px; }}
    .ipa-ds-chip {{ display: inline-block; margin-top: 6px; padding: 3px 10px; border-radius: 999px;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.3px; }}
    .ipa-ds-run {{ background: {COLORS["teal"]}; color: #FFFFFF; }}
    .ipa-ds-legacy {{ background: rgba(255,255,255,0.08); color: #DCE4EE; border: 1px solid rgba(255,255,255,0.18); }}
    .ipa-run-card {{ background: {COLORS["card"]}; border: 1px solid {COLORS["border"]};
        border-radius: 14px; padding: 8px 4px; margin: 4px 0 10px 0; }}
    .ipa-model-chip {{ display: block; text-align: center; padding: 8px 10px; border-radius: 10px;
        font-size: 0.82rem; font-weight: 700; border: 1px solid {COLORS["border"]}; }}
    .ipa-ms-completed {{ background: #E4F5EC; color: {COLORS["success"]}; }}
    .ipa-ms-running {{ background: #E5EDFD; color: {COLORS["blue"]}; }}
    .ipa-ms-failed {{ background: #FAE7E9; color: {COLORS["red"]}; }}
    .ipa-ms-pending {{ background: #EAEEF3; color: {COLORS["slate"]}; }}
    .ipa-winner-card {{ background: {COLORS["card"]}; border: 1px solid {COLORS["border"]};
        border-left: 4px solid {COLORS["teal"]}; border-radius: 12px; padding: 12px 14px; }}
    .ipa-winner-card .lbl {{ font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.6px; color: {COLORS["subtext"]}; }}
    .ipa-winner-card .mdl {{ font-size: 1.05rem; font-weight: 800; color: {COLORS["navy"]}; margin-top: 2px; }}
    .ipa-winner-card .met {{ font-size: 0.78rem; color: {COLORS["subtext"]}; margin-top: 4px; }}

    /* ---------- Phase B: Stockout Risk page ---------- */
    .ipa-rtier {{ display: inline-flex; align-items: center; gap: 6px; border-radius: 999px;
        padding: 2px 10px; font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.4px; }}
    .ipa-rtier-critical {{ background: #FAE7E9; color: {COLORS["red"]}; }}
    .ipa-rtier-high {{ background: #FBF0DC; color: {COLORS["amber"]}; }}
    .ipa-rtier-watch {{ background: #E5EDFD; color: {COLORS["blue"]}; }}
    .ipa-rtier-low, .ipa-rtier-healthy {{ background: #E4F5EC; color: {COLORS["success"]}; }}
    .ipa-rtier-unknown {{ background: #EAEEF3; color: {COLORS["slate"]}; }}
    .ipa-riskstripe {{ height: 4px; border-radius: 6px; margin: 0 0 10px 0; }}
    .ipa-riskname {{ font-weight: 700; color: {COLORS["navy"]}; font-size: 0.95rem;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }}
    .ipa-risksku {{ color: {COLORS["subtext"]}; font-size: 0.72rem; margin: 1px 0 6px 0; }}
    .ipa-riskgrid {{ display: flex; flex-wrap: wrap; gap: 3px 18px; }}
    .ipa-riskgrid .m {{ font-size: 0.75rem; color: {COLORS["subtext"]}; }}
    .ipa-riskgrid .m b {{ color: {COLORS["navy"]}; font-weight: 700; }}
    /* Selected risk card: stronger border + subtle red tint (scoped to keyed container) */
    [class*="st-key-riskcard-sel-"] div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: {COLORS["red"]} !important;
        box-shadow: 0 0 0 1px {COLORS["red"]}, 0 6px 16px rgba(11,31,51,0.10) !important;
        background: #FFF7F7 !important;
    }}
    /* Card "open" buttons: full-width, quiet — scoped to the queue so global buttons are untouched */
    [class*="st-key-riskopen-"] button {{ width: 100% !important; border-radius: 10px !important;
        font-weight: 600 !important; }}
    /* Reason-trace explanation card */
    .ipa-reason {{ background: {COLORS["card"]}; border: 1px solid {COLORS["border"]};
        border-left: 4px solid {COLORS["navy"]}; border-radius: 12px; padding: 14px 16px;
        font-size: 0.9rem; color: {COLORS["text"]}; line-height: 1.55; white-space: normal; }}
    .ipa-reason .h {{ font-weight: 700; color: {COLORS["navy"]}; margin-bottom: 4px;
        display: flex; align-items: center; gap: 7px; }}
    .ipa-dd-head {{ font-size: 1.15rem; font-weight: 800; color: {COLORS["navy"]}; line-height: 1.25;
        margin: 2px 0 1px 0; }}
    .ipa-dd-sub {{ color: {COLORS["subtext"]}; font-size: 0.82rem; margin-bottom: 6px; }}

    /* ---------- Phase C: Inventory & Reorder page ---------- */
    .ipa-action {{ display: inline-flex; align-items: center; gap: 6px; border-radius: 999px;
        padding: 2px 10px; font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.4px; }}
    .ipa-action-order_now {{ background: #FAE7E9; color: {COLORS["red"]}; }}
    .ipa-action-vendor_follow_up {{ background: #E5EDFD; color: {COLORS["blue"]}; }}
    .ipa-action-manual_review {{ background: #FBF0DC; color: {COLORS["amber"]}; }}
    .ipa-action-monitor {{ background: #EAEEF3; color: {COLORS["slate"]}; }}
    .ipa-action-no_order {{ background: #E4F5EC; color: {COLORS["success"]}; }}
    /* selected recommendation card: teal accent (scoped to keyed container) */
    [class*="st-key-recocard-sel-"] div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: {COLORS["teal"]} !important;
        box-shadow: 0 0 0 1px {COLORS["teal"]}, 0 6px 16px rgba(11,31,51,0.10) !important;
        background: #F3FBFB !important;
    }}
    [class*="st-key-recoopen-"] button {{ width: 100% !important; border-radius: 10px !important;
        font-weight: 600 !important; }}
    /* quantity construction flow: raw gap -> MOQ -> pack -> final */
    .ipa-qflow {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px; }}
    .ipa-qstage {{ flex: 1 1 130px; min-width: 120px; background: {COLORS["card"]};
        border: 1px solid {COLORS["border"]}; border-radius: 12px; padding: 10px 12px; }}
    .ipa-qstage .lab {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.4px;
        color: {COLORS["subtext"]}; font-weight: 700; }}
    .ipa-qstage .val {{ font-size: 1.3rem; font-weight: 800; color: {COLORS["navy"]}; margin-top: 2px; }}
    .ipa-qstage .cap {{ font-size: 0.72rem; color: {COLORS["subtext"]}; margin-top: 3px; }}
    .ipa-qstage.final {{ border-color: {COLORS["teal"]}; background: #F3FBFB; }}
    /* approval panel — always "awaiting buyer review" in the pilot */
    .ipa-approval {{ display: flex; align-items: center; gap: 12px; border-radius: 14px;
        padding: 14px 18px; background: #FBF0DC; border: 1px solid #F0D9A6; }}
    .ipa-approval .ico {{ color: {COLORS["amber"]}; display: inline-flex; }}
    .ipa-approval .txt .h {{ font-weight: 800; color: {COLORS["navy"]}; }}
    .ipa-approval .txt .s {{ font-size: 0.82rem; color: {COLORS["subtext"]}; }}
    .ipa-recometrics {{ display: flex; flex-wrap: wrap; gap: 3px 16px; margin-top: 6px; }}
    .ipa-recometrics .m {{ font-size: 0.75rem; color: {COLORS["subtext"]}; }}
    .ipa-recometrics .m b {{ color: {COLORS["navy"]}; font-weight: 700; }}
</style>
"""
