"""Shared presentation helpers for the NEBULA X Streamlit app.

Pure UI/rendering only -- no model loading, feature extraction, prediction
or validation logic lives here. `app/door_view.py` and `app/rail_view.py`
own all of that (via `src/door/*` and `src/rail_corrugation/*`); this module
only draws things, using whatever numbers/DataFrames those pages already
computed. No function here invents a metric.

Every function degrades gracefully: if `styles.css` is missing, or an older
Streamlit lacks a widget this module prefers, the app still renders (with a
plainer look) rather than crashing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

try:
    import streamlit as st
except ImportError:  # pragma: no cover - lets non-UI code (e.g. tests) import this module
    class _MissingStreamlit:
        def __getattr__(self, name):
            raise RuntimeError("streamlit is required to use app.ui.")

    st = _MissingStreamlit()

import plotly.express as px
import plotly.graph_objects as go

APP_DIR = Path(__file__).resolve().parent
STYLES_PATH = APP_DIR / "styles.css"

PRODUCT_NAME = "RailClarity"
PRODUCT_ACCENT = "Clarity"  # the part of PRODUCT_NAME rendered in the accent colour
PRODUCT_DESCRIPTOR = "Explainable Train Condition Monitoring"
TAGLINE = "Every train is already telling us what it needs."
COMPETITION_BADGE = "Built for NEBULA X · PS3 Predictive Fault Detection"

_VALID_TONES = {"neutral", "good", "info", "warning", "critical", "rail"}

# One shared colour system every chart in the app draws from. Status colours
# (green/amber/red) are semantic; class colours are fixed per label so
# "Normal" never looks like a warning and Side I / Side II stay visually
# distinct from each other and from Door's Abnormal resistance.
CLASS_COLORS = {
    "Normal": "#2FB344",              # green -- healthy
    "Side I": "#9254DE",              # purple -- Rail accent
    "Side II": "#168FE5",             # cyan/blue -- informational
    "Abnormal resistance": "#F59E0B",  # amber -- Door fault
}
TONE_COLORS = {
    "neutral": "#707583",
    "good": "#2FB344",
    "info": "#168FE5",
    "warning": "#F59E0B",
    "critical": "#EF4444",
    "rail": "#9254DE",
}


def _tone(tone: str) -> str:
    return tone if tone in _VALID_TONES else "neutral"


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------


def inject_css(path: Path | None = None) -> None:
    """Load styles.css once. Uses `st.html` when available (Streamlit's
    documented, sandboxed way to inject a <style> block since ~1.41);
    falls back to `st.markdown(..., unsafe_allow_html=True)` on older
    installs that don't have `st.html`. Never raises -- a missing/broken
    stylesheet should never take down the app, only make it plainer.
    """
    css_path = path or STYLES_PATH
    try:
        css_text = css_path.read_text(encoding="utf-8")
    except OSError:
        return
    style_block = f"<style>{css_text}</style>"
    if hasattr(st, "html"):
        st.html(style_block)
    else:  # pragma: no cover - only exercised on Streamlit < 1.41
        st.markdown(style_block, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Layout primitives
# ---------------------------------------------------------------------------


def card():
    """A bordered container styled as a white card by styles.css. Usage:

        with ui.card():
            st.write("...")
    """
    return st.container(border=True)


def render_header(status_text: str = "Decision-support prototype") -> None:
    """Compact top header: RailClarity wordmark, descriptor, tagline, and a
    status pill -- no long paragraphs. NEBULA X appears only as a small
    competition-reference badge, never as the main product title.
    """
    wordmark = PRODUCT_NAME
    if PRODUCT_ACCENT and PRODUCT_NAME.endswith(PRODUCT_ACCENT):
        base = PRODUCT_NAME[: -len(PRODUCT_ACCENT)]
        wordmark = f"{base}<span class='nebula-header__accent'>{PRODUCT_ACCENT}</span>"

    left, right = st.columns([3, 1])
    with left:
        st.markdown(f"<div class='nebula-header'><span class='nebula-header__name'>{wordmark}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<p class='nebula-header__descriptor'>{PRODUCT_DESCRIPTOR}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='nebula-tagline'>{TAGLINE}</p>", unsafe_allow_html=True)
    with right:
        st.markdown(
            f"<div class='nebula-header__status'>{_pill_html(status_text, 'neutral')}"
            f"{_pill_html(COMPETITION_BADGE, 'neutral')}</div>",
            unsafe_allow_html=True,
        )


def subsystem_selector(options: list[str], *, key: str = "nebula_subsystem") -> str:
    """A stable-across-reruns subsystem picker.

    Uses `st.segmented_control` when available (Streamlit >= 1.36); falls
    back to a horizontal `st.radio` otherwise. Pre-seeding
    `st.session_state[key]` before creating the widget (the documented
    Streamlit pattern for a persistent default) keeps the selection stable
    across reruns triggered by uploads/downloads elsewhere on the page.
    """
    if key not in st.session_state:
        st.session_state[key] = options[0]

    if hasattr(st, "segmented_control"):
        selected = st.segmented_control("Subsystem", options, key=key, label_visibility="collapsed")
        return selected if selected is not None else st.session_state[key]
    return st.radio("Subsystem", options, key=key, horizontal=True, label_visibility="collapsed")


def render_task_line(text: str) -> None:
    """A single-line task description -- replaces a paragraph."""
    st.markdown(f"<p class='nebula-task-line'>{text}</p>", unsafe_allow_html=True)


def render_section_heading(title: str, help_text: str | None = None, *, level: str = "card") -> None:
    """A compact heading. Long explanations belong in `help_text` (rendered
    as a native tooltip), not as visible body text.

    `level="card"` (default): a card/chart title -- used for the vast
    majority of headings, which sit inside a `ui.card()`.
    `level="section"`: a larger main-dashboard section heading -- reserved
    for the handful of top-level headings that sit above/between cards
    rather than inside one (e.g. "Downloads").
    """
    css_class = "nebula-section-title" if level == "card" else "nebula-section-title--section"
    if help_text:
        st.markdown(f"<span class='{css_class}'>{title}</span>", unsafe_allow_html=True, help=help_text)
    else:
        st.markdown(f"<span class='{css_class}'>{title}</span>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI cards / status pills / banners
# ---------------------------------------------------------------------------


def render_kpi_row(items: Iterable[dict]) -> None:
    """Render a row of compact KPI cards.

    `items`: iterable of dicts with:
      - "label": short header text
      - "value": the headline number/string (already computed by the caller)
      - "sub": optional small caption (e.g. a percentage) -- also computed
        by the caller from real data, never invented here
      - "tone": optional colour category (see TONE_COLORS)
      - "help": optional tooltip text

    Uses `st.columns`, which Streamlit already stacks vertically on narrow
    viewports -- no custom flex/grid CSS needed for cards to wrap.
    """
    items = list(items)
    if not items:
        return
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        accent = TONE_COLORS.get(item.get("tone", "neutral"), TONE_COLORS["neutral"])
        sub_html = f"<div class='nebula-kpi__sub'>{item['sub']}</div>" if item.get("sub") else ""
        with column:
            st.markdown(
                f"<div class='nebula-kpi' style='border-top-color:{accent}' title=\"{item.get('help', '')}\">"
                f"<div class='nebula-kpi__label'>{item['label']}</div>"
                f"<div class='nebula-kpi__value'>{item['value']}</div>"
                f"{sub_html}"
                "</div>",
                unsafe_allow_html=True,
            )


def _pill_html(text: str, tone: str) -> str:
    return f"<span class='nebula-pill nebula-pill--{_tone(tone)}'>{text}</span>"


def render_status_pill(text: str, tone: str = "neutral") -> None:
    """A small rounded status badge. `tone` is a visual category only --
    it must never be read by a caller as an operational/safety threshold.
    """
    st.markdown(_pill_html(text, tone), unsafe_allow_html=True)


def render_legend(items: Iterable[tuple[str, str]]) -> None:
    """A small inline row of status pills used as a colour-key legend."""
    pills = "".join(_pill_html(text, tone) for text, tone in items)
    st.markdown(f"<div class='nebula-legend'>{pills}</div>", unsafe_allow_html=True)


def render_info_banner(text: str, tone: str = "info") -> None:
    st.markdown(f"<div class='nebula-banner nebula-banner--{_tone(tone)}'>{text}</div>", unsafe_allow_html=True)


def render_empty_state(message: str) -> None:
    st.markdown(f"<div class='nebula-empty'>{message}</div>", unsafe_allow_html=True)


def render_footer(disclaimer: str) -> None:
    st.markdown(f"<div class='nebula-footer'>{disclaimer}</div>", unsafe_allow_html=True)


def safe_status_update(status, **kwargs) -> None:
    """`st.status(...)`'s context value is a real StatusContainer during an
    actual Streamlit script run, but can be `None` when code runs outside a
    live ScriptRunContext (e.g. a direct-call smoke test). Guarding this in
    one place keeps every real caller identical while never crashing either way.
    """
    if status is not None:
        status.update(**kwargs)


def render_probability_bars(items: Iterable[tuple[str, float]]) -> None:
    """Compact horizontal probability bars, e.g. [("Normal", 0.92), ...].
    Native `st.progress`, so it's already a horizontal bar -- no chart
    library needed for this one.
    """
    for label, value in items:
        st.progress(min(max(float(value), 0.0), 1.0), text=f"{label}: {value:.1%}")


# ---------------------------------------------------------------------------
# Charts (Plotly) -- presentation only, every value comes from the caller
# ---------------------------------------------------------------------------

_CHART_CONFIG = {"displayModeBar": False}


def _base_layout(fig, *, height: int, showlegend: bool = False) -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#171923", size=12),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#E6E8EF")
    fig.update_yaxes(showgrid=True, gridcolor="#F0F1F6", zeroline=False)


def render_class_bar_chart(labels: list[str], values: list[int], *, height: int = 240) -> None:
    """Vertical bar chart, coloured by the shared CLASS_COLORS map."""
    colors = [CLASS_COLORS.get(label, "#707583") for label in labels]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition="outside"))
    _base_layout(fig, height=height)
    st.plotly_chart(fig, use_container_width=True, config=_CHART_CONFIG)


def render_donut_chart(labels: list[str], values: list[int], *, colors: list[str] | None = None, height: int = 240) -> None:
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.58,
            marker=dict(colors=colors) if colors else None,
            textinfo="percent",
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )
    _base_layout(fig, height=height, showlegend=True)
    st.plotly_chart(fig, use_container_width=True, config=_CHART_CONFIG)


def render_horizontal_bar(
    labels: list[str],
    values: list[float],
    *,
    hover_text: list[str] | None = None,
    value_format: str = "+.3f",
    height: int = 320,
) -> None:
    """Horizontal bar chart for feature contributions/importances.

    Colour encodes sign (positive vs negative) when values straddle zero
    (feature *contributions*); otherwise a single accent colour is used
    (plain, non-negative feature *importances*).
    """
    has_negative = any(v < 0 for v in values)
    if has_negative:
        colors = ["#2FB344" if v >= 0 else "#EF4444" for v in values]
    else:
        colors = ["#9254DE"] * len(values)
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:{value_format}}" for v in values],
            textposition="outside",
            customdata=hover_text or labels,
            hovertemplate="%{customdata}: %{x}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    _base_layout(fig, height=height)
    st.plotly_chart(fig, use_container_width=True, config=_CHART_CONFIG)


def render_timeline(
    frame,
    *,
    start_col: str,
    end_col: str,
    color_col: str,
    hover_cols: list[str],
    height: int = 220,
) -> None:
    """A Gantt-style chronological timeline -- one bar per detected cycle,
    coloured by its predicted class, with the full hover detail supplied
    by the caller (cycle number, start/end, prediction, confidence).
    """
    fig = px.timeline(
        frame,
        x_start=start_col,
        x_end=end_col,
        y=[""] * len(frame),
        color=color_col,
        color_discrete_map=CLASS_COLORS,
        hover_data=hover_cols,
    )
    fig.update_yaxes(visible=False)
    _base_layout(fig, height=height, showlegend=True)
    st.plotly_chart(fig, use_container_width=True, config=_CHART_CONFIG)


def render_line_chart(
    x,
    y,
    *,
    color: str = "#168FE5",
    xlabel: str | None = None,
    ylabel: str | None = None,
    height: int = 260,
) -> None:
    fig = go.Figure(go.Scatter(x=list(x), y=list(y), mode="lines", line=dict(color=color, width=1.2)))
    if xlabel:
        fig.update_xaxes(title=xlabel)
    if ylabel:
        fig.update_yaxes(title=ylabel)
    _base_layout(fig, height=height)
    st.plotly_chart(fig, use_container_width=True, config=_CHART_CONFIG)


__all__ = [
    "PRODUCT_NAME",
    "PRODUCT_DESCRIPTOR",
    "TAGLINE",
    "COMPETITION_BADGE",
    "CLASS_COLORS",
    "TONE_COLORS",
    "inject_css",
    "card",
    "render_header",
    "subsystem_selector",
    "render_task_line",
    "render_section_heading",
    "render_kpi_row",
    "render_status_pill",
    "render_legend",
    "render_info_banner",
    "render_empty_state",
    "render_footer",
    "render_probability_bars",
    "render_class_bar_chart",
    "render_donut_chart",
    "render_horizontal_bar",
    "render_timeline",
    "render_line_chart",
]
