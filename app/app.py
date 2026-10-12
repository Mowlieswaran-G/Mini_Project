"""
AI-Based Electricity Demand Projection — Delhi Power System
NowEnergy Aesthetic Replica | Stage 11 Interactive Web Application
"""
from __future__ import annotations
import importlib, sys
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import utils.data, utils.features
importlib.reload(utils.data)
importlib.reload(utils.features)

from utils.data import (ROOT, find_model, is_model_available,
                        load_history, load_json, load_model, load_predictions)
try:
    from utils.data import HISTORICAL_HIGHLIGHTS
except ImportError:
    HISTORICAL_HIGHLIGHTS = [
        {"title": "🔥 Delhi All-Time Record Peak (8,631.53 MW)",
         "timestamp": "2024-06-19 15:05:00",
         "description": "All-time highest demand during the June 2024 heatwave (44.5 °C).",
         "category": "Record Peak"},
        {"title": "🌙 Summer Midnight AC Peak (7,842.10 MW)",
         "timestamp": "2024-06-25 23:30:00",
         "description": "Late-night domestic air-conditioning surge (36.5 °C).",
         "category": "Night Cooling"},
        {"title": "🌧️ Sultry Monsoon Afternoon (6,415.80 MW)",
         "timestamp": "2024-08-12 14:15:00",
         "description": "Post-monsoon 85% RH sustaining compressor loads.",
         "category": "Monsoon Peak"},
        {"title": "❄️ Winter Morning Peak (4,912.40 MW)",
         "timestamp": "2024-01-15 08:35:00",
         "description": "Heating surge during a dense winter cold spell (8.2 °C).",
         "category": "Winter Heating"},
        {"title": "🍃 Pleasant Spring Afternoon (3,520.15 MW)",
         "timestamp": "2024-03-10 12:45:00",
         "description": "Mild weather (24.2 °C) with minimal HVAC load.",
         "category": "Baseload"},
    ]

from utils.features import (FEATURE_COLUMNS, FEATURE_INFO, LAGS, SCENARIO_PRESETS,
                             WEATHER_COLUMNS, calculate_heat_index, classify_demand_regime,
                             estimate_lags, get_wind_direction_label,
                             make_custom_feature_row, make_feature_row)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="NowEnergy | Delhi Power AI",
                   page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

# ── NowEnergy CSS Theme ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

/* Sage-green outer canvas from the reference image */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #3b876a !important;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

/* Prevent Streamlit top bar from overlapping page content */
header[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 10 !important;
    height: 48px !important;
}

/* Floating White Canvas for Dashboard with clean margins and no clipping */
.block-container {
    background: #ffffff !important;
    border-radius: 24px !important;
    padding: 28px 34px 38px 34px !important;
    margin-top: 8px !important;
    margin-bottom: 24px !important;
    max-width: 98% !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.14) !important;
}

/* Floating White Sidebar Container */
[data-testid="stSidebar"] {
    background-color: #3b876a !important;
    border-right: none !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #ffffff !important;
    border-radius: 22px !important;
    margin: 8px 10px 24px 12px !important;
    padding: 16px 12px !important;
    box-shadow: 0 10px 32px rgba(0,0,0,0.12) !important;
}

/* Typography & Base Text */
.stMarkdown p, .stMarkdown li, .stMarkdown label,
.stText, [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {
    color: #1e293b !important;
    font-size: 0.92rem;
}
h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700;
}

/* Sidebar Navigation Items — Hide ugly default radio dots so buttons look clean */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"] {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 10px 14px !important;
    border-radius: 12px !important;
    margin-bottom: 3px !important;
    transition: all 0.18s ease !important;
    cursor: pointer !important;
    font-size: 0.90rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
    border: 1px solid transparent !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #eef9f2 !important;
    color: #15803d !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #32996e !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(50, 153, 110, 0.3) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * {
    color: #ffffff !important;
}

/* Top Dashboard Bar (Title + Breadcrumb + User Avatar) */
.ne-dash-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 16px;
    margin-bottom: 18px;
    border-bottom: 1.5px solid #f1f5f9;
}
.ne-dash-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.35rem;
    font-weight: 800;
    color: #0f172a;
}
.ne-dash-user {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    color: #334155;
    font-weight: 600;
}
.ne-user-dot {
    width: 22px;
    height: 22px;
    background: #3b82f6;
    color: #fff;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    font-weight: 800;
}

/* NowEnergy Top 5-KPI Cards Row (Responsive auto-fit grid) */
.ne-kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 22px;
}
.ne-kpi-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 8px 8px 12px 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.ne-kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(50, 153, 110, 0.1);
    border-color: #cbd5e1;
}
.ne-kpi-pill {
    background: #32996e;
    color: #ffffff !important;
    border-radius: 8px;
    padding: 6px 8px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.ne-kpi-pill * {
    color: #ffffff !important;
}
.ne-kpi-val {
    font-size: 1.65rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #0f172a;
    margin-top: 10px;
    line-height: 1.1;
}
.ne-kpi-val span {
    font-size: 0.78rem;
    font-weight: 600;
    color: #64748b;
}
.ne-kpi-sub {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 4px;
    font-weight: 500;
}

/* NowEnergy Cards (Section Boxes) */
.ne-box {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
}
.ne-box-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
}

/* Energy Mode Orange Gradient Card */
.ne-energy-mode-card {
    background: linear-gradient(135deg, #f97316 0%, #fb923c 100%);
    border-radius: 18px;
    padding: 22px 20px;
    color: #ffffff;
    box-shadow: 0 8px 24px rgba(249, 115, 22, 0.28);
    margin-bottom: 18px;
    text-align: center;
}
.ne-energy-mode-card * {
    color: #ffffff !important;
}
.ne-em-title {
    font-size: 1.25rem;
    font-weight: 800;
    margin: 0 0 12px 0;
}
.ne-em-icon-box {
    background: rgba(255, 255, 255, 0.25);
    border: 2px solid rgba(255, 255, 255, 0.45);
    border-radius: 14px;
    padding: 12px 24px;
    display: inline-block;
    font-size: 2.2rem;
    margin: 8px 0 14px 0;
}
.ne-em-desc {
    font-size: 0.82rem;
    line-height: 1.5;
    opacity: 0.95;
    margin-bottom: 16px;
}
.ne-em-pill-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ffffff;
    color: #ea580c !important;
    font-weight: 700;
    font-size: 0.82rem;
    padding: 7px 18px;
    border-radius: 9999px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Energy Storage / Stress Card */
.ne-storage-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    margin-bottom: 18px;
}
.ne-sc-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}
.ne-pill-legend {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 8px;
}
.ne-pl-item {
    font-size: 0.74rem;
    font-weight: 600;
    color: #475569;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 4px 10px;
    border-radius: 9999px;
    display: flex;
    align-items: center;
    gap: 5px;
}

/* Prediction Output Banner */
.ne-result-banner {
    background: linear-gradient(135deg, #134e4a 0%, #15803d 50%, #2ea879 100%) !important;
    border-radius: 18px !important;
    padding: 24px 28px !important;
    color: #ffffff !important;
    box-shadow: 0 8px 30px rgba(22, 163, 74, 0.22) !important;
    margin-bottom: 20px !important;
}
.ne-result-banner * {
    color: #ffffff !important;
}
.ne-rb-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #a7f3d0 !important;
    margin-bottom: 5px;
}
.ne-rb-val {
    font-size: 2.75rem;
    font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
}
.ne-rb-val span {
    font-size: 1.4rem;
    color: #86efac !important;
}
.ne-rb-sub {
    font-size: 0.92rem;
    color: #d1fae5 !important;
    margin-top: 4px;
}
.ne-rb-advisory {
    font-size: 0.90rem;
    color: #e0f2fe !important;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.22);
    line-height: 1.6;
}

/* Callout */
.cb {
    background: #f0fdf4 !important;
    border-left: 4px solid #32996e !important;
    border-radius: 0 10px 10px 0 !important;
    padding: 13px 17px !important;
    margin-bottom: 18px !important;
    color: #166534 !important;
    font-size: 0.90rem !important;
    line-height: 1.6 !important;
}
.cb b {
    color: #14532d !important;
}

/* Buttons */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #32996e, #22c55e) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.96rem !important;
    padding: 12px 26px !important;
    box-shadow: 0 4px 14px rgba(50, 153, 110, 0.3) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 22px rgba(50, 153, 110, 0.4) !important;
    background: linear-gradient(135deg, #28805c, #16a34a) !important;
}
[data-testid="stButton"] > button * {
    color: #ffffff !important;
    font-weight: 700 !important;
}
[data-testid="stDownloadButton"] > button {
    background: #ffffff !important;
    color: #32996e !important;
    border: 1.5px solid #32996e !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 10px 20px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #eef9f2 !important;
}
[data-testid="stDownloadButton"] > button * {
    color: #32996e !important;
}

/* Model Cards & Step Cards */
.mc {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    text-align: center !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03) !important;
    margin-bottom: 12px !important;
}
.mv {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    font-family: 'JetBrains Mono', monospace !important;
    display: block !important;
}
.ml {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    display: block !important;
    margin-top: 4px !important;
}
.mm {
    font-size: 0.74rem !important;
    color: #94a3b8 !important;
    display: block !important;
    margin-top: 2px !important;
}
.sc {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
.sc:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(50, 153, 110, 0.12) !important;
    border-color: #86efac !important;
}
.sc .si {
    font-size: 1.9rem !important;
    margin-bottom: 8px !important;
    display: block;
}
.sc h4 {
    font-size: 0.98rem !important;
    font-weight: 700 !important;
    color: #14532d !important;
    margin: 0 0 6px !important;
}
.sc p {
    font-size: 0.83rem !important;
    color: #475569 !important;
    line-height: 1.55 !important;
    margin: 0 !important;
}

/* Native Streamlit Elements */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricValue"] {
    color: #15803d !important;
    font-weight: 800 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.76rem !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600 !important;
    color: #64748b !important;
    font-size: 0.90rem !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #15803d !important;
    font-weight: 700 !important;
    border-bottom: 2.5px solid #32996e !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stTextInput"] input, 
[data-testid="stNumberInput"] input, 
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus, 
[data-testid="stNumberInput"] input:focus, 
[data-testid="stDateInput"] input:focus,
[data-testid="stTimeInput"] input:focus {
    border-color: #32996e !important;
    box-shadow: 0 0 0 2px rgba(50, 153, 110, 0.2) !important;
}
[data-testid="stSelectbox"] > div > div {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
.st2 {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #14532d !important;
    border-left: 4px solid #32996e !important;
    padding-left: 10px !important;
    margin: 20px 0 14px !important;
}
.fg {
    font-size: 0.90rem !important;
    font-weight: 700 !important;
    color: #15803d !important;
    margin-bottom: 8px !important;
    padding-bottom: 6px !important;
    border-bottom: 2px solid #bbf7d0 !important;
}
hr {
    border-color: #f1f5f9 !important;
    margin: 20px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Chart Themes & Helpers ───────────────────────────────────────────────────
PALETTE = ["#32996e", "#f97316", "#10b981", "#fb923c", "#059669", "#ea580c"]

def _wl(fig, title, h=390):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color="#14532d")),
        paper_bgcolor="#ffffff", plot_bgcolor="#fafcfb",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#0f172a"),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", title_font=dict(color="#475569")),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", title_font=dict(color="#475569")),
        height=h, margin=dict(l=14, r=14, t=44, b=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#0f172a"))
    )
    return fig

def line_chart(frame, title, y_cols, labels):
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        color = PALETTE[i % len(PALETTE)]
        fig.add_scatter(x=frame["datetime"], y=frame[col], mode="lines",
                        name=labels.get(col, col),
                        line=dict(width=2.5, color=color))
    return _wl(fig, title)

def nowenergy_combo_chart(frame, title="Energy Usage"):
    """Exact NowEnergy Energy Usage chart: Peach bars with August peak in orange + smooth emerald green spline line."""
    fig = go.Figure()
    # Peach bars
    fig.add_trace(go.Bar(
        x=frame["datetime"], y=frame["Power demand"],
        name="Energy (MW)",
        marker=dict(
            color="rgba(254, 215, 170, 0.7)",
            line=dict(color="#fdba74", width=1)
        )
    ))
    # Green spline for temperature overlay
    if "temp" in frame.columns:
        fig.add_trace(go.Scatter(
            x=frame["datetime"], y=frame["temp"],
            name="Temperature (°C)", yaxis="y2",
            mode="lines+markers",
            line=dict(color="#32996e", width=2.8, shape="spline"),
            marker=dict(size=4.5, color="#32996e", line=dict(color="#ffffff", width=1))
        ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color="#14532d")),
        paper_bgcolor="#ffffff", plot_bgcolor="#fafcfb",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#0f172a"),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        yaxis=dict(title="Energy (MW)", gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right", showgrid=False),
        height=350, margin=dict(l=14, r=14, t=44, b=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def nowenergy_donut_chart(main_val=236, green_val=90, title_total="Total: 336 MW"):
    """Exact NowEnergy Daily Energy Cost Donut: 72% orange + 28% green."""
    tot = main_val + green_val
    fig = go.Figure(go.Pie(
        values=[main_val, green_val],
        labels=["Main Power", "Green Energy"],
        hole=0.68,
        sort=False,
        direction="clockwise",
        marker=dict(colors=["#f97316", "#32996e"], line=dict(color="#ffffff", width=2.5)),
        textinfo="percent",
        textfont=dict(color="#ffffff", size=12, family="Plus Jakarta Sans"),
        hoverinfo="label+value+percent"
    ))
    fig.update_layout(
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{title_total}</b>",
                x=0.5, y=0.5, font=dict(size=14, color="#0f172a", family="Plus Jakarta Sans"),
                showarrow=False
            )
        ],
        height=220, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def nowenergy_capacity_gauge(mw):
    """Exact NowEnergy 75% Capacity circular donut gauge widget."""
    pct = min(100.0, max(0.0, (mw / 8631.53) * 100))
    fig = go.Figure(go.Pie(
        values=[pct, 100 - pct],
        hole=0.74,
        sort=False,
        direction="clockwise",
        marker=dict(colors=["#32996e", "#f1f5f9"], line=dict(color="#ffffff", width=2)),
        textinfo="none",
        hoverinfo="none"
    ))
    fig.update_layout(
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{pct:.0f}%</b><br><span style='font-size:0.68em;color:#64748b'>Capacity</span>",
                x=0.5, y=0.5, font=dict(size=20, color="#0f172a", family="JetBrains Mono"),
                showarrow=False
            )
        ],
        height=180, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def nowenergy_area_carbon(frame, title="Carbon Footprint CO2"):
    """Exact NowEnergy smooth emerald green area chart with data markers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame["datetime"], y=frame["Power demand"],
        mode="lines+markers",
        line=dict(color="#32996e", width=2.5, shape="spline"),
        marker=dict(size=5, color="#32996e", line=dict(color="#ffffff", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(50, 153, 110, 0.15)",
        name="Demand Trajectory"
    ))
    return _wl(fig, title, h=280)

# ── Data Loading ─────────────────────────────────────────────────────────────
try:
    history = load_history()
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}"); st.stop()

def latest_lag_ts(df):
    idx = pd.DatetimeIndex(df["datetime"])
    valid = pd.Series(True, index=idx)
    for d in LAGS.values():
        valid &= (idx - d).isin(idx)
    m = idx[valid.to_numpy()]
    return m[-1] if len(m) else df["datetime"].iloc[-1]

def predict_features(key, fdf):
    m = load_model(key)
    miss = [c for c in FEATURE_COLUMNS if c not in fdf.columns or fdf[c].isna().any()]
    if miss: raise ValueError(f"Missing: {', '.join(miss)}")
    return float(m.predict(fdf[FEATURE_COLUMNS])[0])

def predict_at(key, ts, dfh, rec=None):
    b = dfh.set_index("datetime")
    d = b["Power demand"].copy()
    if rec:
        d = pd.concat([d, pd.Series(rec)]).sort_index()
        d = d[~d.index.duplicated(keep="last")]
    w = b.loc[ts, WEATHER_COLUMNS] if ts in b.index else b.iloc[-1][WEATHER_COLUMNS]
    f_row = make_feature_row(ts, d, w)
    for col in ["demand_lag_5min", "demand_lag_1hour", "demand_lag_24hour", "demand_lag_7day"]:
        if pd.isna(f_row[col].iloc[0]):
            est = estimate_lags(ts.hour, ts.month, int(ts.dayofweek in (5, 6)))
            f_row[col] = est[col]
    return predict_features(key, f_row)

# ── Reusable NowEnergy Dashboard Bar ──────────────────────────────────────────
def render_dash_bar(subtitle="Delhi State Grid Command Center"):
    st.markdown(f"""
    <div class="ne-dash-bar">
      <div class="ne-dash-title">
        <span>📊</span> Dashboard
        <span style="font-size:0.75rem;font-weight:500;color:#64748b;margin-left:8px;">{subtitle}</span>
      </div>
      <div class="ne-dash-user">
        <span class="ne-user-dot">AH</span> Akhsan Habib &nbsp; <b>⋮</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:4px 6px 16px;">
      <div style="width:38px;height:38px;background:#32996e;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#ffffff;font-size:1.2rem;box-shadow:0 4px 10px rgba(50,153,110,0.3);">⚡</div>
      <div>
        <div style="font-size:1.15rem;font-weight:800;color:#0f172a;line-height:1.1;">NowEnergy</div>
        <div style="font-size:0.72rem;color:#64748b;font-weight:500;">Natural Power of Energy</div>
      </div>
    </div>
    <div style="font-size:0.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;padding:0 6px 8px;">menu</div>
    """, unsafe_allow_html=True)

    page = st.radio("menu", [
        "⚡ Demand Prediction & What-If",
        "🏠 Dashboard & System Overview",
        "📊 Historical Data Explorer",
        "🎯 Model Performance & Metrics",
        "🔮 Future Demand Forecasting",
        "🏔️ Peak Demand Analysis",
        "📖 About & Methodology",
    ], index=0, label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**🤖 AI Models Status**")
    lr_ok = is_model_available("linear_regression")
    rf_ok = is_model_available("random_forest")
    a, b2 = st.columns(2)
    a.markdown(f"Linear Reg.<br>{'✅ Ready' if lr_ok else '❌ Missing'}", unsafe_allow_html=True)
    b2.markdown(f"Random Forest<br>{'✅ Ready' if rf_ok else '❌ Missing'}", unsafe_allow_html=True)
    st.caption("Delhi State SLDC Grid • 2021–2024")

# ── Page 1: Demand Prediction & What-If ───────────────────────────────────────
def page_prediction(dfh):
    render_dash_bar("AI Electricity Demand Prediction & What-If Simulator")

    # 5 Top Cards directly matching NowEnergy
    st.markdown("""
    <div class="ne-kpi-row">
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">⚙️ System Accuracy</div>
        <div class="ne-kpi-val">99.90%</div>
        <div class="ne-kpi-sub">Test Set R² Score</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">⚡ 4-Yr Mean Demand</div>
        <div class="ne-kpi-val">4,414 <span>MW</span></div>
        <div class="ne-kpi-sub">Baseload: 1,480 MW</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">🔥 All-Time Peak</div>
        <div class="ne-kpi-val" style="color:#ea580c">8,632 <span>MW</span></div>
        <div class="ne-kpi-sub">19 June 2024 Record</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">🌿 Grid Efficiency</div>
        <div class="ne-kpi-val">25.24 <span>MW</span></div>
        <div class="ne-kpi-sub">Linear Reg. Test MAE</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">⏱️ Dispatch Slot</div>
        <div class="ne-kpi-val">5 <span>Min</span></div>
        <div class="ne-kpi-sub">Continuous Cadence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="cb">
      💡 <b>Interactive What-If Simulation:</b> Adjust atmospheric weather and momentum demand lags below to simulate real-time Delhi electricity dispatches.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        model_choice = st.selectbox("🤖 Primary Forecasting Model",
                                    ["Linear Regression", "Random Forest"])
        pkey = "linear_regression" if model_choice == "Linear Regression" else "random_forest"
    with c2:
        mode = st.radio("📥 Data Input Mode", [
            "🎛️ Interactive Scenario Builder (What-If Analysis)",
            "📂 Historical Record Inspection (Actual Delhi Data)"],
            horizontal=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    feats: dict[str, float] = {}
    actual: float | None = None

    if "Interactive" in mode:
        st.markdown('<div class="st2">Step 1 — Configure All 17 Input Parameters</div>', unsafe_allow_html=True)
        pc, lc = st.columns([3, 1])
        with pc:
            preset = st.selectbox(
                "⚡ Quick Scenario Preset",
                ["— Custom Manual Input —"] + list(SCENARIO_PRESETS.keys()))
        with lc:
            auto = st.checkbox("Auto-Estimate Lags", True,
                               help="Auto-compute 5-min/1-hr/24-hr/7-day lag values.")
        pd2 = SCENARIO_PRESETS.get(preset, {})
        if preset != "— Custom Manual Input —":
            st.info(f"**Scenario:** {pd2.get('desc','')}")

        ct, cw, cl = st.columns(3)
        with ct:
            st.markdown('<div class="fg">📅 1. Date & Time (7 features)</div>', unsafe_allow_html=True)
            idate = st.date_input("Date", date(pd2.get("year", 2024), pd2.get("month", 6), pd2.get("day", 19)))
            itime = st.time_input("Dispatch Time (5-min grid)", time(pd2.get("hour", 15), pd2.get("minute", 5)), step=300)
            fy, fm, fd = float(idate.year), float(idate.month), float(idate.day)
            fh, fmin = float(itime.hour), float(itime.minute - (itime.minute % 5))
            fdow = float(idate.weekday())
            fiw = float(1 if fdow in [5, 6] else 0)
            dnames = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            st.caption(f"**Day:** {dnames[int(fdow)]} | **Slot:** {int(fh):02d}:{int(fmin):02d} | **Weekend:** {bool(fiw)}")

        with cw:
            st.markdown('<div class="fg">🌡️ 2. Weather Conditions (6 features)</div>', unsafe_allow_html=True)
            ftemp = st.slider("Temperature (°C)", 0.0, 52.0, float(pd2.get("temp", 44.5)), 0.5)
            fdwpt = st.slider("Dew Point (°C)", -10.0, 35.0, float(pd2.get("dwpt", 19.5)), 0.5)
            frhum = st.slider("Relative Humidity (%)", 5.0, 100.0, float(pd2.get("rhum", 32.0)), 1.0)
            hi = calculate_heat_index(ftemp, frhum)
            fwspd = st.number_input("Wind Speed (km/h)", 0.0, 75.0, float(pd2.get("wspd", 14.0)), 1.0)
            fwdir = st.number_input("Wind Direction (°)", 0.0, 360.0, float(pd2.get("wdir", 290.0)), 5.0)
            fpres = st.number_input("Surface Pressure (hPa)", 970.0, 1035.0, float(pd2.get("pres", 998.0)), 0.5)
            st.caption(f"**Heat Index:** {hi}°C | **Wind Cardinal:** {get_wind_direction_label(fwdir)}")

        with cl:
            st.markdown('<div class="fg">⚡ 3. Historical Demand Lags (4 features)</div>', unsafe_allow_html=True)
            if auto:
                if preset != "— Custom Manual Input —":
                    est = {k: pd2[k] for k in ["demand_lag_5min", "demand_lag_1hour", "demand_lag_24hour", "demand_lag_7day"]}
                else:
                    est = estimate_lags(int(fh), int(fm), int(fiw))
                fl5, fl1, fl24, fl7 = (est["demand_lag_5min"], est["demand_lag_1hour"],
                                       est["demand_lag_24hour"], est["demand_lag_7day"])
                st.success("✅ Lags estimated automatically.")
                la, lb = st.columns(2)
                la.metric("5-Min Lag", f"{fl5:,.0f} MW")
                la.metric("24-Hr Lag", f"{fl24:,.0f} MW")
                lb.metric("1-Hour Lag", f"{fl1:,.0f} MW")
                lb.metric("7-Day Lag", f"{fl7:,.0f} MW")
            else:
                fl5 = st.number_input("Demand 5-Min Ago (MW)", 1000.0, 9500.0, float(pd2.get("demand_lag_5min", 8500.0)), 10.0)
                fl1 = st.number_input("Demand 1-Hr Ago (MW)", 1000.0, 9500.0, float(pd2.get("demand_lag_1hour", 8300.0)), 10.0)
                fl24 = st.number_input("Demand 24-Hr Ago (MW)", 1000.0, 9500.0, float(pd2.get("demand_lag_24hour", 8100.0)), 10.0)
                fl7 = st.number_input("Demand 7-Days Ago (MW)", 1000.0, 9500.0, float(pd2.get("demand_lag_7day", 7800.0)), 10.0)

        feats = {"year": fy, "month": fm, "day": fd, "hour": fh, "minute": fmin,
                 "day_of_week": fdow, "is_weekend": fiw,
                 "demand_lag_5min": fl5, "demand_lag_1hour": fl1,
                 "demand_lag_24hour": fl24, "demand_lag_7day": fl7,
                 "temp": ftemp, "dwpt": fdwpt, "rhum": frhum,
                 "wdir": fwdir, "wspd": fwspd, "pres": fpres}

    else:
        st.markdown('<div class="st2">Step 1 — Select a Historical Delhi Observation</div>', unsafe_allow_html=True)
        choice2 = st.selectbox(
            "Choose a curated event or a custom date:",
            ["— Choose from Highlights —"] +
            [h["title"] for h in HISTORICAL_HIGHLIGHTS] +
            ["📅 Custom Date & Time Picker"])
        if choice2 == "📅 Custom Date & Time Picker":
            dts = latest_lag_ts(dfh)
            da, db = st.columns(2)
            sd = da.date_input("Historical Date", value=dts.date(),
                               min_value=dfh["datetime"].min().date(),
                               max_value=dfh["datetime"].max().date())
            st2_time = db.time_input("Historical Time", value=time(dts.hour, dts.minute), step=300)
            sel_dt = pd.Timestamp(datetime.combine(sd, st2_time))
        elif choice2 != "— Choose from Highlights —":
            h2 = next(x for x in HISTORICAL_HIGHLIGHTS if x["title"] == choice2)
            sel_dt = pd.Timestamp(h2["timestamp"])
            st.info(f"**Event:** {h2['description']}")
        else:
            sel_dt = pd.Timestamp("2024-06-19 15:05:00")

        hidx = dfh.set_index("datetime")
        if sel_dt in hidx.index:
            row = hidx.loc[sel_dt]
            actual = float(row["Power demand"])
            fd2 = make_feature_row(sel_dt, hidx["Power demand"], row[WEATHER_COLUMNS])
            feats = {c: float(fd2[c].iloc[0]) for c in FEATURE_COLUMNS}
            st.success(f"✅ Loaded **{sel_dt:%d %b %Y, %H:%M}** — Actual: **{actual:,.2f} MW**")
        else:
            nr = dfh.iloc[(dfh["datetime"] - sel_dt).abs().argsort()[:1]]["datetime"].iloc[0]
            row = hidx.loc[nr]
            actual = float(row["Power demand"])
            fd2 = make_feature_row(nr, hidx["Power demand"], row[WEATHER_COLUMNS])
            feats = {c: float(fd2[c].iloc[0]) for c in FEATURE_COLUMNS}
            st.info(f"Loaded nearest: **{nr:%d %b %Y, %H:%M}** — Demand: **{actual:,.2f} MW**")

    # Step 2: Prediction Trigger
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="st2">Step 2 — Run AI Forecast</div>', unsafe_allow_html=True)
    st.button("🚀 Predict Electricity Demand Now", type="primary", use_container_width=True)

    fdf = make_custom_feature_row(feats)
    pl, pr = None, None
    try:
        if is_model_available("linear_regression"):
            pl = predict_features("linear_regression", fdf)
        if is_model_available("random_forest"):
            pr = predict_features("random_forest", fdf)
    except Exception as e:
        st.error(f"Inference error: {e}"); return

    pp = pl if pkey == "linear_regression" else (pr or pl)
    if pp is None:
        st.warning("No model .pkl file found."); return

    # Step 3: NowEnergy Output Layout
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="st2">Step 3 — Forecast Output & Grid Intelligence</div>', unsafe_allow_html=True)
    rg = classify_demand_regime(pp)
    dtr = 8631.53 - pp
    dir_word = "below" if dtr >= 0 else "above"
    dir_sym = "⬇" if dtr >= 0 else "⬆"

    st.markdown(f"""
    <div class="ne-result-banner">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;">
        <div>
          <div class="ne-rb-label">⚡ AI Forecast Output — {model_choice}</div>
          <div class="ne-rb-val">{pp:,.2f} <span>MW</span></div>
          <div class="ne-rb-sub">= <b>{pp/1000:.3f} GW</b> of active electrical demand</div>
        </div>
        <div style="text-align:right;">
          <span style="display:inline-block;padding:5px 16px;border-radius:9999px;font-size:0.86rem;font-weight:700;background:rgba(255,255,255,0.22);color:#fff;border:1.5px solid rgba(255,255,255,0.4)">{rg['badge']}</span>
          <div style="font-size:0.83rem;color:#d1fae5;margin-top:8px;">
            Grid Stress: <b>{rg['stress']}</b> • {rg['percentile']}
          </div>
          <div style="font-size:0.81rem;color:#a7f3d0;margin-top:4px;">
            {dir_sym} <b>{abs(dtr):,.1f} MW</b> {dir_word} all-time record
          </div>
        </div>
      </div>
      <div class="ne-rb-advisory">🛡️ <b>Operational Advisory:</b> {rg['advice']}</div>
    </div>
    """, unsafe_allow_html=True)

    # 2-column layout: Left (Charts & Dual Models) vs Right (Energy Mode & Storage Widgets)
    left_col, right_col = st.columns([1.75, 1.0])

    with left_col:
        st.markdown('<div class="st2" style="margin-top:0">🤖 Dual-Model Projection Comparison</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            lv = f"{pl:,.1f} MW" if pl is not None else "Unavailable"
            st.markdown(f'<div class="mc"><span class="mv" style="color:#32996e">{lv}</span>' +
                        '<span class="ml">Predicted Demand</span>' +
                        '<span class="mm">Linear Regression (Baseline)</span></div>', unsafe_allow_html=True)
        with m2:
            rv2 = f"{pr:,.1f} MW" if pr is not None else "Unavailable"
            rc = "#f97316" if pr is not None else "#9ca3af"
            st.markdown(f'<div class="mc"><span class="mv" style="color:{rc}">{rv2}</span>' +
                        '<span class="ml">Predicted Demand</span>' +
                        '<span class="mm">Random Forest (100 Trees)</span></div>', unsafe_allow_html=True)

        if pl is not None and pr is not None:
            diff = pr - pl; dpct = (diff / pl) * 100
            st.info(f"📊 **Model Divergence:** Random Forest projects **{diff:+,.1f} MW ({dpct:+.2f}%)** relative to Linear Regression.")

        if actual is not None:
            em = pp - actual; ep = (em / actual) * 100; ap = max(0.0, 100.0 - abs(ep))
            st.markdown('<div class="st2">🎯 Ground-Truth Validation</div>', unsafe_allow_html=True)
            v1, v2, v3 = st.columns(3)
            v1.metric("Actual Recorded", f"{actual:,.2f} MW")
            v2.metric("Prediction Error", f"{em:+,.2f} MW", delta=f"{ep:+.2f}%", delta_color="inverse")
            v3.metric("Model Accuracy", f"{ap:.2f}%")

    with right_col:
        # Energy Mode Card from Reference Image
        st.markdown(f"""
        <div class="ne-energy-mode-card">
          <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;opacity:0.9;">⚙️ Energy Mode</div>
          <div class="ne-em-title">Best Power Efficiency</div>
          <div class="ne-em-icon-box">⚡</div>
          <div class="ne-em-desc">This mode will help to extend the power storage efficiency and balance peak HVAC loads.</div>
          <div>
            <span class="ne-em-pill-btn">⚙️ Change Plan</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Energy Storage / Capacity Donut Card from Reference Image
        with st.container(border=True):
            st.markdown('<div class="ne-sc-title">🔋 Energy Storage</div>', unsafe_allow_html=True)
            st.plotly_chart(nowenergy_capacity_gauge(pp), use_container_width=True)
            st.markdown(f"""
            <div class="ne-pill-legend">
              <span class="ne-pl-item"><span style="color:#f97316">●</span> Main Power {min(100, int((pp/8631.53)*70))}%</span>
              <span class="ne-pl-item"><span style="color:#32996e">●</span> Green Energy 25%</span>
            </div>
            """, unsafe_allow_html=True)

    # Step 4: Verification Table
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="st2">Step 4 — Complete Input Verification Table (All 17 Features)</div>', unsafe_allow_html=True)
    rows = []
    for col in FEATURE_COLUMNS:
        info = FEATURE_INFO[col]; v = feats.get(col, float("nan"))
        dv = f"{int(v)}" if col in ["year", "month", "day", "hour", "minute", "day_of_week", "is_weekend"] else f"{v:,.2f}"
        rows.append({"Feature Key": col, "Human Name": info["label"],
                     "Category": info["category"], "Value": dv,
                     "Unit": info["unit"], "Grid Significance": info["desc"]})
    dft = pd.DataFrame(rows)
    tw, tl, tt, ta = st.tabs(["🌡️ Weather (6)", "⚡ Demand Lags (4)", "📅 Temporal (7)", "📋 All 17 Features"])
    cs = ["Human Name", "Value", "Unit", "Grid Significance"]
    with tw: st.dataframe(dft[dft["Category"] == "Atmospheric Weather"][cs], use_container_width=True, hide_index=True)
    with tl: st.dataframe(dft[dft["Category"] == "Grid Demand Momentum (Lags)"][cs], use_container_width=True, hide_index=True)
    with tt: st.dataframe(dft[dft["Category"] == "Temporal / Calendar"][cs], use_container_width=True, hide_index=True)
    with ta: st.dataframe(dft, use_container_width=True, hide_index=True)

# ── Page 2: Dashboard & System Overview ───────────────────────────────────────
def page_home(dfh):
    render_dash_bar("Natural Power of Energy • Delhi SLDC 5-Min Telemetry")

    # The 5 Top Cards exactly like the image
    st.markdown("""
    <div class="ne-kpi-row">
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">⚙️ Overall System Efficiency</div>
        <div class="ne-kpi-val">70%</div>
        <div class="ne-kpi-sub">Optimal Dispatch</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">⚡ Renewable Energy Utilization</div>
        <div class="ne-kpi-val">70%</div>
        <div class="ne-kpi-sub">Clean Baseload</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">🌿 Carbon Emission Reduction</div>
        <div class="ne-kpi-val">40%</div>
        <div class="ne-kpi-sub">Target Benchmark</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">💰 Energy Cost Savings</div>
        <div class="ne-kpi-val">150</div>
        <div class="ne-kpi-sub">metric tons CO2/year</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">🌐 Overall Carbon Footprint</div>
        <div class="ne-kpi-val">150</div>
        <div class="ne-kpi-sub">metric tons CO2/year</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 3-Column Center Layout matching the NowEnergy mockup
    col_left, col_mid, col_right = st.columns([1.6, 1.1, 1.0])

    with col_left:
        with st.container(border=True):
            st.markdown('<div class="ne-box-title">📈 Energy Usage</div>', unsafe_allow_html=True)
            recent_data = dfh.tail(288) # Last 24 hours of 5-min intervals
            st.plotly_chart(nowenergy_combo_chart(recent_data, "Demand (MW) & Ambient Temperature (°C)"), use_container_width=True)
            st.markdown("""
            <div style="font-size:0.75rem;color:#64748b;background:#f8fafc;padding:8px 12px;border-radius:10px;margin-top:6px;line-height:1.5;">
              <b>Energy consumption:</b> 4,414 MW avg &nbsp;|&nbsp; <b>House area:</b> Delhi Capital Region &nbsp;|&nbsp; <b>Average temperature:</b> 32.5°C
            </div>
            """, unsafe_allow_html=True)

    with col_mid:
        with st.container(border=True):
            st.markdown("""
            <div class="ne-box-title">📅 Daily Energy Cost</div>
            <div style="display:flex;justify-content:center;gap:10px;margin-bottom:8px;">
              <span class="ne-pl-item"><span style="color:#f97316">●</span> Main Power 236 MW</span>
              <span class="ne-pl-item"><span style="color:#32996e">●</span> Green energy 90 MW</span>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(nowenergy_donut_chart(236, 90, "Total: 326 MW"), use_container_width=True)

    with col_right:
        # Energy Mode Orange Card
        st.markdown("""
        <div class="ne-energy-mode-card">
          <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;opacity:0.9;">⚙️ Energy Mode</div>
          <div class="ne-em-title">Best Power Efficiency</div>
          <div class="ne-em-icon-box">⚡</div>
          <div class="ne-em-desc">This mode will help to extend the power storage efficiency.</div>
          <div>
            <span class="ne-em-pill-btn">⚙️ Change Plan</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Energy Storage Card
        with st.container(border=True):
            st.markdown('<div class="ne-sc-title">🔋 Energy Storage</div>', unsafe_allow_html=True)
            st.plotly_chart(nowenergy_capacity_gauge(6500), use_container_width=True)
            st.markdown("""
            <div class="ne-pill-legend">
              <span class="ne-pl-item"><span style="color:#f97316">●</span> Main Power 50%</span>
              <span class="ne-pl-item"><span style="color:#32996e">●</span> Green Energy 25%</span>
            </div>
            """, unsafe_allow_html=True)

    # Bottom Full-Width Card: Carbon Footprint CO2
    with st.container(border=True):
        st.markdown('<div class="ne-box-title">🌿 Carbon Footprint CO2 & Demand Trajectory</div>', unsafe_allow_html=True)
        st.plotly_chart(nowenergy_area_carbon(dfh.iloc[::288], "Delhi Power Demand & Emission Baseline (2021–2024)"), use_container_width=True)

    # 4 Pipeline Cards
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="st2">🛠️ Delhi AI Forecasting Pipeline</div>', unsafe_allow_html=True)
    steps = [("🧹", "1. Data Cleaning", "Merged 5-minute telemetry with atmospheric weather; irregular gaps cleaned."),
             ("⚙️", "2. Causal Lags", "17 strictly causal features: 7 temporal, 4 exact timestamp lags, 6 weather."),
             ("🤖", "3. Dual AI Models", "Linear Regression + 100-tree Random Forest on chronological 80/20 split."),
             ("🎯", "4. Peak Projection", "Regime classification, heatwave stress simulation, and recursive future forecasts.")]
    cc = st.columns(4)
    for col, (ico, ttl, dsc) in zip(cc, steps):
        with col:
            st.markdown(f'<div class="sc"><span class="si">{ico}</span><h4>{ttl}</h4><p>{dsc}</p></div>', unsafe_allow_html=True)

# ── Page 3: Historical Data Explorer ──────────────────────────────────────────
def page_explorer(dfh):
    render_dash_bar("Historical Load Profiles, Diurnal Curves & Atmospheric Dependencies")
    mn, mx = dfh.datetime.min().date(), dfh.datetime.max().date()
    dates = st.date_input("📅 Date Range Selection", value=(date(2024, 6, 1), date(2024, 6, 30)), min_value=mn, max_value=mx)
    if len(dates) != 2: st.info("Please select both start and end dates."); return
    sel = dfh[dfh.datetime.dt.date.between(*dates)]
    if sel.empty: st.warning("No records found in this range."); return

    st.markdown(f"""
    <div class="ne-kpi-row">
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">📊 Records</div>
        <div class="ne-kpi-val">{len(sel):,}</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">⚡ Mean Demand</div>
        <div class="ne-kpi-val">{sel['Power demand'].mean():,.1f} <span>MW</span></div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">🔥 Peak Demand</div>
        <div class="ne-kpi-val" style="color:#ea580c">{sel['Power demand'].max():,.1f} <span>MW</span></div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">🌡️ Mean Temp</div>
        <div class="ne-kpi-val">{sel['temp'].mean():.1f} <span>°C</span></div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">☀️ Peak Temp</div>
        <div class="ne-kpi-val" style="color:#ea580c">{sel['temp'].max():.1f} <span>°C</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(nowenergy_combo_chart(sel, f"Demand & Temperature Profile — {dates[0]:%d %b %Y} to {dates[1]:%d %b %Y}"), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    fh2 = px.bar(sel.groupby(sel.datetime.dt.hour)["Power demand"].mean().reset_index(name="D"),
                 x="datetime", y="D", title="<b>Average Hourly Load</b>",
                 labels={"datetime": "Hour of Day", "D": "Demand (MW)"},
                 color="D", color_continuous_scale="Greens")
    _wl(fh2, "Average Hourly Load", 320); c1.plotly_chart(fh2, use_container_width=True)

    fw2 = px.bar(sel.groupby(sel.datetime.dt.dayofweek)["Power demand"].mean().reset_index(name="D"),
                 x="datetime", y="D", title="<b>Day of Week Profile</b>",
                 labels={"datetime": "Day (0=Mon)", "D": "Demand (MW)"},
                 color="D", color_continuous_scale="Oranges")
    _wl(fw2, "Day of Week Profile", 320); c2.plotly_chart(fw2, use_container_width=True)

    fm2 = px.bar(dfh.groupby(dfh.datetime.dt.month)["Power demand"].mean().reset_index(name="D"),
                 x="datetime", y="D", title="<b>Monthly Seasonality (All Years)</b>",
                 labels={"datetime": "Month", "D": "Demand (MW)"},
                 color="D", color_continuous_scale="Teal")
    _wl(fm2, "Monthly Seasonality", 320); c3.plotly_chart(fm2, use_container_width=True)

    st.markdown('<div class="st2">🌡️ Temperature vs Demand Scatter Profile</div>', unsafe_allow_html=True)
    smp = sel.sample(min(len(sel), 4000), random_state=42)
    fs = px.scatter(smp, x="temp", y="Power demand", color="rhum",
                    title="<b>Temperature (°C) vs Power Demand (MW) with Humidity</b>",
                    labels={"temp": "Temperature (°C)", "Power demand": "Demand (MW)", "rhum": "Humidity (%)"},
                    opacity=0.6, color_continuous_scale="YlOrRd")
    _wl(fs, "Temperature vs Demand", 380); st.plotly_chart(fs, use_container_width=True)

    st.markdown('<div class="st2">🏆 Top 10 Peak Records in Selected Range</div>', unsafe_allow_html=True)
    pk = sel.nlargest(10, "Power demand")[["datetime", "Power demand", "temp", "rhum"]].copy()
    pk["Date"] = pk.datetime.dt.strftime("%d %b %Y"); pk["Time"] = pk.datetime.dt.strftime("%H:%M")
    pk = pk.rename(columns={"Power demand": "Demand (MW)", "temp": "Temp (°C)", "rhum": "Humidity (%)"})
    st.dataframe(pk[["Date", "Time", "Demand (MW)", "Temp (°C)", "Humidity (%)"]], use_container_width=True, hide_index=True)

# ── Page 4: Model Performance ─────────────────────────────────────────────────
def page_performance():
    render_dash_bar("Hold-Out Test Benchmark (70,825 Intervals • March–Dec 2024)")
    lm2 = load_json("outputs/linear_regression/linear_regression_metrics.json")["test_metrics"]
    rm2 = load_json("outputs/random_forest/random_forest_metrics.json")["test_metrics"]
    comp = load_json("outputs/model_comparison/model_comparison_summary.json")

    def gv(d, *keys):
        for k in keys:
            if k in d: return d[k]
        return "—"

    st.markdown(f"""
    <div class="ne-kpi-row">
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">⚡ LR MAE</div>
        <div class="ne-kpi-val">{gv(lm2,'MAE'):.2f} <span>MW</span></div>
        <div class="ne-kpi-sub">Linear Baseline</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">⚡ LR RMSE</div>
        <div class="ne-kpi-val">{gv(lm2,'RMSE'):.2f} <span>MW</span></div>
        <div class="ne-kpi-sub">Linear Baseline</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">⚡ LR R² Score</div>
        <div class="ne-kpi-val">{gv(lm2,'R2','R²'):.4f}</div>
        <div class="ne-kpi-sub">99.90% Fit</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">🌲 RF MAE</div>
        <div class="ne-kpi-val" style="color:#ea580c">{gv(rm2,'MAE'):.2f} <span>MW</span></div>
        <div class="ne-kpi-sub">100 Trees Ensemble</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">🌲 RF R² Score</div>
        <div class="ne-kpi-val" style="color:#ea580c">{gv(rm2,'R2','R²'):.4f}</div>
        <div class="ne-kpi-sub">99.83% Fit</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    pk2 = comp.get("peak_comparison", {})
    if pk2:
        st.markdown('<div class="st2">🔥 Performance at All-Time Historical Record Peak (2024-06-19 15:05)</div>', unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Actual Peak", f"{pk2.get('actual_demand_mw', 0.0):,.2f} MW")
        p2.metric("LR Predicted", f"{pk2.get('linear_regression_pred_mw', 0.0):,.2f} MW",
                  delta=f"{pk2.get('linear_regression_error_pct', 0.0):+.2f}%", delta_color="inverse")
        p3.metric("RF Predicted", f"{pk2.get('random_forest_pred_mw', 0.0):,.2f} MW",
                  delta=f"{pk2.get('random_forest_error_pct', 0.0):+.2f}%", delta_color="inverse")
        p4.metric("Peak Timestamp", pk2.get("timestamp", "—"))

    rd = comp.get("regime_analysis", {})
    if rd:
        st.markdown('<div class="st2">⚡ Performance by Operational Demand Regime</div>', unsafe_allow_html=True)
        rr = []
        for rk, ri in rd.items():
            obs = ri.get("observation_count", 0)
            pct = ri.get("percentage_of_test_set", "?")
            lrmae = ri.get("linear_regression_mae_mw", 0)
            rfmae = ri.get("random_forest_mae_mw", 0)
            rr.append({"Regime": rk.replace("_", " ").title(),
                       "Definition": ri.get("definition", ""),
                       "Observations": f"{obs:,} ({pct}%)",
                       "LR MAE": f"{lrmae:.2f} MW",
                       "RF MAE": f"{rfmae:.2f} MW"})
        st.dataframe(pd.DataFrame(rr), hide_index=True, use_container_width=True)

    ms = st.selectbox("Inspect prediction residuals for:", ["linear_regression", "random_forest"])
    pr2 = load_predictions(ms); pr2["residual"] = pr2.actual_demand - pr2.predicted_demand
    st.plotly_chart(line_chart(pr2.iloc[1000:2500],
                               f"Actual vs Predicted — {ms.replace('_',' ').title()} (Zoomed Section)",
                               ["actual_demand", "predicted_demand"],
                               {"actual_demand": "Actual (MW)", "predicted_demand": "Predicted (MW)"}),
                    use_container_width=True)
    fres = px.histogram(pr2, x="residual", nbins=100,
                        title=f"<b>Residual Error Distribution — {ms.replace('_',' ').title()}</b>",
                        labels={"residual": "Error (Actual − Predicted MW)"},
                        color_discrete_sequence=["#32996e"])
    _wl(fres, f"Residuals — {ms.replace('_',' ').title()}"); st.plotly_chart(fres, use_container_width=True)

# ── Page 5: Future Demand Forecasting ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_fast_forecast(key: str, steps2: int) -> pd.DataFrame:
    """High-performance recursive multi-step forecasting engine.
    Uses O(1) buffer indexing and vectorized evaluation to propagate
    forward demand projections in milliseconds instead of minutes.
    """
    dfh = load_history()
    iv2 = dfh.datetime.diff().dropna().mode().iloc[0]
    st3 = dfh.datetime.max() + iv2
    dates = pd.date_range(st3, periods=steps2, freq=iv2)

    # 2016 steps covers 7-day lag (2016 * 5 min = 7 days)
    recent_demands = list(dfh["Power demand"].tail(2016).to_numpy(dtype=float))
    last_w = dfh.iloc[-1][WEATHER_COLUMNS].to_numpy(dtype=float)

    model = load_model(key)

    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        # Instant O(1) dot product for Linear Regression (~0.08s for 288 steps)
        coef = np.asarray(model.coef_, dtype=np.float64)
        intercept = float(model.intercept_)
        preds = []
        f = np.empty(17, dtype=np.float64)
        f[11:17] = last_w
        for ts in dates:
            f[0] = ts.year
            f[1] = ts.month
            f[2] = ts.day
            f[3] = ts.hour
            f[4] = ts.minute
            f[5] = ts.dayofweek
            f[6] = 1.0 if ts.dayofweek in (5, 6) else 0.0
            f[7] = recent_demands[-1]
            f[8] = recent_demands[-12]
            f[9] = recent_demands[-288]
            f[10] = recent_demands[-2016]
            pred = float(np.dot(f, coef) + intercept)
            preds.append(pred)
            recent_demands.append(pred)
    elif hasattr(model, "estimators_"):
        # Direct Cython tree-level evaluation for Random Forest (bypasses DataFrame & threadpool spinup)
        trees = [e.tree_ for e in model.estimators_]
        n_trees = len(trees)
        preds = []
        f_buf = np.empty((1, 17), dtype=np.float32)
        f_buf[0, 11:17] = last_w
        for ts in dates:
            f_buf[0, 0] = ts.year
            f_buf[0, 1] = ts.month
            f_buf[0, 2] = ts.day
            f_buf[0, 3] = ts.hour
            f_buf[0, 4] = ts.minute
            f_buf[0, 5] = ts.dayofweek
            f_buf[0, 6] = 1.0 if ts.dayofweek in (5, 6) else 0.0
            f_buf[0, 7] = recent_demands[-1]
            f_buf[0, 8] = recent_demands[-12]
            f_buf[0, 9] = recent_demands[-288]
            f_buf[0, 10] = recent_demands[-2016]
            pred = float(sum(t.predict(f_buf)[0, 0] for t in trees) / n_trees)
            preds.append(pred)
            recent_demands.append(pred)
    else:
        # Fallback to model.predict
        preds = []
        f_buf = np.empty((1, 17), dtype=np.float64)
        f_buf[0, 11:17] = last_w
        for ts in dates:
            f_buf[0, 0] = ts.year
            f_buf[0, 1] = ts.month
            f_buf[0, 2] = ts.day
            f_buf[0, 3] = ts.hour
            f_buf[0, 4] = ts.minute
            f_buf[0, 5] = ts.dayofweek
            f_buf[0, 6] = 1.0 if ts.dayofweek in (5, 6) else 0.0
            f_buf[0, 7] = recent_demands[-1]
            f_buf[0, 8] = recent_demands[-12]
            f_buf[0, 9] = recent_demands[-288]
            f_buf[0, 10] = recent_demands[-2016]
            pred = float(model.predict(f_buf)[0])
            preds.append(pred)
            recent_demands.append(pred)

    return pd.DataFrame({"datetime": dates, "predicted_demand": preds})

def run_forecast(key, dfh, steps2):
    return run_fast_forecast(key, steps2)

def page_forecast(dfh):
    render_dash_bar("Recursive Multi-Step Forward Forecasting Engine")
    
    col_engine, col_hz = st.columns(2)
    with col_engine:
        ch = st.selectbox("Forecasting Engine", ["Linear Regression (Recommended - Ultra Fast)", "Random Forest"])
        k2 = "linear_regression" if "Linear" in ch else "random_forest"
    with col_hz:
        iv3 = dfh.datetime.diff().dropna().mode().iloc[0]
        hz = {"Next 24 Hours (288 steps)": int(pd.Timedelta(days=1) / iv3),
              "Next 7 Days (2,016 steps)": int(pd.Timedelta(days=7) / iv3),
              "Next 30 Days (8,640 steps)": int(pd.Timedelta(days=30) / iv3)}
        hl = st.selectbox("Forecast Horizon", list(hz.keys()))

    if "linear_regression" in k2:
        st.caption("⚡ **Ultra-Fast Mode:** Linear Regression evaluates in **<0.1s** with 99.90% accuracy.")
    else:
        st.caption("🌲 **Ensemble Mode:** Propagates 100 Decision Trees across recursive lag feedback loops.")

    steps = hz[hl]
    btn = st.button("🚀 Generate Recursive Forecast", type="primary")

    if btn:
        from time import perf_counter
        t_start = perf_counter()
        with st.spinner("Propagating recursive forward steps…"):
            fc = run_forecast(k2, dfh, steps)
        dur = perf_counter() - t_start
        st.session_state["last_forecast"] = {"data": fc, "key": k2, "horizon": hl, "duration": dur}

    if "last_forecast" in st.session_state:
        saved = st.session_state["last_forecast"]
        fc = saved["data"]
        hl_label = saved["horizon"]
        dur = saved.get("duration", 0.0)

        st.success(f"⚡ **Forecast generated in {dur:.3f} seconds** ({len(fc):,} steps propagated).")

        ctx = dfh.tail(min(len(dfh), int(pd.Timedelta(days=7) / iv3)))
        fg = line_chart(ctx, f"Historical Baseline & {hl_label.strip()} Forecast",
                        ["Power demand"], {"Power demand": "Historical Actual (MW)"})
        fg.add_scatter(x=fc.datetime, y=fc.predicted_demand, mode="lines",
                       name="AI Forecast (MW)", line=dict(color="#f97316", width=2.5))
        st.plotly_chart(fg, use_container_width=True)

        st.markdown(f"""
        <div class="ne-kpi-row">
          <div class="ne-kpi-card">
            <div class="ne-kpi-pill">⚡ Forecast Mean</div>
            <div class="ne-kpi-val">{fc.predicted_demand.mean():,.1f} <span>MW</span></div>
          </div>
          <div class="ne-kpi-card">
            <div class="ne-kpi-pill">🔥 Forecast Peak</div>
            <div class="ne-kpi-val" style="color:#ea580c">{fc.predicted_demand.max():,.1f} <span>MW</span></div>
          </div>
          <div class="ne-kpi-card">
            <div class="ne-kpi-pill">🌿 Forecast Min</div>
            <div class="ne-kpi-val">{fc.predicted_demand.min():,.1f} <span>MW</span></div>
          </div>
          <div class="ne-kpi-card">
            <div class="ne-kpi-pill">⏱️ Total Steps</div>
            <div class="ne-kpi-val">{len(fc):,}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.download_button("📥 Download Forecast CSV",
                           fc.to_csv(index=False).encode(),
                           "delhi_demand_forecast.csv", "text/csv")
        st.session_state["forecast"] = fc

# ── Page 6: Peak Demand Analysis ──────────────────────────────────────────────
def page_peaks(dfh):
    render_dash_bar("Historical Highs, Heatwaves & Substation Capacity Margins")
    pr3 = dfh.loc[dfh["Power demand"].idxmax()]

    st.markdown(f"""
    <div class="ne-kpi-row">
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">🔥 All-Time Record</div>
        <div class="ne-kpi-val" style="color:#ea580c">{pr3['Power demand']:,.2f} <span>MW</span></div>
        <div class="ne-kpi-sub">Delhi State Peak</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">📅 Peak Date</div>
        <div class="ne-kpi-val" style="font-size:1.3rem">{pr3.datetime.strftime('%d %b %Y')}</div>
        <div class="ne-kpi-sub">Wednesday</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">⏱️ Peak Time</div>
        <div class="ne-kpi-val">{pr3.datetime.strftime('%H:%M')}</div>
        <div class="ne-kpi-sub">15:05 Afternoon</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">🌡️ Temperature</div>
        <div class="ne-kpi-val" style="color:#ea580c">{pr3['temp']:.1f} <span>°C</span></div>
        <div class="ne-kpi-sub">Heatwave Extreme</div>
      </div>
      <div class="ne-kpi-card">
        <div class="ne-kpi-pill">💧 Humidity</div>
        <div class="ne-kpi-val">{pr3['rhum']:.0f} <span>%</span></div>
        <div class="ne-kpi-sub">Dry Heat Spell</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    ph = ROOT / "outputs" / "peak_analysis" / "peak_hour_analysis.csv"
    if ph.exists():
        st.markdown('<div class="st2">🕒 Peak Demand Distribution by Hour of Day</div>', unsafe_allow_html=True)
        pkdf = pd.read_csv(ph)
        peak_col = "actual_peak_count" if "actual_peak_count" in pkdf.columns else "overall_count"
        fph = px.bar(pkdf, x="hour", y=peak_col,
                     title="<b>Daily Peak Frequency by Hour of Day</b>",
                     labels={"hour": "Hour (0–23)", peak_col: "Peak Occurrences"},
                     color=peak_col, color_continuous_scale="YlOrRd")
        _wl(fph, "Daily Peak Frequency by Hour", 380); st.plotly_chart(fph, use_container_width=True)

    st.markdown('<div class="st2">🏆 Top 20 All-Time Peak Dispatch Intervals</div>', unsafe_allow_html=True)
    t20 = dfh.nlargest(20, "Power demand")[["datetime", "Power demand", "temp", "rhum", "wspd"]].copy()
    t20["Date"] = t20.datetime.dt.strftime("%d %b %Y"); t20["Time"] = t20.datetime.dt.strftime("%H:%M")
    t20 = t20.rename(columns={"Power demand": "Demand (MW)", "temp": "Temp (°C)",
                             "rhum": "Humidity (%)", "wspd": "Wind (km/h)"})
    st.dataframe(t20[["Date", "Time", "Demand (MW)", "Temp (°C)", "Humidity (%)", "Wind (km/h)"]],
                 use_container_width=True, hide_index=True)

# ── Page 7: About & Methodology ───────────────────────────────────────────────
def page_about():
    render_dash_bar("System Architecture, Causal Modeling & Zero Data Leakage Standards")
    st.markdown('<div class="st2">📐 Complete 17 Input Feature Catalog</div>', unsafe_allow_html=True)
    rows2 = []
    for f in FEATURE_COLUMNS:
        i = FEATURE_INFO[f]
        rows2.append({"Feature": f, "Name": i["label"], "Category": i["category"],
                      "Unit": i["unit"], "Operational Range": f"{i['min']} – {i['max']}",
                      "Grid Significance": i["desc"]})
    st.dataframe(pd.DataFrame(rows2), hide_index=True, use_container_width=True)

    st.markdown('<div class="st2">🛡️ Strict Causal Engineering — Zero Data Leakage</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cb">
      All 4 demand lags utilize <b>exact timestamp lookup</b> (<code>t−5m</code>, <code>t−1h</code>, <code>t−24h</code>, <code>t−7d</code>)
      rather than naive positional shifts. The Delhi grid telemetry records ~1,150 irregular gap intervals; positional shifting would silently inject
      temporal distortion and compromise prediction accuracy. Missing slots remain strictly NaN during training to guarantee genuine real-world causality.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="st2">🤖 Model Architecture Comparison</div>', unsafe_allow_html=True)
    ma, mb = st.columns(2)
    with ma:
        st.markdown("""
        <div class="sc">
          <span class="si">📐</span><h4>Linear Regression (Baseline)</h4>
          <p>• Ordinary Least Squares formulation<br>
          • Test MAE: <b>25.24 MW</b> | R²: <b>0.9990</b><br>
          • Unmatched linearity at extreme record peaks during high heatwave temperatures.</p>
        </div>""", unsafe_allow_html=True)
    with mb:
        st.markdown("""
        <div class="sc">
          <span class="si">🌲</span><h4>Random Forest (100 Trees)</h4>
          <p>• 100 Estimators with max depth 20<br>
          • Test MAE: <b>28.17 MW</b> | R²: <b>0.9983</b><br>
          • Captures non-linear meteorological and humidity interactions across normal load tiers.</p>
        </div>""", unsafe_allow_html=True)

# ── Router ────────────────────────────────────────────────────────────────────
if page.startswith("⚡"):
    page_prediction(history)
elif page.startswith("🏠"):
    page_home(history)
elif page.startswith("📊"):
    page_explorer(history)
elif page.startswith("🎯"):
    page_performance()
elif page.startswith("🔮"):
    page_forecast(history)
elif page.startswith("🏔️"):
    page_peaks(history)
else:
    page_about()
