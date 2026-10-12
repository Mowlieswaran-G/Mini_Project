"""Safe, cached project-file loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "cleaned_delhi_powerdemand_5min_2021_to_2024.csv"


def find_model(name: str) -> Path | None:
    """Support the models/ folder and the project root layout."""
    filename = f"{name}_delhi_powerdemand.pkl"
    for candidate in (ROOT / "models" / filename, ROOT / filename):
        if candidate.exists():
            return candidate
    return None


def is_model_available(name: str) -> bool:
    """Checks if a model binary is available locally."""
    return find_model(name) is not None


@st.cache_data(show_spinner="Loading cleaned demand data...")
def load_history() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
    return df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_predictions(model_key: str) -> pd.DataFrame:
    path = ROOT / f"{model_key}_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found at {path}")
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.sort_values("datetime")


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(ROOT / path, encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource(show_spinner="Loading saved model...")
def load_model(model_key: str):
    path = find_model(model_key)
    if path is None:
        raise FileNotFoundError(f"Saved {model_key.replace('_', ' ').title()} model file was not found.")
    return joblib.load(path)


HISTORICAL_HIGHLIGHTS = [
    {
        "title": "🔥 Delhi All-Time Record Peak (8,631.53 MW)",
        "timestamp": "2024-06-19 15:05:00",
        "description": "Historical maximum power demand in Delhi during the historic June 2024 heatwave (44.5°C).",
        "category": "Record Peak",
    },
    {
        "title": "🌙 Summer Midnight AC Peak (7,842.10 MW)",
        "timestamp": "2024-06-25 23:30:00",
        "description": "Intense late-night domestic air conditioning surge with temperature at 36.5°C.",
        "category": "Night Cooling",
    },
    {
        "title": "🌧️ Sultry Monsoon Afternoon (6,415.80 MW)",
        "timestamp": "2024-08-12 14:15:00",
        "description": "Heavy post-monsoon humidity (85% RH) causing sustained compressor loads.",
        "category": "Monsoon Peak",
    },
    {
        "title": "❄️ Winter Morning Peak (4,912.40 MW)",
        "timestamp": "2024-01-15 08:35:00",
        "description": "Peak morning domestic heating and water geyser surge during dense winter cold spell (8.2°C).",
        "category": "Winter Heating",
    },
    {
        "title": "🍃 Pleasant Spring Afternoon (3,520.15 MW)",
        "timestamp": "2024-03-10 12:45:00",
        "description": "Mild, comfortable weather (24.2°C) requiring neither space cooling nor heating.",
        "category": "Baseload",
    },
]
