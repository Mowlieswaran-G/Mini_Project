"""Safe, cached project-file loading helpers with hardware optimization."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]

# Ensure both project root and app dir are in sys.path
for p in (str(ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

DATA_FILE = ROOT / "cleaned_delhi_powerdemand_5min_2021_to_2024.csv"

def is_torch_installed() -> bool:
    """Checks whether PyTorch is installed in the active Python environment."""
    try:
        import torch
        return True
    except ImportError:
        return False

# Attempt to pre-register lstm_model in sys.modules if torch is available
if is_torch_installed():
    try:
        import lstm_model
        sys.modules["lstm_model"] = lstm_model
    except Exception:
        pass

def find_model(name: str) -> Path | None:
    """Locates model binary in models/ folder or project root."""
    filename = f"{name}_delhi_powerdemand.pkl"
    for candidate in (ROOT / "models" / filename, ROOT / filename):
        if candidate.exists():
            return candidate
    return None

def is_lstm_file_present() -> bool:
    """Returns True if the trained LSTM checkpoint / pickle artifact exists on disk."""
    return find_model("lstm") is not None

def is_model_available(name: str) -> bool:
    """Checks if a model binary is available and its runtime dependencies are satisfied."""
    exists = find_model(name) is not None
    if name == "lstm":
        return exists and is_torch_installed()
    return exists

@st.cache_resource(show_spinner="⚡ Power Predict AI: Loading Delhi Grid Telemetry...")
def load_history() -> pd.DataFrame:
    """Loads cleaned 5-minute historical telemetry with memory-efficient typing."""
    df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
    return df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def load_predictions(model_key: str) -> pd.DataFrame:
    """Loads hold-out test predictions for model comparison."""
    path = ROOT / f"{model_key}_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found at {path}")
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.sort_values("datetime")

@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    """Loads metrics or summary JSON safely."""
    with open(ROOT / path, encoding="utf-8") as file:
        return json.load(file)

@st.cache_resource(show_spinner="⚡ Power Predict AI: Initializing Model Engine...")
def load_model(model_key: str):
    """Loads pre-trained model binary with safe memory caching for Intel Core i3 / 8 GB RAM."""
    path = find_model(model_key)
    if path is None:
        raise FileNotFoundError(f"Saved {model_key.replace('_', ' ').title()} model file was not found.")
    
    if model_key == "lstm":
        if not is_torch_installed():
            raise RuntimeError(
                "PyTorch is not installed in the local Python environment. "
                "PyTorch is required to execute interactive inference with the LSTM model. "
                "Install it via: pip install torch (or run via Docker). "
                "Pre-computed test evaluation metrics for LSTM remain viewable across all comparison tabs."
            )
        if "lstm_model" not in sys.modules:
            try:
                import lstm_model
                sys.modules["lstm_model"] = lstm_model
            except Exception as e:
                raise RuntimeError(f"Could not initialize LSTM neural architecture: {e}")

    return joblib.load(path)

HISTORICAL_HIGHLIGHTS = [
    {
        "title": "🔥 Delhi All-Time Test Period Peak (8,631.53 MW)",
        "timestamp": "2024-06-19 15:05:00",
        "description": "Highest recorded dispatch in this dataset during the severe June 2024 heatwave (44.5°C).",
        "category": "Record Peak",
    },
    {
        "title": "🌙 Summer Midnight AC Surge (7,842.10 MW)",
        "timestamp": "2024-06-25 23:30:00",
        "description": "Intense late-night domestic air conditioning surge with outdoor temperature at 36.5°C.",
        "category": "Night Cooling",
    },
    {
        "title": "🌧️ Sultry Monsoon Afternoon (6,415.80 MW)",
        "timestamp": "2024-08-12 14:15:00",
        "description": "Heavy post-monsoon humidity (82% RH) sustaining elevated compressor loads.",
        "category": "Monsoon Peak",
    },
    {
        "title": "❄️ Winter Morning Heating Surge (4,912.40 MW)",
        "timestamp": "2024-01-15 08:35:00",
        "description": "Peak morning domestic heating and water geyser surge during winter cold spell (8.2°C).",
        "category": "Winter Heating",
    },
    {
        "title": "🍃 Pleasant Spring Afternoon (3,520.15 MW)",
        "timestamp": "2024-03-10 12:45:00",
        "description": "Mild, comfortable weather (24.2°C) requiring neither space cooling nor heating.",
        "category": "Baseload",
    },
]
