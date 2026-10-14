"""Feature helpers, physical boundaries, and metadata preserving the project's causal feature logic."""

from __future__ import annotations

import urllib.request
import json
import pandas as pd

FEATURE_COLUMNS = [
    "year", "month", "day", "hour", "minute", "day_of_week", "is_weekend",
    "demand_lag_5min", "demand_lag_1hour", "demand_lag_24hour", "demand_lag_7day",
    "temp", "dwpt", "rhum", "wdir", "wspd", "pres",
]

LAGS = {
    "demand_lag_5min": pd.Timedelta(minutes=5),
    "demand_lag_1hour": pd.Timedelta(hours=1),
    "demand_lag_24hour": pd.Timedelta(hours=24),
    "demand_lag_7day": pd.Timedelta(days=7),
}

WEATHER_COLUMNS = ["temp", "dwpt", "rhum", "wdir", "wspd", "pres"]

# Physical and operational bounds established from 4 years of historical Delhi SLDC telemetry
PHYSICAL_BOUNDS = {
    "temp": {"min": -5.0, "max": 55.0, "hist_min": 4.0, "hist_max": 46.4, "unit": "°C", "label": "Ambient Temperature"},
    "dwpt": {"min": -20.0, "max": 35.0, "hist_min": -8.6, "hist_max": 30.3, "unit": "°C", "label": "Dew Point"},
    "rhum": {"min": 0.0, "max": 100.0, "hist_min": 5.0, "hist_max": 100.0, "unit": "%", "label": "Relative Humidity"},
    "wdir": {"min": 0.0, "max": 360.0, "hist_min": 0.0, "hist_max": 360.0, "unit": "°", "label": "Wind Direction"},
    "wspd": {"min": 0.0, "max": 100.0, "hist_min": 0.0, "hist_max": 63.0, "unit": "km/h", "label": "Wind Speed"},
    "pres": {"min": 950.0, "max": 1050.0, "hist_min": 989.6, "hist_max": 1027.0, "unit": "hPa", "label": "Atmospheric Pressure"},
    "demand": {"min": 1000.0, "max": 9500.0, "hist_min": 1302.08, "hist_max": 8631.53, "unit": "MW", "label": "Power Demand"},
}

FEATURE_INFO = {
    "year": {
        "label": "Year",
        "category": "Temporal / Calendar",
        "unit": "YYYY",
        "min": 2021, "max": 2030, "default": 2024, "step": 1,
        "desc": "Calendar year reflecting long-term urbanization and structural load growth in Delhi."
    },
    "month": {
        "label": "Month",
        "category": "Temporal / Calendar",
        "unit": "1-12",
        "min": 1, "max": 12, "default": 6, "step": 1,
        "desc": "Month of the year capturing strong seasonal weather cycles (Summer, Monsoon, Winter)."
    },
    "day": {
        "label": "Day of Month",
        "category": "Temporal / Calendar",
        "unit": "1-31",
        "min": 1, "max": 31, "default": 19, "step": 1,
        "desc": "Day of the month."
    },
    "hour": {
        "label": "Hour of Day",
        "category": "Temporal / Calendar",
        "unit": "0-23 hrs",
        "min": 0, "max": 23, "default": 15, "step": 1,
        "desc": "Diurnal hour capturing human activity cycles (afternoon peak ~15:00, night AC peak ~23:00)."
    },
    "minute": {
        "label": "Minute",
        "category": "Temporal / Calendar",
        "unit": "0-55 mins",
        "min": 0, "max": 55, "default": 5, "step": 5,
        "desc": "5-minute dispatch interval of the power grid."
    },
    "day_of_week": {
        "label": "Day of Week",
        "category": "Temporal / Calendar",
        "unit": "0=Mon, 6=Sun",
        "min": 0, "max": 6, "default": 2, "step": 1,
        "desc": "Day of the week (commercial and industrial loads drop slightly on weekends)."
    },
    "is_weekend": {
        "label": "Weekend Indicator",
        "category": "Temporal / Calendar",
        "unit": "0 or 1",
        "min": 0, "max": 1, "default": 0, "step": 1,
        "desc": "Binary flag (1 for Saturday/Sunday, 0 for Weekday)."
    },
    "demand_lag_5min": {
        "label": "Demand 5-Min Ago",
        "category": "Grid Demand Momentum (Lags)",
        "unit": "MW",
        "min": 1000.0, "max": 9500.0, "default": 8500.0, "step": 10.0,
        "desc": "Immediate short-term inertia of the power system (t − 5 min)."
    },
    "demand_lag_1hour": {
        "label": "Demand 1-Hour Ago",
        "category": "Grid Demand Momentum (Lags)",
        "unit": "MW",
        "min": 1000.0, "max": 9500.0, "default": 8200.0, "step": 10.0,
        "desc": "Near-term hourly demand level capturing intra-day ramp rate (t − 1 hr)."
    },
    "demand_lag_24hour": {
        "label": "Demand 24-Hours Ago",
        "category": "Grid Demand Momentum (Lags)",
        "unit": "MW",
        "min": 1000.0, "max": 9500.0, "default": 8100.0, "step": 10.0,
        "desc": "Same-time yesterday demand capturing daily 24-hour diurnal load rhythm (t − 24 hr)."
    },
    "demand_lag_7day": {
        "label": "Demand 7-Days Ago",
        "category": "Grid Demand Momentum (Lags)",
        "unit": "MW",
        "min": 1000.0, "max": 9500.0, "default": 7800.0, "step": 10.0,
        "desc": "Same-time last week demand capturing weekly cyclic patterns (t − 7 days)."
    },
    "temp": {
        "label": "Ambient Temperature",
        "category": "Atmospheric Weather",
        "unit": "°C",
        "min": 0.0, "max": 52.0, "default": 44.5, "step": 0.5,
        "desc": "Outdoor dry-bulb temperature; primary driver of summer AC cooling and winter space heating."
    },
    "dwpt": {
        "label": "Dew Point Temperature",
        "category": "Atmospheric Weather",
        "unit": "°C",
        "min": -10.0, "max": 35.0, "default": 20.0, "step": 0.5,
        "desc": "Atmospheric moisture level; combined with temperature determines apparent heat index."
    },
    "rhum": {
        "label": "Relative Humidity",
        "category": "Atmospheric Weather",
        "unit": "%",
        "min": 5.0, "max": 100.0, "default": 35.0, "step": 1.0,
        "desc": "Relative air humidity; high humidity during summer/monsoon severely impedes natural cooling."
    },
    "wdir": {
        "label": "Wind Direction",
        "category": "Atmospheric Weather",
        "unit": "degrees (°)",
        "min": 0.0, "max": 360.0, "default": 290.0, "step": 5.0,
        "desc": "Wind direction (North=0°, East=90°, South=180°, West=270°)."
    },
    "wspd": {
        "label": "Wind Speed",
        "category": "Atmospheric Weather",
        "unit": "km/h",
        "min": 0.0, "max": 75.0, "default": 14.0, "step": 0.5,
        "desc": "Wind velocity influencing surface heat dissipation and building thermal absorption."
    },
    "pres": {
        "label": "Surface Air Pressure",
        "category": "Atmospheric Weather",
        "unit": "hPa",
        "min": 970.0, "max": 1035.0, "default": 998.0, "step": 0.5,
        "desc": "Barometric pressure; lower in extreme hot pre-monsoon lows, higher in cold winter highs."
    },
}

SCENARIO_PRESETS = {
    "☀️ High Summer Afternoon Peak (Severe Heatwave)": {
        "desc": "Simulates Delhi's high summer afternoon peak load during intense 44.5°C heat.",
        "year": 2024, "month": 6, "day": 19, "hour": 15, "minute": 5, "day_of_week": 2, "is_weekend": 0,
        "temp": 44.5, "dwpt": 19.5, "rhum": 32.0, "wdir": 290.0, "wspd": 15.0, "pres": 997.5,
        "demand_lag_5min": 8580.0, "demand_lag_1hour": 8350.0, "demand_lag_24hour": 8100.0, "demand_lag_7day": 7820.0,
    },
    "🌙 Summer Night AC Surge (Late Evening Cooling)": {
        "desc": "Night-time residential air conditioning surge during a warm summer night (25 June, 23:00).",
        "year": 2024, "month": 6, "day": 25, "hour": 23, "minute": 0, "day_of_week": 1, "is_weekend": 0,
        "temp": 36.5, "dwpt": 24.0, "rhum": 52.0, "wdir": 270.0, "wspd": 8.0, "pres": 1000.5,
        "demand_lag_5min": 7450.0, "demand_lag_1hour": 7520.0, "demand_lag_24hour": 7200.0, "demand_lag_7day": 6950.0,
    },
    "🌧️ Monsoon Humid Afternoon (Sultry Weather)": {
        "desc": "High humidity post-rain condition where AC compressor duty cycles remain elevated (12 August, 14:00).",
        "year": 2024, "month": 8, "day": 12, "hour": 14, "minute": 0, "day_of_week": 0, "is_weekend": 0,
        "temp": 33.5, "dwpt": 27.5, "rhum": 82.0, "wdir": 120.0, "wspd": 10.0, "pres": 1002.0,
        "demand_lag_5min": 6120.0, "demand_lag_1hour": 5980.0, "demand_lag_24hour": 5850.0, "demand_lag_7day": 5700.0,
    },
    "❄️ Winter Morning Heating Surge (Cold & Fog)": {
        "desc": "Cold winter morning in January with domestic water geysers and space heaters active (15 January, 08:30).",
        "year": 2024, "month": 1, "day": 15, "hour": 8, "minute": 30, "day_of_week": 0, "is_weekend": 0,
        "temp": 8.5, "dwpt": 7.0, "rhum": 92.0, "wdir": 310.0, "wspd": 5.0, "pres": 1019.5,
        "demand_lag_5min": 4820.0, "demand_lag_1hour": 4450.0, "demand_lag_24hour": 4600.0, "demand_lag_7day": 4500.0,
    },
    "🍃 Pleasant Spring Afternoon (Baseload Conditions)": {
        "desc": "Temperate weather in early spring with minimal HVAC and cooling requirements (10 March, 12:30).",
        "year": 2024, "month": 3, "day": 10, "hour": 12, "minute": 30, "day_of_week": 6, "is_weekend": 1,
        "temp": 24.5, "dwpt": 12.0, "rhum": 45.0, "wdir": 290.0, "wspd": 12.0, "pres": 1013.0,
        "demand_lag_5min": 3480.0, "demand_lag_1hour": 3420.0, "demand_lag_24hour": 3550.0, "demand_lag_7day": 3490.0,
    },
}

def validate_weather_inputs(weather_dict: dict) -> tuple[bool, list[str]]:
    """Validates user-provided weather inputs against physical and historical boundaries.
    
    Returns (is_valid, warnings_list). Never replaces missing inputs silently.
    """
    warnings = []
    is_valid = True

    for var in WEATHER_COLUMNS:
        if var not in weather_dict or weather_dict[var] is None or pd.isna(weather_dict[var]):
            warnings.append(f"Missing required weather feature: '{var}' ({PHYSICAL_BOUNDS[var]['label']}).")
            is_valid = False
            continue

        try:
            val = float(weather_dict[var])
        except (ValueError, TypeError):
            warnings.append(f"Invalid non-numeric value for '{var}': {weather_dict[var]}")
            is_valid = False
            continue

        bounds = PHYSICAL_BOUNDS[var]
        if val < bounds["min"] or val > bounds["max"]:
            warnings.append(
                f"Physical Error: '{bounds['label']}' = {val} {bounds['unit']} is outside absolute physical limits "
                f"[{bounds['min']} to {bounds['max']} {bounds['unit']}]."
            )
            is_valid = False
        elif val < bounds["hist_min"] or val > bounds["hist_max"]:
            warnings.append(
                f"Operational Warning: '{bounds['label']}' = {val} {bounds['unit']} exceeds historical Delhi observed range "
                f"[{bounds['hist_min']} to {bounds['hist_max']} {bounds['unit']}]. Model predictions may have higher uncertainty."
            )

    return is_valid, warnings

def fetch_optional_delhi_weather(timeout_secs: float = 3.0) -> dict | None:
    """Safely queries free Open-Meteo meteorological API for New Delhi (28.6139° N, 77.2090° E).
    
    Completely optional with fail-safe error handling. Returns None on network failure or timeout.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=28.6139&longitude=77.2090"
        "&current=temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,wind_speed_10m,wind_direction_10m"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PowerPredictAI/2.0"})
        with urllib.request.urlopen(req, timeout=timeout_secs) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            current = data.get("current", {})
            return {
                "temp": float(current.get("temperature_2m", 30.0)),
                "rhum": float(current.get("relative_humidity_2m", 50.0)),
                "dwpt": float(current.get("dew_point_2m", 18.0)),
                "pres": float(current.get("surface_pressure", 1000.0)),
                "wspd": float(current.get("wind_speed_10m", 10.0)),
                "wdir": float(current.get("wind_direction_10m", 270.0)),
                "source": "Open-Meteo Weather API (Live Observation)",
                "timestamp": str(current.get("time", "Current"))
            }
    except Exception as e:
        # Fails safely without disrupting the application
        return None

def get_wind_direction_label(degrees: float) -> str:
    """Converts wind degrees to cardinal direction name."""
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((degrees + 11.25) / 22.5) % 16
    return f"{dirs[idx]} ({degrees:.0f}°)"

def calculate_heat_index(temp_c: float, rhum: float) -> float:
    """Computes Rothfusz heat index approximation in Celsius."""
    t_f = temp_c * 9.0 / 5.0 + 32.0
    r = rhum
    if t_f < 80.0:
        hi_f = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (r * 0.094))
    else:
        hi_f = (-42.379 + 2.04901523 * t_f + 10.14333127 * r - 0.22475541 * t_f * r
                - 0.00683783 * t_f**2 - 0.05481717 * r**2 + 0.00122874 * t_f**2 * r
                + 0.00085282 * t_f * r**2 - 0.00000199 * t_f**2 * r**2)
    return round((hi_f - 32.0) * 5.0 / 9.0, 1)

def classify_demand_regime(demand_mw: float, alert_threshold_mw: float | None = None) -> dict:
    """Classifies electricity demand into Delhi system operational tiers with optional user-configured threshold."""
    if demand_mw <= 5589.38:
        info = {
            "tier": "Normal Demand (Baseload to P75)",
            "badge": "🟢 Normal Grid Load",
            "color": "#10b981",
            "stress": "Low",
            "percentile": "≤ 75th Percentile (Historical)",
            "advice": "Standard baseload generation (Thermal & Renewables) suffices. Grid stability optimal.",
            "alert": False
        }
    elif demand_mw <= 6905.92:
        info = {
            "tier": "High Demand (75th to 95th Percentile)",
            "badge": "🟡 High Demand Zone",
            "color": "#f59e0b",
            "stress": "Moderate",
            "percentile": "75th – 95th Percentile (Historical)",
            "advice": "Substantial HVAC load detected. Ensure intermediate thermal plants and bilateral purchases are active.",
            "alert": False
        }
    elif demand_mw <= 7654.84:
        info = {
            "tier": "Extreme Peak Demand (95th to 99th Percentile)",
            "badge": "🟠 Extreme Peak Load",
            "color": "#f97316",
            "stress": "High",
            "percentile": "95th – 99th Percentile (Historical)",
            "advice": "Significant grid stress. Activate gas-based peaking units, maintain spinning reserves, and monitor substation transformers.",
            "alert": False
        }
    else:
        info = {
            "tier": "Top 1% Super Extreme Peak (> 99th Percentile)",
            "badge": "🔴 Critical Peak Zone",
            "color": "#ef4444",
            "stress": "Critical",
            "percentile": "> 99th Percentile (Historical Test Peak Territory)",
            "advice": "Critical system load approaching Delhi's highest observed historical demand (8,631.53 MW). Implement peak demand-response protocols and full reserve mobilization.",
            "alert": True
        }

    # User-configured operational alert check (clearly designated as user-defined)
    if alert_threshold_mw is not None and demand_mw >= alert_threshold_mw:
        info["user_threshold_exceeded"] = True
        info["threshold_warning"] = (
            f"User Alert: Demand ({demand_mw:,.1f} MW) exceeds user-configured threshold ({alert_threshold_mw:,.1f} MW). "
            f"(Note: This is a user-configured simulation threshold, not an official SLDC emergency limit)."
        )
    else:
        info["user_threshold_exceeded"] = False

    return info

def estimate_lags(hour: int, month: int, is_weekend: int, target_base_mw: float | None = None) -> dict[str, float]:
    """Estimates realistic lag values based on seasonal and diurnal patterns in Delhi."""
    if target_base_mw is not None:
        base = target_base_mw
    else:
        is_summer = month in [5, 6, 7]
        is_winter = month in [12, 1]
        if is_summer:
            base = 6500.0 if (12 <= hour <= 17 or hour >= 22) else 5200.0
        elif is_winter:
            base = 4500.0 if (7 <= hour <= 11 or 18 <= hour <= 21) else 3200.0
        else:
            base = 3800.0
        if is_weekend:
            base *= 0.92

    return {
        "demand_lag_5min": round(base * 0.998, 2),
        "demand_lag_1hour": round(base * 0.975, 2),
        "demand_lag_24hour": round(base * 0.960, 2),
        "demand_lag_7day": round(base * 0.950, 2),
    }

def make_custom_feature_row(feature_dict: dict) -> pd.DataFrame:
    """Builds a verified DataFrame with all 17 features ready for model inference."""
    row = {col: float(feature_dict[col]) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)

def make_feature_row(timestamp: pd.Timestamp, demand_history: pd.Series, weather: pd.Series) -> pd.DataFrame:
    """Creates training-compatible row using exact historical timestamp lags."""
    row = {
        "year": timestamp.year,
        "month": timestamp.month,
        "day": timestamp.day,
        "hour": timestamp.hour,
        "minute": timestamp.minute,
        "day_of_week": timestamp.dayofweek,
        "is_weekend": int(timestamp.dayofweek in (5, 6)),
    }
    row.update({name: demand_history.get(timestamp - delta, float("nan")) for name, delta in LAGS.items()})
    row.update({name: weather[name] for name in WEATHER_COLUMNS})
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
