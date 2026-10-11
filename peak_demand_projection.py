"""
Peak Demand Projection Module for Delhi Power System
Project: AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System

Purpose:
A transparent, standalone inference module that loads the trained Linear Regression model
and allows users/operators to evaluate model-based demand projections under user-defined scenarios.

Inputs:
- Date and Time
- Required historical demand lags:
    * demand_lag_5min (MW)
    * demand_lag_1hour (MW)
    * demand_lag_24hour (MW)
    * demand_lag_7day (MW)
- Weather inputs:
    * temp (Ambient Temperature in °C)
    * dwpt (Dew Point in °C)
    * rhum (Relative Humidity in %)
    * wdir (Wind Direction in degrees)
    * wspd (Wind Speed in km/h)
    * pres (Surface Atmospheric Pressure in hPa)

Outputs:
- outputs/peak_projection/example_projection.csv
- outputs/peak_projection/peak_projection_report.md
- outputs/peak_projection/projected_peak_scenario.png
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join("outputs", "peak_projection")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "linear_regression_delhi_powerdemand.pkl"

# Exact 17-feature schema used during model training
MODEL_FEATURE_COLS = [
    "year", "month", "day", "hour", "minute", "day_of_week", "is_weekend",
    "demand_lag_5min", "demand_lag_1hour", "demand_lag_24hour", "demand_lag_7day",
    "temp", "dwpt", "rhum", "wdir", "wspd", "pres"
]

# Historical physical observation boundaries for input validation
HISTORICAL_BOUNDS = {
    "demand": (1000.0, 9500.0),    # MW (Delhi observed historical range: 1,302 to 8,632 MW)
    "temp": (-2.0, 50.0),          # °C (Delhi observed range: 4.0 to 46.4 °C)
    "dwpt": (-15.0, 35.0),         # °C (Delhi observed range: -8.6 to 30.3 °C)
    "rhum": (0.0, 100.0),          # % (Delhi observed range: 5.0 to 100.0 %)
    "wdir": (0.0, 360.0),          # Degrees (Delhi observed range: 0.0 to 360.0 °)
    "wspd": (0.0, 80.0),           # km/h (Delhi observed range: 0.0 to 63.0 km/h)
    "pres": (970.0, 1040.0)        # hPa (Delhi observed range: 989.6 to 1027.0 hPa)
}

class PeakDemandProjector:
    """
    Transparent inference engine for projecting electricity demand in Delhi.
    Loads the validated baseline Linear Regression model without modifying it.
    """
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model pickle not found at: {model_path}")
        print(f"Loading trained Linear Regression model from: {model_path}...")
        self.model = joblib.load(model_path)
        self.feature_names = MODEL_FEATURE_COLS
        print("Model loaded successfully without retraining.")

    def validate_inputs(self, input_dict: dict) -> pd.Timestamp:
        """
        Validates date, time, demand lags, and weather features against physical bounds.
        """
        # 1. Datetime parsing
        if "date" not in input_dict or "time" not in input_dict:
            raise KeyError("Input must contain 'date' (YYYY-MM-DD) and 'time' (HH:MM:SS or HH:MM).")
        dt_str = f"{input_dict['date']} {input_dict['time']}"
        try:
            dt = pd.to_datetime(dt_str)
        except Exception as e:
            raise ValueError(f"Invalid date/time format: {dt_str}. Error: {e}")

        # 2. Demand lags check
        lag_keys = ["demand_lag_5min", "demand_lag_1hour", "demand_lag_24hour", "demand_lag_7day"]
        for k in lag_keys:
            if k not in input_dict:
                raise KeyError(f"Missing required historical lag input: '{k}'.")
            val = float(input_dict[k])
            min_d, max_d = HISTORICAL_BOUNDS["demand"]
            if not (min_d <= val <= max_d):
                print(f"  [Validation Warning] {k} = {val:.1f} MW is outside historical bounds ({min_d}-{max_d} MW).")

        # 3. Weather check
        weather_keys = ["temp", "dwpt", "rhum", "wdir", "wspd", "pres"]
        for wk in weather_keys:
            if wk not in input_dict:
                raise KeyError(f"Missing required weather input: '{wk}'.")
            val = float(input_dict[wk])
            min_w, max_w = HISTORICAL_BOUNDS[wk]
            if not (min_w <= val <= max_w):
                print(f"  [Validation Warning] {wk} = {val} is outside historical bounds ({min_w}-{max_w}).")

        return dt

    def build_feature_vector(self, input_dict: dict, dt: pd.Timestamp) -> pd.DataFrame:
        """
        Derives calendar features and arranges all 17 features in exact training order.
        """
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour
        minute = dt.minute
        day_of_week = dt.dayofweek  # 0=Monday, 6=Sunday
        is_weekend = int(day_of_week in [5, 6])

        row_data = {
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "demand_lag_5min": float(input_dict["demand_lag_5min"]),
            "demand_lag_1hour": float(input_dict["demand_lag_1hour"]),
            "demand_lag_24hour": float(input_dict["demand_lag_24hour"]),
            "demand_lag_7day": float(input_dict["demand_lag_7day"]),
            "temp": float(input_dict["temp"]),
            "dwpt": float(input_dict["dwpt"]),
            "rhum": float(input_dict["rhum"]),
            "wdir": float(input_dict["wdir"]),
            "wspd": float(input_dict["wspd"]),
            "pres": float(input_dict["pres"])
        }

        # Return single-row DataFrame with strict column order
        return pd.DataFrame([row_data], columns=self.feature_names)

    def project_single_point(self, input_dict: dict) -> dict:
        """
        Projects demand for a single scenario time point.
        """
        dt = self.validate_inputs(input_dict)
        X = self.build_feature_vector(input_dict, dt)
        pred_mw = float(self.model.predict(X)[0])

        return {
            "timestamp": str(dt),
            "date": str(dt.date()),
            "time": str(dt.strftime("%H:%M:%S")),
            "projected_demand_mw": round(pred_mw, 2),
            "nature_of_value": "MODEL-BASED SCENARIO PROJECTION (Not an observed value)",
            "inputs": input_dict
        }

    def project_peak_scenario(self, scenario_records: list) -> dict:
        """
        Evaluates a collection of scenario time points, identifies the maximum projected peak,
        and returns tabular projections.
        """
        projections = []
        for item in scenario_records:
            proj = self.project_single_point(item)
            projections.append({
                "timestamp": proj["timestamp"],
                "date": proj["date"],
                "time": proj["time"],
                "temp_degC": item["temp"],
                "dwpt_degC": item["dwpt"],
                "rhum_pct": item["rhum"],
                "wspd_kmh": item["wspd"],
                "pres_hpa": item["pres"],
                "demand_lag_5min_mw": item["demand_lag_5min"],
                "projected_demand_mw": proj["projected_demand_mw"]
            })

        proj_df = pd.DataFrame(projections)

        # Identify maximum projected peak
        max_idx = proj_df["projected_demand_mw"].idxmax()
        peak_row = proj_df.loc[max_idx]

        return {
            "projections_df": proj_df,
            "projected_peak": {
                "peak_timestamp": peak_row["timestamp"],
                "peak_date": peak_row["date"],
                "peak_time": peak_row["time"],
                "maximum_projected_demand_mw": float(peak_row["projected_demand_mw"]),
                "temp_at_peak_degC": float(peak_row["temp_degC"]),
                "nature_of_value": "MODEL-BASED SCENARIO PROJECTION (Maximum of evaluated scenario)"
            }
        }

def run_example_demonstration():
    print("\n=== RUNNING STEP 11: PEAK DEMAND PROJECTION DEMONSTRATION ===")
    projector = PeakDemandProjector()

    # Illustrative scenario: A high-temperature summer afternoon sequence
    # Parameters are strictly within historical observation bounds (temperature 42.0 to 43.8 °C)
    # This is clearly labeled as an illustrative scenario, not a real weather forecast.
    scenario_points = [
        {"date": "2025-06-20", "time": "14:00:00", "temp": 42.0, "dwpt": 23.5, "rhum": 33.0, "wdir": 270.0, "wspd": 6.5, "pres": 997.5, "demand_lag_5min": 8150.0, "demand_lag_1hour": 7920.0, "demand_lag_24hour": 8080.0, "demand_lag_7day": 7850.0},
        {"date": "2025-06-20", "time": "14:15:00", "temp": 42.4, "dwpt": 23.8, "rhum": 33.5, "wdir": 270.0, "wspd": 6.8, "pres": 997.3, "demand_lag_5min": 8220.0, "demand_lag_1hour": 7960.0, "demand_lag_24hour": 8110.0, "demand_lag_7day": 7870.0},
        {"date": "2025-06-20", "time": "14:30:00", "temp": 42.8, "dwpt": 24.0, "rhum": 34.0, "wdir": 275.0, "wspd": 7.0, "pres": 997.0, "demand_lag_5min": 8295.0, "demand_lag_1hour": 8010.0, "demand_lag_24hour": 8150.0, "demand_lag_7day": 7900.0},
        {"date": "2025-06-20", "time": "14:45:00", "temp": 43.2, "dwpt": 24.2, "rhum": 34.0, "wdir": 275.0, "wspd": 7.2, "pres": 996.8, "demand_lag_5min": 8380.0, "demand_lag_1hour": 8090.0, "demand_lag_24hour": 8200.0, "demand_lag_7day": 7930.0},
        {"date": "2025-06-20", "time": "15:00:00", "temp": 43.6, "dwpt": 24.5, "rhum": 34.5, "wdir": 280.0, "wspd": 7.5, "pres": 996.5, "demand_lag_5min": 8470.0, "demand_lag_1hour": 8150.0, "demand_lag_24hour": 8260.0, "demand_lag_7day": 7960.0},
        {"date": "2025-06-20", "time": "15:15:00", "temp": 43.8, "dwpt": 24.6, "rhum": 35.0, "wdir": 280.0, "wspd": 7.8, "pres": 996.3, "demand_lag_5min": 8550.0, "demand_lag_1hour": 8210.0, "demand_lag_24hour": 8310.0, "demand_lag_7day": 7990.0},
        {"date": "2025-06-20", "time": "15:30:00", "temp": 43.5, "dwpt": 24.5, "rhum": 35.0, "wdir": 280.0, "wspd": 7.6, "pres": 996.5, "demand_lag_5min": 8510.0, "demand_lag_1hour": 8250.0, "demand_lag_24hour": 8290.0, "demand_lag_7day": 7980.0},
        {"date": "2025-06-20", "time": "15:45:00", "temp": 43.1, "dwpt": 24.2, "rhum": 34.5, "wdir": 275.0, "wspd": 7.2, "pres": 996.8, "demand_lag_5min": 8420.0, "demand_lag_1hour": 8240.0, "demand_lag_24hour": 8240.0, "demand_lag_7day": 7950.0},
        {"date": "2025-06-20", "time": "16:00:00", "temp": 42.7, "dwpt": 24.0, "rhum": 34.0, "wdir": 270.0, "wspd": 6.8, "pres": 997.0, "demand_lag_5min": 8310.0, "demand_lag_1hour": 8210.0, "demand_lag_24hour": 8180.0, "demand_lag_7day": 7920.0}
    ]

    result = projector.project_peak_scenario(scenario_points)
    proj_df = result["projections_df"]
    peak_info = result["projected_peak"]

    print("\n--- SCENARIO PROJECTION RESULTS ---")
    for _, r in proj_df.iterrows():
        print(f"Time: {r['time']} | Temp: {r['temp_degC']:.1f} °C | Lag 5min: {r['demand_lag_5min_mw']:.1f} MW | Projected: {r['projected_demand_mw']:.2f} MW")

    print("\n--- PROJECTED PEAK IDENTIFICATION ---")
    print(f"Projected Peak Timestamp : {peak_info['peak_timestamp']}")
    print(f"Projected Peak Demand    : {peak_info['maximum_projected_demand_mw']:.2f} MW")
    print(f"Temperature at Peak      : {peak_info['temp_at_peak_degC']:.1f} °C")
    print(f"Label                    : {peak_info['nature_of_value']}")

    # 1. Save example_projection.csv
    csv_out_path = os.path.join(OUTPUT_DIR, "example_projection.csv")
    proj_df.to_csv(csv_out_path, index=False)
    print(f"\nSaved scenario projections table to: {csv_out_path}")

    # 2. Generate Plot
    fig, ax1 = plt.subplots(figsize=(12, 5.5), dpi=150)
    color_demand = "#D32F2F"
    color_temp = "#E65100"

    times = proj_df["time"]
    ax1.plot(times, proj_df["projected_demand_mw"], color=color_demand, marker="o", linewidth=2.2, label="Model-Based Projected Demand (MW)")
    ax1.set_xlabel("Scenario Time (June 20, 2025)", fontsize=11)
    ax1.set_ylabel("Projected Demand (MW)", color=color_demand, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color_demand)
    ax1.set_ylim(7800, 9200)

    # Highlight peak
    p_time = peak_info["peak_time"]
    p_val = peak_info["maximum_projected_demand_mw"]
    ax1.scatter(p_time, p_val, color="#B71C1C", s=90, zorder=5, label=f"Projected Peak: {p_val:.1f} MW ({p_time})")

    # Secondary axis for temperature
    ax2 = ax1.twinx()
    ax2.plot(times, proj_df["temp_degC"], color=color_temp, linestyle="--", marker="s", linewidth=1.5, alpha=0.75, label="Scenario Ambient Temp (°C)")
    ax2.set_ylabel("Temperature (°C)", color=color_temp, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color_temp)
    ax2.set_ylim(35.0, 48.0)

    ax1.set_title("Peak Demand Projection Module: Illustrative Summer Heatwave Scenario\n(Model-Based Scenario Projection - Not an Observed Value)",
                  fontsize=12, fontweight='bold', pad=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "projected_peak_scenario.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved scenario plot to: {plot_path}")

    # 3. Save Markdown Report
    def to_md_table(df_sub):
        headers = list(df_sub.columns)
        h = "| " + " | ".join([str(c) for c in headers]) + " |"
        s = "| " + " | ".join(["---"] * len(headers)) + " |"
        rows = ["| " + " | ".join([str(val) for val in row]) + " |" for row in df_sub.values]
        return "\n".join([h, s] + rows)

    report_path = os.path.join(OUTPUT_DIR, "peak_projection_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Peak Demand Projection Module: Scenario Analysis Report\n")
        f.write("## Delhi Electricity Demand and Peak Demand Projection\n\n")

        f.write("### 1. Important Analytical Disclaimer\n")
        f.write("> **LABEL:** `MODEL-BASED SCENARIO PROJECTION`\n")
        f.write("> - The projections generated by this module are **mathematical model simulations** conditioned upon hypothetical, user-specified exogenous inputs.\n")
        f.write("> - This demonstration **does NOT constitute an official weather forecast** or an observed grid dispatch record.\n")
        f.write("> - All input parameters used in this demonstration are strictly bound within the historical range observed in the Delhi power grid (2021–2024).\n\n")

        f.write("### 2. Evaluated Illustrative Scenario Table\n")
        f.write("- **Scenario Context:** High summer afternoon peak window (Friday, June 20, 2025, 14:00 to 16:00).\n\n")
        f.write(to_md_table(proj_df))

        f.write("\n\n### 3. Projected Peak Identification\n")
        f.write(f"- **Maximum Projected Peak Demand:** **`{peak_info['maximum_projected_demand_mw']:.2f} MW`**\n")
        f.write(f"- **Projected Peak Timestamp:** **`{peak_info['peak_timestamp']}`**\n")
        f.write(f"- **Scenario Temperature at Peak:** **`{peak_info['temp_at_peak_degC']:.1f} °C`**\n")
        f.write(f"- **Scenario Dew Point at Peak:** `24.6 °C` (Humidity: 35.0%)\n")
        f.write(f"- **Classification:** `{peak_info['nature_of_value']}`\n\n")

        f.write("### 4. Technical Architecture and Validation\n")
        f.write("- **Model Loaded:** Trained baseline Linear Regression (`linear_regression_delhi_powerdemand.pkl`).\n")
        f.write("- **Model Training State:** **Zero Retraining**; original weights and biases strictly preserved.\n")
        f.write("- **Feature Mapping:** All 17 features (`year`, `month`, `day`, `hour`, `minute`, `day_of_week`, `is_weekend`, 4 exact historical lags, and 6 weather features) arranged in identical schema.\n\n")

        f.write("### 5. Limitations of Scenario Projections\n")
        f.write("1. **Autoregressive Dependency:** The model relies heavily on `demand_lag_5min` ($w \approx 1.046$). When projecting consecutive multi-hour sequences into the distant future without real-time SCADA telemetry, simulated autoregressive recursive feedback can accumulate drift.\n")
        f.write("2. **Linear Weather Sensitivity:** Linear Regression models temperature with a constant linear slope ($w_{temp} \approx +0.395$ MW/°C), which does not fully capture the exponential 'hockey stick' non-linearity observed when Delhi temperatures exceed 42 °C–45 °C.\n")
        f.write("3. **Exogenous Accuracy:** Projection reliability is fundamentally bounded by the quality and accuracy of upstream meteorological forecasts provided to the model.\n")

    print(f"Saved scenario report to: {report_path}")
    print("\nPeak Demand Projection module executed successfully!")

if __name__ == "__main__":
    run_example_demonstration()
