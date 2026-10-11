"""
Baseline Linear Regression Model Training and Evaluation
Project: AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System

Inputs:
- train_delhi_powerdemand.csv
- test_delhi_powerdemand.csv

Outputs:
- linear_regression_delhi_powerdemand.pkl (Trained model file)
- linear_regression_test_predictions.csv (Datetime, Actual, Predicted)
- outputs/linear_regression/1_actual_vs_predicted_full_test.png
- outputs/linear_regression/2_actual_vs_predicted_zoomed_peak.png
- outputs/linear_regression/3_actual_vs_predicted_scatter.png
- outputs/linear_regression/linear_regression_metrics.json
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.autolayout'] = True

OUTPUT_DIR = os.path.join("outputs", "linear_regression")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_and_evaluate_linear_regression():
    print("Loading training and testing datasets...")
    train_df = pd.read_csv("train_delhi_powerdemand.csv")
    test_df = pd.read_csv("test_delhi_powerdemand.csv")

    train_df["datetime"] = pd.to_datetime(train_df["datetime"])
    test_df["datetime"] = pd.to_datetime(test_df["datetime"])

    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # Define 17 input features
    feature_cols = [
        "year", "month", "day", "hour", "minute", "day_of_week", "is_weekend",
        "demand_lag_5min", "demand_lag_1hour", "demand_lag_24hour", "demand_lag_7day",
        "temp", "dwpt", "rhum", "wdir", "wspd", "pres"
    ]
    target_col = "Power demand"

    print(f"Number of input features: {len(feature_cols)}")
    print(f"Features: {feature_cols}")

    # Separate X and y without shuffling
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Train Linear Regression model
    print("\nTraining Linear Regression model on training set...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model training completed.")

    # Model coefficients inspection
    intercept = float(model.intercept_)
    coefs = dict(zip(feature_cols, [float(c) for c in model.coef_]))

    # Predict on Train and Test
    print("Generating predictions on train and test sets...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute Evaluation Metrics
    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    train_rmse = float(root_mean_squared_error(y_train, y_train_pred))
    train_r2 = float(r2_score(y_train, y_train_pred))

    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_rmse = float(root_mean_squared_error(y_test, y_test_pred))
    test_r2 = float(r2_score(y_test, y_test_pred))

    # Min and Max values in Test Set
    actual_min = float(y_test.min())
    actual_max = float(y_test.max())
    pred_min = float(y_test_pred.min())
    pred_max = float(y_test_pred.max())

    # Peak demand observation in Test Set
    max_test_idx = int(np.argmax(y_test.values))
    max_test_row = test_df.iloc[max_test_idx]
    max_test_dt = str(max_test_row["datetime"])
    max_actual_demand = float(max_test_row[target_col])
    pred_at_max_demand = float(y_test_pred[max_test_idx])
    peak_prediction_error = pred_at_max_demand - max_actual_demand

    print("\n=== EVALUATION RESULTS ===")
    print(f"Train Metrics: MAE = {train_mae:.2f} MW | RMSE = {train_rmse:.2f} MW | R² = {train_r2:.4f}")
    print(f"Test Metrics : MAE = {test_mae:.2f} MW | RMSE = {test_rmse:.2f} MW | R² = {test_r2:.4f}")
    print(f"\nTest Demand Ranges:")
    print(f"  Actual Min : {actual_min:.2f} MW | Actual Max : {actual_max:.2f} MW")
    print(f"  Pred Min   : {pred_min:.2f} MW | Pred Max   : {pred_max:.2f} MW")
    print(f"\nPeak Demand Event in Test Set:")
    print(f"  Timestamp        : {max_test_dt}")
    print(f"  Actual Demand    : {max_actual_demand:.2f} MW")
    print(f"  Predicted Demand : {pred_at_max_demand:.2f} MW")
    print(f"  Difference (Pred - Actual): {peak_prediction_error:+.2f} MW")

    # Save Test Predictions CSV
    pred_df = pd.DataFrame({
        "datetime": test_df["datetime"],
        "actual_demand": y_test.values,
        "predicted_demand": y_test_pred
    })
    pred_csv_path = "linear_regression_test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"\nSaved test predictions to: {pred_csv_path}")

    # Save Model Pickle
    model_pkl_path = "linear_regression_delhi_powerdemand.pkl"
    joblib.dump(model, model_pkl_path)
    print(f"Saved trained model to: {model_pkl_path}")

    # ==========================================
    # Visualizations
    # ==========================================
    print("\nGenerating evaluation plots...")

    # 1. Full Test Period Plot (Actual vs Predicted)
    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    # Plot daily aggregated or downsampled to keep crisp visualization
    pred_df_indexed = pred_df.set_index("datetime")
    daily_actual = pred_df_indexed["actual_demand"].resample("D").mean()
    daily_pred = pred_df_indexed["predicted_demand"].resample("D").mean()

    ax.plot(daily_actual.index, daily_actual.values, color="#1565C0", label="Actual Demand (Daily Mean)", linewidth=1.5)
    ax.plot(daily_pred.index, daily_pred.values, color="#D32F2F", linestyle="--", label="Linear Regression Predicted (Daily Mean)", linewidth=1.5, alpha=0.9)

    ax.set_title("Linear Regression Baseline: Actual vs. Predicted Demand (Complete Test Period: Mar - Dec 2024)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Power Demand (MW)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plot1_path = os.path.join(OUTPUT_DIR, "1_actual_vs_predicted_full_test.png")
    plt.savefig(plot1_path)
    plt.close()
    print(f"Saved: {plot1_path}")

    # 2. Zoomed-In Plot Around the Highest Demand Peak Period (June 17 to June 21, 2024)
    peak_dt = pd.to_datetime(max_test_dt)
    zoom_start = peak_dt - pd.Timedelta(days=2)
    zoom_end = peak_dt + pd.Timedelta(days=2)

    zoom_mask = (pred_df["datetime"] >= zoom_start) & (pred_df["datetime"] <= zoom_end)
    zoom_df = pred_df[zoom_mask]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(zoom_df["datetime"], zoom_df["actual_demand"], color="#0D47A1", label="Actual Demand (5-min)", linewidth=2.0)
    ax.plot(zoom_df["datetime"], zoom_df["predicted_demand"], color="#E65100", linestyle="--", label="Linear Regression Predicted (5-min)", linewidth=1.8)

    ax.scatter(peak_dt, max_actual_demand, color="#D32F2F", s=80, zorder=5, label=f"Peak Actual: {max_actual_demand:.1f} MW")
    ax.scatter(peak_dt, pred_at_max_demand, color="#E65100", s=80, marker="x", zorder=5, label=f"Peak Predicted: {pred_at_max_demand:.1f} MW")

    ax.set_title(f"Zoomed View Around All-Time Peak Period (June 17 - 21, 2024)\nActual: {max_actual_demand:.1f} MW vs Predicted: {pred_at_max_demand:.1f} MW (Error: {peak_prediction_error:+.1f} MW)",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Timestamp", fontsize=11)
    ax.set_ylabel("Power Demand (MW)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, "2_actual_vs_predicted_zoomed_peak.png")
    plt.savefig(plot2_path)
    plt.close()
    print(f"Saved: {plot2_path}")

    # 3. Actual vs Predicted Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    sample_sub = pred_df.sample(n=min(10000, len(pred_df)), random_state=42)
    ax.scatter(sample_sub["actual_demand"], sample_sub["predicted_demand"], alpha=0.2, color="#1976D2", s=10, label="Predictions (10k Sample)")

    lims = [
        min(actual_min, pred_min) - 200,
        max(actual_max, pred_max) + 200
    ]
    ax.plot(lims, lims, color="#D32F2F", linestyle="--", linewidth=1.5, label="Perfect Forecast (y = x)")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f"Linear Regression: Actual vs. Predicted Demand (Test Set)\nR² = {test_r2:.4f} | RMSE = {test_rmse:.2f} MW",
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Actual Power Demand (MW)", fontsize=11)
    ax.set_ylabel("Predicted Power Demand (MW)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plot3_path = os.path.join(OUTPUT_DIR, "3_actual_vs_predicted_scatter.png")
    plt.savefig(plot3_path)
    plt.close()
    print(f"Saved: {plot3_path}")

    # Save summary metrics to JSON
    metrics_summary = {
        "model": "Linear Regression Baseline",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "input_features_count": len(feature_cols),
        "train_metrics": {
            "MAE": round(train_mae, 2),
            "RMSE": round(train_rmse, 2),
            "R2": round(train_r2, 4)
        },
        "test_metrics": {
            "MAE": round(test_mae, 2),
            "RMSE": round(test_rmse, 2),
            "R2": round(test_r2, 4)
        },
        "test_demand_actual_min": round(actual_min, 2),
        "test_demand_actual_max": round(actual_max, 2),
        "test_demand_predicted_min": round(pred_min, 2),
        "test_demand_predicted_max": round(pred_max, 2),
        "peak_observation": {
            "datetime": max_test_dt,
            "actual_demand": round(max_actual_demand, 2),
            "predicted_demand": round(pred_at_max_demand, 2),
            "prediction_error_mw": round(peak_prediction_error, 2),
            "percentage_error": round((peak_prediction_error / max_actual_demand) * 100, 2)
        },
        "coefficients": coefs,
        "intercept": round(intercept, 4),
        "execution_status": "Success - Zero Errors"
    }

    metrics_path = os.path.join(OUTPUT_DIR, "linear_regression_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics summary to: {metrics_path}")

    return metrics_summary

if __name__ == "__main__":
    summary = train_and_evaluate_linear_regression()
    print("\nExecution finished successfully.")
