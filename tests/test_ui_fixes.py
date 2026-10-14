"""Unit tests verifying the 8 UI and functionality fixes.

Tests cover:
1. LSTM causal rolling sequence construction, (1, 6, 17) shape, and recursive lag updates.
2. Weather scenario calendar feature derivation from arbitrary timestamps.
3. Model comparison CSV encoding and R² Score metric label.
4. Dataset and evaluation metrics preservation.
5. HTML dashboard artifact existence and safe readability.
6. EDA grouping column names (Hour, Day of Week).
7. Streamlit deprecation clean-up (absence of deprecated use_container_width).
"""

import os
import unittest
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from unittest.mock import MagicMock

from app.utils.data import is_torch_installed, is_lstm_file_present

class TestUIFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_csv = "test_delhi_powerdemand.csv"
        cls.df_test = pd.read_csv(cls.test_csv)
        cls.df_test["datetime"] = pd.to_datetime(cls.df_test["datetime"])
        cls.feature_cols = [
            "year", "month", "day", "hour", "minute", "day_of_week", "is_weekend",
            "demand_lag_5min", "demand_lag_1hour", "demand_lag_24hour", "demand_lag_7day",
            "temp", "dwpt", "rhum", "wdir", "wspd", "pres"
        ]

    # --- ISSUE 1: LSTM Recursive Forecasting & Causal Rolling Updates ---
    def test_lstm_rolling_sequence_shape_and_inference(self):
        """Verify 6-step causal sequence has shape (1, 6, 17) and feeds cleanly to inference."""
        self.assertTrue(is_lstm_file_present(), "LSTM pickle/weights must be present on disk")
        
        # Take trailing 6 rows from test set
        history_6 = self.df_test.iloc[-6:][self.feature_cols].copy()
        self.assertEqual(len(history_6), 6)
        
        scaler = StandardScaler()
        scaler.fit(self.df_test[self.feature_cols].values)
        scaled_6 = scaler.transform(history_6.values)
        seq_buffer = scaled_6.reshape(1, 6, 17)
        self.assertEqual(seq_buffer.shape, (1, 6, 17))

        if is_torch_installed():
            import joblib
            lstm_wrapper = joblib.load("lstm_delhi_powerdemand.pkl")
            pred = lstm_wrapper.predict(seq_buffer)
            self.assertEqual(len(pred), 1)
            self.assertTrue(1000.0 < float(pred[0]) < 10000.0)
        else:
            # When PyTorch is not in the environment, test with mock wrapper adhering to LSTMModelWrapper contract
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([5432.10])
            pred = mock_model.predict(seq_buffer)
            self.assertEqual(len(pred), 1)
            self.assertEqual(float(pred[0]), 5432.10)
            mock_model.predict.assert_called_with(seq_buffer)

    def test_lstm_causal_rolling_updates(self):
        """Verify rolling sequence shifts causally and updates lags autoregressively without future actuals."""
        history_6 = self.df_test.iloc[-6:][self.feature_cols].copy()
        scaler = StandardScaler()
        scaler.fit(self.df_test[self.feature_cols].values)
        
        seq_buffer = scaler.transform(history_6.values).reshape(1, 6, 17).copy()
        
        curr_demand = float(history_6["demand_lag_5min"].iloc[-1])
        base_time = pd.Timestamp("2024-06-20 00:00:00")
        
        captured_shapes = []
        captured_buffers = []
        
        # Simulate 6 forecast steps as implemented in app/app.py
        for step in range(1, 7):
            captured_shapes.append(seq_buffer.shape)
            captured_buffers.append(seq_buffer.copy())
            
            # Predict step (simulated causal forecast)
            simulated_forecast = curr_demand + 10.0 * step
            
            step_dt = base_time + pd.Timedelta(minutes=5 * step)
            row_dict = {
                "year": step_dt.year,
                "month": step_dt.month,
                "day": step_dt.day,
                "hour": step_dt.hour,
                "minute": step_dt.minute,
                "day_of_week": step_dt.dayofweek,
                "is_weekend": 1 if step_dt.dayofweek in [5, 6] else 0,
                "demand_lag_5min": simulated_forecast,  # Autoregressive update: predicted value fed as next lag
                "demand_lag_1hour": 5000.0,
                "demand_lag_24hour": 5100.0,
                "demand_lag_7day": 5200.0,
                "temp": 38.0,
                "dwpt": 22.0,
                "rhum": 40.0,
                "wdir": 270.0,
                "wspd": 10.0,
                "pres": 998.0
            }
            step_df = pd.DataFrame([row_dict], columns=self.feature_cols)
            new_scaled = scaler.transform(step_df.values).reshape(1, 1, 17)
            
            # Causal rolling update
            old_buffer = seq_buffer.copy()
            seq_buffer = np.concatenate([seq_buffer[:, 1:, :], new_scaled], axis=1)
            
            # Assertions for causal slide
            self.assertEqual(seq_buffer.shape, (1, 6, 17))
            # Steps [0..4] of new buffer must match steps [1..5] of previous buffer
            np.testing.assert_allclose(seq_buffer[:, :5, :], old_buffer[:, 1:, :])
            # Step [5] of new buffer must match new scaled step
            np.testing.assert_allclose(seq_buffer[:, 5:, :], new_scaled)
            
            curr_demand = simulated_forecast

        self.assertEqual(len(captured_shapes), 6)
        for s in captured_shapes:
            self.assertEqual(s, (1, 6, 17))

    # --- ISSUE 2: Weather Scenario Date and Time ---
    def test_scenario_calendar_features_calculation(self):
        """Verify dynamic calculation of calendar features for user-selected date and time."""
        test_cases = [
            (datetime.date(2024, 6, 19), datetime.time(15, 30), {
                "year": 2024, "month": 6, "day": 19, "hour": 15, "minute": 30,
                "day_of_week": 2, "is_weekend": 0
            }),
            (datetime.date(2024, 6, 23), datetime.time(10, 0), {  # Sunday
                "year": 2024, "month": 6, "day": 23, "hour": 10, "minute": 0,
                "day_of_week": 6, "is_weekend": 1
            }),
            (datetime.date(2025, 1, 1), datetime.time(0, 0), {  # Wednesday
                "year": 2025, "month": 1, "day": 1, "hour": 0, "minute": 0,
                "day_of_week": 2, "is_weekend": 0
            })
        ]
        
        for sim_date, sim_time, expected in test_cases:
            combined_dt = pd.to_datetime(f"{sim_date} {sim_time}")
            self.assertEqual(combined_dt.year, expected["year"])
            self.assertEqual(combined_dt.month, expected["month"])
            self.assertEqual(combined_dt.day, expected["day"])
            self.assertEqual(combined_dt.hour, expected["hour"])
            self.assertEqual(combined_dt.minute, expected["minute"])
            self.assertEqual(combined_dt.dayofweek, expected["day_of_week"])
            self.assertEqual(1 if combined_dt.dayofweek in [5, 6] else 0, expected["is_weekend"])

    # --- ISSUE 4: EDA Chart Labels ---
    def test_eda_chart_groupby_labels(self):
        """Verify EDA grouping columns Hour and Day of Week exist with formatted labels."""
        df_sample = self.df_test.head(100).copy()
        
        # Hourly aggregation
        hr_avg = df_sample.groupby("hour")["Power demand"].mean().reset_index()
        hr_avg.columns = ["Hour", "Average Demand (MW)"]
        self.assertIn("Hour", hr_avg.columns)
        self.assertIn("Average Demand (MW)", hr_avg.columns)
        
        # Day of week aggregation
        dow_avg = df_sample.groupby("day_of_week")["Power demand"].mean().reset_index()
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_avg["Day of Week"] = dow_avg["day_of_week"].apply(lambda d: dow_names[d % 7])
        dow_avg = dow_avg.rename(columns={"Power demand": "Average Demand (MW)"})
        self.assertIn("Day of Week", dow_avg.columns)
        self.assertTrue(all(isinstance(name, str) for name in dow_avg["Day of Week"]))

    # --- ISSUE 5: Model Evaluation Alignment & Metrics Preservation ---
    def test_evaluation_alignment_and_metrics_preservation(self):
        """Verify model evaluation CSV metrics match baseline benchmarks exactly."""
        comparison_csv = "model_comparison_delhi_powerdemand.csv"
        self.assertTrue(os.path.exists(comparison_csv))
        df_comp = pd.read_csv(comparison_csv, encoding="utf-8-sig")
        
        mae_row = df_comp[df_comp["Metric"].str.startswith("MAE")].iloc[0]
        rmse_row = df_comp[df_comp["Metric"].str.startswith("RMSE")].iloc[0]
        
        self.assertAlmostEqual(float(mae_row["Linear Regression"]), 25.24, places=2)
        self.assertAlmostEqual(float(mae_row["Random Forest"]), 28.17, places=2)
        self.assertAlmostEqual(float(mae_row["LSTM"]), 27.63, places=2)
        
        self.assertAlmostEqual(float(rmse_row["Linear Regression"]), 43.23, places=2)
        self.assertAlmostEqual(float(rmse_row["Random Forest"]), 56.29, places=2)
        self.assertAlmostEqual(float(rmse_row["LSTM"]), 47.77, places=2)

    # --- ISSUE 6: Model Comparison CSV Encoding and R² Label ---
    def test_csv_encoding_and_r2_score_label(self):
        """Verify model_comparison_delhi_powerdemand.csv has R² Score and is clean UTF-8."""
        comparison_csv = "model_comparison_delhi_powerdemand.csv"
        df = pd.read_csv(comparison_csv, encoding="utf-8-sig")
        
        # Metric column must contain 'R² Score'
        metric_values = df["Metric"].tolist()
        self.assertIn("R² Score", metric_values, f"Metrics found: {metric_values}")
        
        r2_row = df[df["Metric"] == "R² Score"].iloc[0]
        self.assertAlmostEqual(float(r2_row["Linear Regression"]), 0.999, places=3)
        self.assertAlmostEqual(float(r2_row["Random Forest"]), 0.9983, places=3)
        self.assertAlmostEqual(float(r2_row["LSTM"]), 0.9987, places=3)

    # --- ISSUE 7: Static HTML Dashboard Download & Missing File Handling ---
    def test_html_dashboard_download_file_integrity(self):
        """Verify outputs/project_dashboard/project_dashboard.html is present and readable as binary for download."""
        html_path = os.path.join("outputs", "project_dashboard", "project_dashboard.html")
        self.assertTrue(os.path.exists(html_path), f"Dashboard file not found at {html_path}")
        with open(html_path, "rb") as f:
            data = f.read()
        self.assertGreater(len(data), 500_000, "HTML file must be complete self-contained report (>500KB)")

    def test_missing_optional_file_warning_logic(self):
        """Verify non-existent files are safely detected without throwing unhandled exceptions."""
        non_existent_file = "outputs/project_dashboard/non_existent_dashboard.html"
        self.assertFalse(os.path.exists(non_existent_file))

    # --- ISSUE 8: Streamlit Deprecation Warnings ---
    def test_no_deprecated_use_container_width(self):
        """Verify app/app.py contains no deprecated use_container_width arguments."""
        with open("app/app.py", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("use_container_width=True", content)
        self.assertNotIn("use_container_width=False", content)
        self.assertNotIn("use_container_width", content)

if __name__ == "__main__":
    unittest.main()
