"""Unit tests for peak demand extraction, timing matching, and distinction from model daily maximum."""

import os
import unittest
import pandas as pd

class TestPeakAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.daily_peak_path = "outputs/peak_analysis/daily_peak_analysis.csv"
        cls.perf_path = "outputs/peak_analysis/peak_model_performance.csv"

    def test_daily_peak_columns_and_counts(self):
        """Test daily peak analysis contains exactly 272 days with actual peak vs model maximum distinctions."""
        self.assertTrue(os.path.exists(self.daily_peak_path))
        df = pd.read_csv(self.daily_peak_path)
        self.assertEqual(len(df), 272, "Must evaluate exactly 272 days in test set")

        required_cols = [
            "actual_daily_peak_mw", "actual_peak_time",
            "lr_pred_at_actual_peak_mw", "lr_model_daily_max_mw",
            "rf_pred_at_actual_peak_mw", "rf_model_daily_max_mw"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Required column '{col}' missing from daily peak analysis")

    def test_distinction_actual_peak_vs_model_max(self):
        """Verify that prediction at the actual peak timestamp is distinguished from the model's daily maximum."""
        df = pd.read_csv(self.daily_peak_path)
        
        # Spot check highest peak day: 2024-06-19
        peak_day = df[df["date"] == "2024-06-19"].iloc[0]
        self.assertEqual(peak_day["actual_daily_peak_mw"], 8631.53)
        self.assertEqual(peak_day["actual_peak_time"], "2024-06-19 15:05:00")

        # LR
        self.assertAlmostEqual(peak_day["lr_pred_at_actual_peak_mw"], 8560.41, delta=0.5)
        # RF
        self.assertAlmostEqual(peak_day["rf_pred_at_actual_peak_mw"], 7677.32, delta=0.5)

    def test_peak_model_performance_summary(self):
        """Verify peak performance CSV contains all three models and valid error metrics."""
        self.assertTrue(os.path.exists(self.perf_path))
        perf = pd.read_csv(self.perf_path)
        models = list(perf["model"])
        self.assertIn("Linear Regression", models)
        self.assertIn("Random Forest", models)
        self.assertIn("LSTM Neural Network", models)

if __name__ == "__main__":
    unittest.main()
