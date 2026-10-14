"""Unit tests for feature engineering, exact lag timestamp matching, and zero lookahead bias."""

import os
import unittest
import pandas as pd
import numpy as np

class TestFeatureEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feat_path = "feature_engineered_delhi_powerdemand_5min_2021_to_2024.csv"
        cls.expected_features = [
            "year", "month", "day", "hour", "minute", "day_of_week", "is_weekend",
            "demand_lag_5min", "demand_lag_1hour", "demand_lag_24hour", "demand_lag_7day",
            "temp", "dwpt", "rhum", "wdir", "wspd", "pres"
        ]

    def test_feature_columns_presence(self):
        """Verify all 17 causal input features and target are present."""
        self.assertTrue(os.path.exists(self.feat_path))
        df = pd.read_csv(self.feat_path, nrows=50)
        for col in self.expected_features:
            self.assertIn(col, df.columns, f"Feature '{col}' missing from feature dataset")
        self.assertIn("Power demand", df.columns)
        self.assertIn("datetime", df.columns)

    def test_causal_lag_exact_matching(self):
        """Spot check exact retrospective lag values at test period peak."""
        df = pd.read_csv(self.feat_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        lookup = df.set_index("datetime")["Power demand"]

        peak_dt = pd.Timestamp("2024-06-19 15:05:00")
        sub = df[df["datetime"] == peak_dt]
        self.assertFalse(sub.empty, "Peak timestamp must exist in feature dataset")
        row = sub.iloc[0]

        # Check that lag_5min matches t - 5 min
        expected_5m = lookup.get(peak_dt - pd.Timedelta(minutes=5))
        self.assertTrue(np.isclose(row["demand_lag_5min"], expected_5m, atol=1e-3))

        # Check that lag_1h matches t - 1 hour
        expected_1h = lookup.get(peak_dt - pd.Timedelta(hours=1))
        self.assertTrue(np.isclose(row["demand_lag_1hour"], expected_1h, atol=1e-3))

        # Check that lag_24h matches t - 24 hours
        expected_24h = lookup.get(peak_dt - pd.Timedelta(hours=24))
        self.assertTrue(np.isclose(row["demand_lag_24hour"], expected_24h, atol=1e-3))

        # Check that lag_7d matches t - 7 days
        expected_7d = lookup.get(peak_dt - pd.Timedelta(days=7))
        self.assertTrue(np.isclose(row["demand_lag_7day"], expected_7d, atol=1e-3))

if __name__ == "__main__":
    unittest.main()
