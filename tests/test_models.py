"""Unit tests for model loading and single-instance inference."""

import os
import unittest
import joblib
import pandas as pd
import numpy as np

class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lr_model_path = "linear_regression_delhi_powerdemand.pkl"
        cls.rf_model_path = "random_forest_delhi_powerdemand.pkl"
        cls.lstm_pt_path = "lstm_delhi_powerdemand.pt"
        cls.feature_cols = [
            "year", "month", "day", "hour", "minute", "day_of_week", "is_weekend",
            "demand_lag_5min", "demand_lag_1hour", "demand_lag_24hour", "demand_lag_7day",
            "temp", "dwpt", "rhum", "wdir", "wspd", "pres"
        ]

    def test_linear_regression_loading_and_prediction(self):
        """Test Linear Regression model loads and predicts within plausible MW bounds."""
        self.assertTrue(os.path.exists(self.lr_model_path))
        lr = joblib.load(self.lr_model_path)
        sample_input = pd.DataFrame([{
            "year": 2024, "month": 6, "day": 19, "hour": 15, "minute": 5, "day_of_week": 2, "is_weekend": 0,
            "demand_lag_5min": 8580.0, "demand_lag_1hour": 8350.0, "demand_lag_24hour": 8100.0, "demand_lag_7day": 7820.0,
            "temp": 44.5, "dwpt": 19.5, "rhum": 32.0, "wdir": 290.0, "wspd": 15.0, "pres": 997.5
        }], columns=self.feature_cols)

        pred = lr.predict(sample_input)[0]
        self.assertTrue(7000.0 < pred < 9500.0, f"Expected realistic prediction for summer peak, got {pred}")

    def test_random_forest_loading(self):
        """Test Random Forest model exists on disk and has 100 estimators."""
        self.assertTrue(os.path.exists(self.rf_model_path))
        rf = joblib.load(self.rf_model_path)
        self.assertEqual(len(rf.estimators_), 100, "Random Forest must contain 100 trees")

    def test_lstm_checkpoint_existence(self):
        """Test PyTorch LSTM checkpoint and wrapper files exist on disk."""
        self.assertTrue(os.path.exists(self.lstm_pt_path), "PyTorch checkpoint lstm_delhi_powerdemand.pt missing")
        self.assertTrue(os.path.exists("lstm_delhi_powerdemand.pkl"), "LSTM wrapper lstm_delhi_powerdemand.pkl missing")

if __name__ == "__main__":
    unittest.main()
