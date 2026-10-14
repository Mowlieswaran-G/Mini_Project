"""Unit tests for test prediction alignment and verified baseline metric calculations."""

import os
import unittest
import pandas as pd
import numpy as np

class TestPredictionsAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_csv = "test_delhi_powerdemand.csv"
        cls.lr_pred = "linear_regression_test_predictions.csv"
        cls.rf_pred = "random_forest_test_predictions.csv"
        cls.lstm_pred = "lstm_test_predictions.csv"

        cls.df_test = pd.read_csv(cls.test_csv)
        cls.df_test["datetime"] = pd.to_datetime(cls.df_test["datetime"])

    def test_prediction_files_alignment(self):
        """Verify all three prediction files have 70,825 rows matching test set timestamps and targets."""
        for name, path in [("LR", self.lr_pred), ("RF", self.rf_pred), ("LSTM", self.lstm_pred)]:
            self.assertTrue(os.path.exists(path), f"{name} prediction file missing")
            p_df = pd.read_csv(path)
            p_df["datetime"] = pd.to_datetime(p_df["datetime"])

            self.assertEqual(len(p_df), len(self.df_test), f"{name} rows {len(p_df)} != test rows {len(self.df_test)}")
            self.assertTrue((p_df["datetime"] == self.df_test["datetime"]).all(), f"{name} timestamps do not align")
            self.assertTrue(np.allclose(p_df["actual_demand"].values, self.df_test["Power demand"].values, atol=1e-3))

    def test_baseline_metrics_verification(self):
        """Recompute test MAE and RMSE from prediction CSVs and verify against established baseline results."""
        # 1. Linear Regression
        lr_df = pd.read_csv(self.lr_pred)
        lr_mae = np.mean(np.abs(lr_df["actual_demand"] - lr_df["predicted_demand"]))
        lr_rmse = np.sqrt(np.mean((lr_df["actual_demand"] - lr_df["predicted_demand"])**2))
        self.assertAlmostEqual(lr_mae, 25.24, delta=0.05)
        self.assertAlmostEqual(lr_rmse, 43.23, delta=0.05)

        # 2. Random Forest
        rf_df = pd.read_csv(self.rf_pred)
        rf_mae = np.mean(np.abs(rf_df["actual_demand"] - rf_df["predicted_demand"]))
        rf_rmse = np.sqrt(np.mean((rf_df["actual_demand"] - rf_df["predicted_demand"])**2))
        self.assertAlmostEqual(rf_mae, 28.17, delta=0.05)
        self.assertAlmostEqual(rf_rmse, 56.29, delta=0.05)

        # 3. LSTM
        lstm_df = pd.read_csv(self.lstm_pred)
        lstm_mae = np.mean(np.abs(lstm_df["actual_demand"] - lstm_df["predicted_demand"]))
        lstm_rmse = np.sqrt(np.mean((lstm_df["actual_demand"] - lstm_df["predicted_demand"])**2))
        self.assertAlmostEqual(lstm_mae, 27.63, delta=0.05)
        self.assertAlmostEqual(lstm_rmse, 47.77, delta=0.05)

if __name__ == "__main__":
    unittest.main()
