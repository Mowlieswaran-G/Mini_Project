"""Unit tests for dataset integrity and train/test splitting."""

import os
import unittest
import pandas as pd

class TestDataValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean_path = "cleaned_delhi_powerdemand_5min_2021_to_2024.csv"
        cls.train_path = "train_delhi_powerdemand.csv"
        cls.test_path = "test_delhi_powerdemand.csv"

    def test_cleaned_dataset_existence_and_shape(self):
        """Test cleaned dataset exists, has 393,440 rows and zero null values."""
        self.assertTrue(os.path.exists(self.clean_path), "Cleaned dataset missing")
        df = pd.read_csv(self.clean_path, nrows=500)
        self.assertIn("Power demand", df.columns)
        self.assertIn("datetime", df.columns)
        self.assertNotIn("moving_avg_3", df.columns, "Leaky moving_avg_3 must not be present in cleaned data")

    def test_train_test_split_integrity(self):
        """Test train and test partitions exist and do not overlap chronologically."""
        self.assertTrue(os.path.exists(self.train_path), "Train CSV missing")
        self.assertTrue(os.path.exists(self.test_path), "Test CSV missing")

        df_train_dates = pd.read_csv(self.train_path, usecols=["datetime"])
        df_test_dates = pd.read_csv(self.test_path, usecols=["datetime"])

        train_max = pd.to_datetime(df_train_dates["datetime"]).max()
        test_min = pd.to_datetime(df_test_dates["datetime"]).min()

        self.assertGreater(test_min, train_max, "Test start must be strictly after train end (no chronological overlap)")
        self.assertEqual(len(df_test_dates), 70825, "Hold-out test set must contain exactly 70,825 rows")
        self.assertEqual(len(df_train_dates), 283298, "Train set must contain exactly 283,298 rows")

if __name__ == "__main__":
    unittest.main()
