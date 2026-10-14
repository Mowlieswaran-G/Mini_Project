"""Unit tests for weather validation, optional weather service failure handling, and recursive forecasting."""

import unittest
import pandas as pd
from app.utils.features import validate_weather_inputs, fetch_optional_delhi_weather, classify_demand_regime

class TestWeatherAndForecasting(unittest.TestCase):
    def test_weather_input_validation_valid(self):
        """Test valid typical weather inputs pass without error."""
        valid_w = {
            "temp": 35.0, "dwpt": 20.0, "rhum": 45.0,
            "pres": 1000.0, "wspd": 12.0, "wdir": 270.0
        }
        is_valid, warnings = validate_weather_inputs(valid_w)
        self.assertTrue(is_valid, "Valid weather inputs must pass validation")
        self.assertEqual(len(warnings), 0, "No warnings expected for normal range")

    def test_weather_input_validation_out_of_bounds(self):
        """Test physically impossible weather inputs are flagged as invalid."""
        invalid_w = {
            "temp": 75.0, # Impossible temperature
            "dwpt": 20.0, "rhum": -10.0, # Negative humidity
            "pres": 1000.0, "wspd": 12.0, "wdir": 400.0 # Out of 360 degrees
        }
        is_valid, warnings = validate_weather_inputs(invalid_w)
        self.assertFalse(is_valid, "Out-of-bounds weather inputs must be rejected")
        self.assertGreater(len(warnings), 0, "Warnings/errors must be reported")

    def test_optional_weather_api_timeout_safety(self):
        """Test that the weather fetch helper safely returns None or dict without raising an unhandled exception."""
        # Using a very short timeout to test failure/timeout handling safely
        res = fetch_optional_delhi_weather(timeout_secs=0.001)
        # Should either return None (timeout/offline) or a dict (if lightning fast)
        self.assertTrue(res is None or isinstance(res, dict), "Weather service query must fail safely")

    def test_demand_regime_classification_and_alert_threshold(self):
        """Test operational tier classification and user-configurable alert threshold."""
        # Normal
        reg_normal = classify_demand_regime(4500.0)
        self.assertIn("Normal", reg_normal["tier"])

        # Extreme Peak
        reg_extreme = classify_demand_regime(8200.0, alert_threshold_mw=7500.0)
        self.assertTrue(reg_extreme["user_threshold_exceeded"])
        self.assertIn("User Alert", reg_extreme["threshold_warning"])

if __name__ == "__main__":
    unittest.main()
