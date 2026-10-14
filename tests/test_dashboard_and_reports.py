"""Unit tests for static dashboard existence, report generation, and Streamlit module imports."""

import os
import unittest

class TestDashboardAndReports(unittest.TestCase):
    def test_static_html_dashboard_existence_and_size(self):
        """Test static HTML dashboard exists, contains all 3 models and project limitations."""
        html_path = "outputs/project_dashboard/project_dashboard.html"
        self.assertTrue(os.path.exists(html_path), "Static HTML dashboard missing")
        self.assertGreater(os.path.getsize(html_path), 500_000, "HTML dashboard must be populated with self-contained assets (>500KB)")

        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Linear Regression", content)
            self.assertIn("Random Forest", content)
            self.assertIn("PyTorch LSTM", content)
            self.assertIn("Documented Project Assumptions & Limitations", content)
            self.assertIn("Evaluation Period:", content)

    def test_csv_reports_existence(self):
        """Test all primary benchmark CSV files exist on disk."""
        expected_csvs = [
            "model_comparison_delhi_powerdemand.csv",
            "outputs/peak_analysis/daily_peak_analysis.csv",
            "outputs/peak_analysis/peak_model_performance.csv",
            "outputs/peak_analysis/monthly_peak_analysis.csv",
            "outputs/peak_analysis/peak_hour_analysis.csv",
            "outputs/peak_projection/example_projection.csv"
        ]
        for csv_file in expected_csvs:
            self.assertTrue(os.path.exists(csv_file), f"Expected CSV report missing: {csv_file}")

    def test_streamlit_app_and_utils_importability(self):
        """Test that app utility modules and app files can be imported cleanly without error."""
        import app.utils.data as data_mod
        import app.utils.features as feat_mod
        import lstm_model

        self.assertTrue(hasattr(data_mod, "load_history"))
        self.assertTrue(hasattr(feat_mod, "validate_weather_inputs"))
        self.assertTrue(hasattr(lstm_model, "DemandLSTM"))

if __name__ == "__main__":
    unittest.main()
