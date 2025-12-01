"""Tests for base model class"""

import pytest
import pandas as pd
from unittest.mock import patch
from src.models.base_model import ElectionForecastModel


class MockModel(ElectionForecastModel):
    """Mock model for testing base class functionality"""

    def __init__(self):
        super().__init__("mock_model")

    def fit_and_forecast(
        self, state_polls, forecast_date, election_date, actual_margin, rng=None
    ):
        """Simple mock forecast"""
        avg_margin = state_polls["margin"].mean()
        return {
            "win_probability": 0.5 + avg_margin,
            "predicted_margin": avg_margin,
            "margin_std": 0.05,
        }


class TestElectionForecastModel:
    """Tests for ElectionForecastModel base class"""

    def test_model_initialization(self):
        """Test that model initializes correctly"""
        model = MockModel()
        assert model.name == "mock_model"
        assert model.predictions == []

    @patch("src.models.base_model.load_polling_data")
    @patch("src.models.base_model.load_election_results")
    def test_load_data(self, mock_results, mock_polls):
        """Test data loading"""
        mock_polls.return_value = pd.DataFrame({"state_code": ["FL"]})
        mock_results.return_value = {"FL": 0.012}

        model = MockModel()
        polls, results = model.load_data()

        assert mock_polls.called
        assert mock_results.called
        assert isinstance(polls, pd.DataFrame)
        assert isinstance(results, dict)

    def test_run_forecast_basic(self, sample_polls, sample_actual_results):
        """Test basic forecast run"""
        model = MockModel()

        with patch.object(
            model, "load_data", return_value=(sample_polls, sample_actual_results)
        ):
            forecast_dates = [pd.to_datetime("2016-10-15")]
            result = model.run_forecast(forecast_dates=forecast_dates, min_polls=5)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "state" in result.columns
        assert "forecast_date" in result.columns
        assert "win_probability" in result.columns
        assert "predicted_margin" in result.columns

    def test_run_forecast_min_polls_filter(self, sample_polls, sample_actual_results):
        """Test that states with insufficient polls are filtered"""
        model = MockModel()

        with patch.object(
            model, "load_data", return_value=(sample_polls, sample_actual_results)
        ):
            forecast_dates = [pd.to_datetime("2016-10-15")]
            result = model.run_forecast(forecast_dates=forecast_dates, min_polls=100)

        # Should have no predictions with min_polls=100
        assert len(result) == 0

    def test_run_forecast_verbose(self, sample_polls, sample_actual_results, caplog):
        """Test verbose output"""
        import logging

        caplog.set_level(logging.INFO)

        model = MockModel()

        with patch.object(
            model, "load_data", return_value=(sample_polls, sample_actual_results)
        ):
            forecast_dates = [pd.to_datetime("2016-10-15")]
            model.run_forecast(forecast_dates=forecast_dates, min_polls=5, verbose=True)

        assert "Processing" in caplog.text

    def test_save_results(
        self, sample_polls, sample_actual_results, tmp_path, monkeypatch
    ):
        """Test saving results"""
        monkeypatch.chdir(tmp_path)
        model = MockModel()

        with patch.object(
            model, "load_data", return_value=(sample_polls, sample_actual_results)
        ):
            forecast_dates = [pd.to_datetime("2016-10-15")]
            model.run_forecast(forecast_dates=forecast_dates, min_polls=5)
            metrics_df = model.save_results()

        assert (tmp_path / "predictions" / "mock_model.csv").exists()
        assert (tmp_path / "metrics" / "mock_model.txt").exists()
        assert isinstance(metrics_df, pd.DataFrame)

    def test_plot_state_creates_file(
        self, sample_polls, sample_actual_results, tmp_path, monkeypatch
    ):
        """Test that plot_state creates output file"""
        monkeypatch.chdir(tmp_path)
        model = MockModel()

        with patch.object(
            model, "load_data", return_value=(sample_polls, sample_actual_results)
        ):
            forecast_dates = [
                pd.to_datetime("2016-10-15"),
                pd.to_datetime("2016-11-01"),
            ]
            model.run_forecast(forecast_dates=forecast_dates, min_polls=5)
            model.plot_state("FL")

        assert (tmp_path / "plots" / "mock_model" / "FL.png").exists()

    def test_fit_and_forecast_not_implemented(self):
        """Test that base class fit_and_forecast is abstract"""

        # Cannot instantiate abstract class without implementing abstract method
        with pytest.raises(TypeError):

            class IncompleteModel(ElectionForecastModel):
                def __init__(self):
                    super().__init__("incomplete")

            _model = IncompleteModel()
