"""Tests for forecasting models"""
import pytest
import pandas as pd
import numpy as np
from election_forecasting.models.poll_average import PollAverageModel
from election_forecasting.models.kalman_diffusion import KalmanDiffusionModel
from election_forecasting.models.improved_kalman import ImprovedKalmanModel
from election_forecasting.models.hierarchical_bayes import HierarchicalBayesModel


class TestPollAverageModel:
    """Tests for PollAverageModel"""

    def test_initialization(self):
        """Test model initialization"""
        model = PollAverageModel()
        assert model.name == "poll_average"

    def test_fit_and_forecast_returns_dict(self, sample_polls, election_date):
        """Test that fit_and_forecast returns proper structure"""
        model = PollAverageModel()
        forecast_date = pd.to_datetime("2016-10-15")

        result = model.fit_and_forecast(
            sample_polls, forecast_date, election_date, actual_margin=0.012
        )

        assert isinstance(result, dict)
        assert "win_probability" in result
        assert "predicted_margin" in result
        assert "margin_std" in result

    def test_fit_and_forecast_probability_range(self, sample_polls, election_date):
        """Test that win probability is in valid range"""
        model = PollAverageModel()
        forecast_date = pd.to_datetime("2016-10-15")

        result = model.fit_and_forecast(
            sample_polls, forecast_date, election_date, actual_margin=0.012
        )

        assert 0 <= result["win_probability"] <= 1

    def test_fit_and_forecast_with_few_polls(self, sample_polls, election_date):
        """Test forecast with limited polling data"""
        model = PollAverageModel()
        forecast_date = pd.to_datetime("2016-10-15")
        limited_polls = sample_polls.head(2)

        result = model.fit_and_forecast(
            limited_polls, forecast_date, election_date, actual_margin=0.012
        )

        assert isinstance(result, dict)
        assert "win_probability" in result


class TestKalmanDiffusionModel:
    """Tests for KalmanDiffusionModel"""

    def test_initialization(self):
        """Test model initialization"""
        model = KalmanDiffusionModel()
        assert model.name == "kalman_diffusion"

    def test_fit_and_forecast_returns_dict(self, sample_polls, election_date):
        """Test that fit_and_forecast returns proper structure"""
        model = KalmanDiffusionModel()
        forecast_date = pd.to_datetime("2016-10-15")

        result = model.fit_and_forecast(
            sample_polls, forecast_date, election_date, actual_margin=0.012
        )

        assert isinstance(result, dict)
        assert "win_probability" in result
        assert "predicted_margin" in result
        assert "margin_std" in result

    def test_kalman_filter_basic(self, sample_polls, election_date):
        """Test Kalman filter runs without errors"""
        model = KalmanDiffusionModel()
        forecast_date = pd.to_datetime("2016-10-15")

        result = model.fit_and_forecast(
            sample_polls, forecast_date, election_date, actual_margin=0.012
        )

        assert result["margin_std"] > 0

    def test_kalman_filter_smoother(self):
        """Test Kalman filter smoother runs without errors"""
        model = KalmanDiffusionModel()

        # Simple synthetic data
        dates = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        observations = np.array([0.0, 0.01, 0.02, 0.015, 0.025])
        obs_variance = np.array([0.001, 0.001, 0.001, 0.001, 0.001])
        mu = 0.005
        sigma2 = 0.0001

        x_smooth, P_smooth = model.kalman_filter_smoother(
            dates, observations, obs_variance, mu, sigma2
        )

        assert len(x_smooth) == len(dates)
        assert len(P_smooth) == len(dates)
        assert all(np.isfinite(x_smooth))
        assert all(P_smooth >= 0)


class TestImprovedKalmanModel:
    """Tests for ImprovedKalmanModel"""

    def test_initialization(self):
        """Test model initialization"""
        model = ImprovedKalmanModel()
        assert model.name == "improved_kalman"

    def test_fit_and_forecast_returns_dict(self, sample_polls, election_date):
        """Test that fit_and_forecast returns proper structure"""
        model = ImprovedKalmanModel()
        forecast_date = pd.to_datetime("2016-10-15")

        result = model.fit_and_forecast(
            sample_polls, forecast_date, election_date, actual_margin=0.012
        )

        assert isinstance(result, dict)
        assert "win_probability" in result
        assert "predicted_margin" in result
        assert "margin_std" in result

    def test_kalman_filter_rts(self):
        """Test Kalman filter RTS smoother"""
        model = ImprovedKalmanModel()

        # Synthetic data with trend
        dates = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        observations = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        obs_variance = np.array([0.001, 0.001, 0.001, 0.001, 0.001])
        mu = 0.01
        sigma2 = 0.0001

        x_smooth, P_smooth = model.kalman_filter_rts(
            dates, observations, obs_variance, mu, sigma2
        )

        assert np.isfinite(x_smooth)
        assert P_smooth > 0
        assert np.isfinite(P_smooth)


class TestHierarchicalBayesModel:
    """Tests for HierarchicalBayesModel"""

    def test_initialization(self):
        """Test model initialization"""
        model = HierarchicalBayesModel()
        assert model.name == "hierarchical_bayes"
        assert model.fundamentals is not None

    def test_fit_and_forecast_returns_dict(self, sample_polls, election_date):
        """Test that fit_and_forecast returns proper structure"""
        model = HierarchicalBayesModel()
        forecast_date = pd.to_datetime("2016-10-15")

        result = model.fit_and_forecast(
            sample_polls, forecast_date, election_date, actual_margin=0.012
        )

        assert isinstance(result, dict)
        assert "win_probability" in result
        assert "predicted_margin" in result
        assert "margin_std" in result

    def test_estimate_house_effects(self, sample_polls):
        """Test house effects estimation"""
        model = HierarchicalBayesModel()

        # Create multi-state polls for house effect estimation
        multi_state_polls = pd.concat([
            sample_polls,
            sample_polls.copy().assign(state_code="PA"),
        ])

        house_effects = model.estimate_house_effects(multi_state_polls)

        assert isinstance(house_effects, dict)
        assert all(isinstance(v, float) for v in house_effects.values())

    def test_kalman_filter_rts_smoother(self):
        """Test Kalman filter with RTS smoother"""
        model = HierarchicalBayesModel()

        dates = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        observations = np.array([0.0, 0.01, 0.02, 0.015, 0.025])
        obs_variance = np.array([0.001, 0.001, 0.001, 0.001, 0.001])
        mu = 0.005
        sigma2 = 0.0001

        x_smooth, P_smooth = model.kalman_filter_rts(
            dates, observations, obs_variance, mu, sigma2
        )

        assert np.isfinite(x_smooth)
        assert P_smooth > 0
        assert np.isfinite(P_smooth)

    def test_fit_and_forecast_with_fundamentals(self, sample_polls, election_date):
        """Test forecast incorporates fundamentals"""
        model = HierarchicalBayesModel()
        forecast_date = pd.to_datetime("2016-10-15")

        # Should work even if state not in fundamentals
        sample_polls["state_code"] = "XX"  # Non-existent state
        result = model.fit_and_forecast(
            sample_polls, forecast_date, election_date, actual_margin=0.012
        )

        assert isinstance(result, dict)
        assert "predicted_margin" in result


@pytest.mark.parametrize(
    "model_class",
    [PollAverageModel, KalmanDiffusionModel, ImprovedKalmanModel, HierarchicalBayesModel],
)
class TestAllModels:
    """Common tests for all models"""

    def test_model_has_correct_name(self, model_class):
        """Test that model has correct name attribute"""
        model = model_class()
        assert isinstance(model.name, str)
        assert len(model.name) > 0

    def test_model_can_run_full_forecast(self, model_class, sample_polls, sample_actual_results):
        """Test that model can run full forecast pipeline"""
        model = model_class()

        from unittest.mock import patch

        with patch.object(model, "load_data", return_value=(sample_polls, sample_actual_results)):
            forecast_dates = [pd.to_datetime("2016-10-15")]
            result = model.run_forecast(forecast_dates=forecast_dates, min_polls=5)

        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "win_probability" in result.columns
            assert "predicted_margin" in result.columns
