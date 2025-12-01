"""Tests for data utility functions"""
import pytest
import pandas as pd
import numpy as np
from election_forecasting.utils.data_utils import (
    load_polling_data,
    load_election_results,
    load_fundamentals,
    compute_metrics,
    get_state_list,
)


class TestLoadPollingData:
    """Tests for load_polling_data function"""

    def test_load_polling_data_returns_dataframe(self):
        """Test that load_polling_data returns a DataFrame"""
        polls = load_polling_data()
        assert isinstance(polls, pd.DataFrame)

    def test_polling_data_has_required_columns(self):
        """Test that polling data has all required columns"""
        polls = load_polling_data()
        required_cols = [
            "middate",
            "dem",
            "rep",
            "margin",
            "dem_proportion",
            "samplesize",
            "pollster",
            "state_code",
        ]
        for col in required_cols:
            assert col in polls.columns

    def test_polling_data_margin_calculation(self):
        """Test that margin is correctly calculated"""
        polls = load_polling_data()
        # Check a few samples
        sample = polls.head(10)
        expected_margin = (sample["dem"] - sample["rep"]) / (
            sample["dem"] + sample["rep"]
        )
        np.testing.assert_allclose(sample["margin"], expected_margin, rtol=1e-10)

    def test_polling_data_dates_are_datetime(self):
        """Test that dates are properly converted to datetime"""
        polls = load_polling_data()
        assert pd.api.types.is_datetime64_any_dtype(polls["middate"])

    def test_polling_data_state_codes_valid(self):
        """Test that state codes are two-letter codes"""
        polls = load_polling_data()
        state_codes = polls["state_code"].dropna().unique()
        for code in state_codes:
            assert len(code) == 2
            assert code.isupper()


class TestLoadElectionResults:
    """Tests for load_election_results function"""

    def test_load_election_results_returns_dict(self):
        """Test that load_election_results returns a dict"""
        results = load_election_results()
        assert isinstance(results, dict)

    def test_election_results_has_states(self):
        """Test that election results contain expected states"""
        results = load_election_results()
        assert len(results) > 40  # Should have most states

    def test_election_results_margins_valid_range(self):
        """Test that margins are in valid range [-1, 1]"""
        results = load_election_results()
        for state, margin in results.items():
            assert -1 <= margin <= 1

    def test_election_results_key_battleground_states(self):
        """Test that key battleground states are present"""
        results = load_election_results()
        battleground_states = ["FL", "PA", "MI", "WI", "NC", "AZ"]
        for state in battleground_states:
            assert state in results


class TestLoadFundamentals:
    """Tests for load_fundamentals function"""

    def test_load_fundamentals_returns_dict(self):
        """Test that load_fundamentals returns a dict"""
        fundamentals = load_fundamentals()
        assert isinstance(fundamentals, dict)

    def test_fundamentals_structure(self):
        """Test that fundamentals have correct structure"""
        fundamentals = load_fundamentals()
        for state, data in fundamentals.items():
            assert "margin" in data
            assert "margin_2012" in data
            assert isinstance(data["margin"], float)

    def test_fundamentals_weighted_average(self):
        """Test that margin is weighted average of 2012 and 2008"""
        fundamentals = load_fundamentals()
        for state, data in fundamentals.items():
            if data["margin_2008"] is not None:
                expected = 0.7 * data["margin_2012"] + 0.3 * data["margin_2008"]
                assert abs(data["margin"] - expected) < 1e-10


class TestComputeMetrics:
    """Tests for compute_metrics function"""

    def test_compute_metrics_basic(self):
        """Test basic metrics computation"""
        pred_df = pd.DataFrame({
            "forecast_date": [pd.to_datetime("2016-10-15")] * 5,
            "state": ["FL", "PA", "MI", "WI", "NC"],
            "win_probability": [0.6, 0.4, 0.45, 0.3, 0.55],
            "predicted_margin": [0.02, -0.01, -0.005, -0.02, 0.01],
            "actual_margin": [0.012, -0.007, -0.002, -0.008, -0.036],
        })

        metrics = compute_metrics(pred_df)
        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) == 1

    def test_compute_metrics_columns(self):
        """Test that metrics have required columns"""
        pred_df = pd.DataFrame({
            "forecast_date": [pd.to_datetime("2016-10-15")] * 3,
            "state": ["FL", "PA", "MI"],
            "win_probability": [0.6, 0.4, 0.5],
            "predicted_margin": [0.02, -0.01, 0.0],
            "actual_margin": [0.012, -0.007, -0.002],
        })

        metrics = compute_metrics(pred_df)
        required_cols = ["forecast_date", "n_states", "brier_score", "log_loss", "mae_margin"]
        for col in required_cols:
            assert col in metrics.columns

    def test_compute_metrics_multiple_dates(self):
        """Test metrics computation for multiple forecast dates"""
        pred_df = pd.DataFrame({
            "forecast_date": [pd.to_datetime("2016-10-15")] * 3
            + [pd.to_datetime("2016-11-01")] * 3,
            "state": ["FL", "PA", "MI"] * 2,
            "win_probability": [0.6, 0.4, 0.5] * 2,
            "predicted_margin": [0.02, -0.01, 0.0] * 2,
            "actual_margin": [0.012, -0.007, -0.002] * 2,
        })

        metrics = compute_metrics(pred_df)
        assert len(metrics) == 2

    def test_compute_metrics_brier_score_range(self):
        """Test that Brier score is in valid range [0, 1]"""
        pred_df = pd.DataFrame({
            "forecast_date": [pd.to_datetime("2016-10-15")] * 3,
            "state": ["FL", "PA", "MI"],
            "win_probability": [0.6, 0.4, 0.5],
            "predicted_margin": [0.02, -0.01, 0.0],
            "actual_margin": [0.012, -0.007, -0.002],
        })

        metrics = compute_metrics(pred_df)
        assert 0 <= metrics["brier_score"].iloc[0] <= 1


class TestGetStateList:
    """Tests for get_state_list function"""

    def test_get_state_list(self, sample_polls, sample_actual_results):
        """Test getting list of valid states"""
        states = get_state_list(sample_polls, sample_actual_results)
        assert isinstance(states, list)
        assert "FL" in states

    def test_get_state_list_filters_missing_results(self, sample_polls):
        """Test that states without results are filtered"""
        actual_results = {"PA": -0.007}
        states = get_state_list(sample_polls, actual_results)
        assert "FL" not in states
        assert "PA" not in states  # FL polls, PA results - no match
