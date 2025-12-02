#!/usr/bin/env python3
"""
Simple Poll-of-Polls Average Model
Weighted average of recent polls with empirical uncertainty
"""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import norm
from src.models.base_model import ElectionForecastModel


class PollAverageModel(ElectionForecastModel):
    """Simple weighted poll average baseline"""

    def __init__(self, seed=None):
        super().__init__("poll_average", seed=seed)

    def fit_and_forecast(
        self, state_polls, forecast_date, election_date, actual_margin, rng=None
    ):
        """Compute weighted poll average with empirical uncertainty"""
        window_days = 14
        cutoff = forecast_date - pd.Timedelta(days=window_days)
        recent_polls = state_polls[state_polls["middate"] >= cutoff].copy()

        if len(recent_polls) < 3:
            recent_polls = state_polls.tail(5)

        # Weight by sample size
        weights = recent_polls["samplesize"].values
        weights = weights / weights.sum()

        avg_margin = np.average(recent_polls["margin"].values, weights=weights)

        # Uncertainty estimation
        empirical_std = np.std(recent_polls["margin"].values, ddof=1)
        avg_sample_size = np.average(recent_polls["samplesize"].values, weights=weights)
        sampling_std = 1.0 / np.sqrt(avg_sample_size)

        # Forecast horizon uncertainty
        days_to_election = (election_date - forecast_date).days
        horizon_uncertainty = 0.001 * days_to_election

        # Combine uncertainties
        total_std = max(empirical_std, sampling_std, 0.02)
        total_std = np.sqrt(total_std**2 + horizon_uncertainty**2)

        # Win probability via normal CDF
        win_prob = norm.cdf(avg_margin / total_std)
        win_prob = np.clip(win_prob, 0.05, 0.95)

        return {
            "win_probability": win_prob,
            "predicted_margin": avg_margin,
            "margin_std": total_std,
        }


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging(__name__)

    model = PollAverageModel()
    pred_df = model.run_forecast()
    metrics_df = model.save_results()
    model.logger.info(f"Total predictions: {len(pred_df)}")
    model.logger.info(f"\n{metrics_df.to_string(index=False)}")
