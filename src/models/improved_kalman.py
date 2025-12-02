#!/usr/bin/env python3
"""
Improved Kalman Diffusion Model

Key improvements over basic Kalman:
- Increased minimum diffusion variance
- Better regularized pollster biases
- Smaller forecast horizon uncertainty
- More conservative probability clipping
"""

import warnings
import numpy as np
from src.models.base_model import ElectionForecastModel


class ImprovedKalmanModel(ElectionForecastModel):
    """Improved Kalman filter diffusion model"""

    def __init__(self, seed=None):
        super().__init__("improved_kalman", seed=seed)

    def kalman_filter_rts(self, dates, observations, obs_variance, mu, sigma2):
        """Kalman filter + RTS smoother"""
        T = len(dates)
        x_filt = np.zeros(T)
        P_filt = np.zeros(T)

        # Initial state
        x_filt[0] = observations[0]
        P_filt[0] = obs_variance[0]

        # Forward filter
        for t in range(1, T):
            dt = max(dates[t] - dates[t - 1], 1.0)

            # Predict
            x_pred = x_filt[t - 1] + mu * dt
            P_pred = P_filt[t - 1] + sigma2 * dt

            # Update
            K = P_pred / (P_pred + obs_variance[t])
            x_filt[t] = x_pred + K * (observations[t] - x_pred)
            P_filt[t] = (1 - K) * P_pred

        # Backward RTS smoother
        x_smooth = np.copy(x_filt)
        P_smooth = np.copy(P_filt)

        for t in range(T - 2, -1, -1):
            dt = max(dates[t + 1] - dates[t], 1.0)
            P_pred = P_filt[t] + sigma2 * dt

            if P_pred > 0:
                J = P_filt[t] / P_pred
                x_smooth[t] = x_filt[t] + J * (x_smooth[t + 1] - x_filt[t] - mu * dt)
                P_smooth[t] = P_filt[t] + J**2 * (P_smooth[t + 1] - P_pred)

        return x_smooth[-1], max(P_smooth[-1], 1e-6)

    def fit_and_forecast(
        self, state_polls, forecast_date, election_date, actual_margin, rng=None
    ):
        """Fit improved Kalman diffusion and forecast"""

        # Use recent 1/3 of polls
        recent_polls = state_polls.tail(max(len(state_polls) // 3, 10))

        dates = (
            recent_polls["middate"] - recent_polls["middate"].min()
        ).dt.days.values.astype(float)
        margins = recent_polls["margin"].values
        sample_sizes = recent_polls["samplesize"].values
        pollsters = recent_polls["pollster"].values

        # Observation variance
        tau_extra2 = 0.015**2
        obs_variance = 1.0 / sample_sizes + tau_extra2

        # Estimate pollster biases with STRONGER regularization
        pollster_bias = {}
        shrinkage = 0.7  # Increased from 0.5
        for pol in np.unique(pollsters):
            mask = pollsters == pol
            if np.sum(mask) >= 2:
                raw_bias = np.mean(margins[mask]) - np.mean(margins)
                pollster_bias[pol] = shrinkage * raw_bias
            else:
                pollster_bias[pol] = 0.0

        # Adjust for pollster bias
        adjusted_margins = margins - np.array([pollster_bias[p] for p in pollsters])

        # Simple parameter estimation
        mu = 0.0  # Assume no systematic drift
        sigma2 = 0.0008**2  # Increased minimum diffusion (was 0.0005)

        poll_mean, poll_var = self.kalman_filter_rts(
            dates, adjusted_margins, obs_variance, mu, sigma2
        )

        # Forecast forward
        days_to_election = (election_date - forecast_date).days

        # REDUCED forecast horizon uncertainty (was 0.001)
        forecast_uncertainty = 0.0005 * days_to_election
        P_current = poll_var + forecast_uncertainty**2

        # Simulate forward
        final_margins = self.simulate_forward(
            poll_mean, P_current, mu, sigma2, days_to_election, N=2000, rng=rng
        )

        # Win probability with tighter clipping
        win_prob = np.mean(final_margins > 0)
        win_prob = np.clip(win_prob, 0.01, 0.99)

        return {
            "win_probability": win_prob,
            "predicted_margin": np.mean(final_margins),
            "margin_std": np.std(final_margins),
        }

    def simulate_forward(self, x_start, P_start, mu, sigma2, days, N=2000, rng=None):
        """Simulate forward with Euler-Maruyama

        Args:
            x_start: Initial state estimate
            P_start: Initial state variance
            mu: Drift parameter
            sigma2: Diffusion variance
            days: Number of days to simulate forward
            N: Number of simulation samples
            rng: NumPy random generator (default: None uses default_rng)

        Returns:
            Array of final margin values (length N)
        """
        if rng is None:
            rng = np.random.default_rng()

        X = np.zeros((N, days + 1))
        X[:, 0] = rng.normal(x_start, np.sqrt(max(P_start, 0)), N)

        dt = 1.0
        for t in range(days):
            drift = mu * dt
            diffusion = np.sqrt(max(sigma2 * dt, 0))
            dW = rng.normal(0, 1, N)
            X[:, t + 1] = X[:, t] + drift + diffusion * dW

        return X[:, -1]


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    warnings.filterwarnings("ignore")
    setup_logging(__name__)

    model = ImprovedKalmanModel()
    pred_df = model.run_forecast()
    metrics_df = model.save_results()
    model.logger.info(f"Total predictions: {len(pred_df)}")
    model.logger.info(f"\n{metrics_df.to_string(index=False)}")
