#!/usr/bin/env python3
"""
Kalman Filter Diffusion Model with Improved Regularization
Brownian motion with drift + pollster biases + fundamentals prior
"""

import numpy as np
from election_forecasting.models.base_model import ElectionForecastModel


class KalmanDiffusionModel(ElectionForecastModel):
    """Improved diffusion model with Kalman filter/RTS smoother"""

    def __init__(self):
        super().__init__("kalman_diffusion")

    def kalman_filter_smoother(self, dates, observations, obs_variance, mu, sigma2):
        """
        Kalman filter + RTS smoother for Brownian motion with drift

        Args:
            dates: Array of time points (in days)
            observations: Array of poll margins
            obs_variance: Array of observation variances
            mu: Drift parameter
            sigma2: Diffusion variance

        Returns:
            tuple of (x_smooth, P_smooth): smoothed state estimates and variances
        """
        T = len(dates)
        x_filt = np.zeros(T)
        P_filt = np.zeros(T)
        x_pred = np.zeros(T)
        P_pred = np.zeros(T)

        # Initial state
        x_filt[0] = observations[0]
        P_filt[0] = obs_variance[0]

        # Forward filter
        for t in range(1, T):
            dt = dates[t] - dates[t - 1]
            x_pred[t] = x_filt[t - 1] + mu * dt
            P_pred[t] = P_filt[t - 1] + sigma2 * dt

            K = P_pred[t] / (P_pred[t] + obs_variance[t])
            x_filt[t] = x_pred[t] + K * (observations[t] - x_pred[t])
            P_filt[t] = (1 - K) * P_pred[t]

        # Backward RTS smoother
        x_smooth = np.copy(x_filt)
        P_smooth = np.copy(P_filt)

        for t in range(T - 2, -1, -1):
            dt = dates[t + 1] - dates[t]
            J = P_filt[t] / (P_filt[t] + sigma2 * dt)
            x_smooth[t] = x_filt[t] + J * (x_smooth[t + 1] - x_filt[t] - mu * dt)
            P_smooth[t] = P_filt[t] + J**2 * (P_smooth[t + 1] - P_filt[t] - sigma2 * dt)

        return x_smooth, P_smooth

    def fit_state_diffusion(self, state_polls, prior_mean=0.0, max_iter=10):
        """
        Fit diffusion model with EM algorithm

        Args:
            state_polls: DataFrame of polls for a single state
            prior_mean: Prior mean for fundamentals
            max_iter: Maximum number of EM iterations

        Returns:
            tuple of (mu, sigma2, pollster_bias, x_smooth, P_smooth, dates)
        """
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

        # Estimate pollster biases with regularization
        pollster_bias = {}
        shrinkage = 0.5
        for pol in np.unique(pollsters):
            mask = pollsters == pol
            if np.sum(mask) >= 2:
                raw_bias = np.mean(margins[mask]) - np.mean(margins)
                pollster_bias[pol] = shrinkage * raw_bias
            else:
                pollster_bias[pol] = 0.0

        # Adjust for pollster bias
        adjusted_margins = margins - np.array([pollster_bias[p] for p in pollsters])

        # EM algorithm
        mu = 0.0
        sigma2 = 0.0005

        for iteration in range(max_iter):
            x_smooth, P_smooth = self.kalman_filter_smoother(
                dates, adjusted_margins, obs_variance, mu, sigma2
            )

            # M-step: update parameters
            mu_vals = []
            sigma2_vals = []
            for t in range(1, len(dates)):
                dt = dates[t] - dates[t - 1]
                if dt > 0:
                    mu_vals.append((x_smooth[t] - x_smooth[t - 1]) / dt)
                    sigma2_vals.append(max((P_smooth[t] + P_smooth[t - 1]) / dt, 1e-6))

            mu = np.mean(mu_vals) if mu_vals else 0.0
            sigma2 = max(np.mean(sigma2_vals), 0.0005) if sigma2_vals else 0.0005

        # Incorporate fundamentals prior
        prior_weight = 0.1
        x_smooth = (1 - prior_weight) * x_smooth + prior_weight * prior_mean

        return mu, sigma2, pollster_bias, x_smooth, P_smooth, dates

    def simulate_forward(self, x_start, P_start, mu, sigma2, days, N=2000):
        """
        Simulate forward with Euler-Maruyama method

        Args:
            x_start: Initial state estimate
            P_start: Initial state variance
            mu: Drift parameter
            sigma2: Diffusion variance
            days: Number of days to simulate forward
            N: Number of simulation samples

        Returns:
            Array of final margin values (length N)
        """
        X = np.zeros((N, days + 1))
        X[:, 0] = np.random.normal(x_start, np.sqrt(P_start), N)

        dt = 1.0
        for t in range(days):
            drift = mu * dt
            diffusion = np.sqrt(max(sigma2 * dt, 0))
            dW = np.random.normal(0, 1, N)
            X[:, t + 1] = X[:, t] + drift + diffusion * dW

        return X[:, -1]

    def fit_and_forecast(
        self, state_polls, forecast_date, election_date, actual_margin
    ):
        """Fit Kalman diffusion and forecast election outcome"""
        mu, sigma2, pollster_bias, x_smooth, P_smooth, dates = self.fit_state_diffusion(
            state_polls, prior_mean=0.0
        )

        # Current state estimate
        x_current = x_smooth[-1]
        P_current = P_smooth[-1]

        # Forecast forward
        days_to_election = (election_date - forecast_date).days
        forecast_uncertainty = 0.001 * days_to_election
        P_current = P_current + forecast_uncertainty**2

        final_margins = self.simulate_forward(
            x_current, P_current, mu, sigma2, days_to_election, N=2000
        )

        # Win probability
        win_prob = np.mean(final_margins > 0)
        win_prob = np.clip(win_prob, 0.01, 0.99)

        return {
            "win_probability": win_prob,
            "predicted_margin": np.mean(final_margins),
            "margin_std": np.std(final_margins),
        }


if __name__ == "__main__":
    model = KalmanDiffusionModel()
    pred_df = model.run_forecast()
    metrics_df = model.save_results()
    print(f"\nTotal predictions: {len(pred_df)}")
    print(metrics_df.to_string(index=False))
