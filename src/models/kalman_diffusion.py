#!/usr/bin/env python3
"""
Kalman Filter Diffusion Model with Improved Regularization
Brownian motion with drift + pollster biases + fundamentals prior
"""

import numpy as np
from src.models.base_model import ElectionForecastModel


class KalmanDiffusionModel(ElectionForecastModel):
    """Improved diffusion model with Kalman filter/RTS smoother"""

    def __init__(self, seed=None):
        super().__init__("kalman_diffusion", seed=seed)

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
            dt = max(dates[t] - dates[t - 1], 1.0)

            # Predict
            x_pred[t] = x_filt[t - 1] + mu * dt
            P_pred[t] = P_filt[t - 1] + sigma2 * dt

            # Update
            S = P_pred[t] + obs_variance[t]
            K = P_pred[t] / S
            x_filt[t] = x_pred[t] + K * (observations[t] - x_pred[t])
            P_filt[t] = (1.0 - K) * P_pred[t]

        # Backward RTS smoother
        x_smooth = np.copy(x_filt)
        P_smooth = np.copy(P_filt)

        for t in range(T - 2, -1, -1):
            dt = max(dates[t + 1] - dates[t], 1.0)
            denom = P_filt[t] + sigma2 * dt
            if denom <= 0:
                J = 0.0
            else:
                J = P_filt[t] / denom

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
        # Use recent 1/3 of polls (at least 10)
        recent_polls = state_polls.tail(max(len(state_polls) // 3, 10))

        dates = (
            recent_polls["middate"] - recent_polls["middate"].min()
        ).dt.days.values.astype(float)
        margins = recent_polls["margin"].values
        sample_sizes = recent_polls["samplesize"].values
        pollsters = recent_polls["pollster"].values

        # Observation variance: sampling error + extra noise
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

        # EM algorithm for drift mu and diffusion variance sigma2
        mu = 0.0
        sigma2 = 0.0005  # initial daily diffusion variance

        for _ in range(max_iter):
            x_smooth, P_smooth = self.kalman_filter_smoother(
                dates, adjusted_margins, obs_variance, mu, sigma2
            )

            # Keep variances in a sane range
            P_smooth = np.clip(P_smooth, 1e-8, 1.0)

            # M-step: update parameters
            mu_vals = []
            sigma2_vals = []
            for t in range(1, len(dates)):
                dt = dates[t] - dates[t - 1]
                if dt <= 0:
                    continue

                # Drift: change in state per day
                mu_vals.append((x_smooth[t] - x_smooth[t - 1]) / dt)

                # Diffusion variance candidate from smoothed variances
                sigma2_candidate = (P_smooth[t] + P_smooth[t - 1]) / dt
                sigma2_candidate = np.clip(sigma2_candidate, 1e-6, 0.005)
                sigma2_vals.append(sigma2_candidate)

            if mu_vals:
                mu = float(np.mean(mu_vals))
            else:
                mu = 0.0

            if sigma2_vals:
                sigma2 = float(np.mean(sigma2_vals))
            else:
                sigma2 = 0.0005

            # Final clamp on sigma2 to avoid explosions
            sigma2 = float(np.clip(sigma2, 1e-6, 0.005))

        # Incorporate fundamentals prior (weakly)
        prior_weight = 0.1
        x_smooth = (1.0 - prior_weight) * x_smooth + prior_weight * prior_mean

        return mu, sigma2, pollster_bias, x_smooth, P_smooth, dates

    def simulate_forward(self, x_start, P_start, mu, sigma2, days, N=2000, rng=None):
        """
        Simulate forward with Euler-Maruyama method

        Args:
            x_start: Initial state estimate
            P_start: Initial state variance
            mu: Drift parameter
            sigma2: Diffusion variance
            days: Number of days to simulate forward
            N: Number of simulation samples
            rng: NumPy random generator (default: None uses default_rng)

        Returns:
            Array of final margin values (length N), clipped to [-1, 1]
        """
        if rng is None:
            rng = np.random.default_rng()

        days = max(int(days), 0)

        # Ensure P_start and sigma2 are in a safe range
        P_start = float(max(P_start, 1e-10))
        sigma2 = float(np.clip(sigma2, 1e-6, 0.005))

        X = np.zeros((N, days + 1))
        X[:, 0] = rng.normal(x_start, np.sqrt(P_start), N)

        dt = 1.0
        for t in range(days):
            drift = mu * dt
            diffusion = np.sqrt(sigma2 * dt)
            dW = rng.normal(0.0, 1.0, N)
            X[:, t + 1] = X[:, t] + drift + diffusion * dW

            # Keep simulated paths within a generous band to avoid runaway paths
            X[:, t + 1] = np.clip(X[:, t + 1], -1.5, 1.5)

        final = X[:, -1]
        # Final clip to physically meaningful margin range [-1, 1]
        final = np.clip(final, -1.0, 1.0)
        return final

    def fit_and_forecast(
        self, state_polls, forecast_date, election_date, actual_margin, rng=None
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
        days_to_election = max(int(days_to_election), 0)

        # Add extra forecast uncertainty for the remaining time
        forecast_uncertainty = 0.001 * days_to_election
        P_current = max(P_current + forecast_uncertainty**2, 1e-10)

        final_margins = self.simulate_forward(
            x_current, P_current, mu, sigma2, days_to_election, N=2000, rng=rng
        )

        # Win probability
        win_prob = float(np.mean(final_margins > 0.0))
        win_prob = float(np.clip(win_prob, 0.01, 0.99))

        return {
            "win_probability": win_prob,
            "predicted_margin": float(np.mean(final_margins)),
            "margin_std": float(np.std(final_margins)),
        }


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging(__name__)

    model = KalmanDiffusionModel()
    pred_df = model.run_forecast()
    metrics_df = model.save_results()
    model.logger.info(f"Total predictions: {len(pred_df)}")
    model.logger.info(f"\n{metrics_df.to_string(index=False)}")
