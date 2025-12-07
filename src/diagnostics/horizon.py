"""
Diagnostics by forecast horizon (days until election).

This module computes metrics grouped by how many days remain until Election Day.
It expects a stacked predictions DataFrame with ONE row per (state, forecast_date, model),
and at least these columns:

    - state
    - forecast_date
    - win_probability
    - predicted_margin
    - actual_margin
    - model

Election Day is assumed to be 2016-11-08 by default.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

ELECTION_DATE_DEFAULT = "2016-11-08"


def add_horizon_column(
    predictions: pd.DataFrame,
    election_date: Union[str, pd.Timestamp] = ELECTION_DATE_DEFAULT,
) -> pd.DataFrame:
    """
    Add a 'days_until_election' column to the predictions DataFrame.

    days_until_election = (election_date - forecast_date).days
    """
    df = predictions.copy()

    if "forecast_date" not in df.columns:
        raise ValueError("Expected column 'forecast_date' in predictions DataFrame.")

    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    election_date = pd.to_datetime(election_date)

    df["days_until_election"] = (election_date - df["forecast_date"]).dt.days.astype(
        int
    )
    return df


def _metrics_for_group(group: pd.DataFrame) -> pd.Series:
    """
    Compute horizon-level metrics for a single (model, days_until_election) group.
    """
    # Binary outcome: Democrat win?
    y = (group["actual_margin"] > 0).astype(float).to_numpy()
    p = group["win_probability"].to_numpy()

    # Margin predictions
    margin_pred = group["predicted_margin"].to_numpy()
    margin_true = group["actual_margin"].to_numpy()

    # Brier score
    brier = float(np.mean((p - y) ** 2))

    # Log loss (with clipping for numerical stability)
    eps = 1e-6
    p_clipped = np.clip(p, eps, 1 - eps)
    logloss = float(-np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)))

    # MAE for margins
    mae_margin = float(np.mean(np.abs(margin_pred - margin_true)))

    # Simple calibration-style summaries
    mean_pred_prob = float(np.mean(p))
    empirical_win_rate = float(np.mean(y))
    mean_predicted_margin = float(np.mean(margin_pred))
    mean_actual_margin = float(np.mean(margin_true))

    return pd.Series(
        {
            "n_obs": len(group),
            "brier_score": brier,
            "log_loss": logloss,
            "mae_margin": mae_margin,
            "mean_pred_prob": mean_pred_prob,
            "empirical_win_rate": empirical_win_rate,
            "mean_predicted_margin": mean_predicted_margin,
            "mean_actual_margin": mean_actual_margin,
        }
    )


def compute_horizon_metrics(
    predictions: pd.DataFrame,
    election_date: Union[str, pd.Timestamp] = ELECTION_DATE_DEFAULT,
) -> pd.DataFrame:
    """
    Compute diagnostics by forecast horizon.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must contain columns:
            - state
            - forecast_date
            - win_probability
            - predicted_margin
            - actual_margin
            - model
    election_date : str or Timestamp, optional
        Election date (default '2016-11-08').

    Returns
    -------
    pd.DataFrame
        One row per (model, days_until_election) with:
            - model
            - days_until_election
            - n_obs
            - brier_score
            - log_loss
            - mae_margin
            - mean_pred_prob
            - empirical_win_rate
            - mean_predicted_margin
            - mean_actual_margin
    """
    required_cols = {
        "state",
        "forecast_date",
        "win_probability",
        "predicted_margin",
        "actual_margin",
        "model",
    }
    missing = required_cols - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Predictions DataFrame is missing required columns: {sorted(missing)}"
        )

    df = add_horizon_column(predictions, election_date=election_date)

    metrics_df = (
        df.groupby(["model", "days_until_election"], as_index=False)
        .apply(_metrics_for_group)
        .reset_index(drop=True)
    )

    metrics_df = metrics_df.sort_values(["model", "days_until_election"]).reset_index(
        drop=True
    )

    return metrics_df
