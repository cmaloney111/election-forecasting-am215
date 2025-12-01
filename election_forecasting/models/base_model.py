#!/usr/bin/env python3
"""
Base class for election forecasting models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from abc import ABC, abstractmethod
from election_forecasting.utils.data_utils import (
    load_polling_data,
    load_election_results,
    compute_metrics,
)


class ElectionForecastModel(ABC):
    """Abstract base class for election forecasting models"""

    def __init__(self, name):
        self.name = name
        self.predictions = []

    @abstractmethod
    def fit_and_forecast(
        self, state_polls, forecast_date, election_date, actual_margin
    ):
        """
        Fit model on polls up to forecast_date and predict election outcome.

        Args:
            state_polls: DataFrame with columns [middate, dem_proportion, margin, samplesize, pollster]
            forecast_date: pd.Timestamp - date to make forecast from
            election_date: pd.Timestamp - election day
            actual_margin: float - actual two-party margin (for evaluation)

        Returns:
            dict with keys: win_probability, predicted_margin, margin_std
        """
        pass

    def load_data(self):
        """
        Load polling and election results data

        Returns:
            tuple of (polls DataFrame, actual_margin dict)
        """
        polls = load_polling_data()
        actual_margin = load_election_results()
        return polls, actual_margin

    def run_forecast(self, forecast_dates=None, min_polls=10, verbose=False):
        """
        Run forecast across multiple dates and states

        Args:
            forecast_dates: List of pd.Timestamp dates to forecast from (default: 4 dates in Oct-Nov 2016)
            min_polls: Minimum number of polls required to forecast a state
            verbose: If True, print processing status for each state

        Returns:
            DataFrame with columns: state, forecast_date, win_probability, predicted_margin, margin_std, actual_margin
        """
        if forecast_dates is None:
            forecast_dates = [
                pd.to_datetime(d)
                for d in ["2016-10-01", "2016-10-15", "2016-11-01", "2016-11-07"]
            ]

        polls, actual_margin = self.load_data()
        election_date = pd.to_datetime("2016-11-08")

        states = [
            s
            for s in polls["state_code"].unique()
            if pd.notna(s) and s in actual_margin
        ]

        self.predictions = []

        for state in states:
            state_polls = polls[polls["state_code"] == state].copy()
            if len(state_polls) < min_polls:
                continue

            if verbose:
                print(f"Processing {state}: {len(state_polls)} polls")

            for forecast_date in forecast_dates:
                train_polls = state_polls[
                    state_polls["middate"] <= forecast_date
                ].copy()
                if len(train_polls) < min_polls:
                    continue

                days_to_election = (election_date - forecast_date).days
                if days_to_election <= 0:
                    continue

                try:
                    result = self.fit_and_forecast(
                        train_polls,
                        forecast_date,
                        election_date,
                        actual_margin.get(state),
                    )

                    self.predictions.append(
                        {
                            "state": state,
                            "forecast_date": forecast_date,
                            "win_probability": result["win_probability"],
                            "predicted_margin": result["predicted_margin"],
                            "margin_std": result.get("margin_std", np.nan),
                            "actual_margin": actual_margin.get(state, np.nan),
                        }
                    )
                except Exception as e:
                    print(f"  Error in {state} on {forecast_date.date()}: {e}")
                    continue

        return pd.DataFrame(self.predictions)

    def save_results(self):
        """
        Save predictions and metrics to CSV and text files

        Creates predictions/{model_name}.csv and metrics/{model_name}.txt

        Returns:
            DataFrame with columns: forecast_date, n_states, brier_score, log_loss, mae_margin
        """
        # Create output directories if they don't exist
        Path("predictions").mkdir(parents=True, exist_ok=True)
        Path("metrics").mkdir(parents=True, exist_ok=True)

        pred_df = pd.DataFrame(self.predictions)
        pred_df.to_csv(f"predictions/{self.name}.csv", index=False)

        metrics_df = compute_metrics(pred_df)

        with open(f"metrics/{self.name}.txt", "w") as f:
            f.write(f"{self.name} - Evaluation Metrics\n")
            for _, row in metrics_df.iterrows():
                f.write(f"Forecast Date: {row['forecast_date']}\n")
                f.write(f"  States: {row['n_states']}\n")
                f.write(f"  Brier Score: {row['brier_score']:.4f}\n")
                f.write(f"  Log Loss: {row['log_loss']:.4f}\n")
                f.write(f"  MAE (Margin): {row['mae_margin']:.4f}\n\n")

        return metrics_df

    def plot_state(self, state):
        """
        Create time-series plot for a specific state showing model predictions over time

        Args:
            state: Two-letter state code (e.g., 'FL', 'PA')

        Saves:
            PNG file to plots/{model_name}/{state}.png
        """
        polls, actual_margin = self.load_data()
        state_polls = polls[polls["state_code"] == state].copy()

        if len(state_polls) < 10:
            return

        # Get predictions for this state
        pred_df = pd.DataFrame(self.predictions)
        if len(pred_df) == 0:
            return

        state_preds = pred_df[pred_df["state"] == state].copy()
        if len(state_preds) == 0:
            return

        state_preds = state_preds.sort_values("forecast_date")

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot model predictions over time
        forecast_dates = pd.to_datetime(state_preds["forecast_date"].values)
        predicted_margins = state_preds["predicted_margin"].values
        margin_stds = state_preds["margin_std"].values

        # Add uncertainty bands (90% CI); plot first so it's in background
        ax.fill_between(
            forecast_dates,
            predicted_margins - 1.645 * margin_stds,
            predicted_margins + 1.645 * margin_stds,
            alpha=0.25,
            color="lightblue",
            label="90% confidence interval",
            zorder=1,
        )

        # Plot raw polls
        ax.scatter(
            state_polls["middate"],
            state_polls["margin"],
            alpha=0.5,
            s=40,
            label="Raw polls",
            color="gray",
            zorder=2,
            marker="o",
        )

        # Plot model forecast line on top of everything
        ax.plot(
            forecast_dates,
            predicted_margins,
            "b-o",
            linewidth=3,
            markersize=10,
            label=f"{self.name} forecast",
            zorder=4,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

        # Add reference lines
        ax.axhline(0, color="k", linestyle="--", alpha=0.5, linewidth=1, zorder=0)
        if state in actual_margin:
            ax.axhline(
                actual_margin[state],
                color="red",
                linestyle="--",
                linewidth=2,
                label="Actual result",
                zorder=4,
            )

        # Set x-axis limits to focus on the forecast period (with some padding)
        election_date = pd.to_datetime("2016-11-08")
        start_date = forecast_dates.min() - pd.Timedelta(days=14)
        ax.set_xlim(start_date, election_date + pd.Timedelta(days=2))

        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Democratic Margin (%)", fontsize=11)
        ax.set_title(
            f"{state} - {self.name} Forecast Evolution", fontsize=13, fontweight="bold"
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3, zorder=0)
        plt.tight_layout()

        Path(f"plots/{self.name}").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"plots/{self.name}/{state}.png")
        plt.close()
