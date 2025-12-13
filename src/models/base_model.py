#!/usr/bin/env python3
"""
Base class for election forecasting models.

This version has been slightly generalized so that the election year and
election date are no longer hard-coded to 2016. Instead, the date is taken
from `src.utils.data_utils.get_current_election_date()`, which in turn is
controlled by `set_election_config(...)`.
"""

from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import matplotlib.pyplot as plt

from src.utils.data_utils import (
    load_polling_data,
    load_election_results,
    compute_metrics,
    get_current_election_date,
)
from src.utils.logging_config import get_logger


class ElectionForecastModel(ABC):
    """Abstract base class for election forecasting models."""

    def __init__(self, name: str, seed: Optional[int] = None) -> None:
        """
        Initialize the model.

        Args:
            name: Model name.
            seed: Random seed for reproducibility
                  (default: None for non-deterministic).
        """
        self.name = name
        self.predictions: List[Dict[str, Any]] = []
        self.logger = get_logger(f"{__name__}.{name}")
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def fit_and_forecast(
        self,
        state_polls: pd.DataFrame,
        forecast_date: pd.Timestamp,
        election_date: pd.Timestamp,
        actual_margin: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        """
        Fit model on polls up to `forecast_date` and predict election outcome.
        """
        raise NotImplementedError

    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Load polling and election results data.

        Returns:
            tuple of (polls DataFrame, actual_margin dict)
        """
        polls = load_polling_data()
        actual_margin = load_election_results()
        return polls, actual_margin

    def _forecast_single_date(
        self,
        forecast_date: pd.Timestamp,
        polls: pd.DataFrame,
        actual_margin: Dict[str, float],
        election_date: pd.Timestamp,
        min_polls: int,
        states: List[str],
    ) -> List[Dict[str, Any]]:
        """Helper method to forecast all states for a single date (for parallelization)."""
        results: List[Dict[str, Any]] = []

        for state in states:
            state_polls = polls[polls["state_code"] == state].copy()
            if len(state_polls) < min_polls:
                continue

            train_polls = state_polls[state_polls["middate"] <= forecast_date].copy()
            if len(train_polls) < min_polls:
                continue

            days_to_election = (election_date - forecast_date).days
            if days_to_election <= 0:
                continue

            try:
                state_margin = actual_margin.get(state, 0.0)
                result = self.fit_and_forecast(
                    train_polls,
                    forecast_date,
                    election_date,
                    state_margin,
                    rng=self.rng,
                )
                results.append(
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
                self.logger.error(
                    f"Error in {state} on {forecast_date.date()}: {e}"
                )

        return results

    def run_forecast(
        self,
        forecast_dates: Optional[List[pd.Timestamp]] = None,
        min_polls: int = 10,
        verbose: bool = False,
        n_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run forecast across multiple dates and states.
        """
        election_date_str = get_current_election_date()
        election_date = pd.to_datetime(election_date_str)
        election_year = int(election_date.year)

        if forecast_dates is None:
            default_dates = [
                f"{election_year}-10-01",
                f"{election_year}-10-15",
                f"{election_year}-11-01",
                f"{election_year}-11-07",
            ]
            forecast_dates = [pd.to_datetime(d) for d in default_dates]

        polls, actual_margin = self.load_data()
        states = [
            s for s in polls["state_code"].unique() if pd.notna(s) and s in actual_margin
        ]

        self.predictions = []

        if n_workers is None or n_workers <= 1:
            # Sequential execution
            for state in states:
                state_polls = polls[polls["state_code"] == state].copy()
                if len(state_polls) < min_polls:
                    continue

                if verbose:
                    self.logger.info(f"Processing {state}: {len(state_polls)} polls")

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
                        state_margin = actual_margin.get(state, 0.0)
                        result = self.fit_and_forecast(
                            train_polls,
                            forecast_date,
                            election_date,
                            state_margin,
                            rng=self.rng,
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
                        self.logger.error(
                            f"Error in {state} on {forecast_date.date()}: {e}"
                        )
                        continue
        else:
            # Parallel execution using ProcessPoolExecutor (parallelized by date)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {}
                for forecast_date in forecast_dates:
                    if verbose:
                        self.logger.info(
                            f"Submitting forecast for {forecast_date.date()}"
                        )
                    future = executor.submit(
                        self._forecast_single_date,
                        forecast_date,
                        polls,
                        actual_margin,
                        election_date,
                        min_polls,
                        states,
                    )
                    futures[future] = forecast_date

                for future in as_completed(futures):
                    forecast_date = futures[future]
                    try:
                        date_results = future.result()
                        self.predictions.extend(date_results)
                        if verbose:
                            self.logger.info(
                                f"Completed {forecast_date.date()} "
                                f"({len(date_results)} predictions)"
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Failed to process {forecast_date.date()}: {e}"
                        )

        return pd.DataFrame(self.predictions)

    def save_results(self) -> pd.DataFrame:
        """
        Save predictions and metrics to CSV and text files.
        """
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

    def plot_state(self, state: str) -> None:
        """
        Create time-series plot for a specific state showing model predictions over time.
        """
        polls, actual_margin = self.load_data()
        state_polls = polls[polls["state_code"] == state].copy()

        if len(state_polls) < 10:
            return

        pred_df = pd.DataFrame(self.predictions)
        if pred_df.empty:
            return

        state_preds = pred_df[pred_df["state"] == state].copy()
        if state_preds.empty:
            return

        state_preds = state_preds.sort_values("forecast_date")

        fig, ax = plt.subplots(figsize=(12, 6))

        forecast_dates = pd.to_datetime(state_preds["forecast_date"].values)
        predicted_margins = state_preds["predicted_margin"].values
        margin_stds = state_preds["margin_std"].values

        ax.fill_between(
            forecast_dates,
            predicted_margins - 1.645 * margin_stds,
            predicted_margins + 1.645 * margin_stds,
            alpha=0.25,
            color="lightblue",
            label="90% confidence interval",
            zorder=1,
        )

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

        # Use the currently configured election date to set x-limits
        election_date = pd.to_datetime(get_current_election_date())
        start_date = forecast_dates.min() - pd.Timedelta(days=14)
        ax.set_xlim(start_date, election_date + pd.Timedelta(days=2))

        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Democratic Margin (%)", fontsize=11)
        ax.set_title(
            f"{state} - {self.name} Forecast Evolution",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3, zorder=0)
        plt.tight_layout()

        # === NEW: save into year-specific subfolder, e.g. plots/hierarchical_bayes/2016/FL.png ===
        election_year = int(election_date.year)
        out_dir = Path("plots") / self.name / str(election_year)
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"{state}.png")
        plt.close()
