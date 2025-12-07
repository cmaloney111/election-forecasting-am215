#!/usr/bin/env python3
"""
Base class for election forecasting models
"""

from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
from pathlib import Path
from abc import ABC, abstractmethod
from src.utils.data_utils import (
    load_polling_data,
    load_election_results,
    compute_metrics,
)
from src.utils.logging_config import get_logger

def plot_state(self, state: str) -> None:
    """
    Create time-series plot for a specific state showing model predictions over time.

    Args:
        state: Two-letter state code (e.g., 'FL', 'PA').

    Saves:
        PNG file to plots/{model_name}/{state}.png
    """
    # Load data
    polls, actual_margin = self.load_data()
    state_polls = polls[polls["state_code"] == state].copy()

    # Require at least a few polls to make the plot meaningful
    if len(state_polls) < 10:
        return

    # Get all predictions as a DataFrame
    pred_df = pd.DataFrame(self.predictions)
    if pred_df.empty:
        return

    # Filter predictions for this state
    state_preds = pred_df.loc[pred_df["state"] == state].copy()
    if state_preds.empty:
        return

    # Sort by forecast_date so the line plot is in time order
    state_preds = state_preds.sort_values(by="forecast_date")

    # Convert columns to numpy arrays so numpy / matplotlib / mypy are all happy
    forecast_dates = state_preds["forecast_date"].to_numpy()
    predicted_margins = state_preds["predicted_margin"].to_numpy(dtype=float)
    margin_stds = state_preds["margin_std"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Add uncertainty bands (90% CI) — plot first so it's in the background
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
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.5, linewidth=1, zorder=0)
    if state in actual_margin:
        ax.axhline(
            float(actual_margin[state]),
            color="red",
            linestyle="--",
            linewidth=2,
            label="Actual result",
            zorder=4,
        )

    # Set x-axis limits to focus on the forecast period (with some padding)
    # Use numpy datetime64 here to avoid pandas Timestamp typing issues
    if forecast_dates.size > 0:
        start_date = forecast_dates.min() - np.timedelta64(14, "D")
        end_date = forecast_dates.max() + np.timedelta64(2, "D")
        ax.set_xlim(start_date, end_date)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Democratic Margin (%)", fontsize=11)
    ax.set_title(
        f"{state} - {self.name} Forecast Evolution", fontsize=13, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, zorder=0)
    plt.tight_layout()

    # Ensure output directory exists and save
    Path(f"plots/{self.name}").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"plots/{self.name}/{state}.png")
    plt.close()
