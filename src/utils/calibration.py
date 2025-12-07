"""
Calibration utilities for election-forecasting-am215.

These functions work with the predictions CSVs produced by the models.
Each predictions file is expected to have at least:

    - "win_probability": model's probability of Democratic win (0–1)
    - "actual_margin": realized Democratic two-party margin (% points),
                       where > 0 means Democratic win, < 0 Republican.

We convert margins to a binary outcome (Dem win = 1, loss = 0) and
compute reliability / calibration curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


@dataclass
class CalibrationResult:
    """Summary calibration statistics for a set of predictions."""

    reliability: DataFrame
    brier_score: float
    log_loss: float


def _prepare_binary_outcomes(
    predictions_df: DataFrame,
    prob_col: str = "win_probability",
    margin_col: str = "actual_margin",
) -> DataFrame:
    """
    Return a clean DataFrame with probability and binary outcome.

    outcome = 1 if actual_margin > 0 (Democratic win), else 0.
    """
    if prob_col not in predictions_df.columns:
        raise KeyError(f"Column '{prob_col}' not found in predictions_df")

    if margin_col not in predictions_df.columns:
        raise KeyError(f"Column '{margin_col}' not found in predictions_df")

    # Keep only the columns we need
    df: DataFrame = predictions_df[[prob_col, margin_col]].copy()  # type: ignore[assignment]

    # Drop rows with missing values (DataFrame.dropna with subset)
    df = df.dropna(subset=[prob_col, margin_col])

    # Binary outcome: Dem win = 1, loss = 0
    outcome: Series = (df[margin_col] > 0).astype(float)

    # Ensure probabilities in [0, 1]
    prob: Series = df[prob_col].astype(float).clip(0.0, 1.0)

    # Build and return a new DataFrame with consistent column types
    result: DataFrame = pd.DataFrame({"prob": prob, "outcome": outcome})
    return result


def compute_reliability_curve(
    predictions_df: DataFrame,
    prob_col: str = "win_probability",
    margin_col: str = "actual_margin",
    n_bins: int = 10,
) -> DataFrame:
    """
    Compute a reliability (calibration) curve.

    Returns
    -------
    DataFrame with columns:
        - bin_lower
        - bin_upper
        - count
        - mean_predicted
        - empirical_win_rate
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    df = _prepare_binary_outcomes(predictions_df, prob_col, margin_col)

    # Define bin edges 0..1
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    df = df.copy()
    df["bin"] = pd.cut(df["prob"], bin_edges, include_lowest=True)

    grouped = df.groupby("bin", observed=True)

    reliability: DataFrame = grouped.agg(
        count=("outcome", "size"),
        mean_predicted=("prob", "mean"),
        empirical_win_rate=("outcome", "mean"),
    ).reset_index()

    # Drop empty bins (count == 0) if any
    reliability = reliability[reliability["count"] > 0]  # type: ignore[assignment]

    # Extract numeric bounds of each interval for convenience
    bins: List[pd.Interval] = list(reliability["bin"])
    bin_lower: List[float] = [float(b.left) for b in bins]
    bin_upper: List[float] = [float(b.right) for b in bins]

    # Assign new columns instead of reassigning reliability to a Series
    reliability["bin_lower"] = bin_lower
    reliability["bin_upper"] = bin_upper

    # Order columns
    ordered: DataFrame = reliability[  # type: ignore[assignment]
        ["bin_lower", "bin_upper", "count", "mean_predicted", "empirical_win_rate"]
    ]  # type: ignore[assignment]

    return ordered


def compute_brier_score(
    predictions_df: DataFrame,
    prob_col: str = "win_probability",
    margin_col: str = "actual_margin",
) -> float:
    """Compute the Brier score for binary outcomes."""
    df = _prepare_binary_outcomes(predictions_df, prob_col, margin_col)
    return float(np.mean((df["prob"] - df["outcome"]) ** 2))


def compute_log_loss(
    predictions_df: DataFrame,
    prob_col: str = "win_probability",
    margin_col: str = "actual_margin",
    eps: float = 1e-15,
) -> float:
    """Compute the (natural-log) log loss for binary outcomes."""
    df = _prepare_binary_outcomes(predictions_df, prob_col, margin_col)
    p = df["prob"].clip(eps, 1.0 - eps)
    y = df["outcome"]
    log_loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(log_loss)


def summarize_calibration(
    predictions_df: DataFrame,
    prob_col: str = "win_probability",
    margin_col: str = "actual_margin",
    n_bins: int = 10,
) -> CalibrationResult:
    """Convenience wrapper to compute curve + scalar scores."""
    reliability = compute_reliability_curve(
        predictions_df,
        prob_col=prob_col,
        margin_col=margin_col,
        n_bins=n_bins,
    )
    brier = compute_brier_score(
        predictions_df,
        prob_col=prob_col,
        margin_col=margin_col,
    )
    log_loss_val = compute_log_loss(
        predictions_df,
        prob_col=prob_col,
        margin_col=margin_col,
    )
    return CalibrationResult(
        reliability=reliability,
        brier_score=brier,
        log_loss=log_loss_val,
    )


def plot_reliability_diagram(
    predictions_df: DataFrame,
    prob_col: str = "win_probability",
    margin_col: str = "actual_margin",
    n_bins: int = 10,
    model_name: Optional[str] = None,
    output_path: Optional[str] = None,
):
    """
    Plot and optionally save a reliability (calibration) diagram.
    """
    # Local import to avoid hard dependency at import time
    import matplotlib.pyplot as plt

    reliability = compute_reliability_curve(
        predictions_df,
        prob_col=prob_col,
        margin_col=margin_col,
        n_bins=n_bins,
    )

    fig, ax = plt.subplots()

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.0,
        label="Perfect calibration",
    )
    ax.plot(
        reliability["mean_predicted"],
        reliability["empirical_win_rate"],
        marker="o",
        linewidth=1.5,
        label="Empirical",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted win probability (Dem)")
    ax.set_ylabel("Empirical win rate (Dem)")
    title = "Calibration diagram"
    if model_name is not None:
        title += f" — {model_name}"
    ax.set_title(title)
    ax.legend()

    ax.grid(True, alpha=0.3)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    return fig, ax
