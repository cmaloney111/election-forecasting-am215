"""
Per-state (per-race) error & calibration diagnostics.

Usage (from repo root, inside your .venv):

    python src/scripts/per_state_calibration.py

Outputs go to:
    diagnostics/per_state/per_state_metrics.csv
    diagnostics/per_state/per_state_calibration.csv
    diagnostics/per_state/calibration_{model}_{state}.png
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

# This file lives in src/scripts/, so:
#   parents[0] = scripts
#   parents[1] = src
#   parents[2] = repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
OUTPUT_DIR = PROJECT_ROOT / "diagnostics" / "per_state"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_all_predictions(predictions_dir: Path = PREDICTIONS_DIR) -> pd.DataFrame:
    """
    Load all prediction CSVs from predictions/.

    Assumes each CSV has columns:
        state, forecast_date, win_probability, predicted_margin, margin_std, actual_margin

    Adds a 'model' column based on the file name.
    """
    csv_files = sorted(predictions_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No prediction CSVs found in {predictions_dir}. "
            "Run `election-run-all` (or `python -m src.scripts.run_all_models`) first."
        )

    dfs = []
    for path in csv_files:
        df = pd.read_csv(path)
        df["model"] = path.stem  # e.g. hierarchical_bayes, poll_average, etc.
        dfs.append(df)

    all_preds = pd.concat(dfs, ignore_index=True)

    # Make sure forecast_date is a datetime
    if "forecast_date" in all_preds.columns:
        all_preds["forecast_date"] = pd.to_datetime(all_preds["forecast_date"])

    return all_preds


def add_binary_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary Democratic win indicator based on actual_margin.

    Convention from the package:
        actual_margin = Dem two-party margin (Dem % - Rep %)
        > 0 means Dem win.
    """
    if "actual_margin" not in df.columns:
        raise ValueError(
            "Expected an 'actual_margin' column in predictions. "
            "Check that your predictions CSVs match the library output."
        )

    out = df.copy()
    out["dem_win"] = (out["actual_margin"] > 0).astype(int)
    return out


def safe_log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    """
    Binary log loss, implemented manually to avoid extra dependencies.

    y_true: 0/1 array
    p: predicted probability in [0,1]
    """
    eps = 1e-15
    p_clipped = np.clip(p, eps, 1 - eps)
    return -np.mean(y_true * np.log(p_clipped) + (1 - y_true) * np.log(1 - p_clipped))


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    """
    Classic Brier score for binary events.
    """
    return float(np.mean((p - y_true) ** 2))


# ---------------------------------------------------------------------
# Per-state error metrics
# ---------------------------------------------------------------------

def compute_state_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-state, per-model error metrics.

    For each (model, state), we aggregate over all forecast dates and compute:
        - n_forecasts
        - Brier score
        - log loss
        - MAE of vote margin
    """
    df = add_binary_outcome(df)

    required_cols = ["model", "state", "win_probability", "predicted_margin", "actual_margin", "dem_win"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in predictions DataFrame: {missing}")

    rows = []
    for (model, state), g in df.groupby(["model", "state"]):
        y = df["dem_win"].to_numpy()
        p = df["win_probability"].to_numpy()
        
        brier = brier_score(y, p)
        logloss = safe_log_loss(y, p)
        mae_margin = float(np.mean(np.abs(g["predicted_margin"] - g["actual_margin"])))

        rows.append(
            {
                "model": model,
                "state": state,
                "n_forecasts": len(g),
                "brier_score": brier,
                "log_loss": logloss,
                "mae_margin": mae_margin,
            }
        )

    metrics_df = pd.DataFrame(rows)
    return metrics_df.sort_values(["model", "brier_score", "mae_margin"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Per-state calibration curves
# ---------------------------------------------------------------------

def compute_state_calibration(
    df: pd.DataFrame,
    n_bins: int = 10,
    min_points_per_bin: int = 1,
) -> pd.DataFrame:
    """
    Compute per-state calibration statistics.

    For each (model, state, probability bin), we compute:
        - bin_lower, bin_upper
        - mean_pred_prob
        - empirical_win_rate
        - n (number of forecasts in the bin)

    We keep even small bins (min_points_per_bin defaults to 1) and make sure we
    return a DataFrame with the expected columns even if there are no rows.
    """
    df = add_binary_outcome(df)

    if "win_probability" not in df.columns:
        raise ValueError("Predictions DataFrame must contain 'win_probability'.")

    # Define probability bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    df = df.copy()
    df["prob_bin"] = pd.cut(
        df["win_probability"],
        bins=bins,
        include_lowest=True,
        duplicates="drop",
    )

    rows = []
    grouped = df.groupby(["model", "state", "prob_bin"], observed=True)
    for (model, state, bin_interval), g in grouped:
        n = len(g)
        if n < min_points_per_bin:
            # Skip tiny bins if requested
            continue

        # bin_interval is a pandas Interval
        rows.append(
            {
                "model": model,
                "state": state,
                "bin_lower": float(bin_interval.left),
                "bin_upper": float(bin_interval.right),
                "mean_pred_prob": g["win_probability"].mean(),
                "empirical_win_rate": g["dem_win"].mean(),
                "n": n,
            }
        )

    # If no rows, return an empty DF *with* the right columns
    columns = [
        "model",
        "state",
        "bin_lower",
        "bin_upper",
        "mean_pred_prob",
        "empirical_win_rate",
        "n",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    calib_df = pd.DataFrame(rows, columns=columns)
    return calib_df.sort_values(["model", "state", "bin_lower"]).reset_index(drop=True)

def plot_state_calibration(
    calib_df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    min_total_points: int = 20,
) -> None:
    """
    Create a calibration plot for each (model, state) pair with enough data.

    Points are sized by the number of forecasts in that probability bin.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if calib_df.empty:
        print("No calibration data to plot (calib_df is empty).")
        return

    for (model, state), g in calib_df.groupby(["model", "state"]):
        total_points = int(g["n"].sum())
        if total_points < min_total_points:
            # Not enough data for a meaningful plot
            continue

        fig, ax = plt.subplots(figsize=(5, 5))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")

        # Scatter of mean predicted probability vs empirical win rate
        sizes = 20 + 5 * np.sqrt(g["n"].astype(float))
        ax.scatter(
            g["mean_pred_prob"],
            g["empirical_win_rate"],
            s=sizes,
            alpha=0.7,
            edgecolor="k",
            linewidth=0.5,
            label="Bins",
        )

        ax.set_xlabel("Predicted Dem win probability")
        ax.set_ylabel("Empirical Dem win rate")
        ax.set_title(f"Calibration: {model} – {state}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

        fig.tight_layout()
        filename = f"calibration_{model}_{state}.png".replace(" ", "_")
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def main() -> None:
    print(f"Project root:        {PROJECT_ROOT}")
    print(f"Predictions dir:     {PREDICTIONS_DIR}")
    print(f"Output (per-state):  {OUTPUT_DIR}")
    print()

    # 1. Load predictions
    preds = load_all_predictions()
    print(f"Loaded {len(preds):,} prediction rows from {PREDICTIONS_DIR}")
    print(f"Models found: {sorted(preds['model'].unique())}")
    print()

    # 2. Per-state metrics
    state_metrics = compute_state_metrics(preds)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "per_state_metrics.csv"
    state_metrics.to_csv(metrics_path, index=False)
    print(f"Saved per-state metrics to: {metrics_path}")

    # 3. Per-state calibration table
    calib_df = compute_state_calibration(preds, n_bins=10, min_points_per_bin=1)

    if calib_df.empty:
        print("Warning: no per-state calibration data produced "
              "(very few predictions per state / probability bin).")
    else:
        calib_path = OUTPUT_DIR / "per_state_calibration.csv"
        calib_df.to_csv(calib_path, index=False)
        print(f"Saved per-state calibration table to: {calib_path}")

        # 4. Per-state calibration plots
        print("Generating per-state calibration plots...")
        plot_state_calibration(calib_df)
        print(f"Calibration plots written under: {OUTPUT_DIR}")

    print("\nDone.")



if __name__ == "__main__":
    main()
