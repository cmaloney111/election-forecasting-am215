#!/usr/bin/env python3
"""
Generate per-state calibration plots from diagnostics/per_state CSVs.

Inputs (expected paths, relative to repo root):
  diagnostics/per_state/per_state_calibration.csv
  diagnostics/per_state/per_state_metrics.csv

Outputs:
  diagnostics/per_state/{STATE}_calibration.png

Each plot shows, for a single state:
  - Reliability curves for each model (mean predicted prob vs empirical win rate)
  - 45-degree perfect-calibration line
  - Legend including the model name (and Brier score if available)
"""

from pathlib import Path
import argparse

import pandas as pd  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-state calibration plots from diagnostics/per_state CSVs"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively instead of just saving to files",
    )
    args = parser.parse_args()

    # Base directory = repo root (one level above src/)
    base_dir = Path(__file__).resolve().parents[2]

    per_state_dir = base_dir / "diagnostics" / "per_state"
    per_state_dir.mkdir(parents=True, exist_ok=True)

    calib_path = per_state_dir / "per_state_calibration.csv"
    metrics_path = per_state_dir / "per_state_metrics.csv"

    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    calib_df = pd.read_csv(calib_path)
    metrics_df = pd.read_csv(metrics_path)

    # Ensure the columns we expect are actually there
    required_calib_cols = {
        "model",
        "state",
        "bin_lower",
        "bin_upper",
        "mean_pred_prob",
        "empirical_win_rate",
        "n",
    }
    missing = required_calib_cols - set(calib_df.columns)
    if missing:
        raise ValueError(f"Missing columns in calibration CSV: {missing}")

    required_metrics_cols = {
        "model",
        "state",
        "n_forecasts",
        "brier_score",
        "log_loss",
        "mae_margin",
    }
    missing = required_metrics_cols - set(metrics_df.columns)
    if missing:
        raise ValueError(f"Missing columns in metrics CSV: {missing}")

    # Unique states and models
    states = sorted(calib_df["state"].unique())
    models = sorted(calib_df["model"].unique())

    print(f"Found {len(states)} states and {len(models)} models.")
    print(f"Saving plots to: {per_state_dir}")

    # Color / marker cycle (just for nicer-looking plots)
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]
    # Let matplotlib pick colors; we just vary markers.

    for state in states:
        state_calib = calib_df[calib_df["state"] == state].copy()

        # Skip if there is basically no data
        if state_calib["n"].sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot per-model reliability curves
        for i, model_name in enumerate(models):
            sub = state_calib[state_calib["model"] == model_name].copy()
            # Only keep bins with at least 1 forecast
            sub = sub[sub["n"] > 0]

            if sub.empty:
                continue

            # Sort by mean predicted probability for a nicer curve
            sub = sub.sort_values("mean_pred_prob")

            # Get per-state metrics for title / legend
            m_row = metrics_df[
                (metrics_df["state"] == state)
                & (metrics_df["model"] == model_name)
            ]
            if not m_row.empty:
                brier = float(m_row["brier_score"].iloc[0])
                label = f"{model_name} (Brier={brier:.3f})"
            else:
                label = model_name

            marker = marker_cycle[i % len(marker_cycle)]

            ax.plot(
                sub["mean_pred_prob"],
                sub["empirical_win_rate"],
                marker + "-",
                linewidth=2,
                markersize=6,
                label=label,
            )

        # 45-degree perfect-calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.7, label="Perfect")

        ax.set_xlabel("Mean Predicted Win Probability", fontsize=11)
        ax.set_ylabel("Empirical Win Rate", fontsize=11)
        ax.set_title(f"{state} – Per-State Calibration", fontsize=13, fontweight="bold")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()

        out_path = per_state_dir / f"{state}_calibration.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
        if args.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
