"""
Diagnostics by forecast horizon (days until election).

This script:
  1. Loads all model prediction CSVs from predictions/
  2. Stacks them into a single DataFrame with a 'model' column
  3. Computes metrics by (model, days_until_election)
  4. Saves a CSV summary to diagnostics/horizon/horizon_metrics.csv
  5. Generates simple plots of Brier score and MAE vs days_until_election
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.diagnostics.horizon import compute_horizon_metrics


ELECTION_DATE = "2016-11-08"


def get_project_root() -> Path:
    # This file lives in src/scripts/, so the project root is two levels up
    return Path(__file__).resolve().parents[2]


def load_predictions(predictions_dir: Path) -> pd.DataFrame:
    """
    Load and stack all predictions/*.csv files, adding a 'model' column
    inferred from the filename.
    """
    csv_files = sorted(predictions_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No prediction CSVs found in {predictions_dir}")

    frames = []
    for path in csv_files:
        model_name = path.stem  # e.g. 'hierarchical_bayes'
        df = pd.read_csv(path)
        df["model"] = model_name
        frames.append(df)

    stacked = pd.concat(frames, ignore_index=True)
    return stacked


def ensure_output_dir(root: Path) -> Path:
    out_dir = root / "diagnostics" / "horizon"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_metric_by_horizon(
    metrics_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_dir: Path,
) -> None:
    """
    For each model, plot `metric` vs days_until_election and save a PNG.
    """
    for model, sub in metrics_df.groupby("model"):
        sub = sub.sort_values("days_until_election")

        fig, ax = plt.subplots()
        ax.plot(
            sub["days_until_election"],
            sub[metric],
            marker="o",
        )
        ax.set_xlabel("Days until election")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} vs horizon — {model}")
        # Optional: show time flowing towards Election Day (larger -> smaller)
        ax.invert_xaxis()

        fig.tight_layout()
        out_path = output_dir / f"horizon_{metric}_{model}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main() -> None:
    root = get_project_root()
    predictions_dir = root / "predictions"

    print(f"Project root:    {root}")
    print(f"Predictions dir: {predictions_dir}")

    try:
        preds = load_predictions(predictions_dir)
    except FileNotFoundError as e:
        print(str(e))
        print("Run `election-run-all` first to generate predictions.")
        sys.exit(1)

    print(f"Loaded {len(preds)} prediction rows.")
    print(f"Models found: {sorted(preds['model'].unique())}")

    out_dir = ensure_output_dir(root)
    print(f"Output (horizon): {out_dir}")

    # Compute metrics by horizon
    metrics_df = compute_horizon_metrics(preds, election_date=ELECTION_DATE)

    # Save CSV
    csv_path = out_dir / "horizon_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved horizon metrics to: {csv_path}")

    # Generate plots
    plot_metric_by_horizon(metrics_df, "brier_score", "Brier score", out_dir)
    plot_metric_by_horizon(metrics_df, "mae_margin", "MAE (margin)", out_dir)
    plot_metric_by_horizon(metrics_df, "log_loss", "Log loss", out_dir)

    print("Generated horizon diagnostic plots (Brier, MAE, Log loss).")


if __name__ == "__main__":
    main()
