"""
Run calibration diagnostics for all model predictions.

This script expects that you have already run the models so that
predictions/*.csv exists (e.g. via `python -m src.models.scripts.run_all_models`).

It will:
  * read each predictions file
  * compute calibration metrics (Brier, log loss)
  * save a calibration plot under plots/calibration/<model>.png
  * write a CSV summary to metrics/calibration_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.utils.calibration import summarize_calibration, plot_reliability_diagram


def get_project_root() -> Path:
    # This file lives in src/models/scripts/, so the repo root is three levels up.
    return Path(__file__).resolve().parents[2]


def find_prediction_files(predictions_dir: Path) -> List[Path]:
    return sorted(predictions_dir.glob("*.csv"))


def main() -> None:
    root = get_project_root()
    predictions_dir = root / "predictions"
    plots_dir = root / "plots" / "calibration"
    metrics_dir = root / "metrics"

    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = find_prediction_files(predictions_dir)

    if not prediction_files:
        raise SystemExit(
            f"No prediction files found in {predictions_dir}. "
            "Run `election-run-all` first."
        )

    records = []

    for path in prediction_files:
        model_name = path.stem  # e.g. "hierarchical_bayes"
        print(f"Processing calibration for {model_name} from {path} ...")

        df = pd.read_csv(path)

        result = summarize_calibration(df)

        # Save plot
        output_plot = plots_dir / f"{model_name}_calibration.png"
        plot_reliability_diagram(
            df,
            model_name=model_name,
            output_path=str(output_plot),
        )

        print(
            f"  Brier score: {result.brier_score:.4f}, "
            f"log loss: {result.log_loss:.4f}, "
            f"bins: {len(result.reliability)}"
        )

        record = {
            "model": model_name,
            "brier_score": result.brier_score,
            "log_loss": result.log_loss,
            "n_bins": len(result.reliability),
        }
        records.append(record)

    summary_df = pd.DataFrame.from_records(records)
    summary_path = metrics_dir / "calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote calibration summary to {summary_path}")
    print(f"Calibration plots saved under {plots_dir}")


if __name__ == "__main__":
    main()
