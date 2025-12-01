#!/usr/bin/env python3
"""
Compare all forecasting models

Generates comparison tables, rankings, and plots for all models
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob


def parse_metrics(filename):
    """
    Parse metrics from text file

    Args:
        filename: Path to metrics text file

    Returns:
        DataFrame with columns: date, brier, log_loss, mae
    """
    metrics = []
    with open(filename) as f:
        lines = f.readlines()
        current = {}
        for line in lines:
            if "Forecast Date:" in line:
                if current:
                    metrics.append(current)
                current = {"date": line.split(":")[1].strip()}
            elif "Brier Score:" in line:
                current["brier"] = float(line.split(":")[1].strip())
            elif "Log Loss:" in line:
                current["log_loss"] = float(line.split(":")[1].strip())
            elif "MAE (Margin):" in line:
                current["mae"] = float(line.split(":")[1].strip())
        if current:
            metrics.append(current)
    return pd.DataFrame(metrics)


def main():
    """Load all model metrics, compare performance, and generate visualizations"""
    metrics_files = glob.glob("metrics/*.txt")

    if len(metrics_files) == 0:
        print("No metrics files found. Run models first.")
        return

    all_metrics = []
    for metrics_file in metrics_files:
        model_name = Path(metrics_file).stem
        df = parse_metrics(metrics_file)
        df["model"] = model_name
        all_metrics.append(df)

    all_metrics = pd.concat(all_metrics, ignore_index=True)

    # Create comparison tables
    print("Brier Score (lower is better):")
    pivot_brier = all_metrics.pivot(index="date", columns="model", values="brier")
    print(pivot_brier.to_string())

    print("\nLog Loss (lower is better):")
    pivot_ll = all_metrics.pivot(index="date", columns="model", values="log_loss")
    print(pivot_ll.to_string())

    print("\nMAE Margin (lower is better):")
    pivot_mae = all_metrics.pivot(index="date", columns="model", values="mae")
    print(pivot_mae.to_string())

    print("\nAverage performance across all forecast dates:")
    summary = all_metrics.groupby("model")[["brier", "log_loss", "mae"]].mean()
    summary = summary.round(4)
    print(summary.to_string())

    print("\nModel rankings (1 = best)")
    rankings = pd.DataFrame(
        {
            "Brier Score": summary["brier"].rank(),
            "Log Loss": summary["log_loss"].rank(),
            "MAE": summary["mae"].rank(),
        }
    )
    rankings["Average Rank"] = rankings.mean(axis=1)
    rankings = rankings.sort_values("Average Rank")
    print(rankings.to_string())

    comparison_table = all_metrics.pivot_table(
        index="date", columns="model", values=["brier", "log_loss", "mae"]
    )
    comparison_table.to_csv("model_comparison.csv")

    models = all_metrics["model"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    markers = ["o", "s", "^", "d", "v", "*", "p"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        model_data = all_metrics[all_metrics["model"] == model].sort_values("date")
        model_dates = pd.to_datetime(model_data["date"])
        axes[0].plot(
            model_dates,
            model_data["brier"].values,
            marker=markers[i % len(markers)],
            label=model,
            linewidth=2,
            color=colors[i],
            markersize=8,
        )
        axes[1].plot(
            model_dates,
            model_data["log_loss"].values,
            marker=markers[i % len(markers)],
            label=model,
            linewidth=2,
            color=colors[i],
            markersize=8,
        )
        axes[2].plot(
            model_dates,
            model_data["mae"].values,
            marker=markers[i % len(markers)],
            label=model,
            linewidth=2,
            color=colors[i],
            markersize=8,
        )

    axes[0].set_xlabel("Forecast Date")
    axes[0].set_ylabel("Brier Score")
    axes[0].set_title("Brier Score Over Time")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].set_xlabel("Forecast Date")
    axes[1].set_ylabel("Log Loss")
    axes[1].set_title("Log Loss Over Time")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].set_xlabel("Forecast Date")
    axes[2].set_ylabel("MAE (Margin)")
    axes[2].set_title("Margin Error Over Time")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("model_comparison.png")


if __name__ == "__main__":
    main()
