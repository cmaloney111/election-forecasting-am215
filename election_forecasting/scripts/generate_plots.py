#!/usr/bin/env python3
"""
Generate state-level plots for all models

Usage:
    election-plot              # Default: plot key swing states
    election-plot --all        # Plot all states with sufficient data
    election-plot --states FL PA MI WI  # Plot specific states
"""

import importlib
import inspect
import argparse
import traceback
import pandas as pd
from importlib import resources
from pathlib import Path

import election_forecasting.models as models_package
from election_forecasting.models.base_model import ElectionForecastModel
from election_forecasting.utils.logging_config import setup_logging, get_logger
from election_forecasting.utils.data_utils import load_polling_data

logger = get_logger(__name__)


def discover_models():
    """Auto-discover all model classes using importlib.resources"""
    models = []

    try:
        for item in resources.files(models_package).iterdir():
            if not item.is_file():
                continue
            if not item.name.endswith(".py"):
                continue
            if item.name.startswith("_") or item.name == "base_model.py":
                continue

            module_name = f"election_forecasting.models.{item.name[:-3]}"
            try:
                module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, ElectionForecastModel)
                        and obj != ElectionForecastModel
                        and obj.__module__ == module_name
                    ):
                        models.append((name, obj))
            except Exception as e:
                logger.info(f"Warning: Could not import {module_name}: {e}")

    except Exception as e:
        logger.info(f"Error discovering models: {e}")

    return sorted(models, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(
        description="Generate state-level forecast plots for all models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  election-plot                    # Plot key swing states
  election-plot --all              # Plot all states
  election-plot --states FL PA MI  # Plot specific states
        """,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate plots for all states with sufficient polling data",
    )
    parser.add_argument(
        "--states", nargs="+", help="Specific state codes to plot (e.g., FL PA MI WI)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(__name__)

    # Determine which states to plot
    if args.states:
        states_to_plot = [s.upper() for s in args.states]
        logger.info(f"Plotting {len(states_to_plot)} specified states")
    elif args.all:
        # Get all states with polling data

        polls = load_polling_data()
        states_to_plot = sorted([s for s in polls["state_code"].unique() if s])
        logger.info(f"Plotting all {len(states_to_plot)} states with polling data")
    else:
        # Default: key swing states
        states_to_plot = ["FL", "PA", "MI", "WI", "NC", "AZ", "NV", "GA", "OH", "VA"]
        logger.info(f"Plotting {len(states_to_plot)} key swing states")

    logger.info(f"States: {', '.join(states_to_plot)}\n")

    # Discover models
    logger.info("Discovering models...")
    model_classes = discover_models()

    if not model_classes:
        logger.info("No models found in election_forecasting.models")
        return

    logger.info(f"Found {len(model_classes)} model(s):")
    for name, _ in model_classes:
        logger.info(f"  - {name}")
    logger.info()

    # Generate plots for each model
    total_plots = 0
    for model_name, ModelClass in model_classes:
        logger.info(f"Generating plots for {model_name}...")
        try:
            model = ModelClass()

            # Load predictions from CSV if they exist

            pred_file = Path(f"predictions/{model.name}.csv")
            if pred_file.exists():
                pred_df = pd.read_csv(pred_file)
                # Convert forecast_date to datetime
                pred_df["forecast_date"] = pd.to_datetime(pred_df["forecast_date"])
                model.predictions = pred_df.to_dict("records")
            else:
                logger.info(f"  Warning: No predictions found at {pred_file}")
                logger.info("  Run 'election-forecast' first to generate predictions")
                continue

            for state in states_to_plot:
                try:
                    model.plot_state(state)
                    total_plots += 1
                except Exception as e:
                    logger.info(f"  Warning: Could not plot {state}: {e}")
            logger.info(f"  ✓ Saved to plots/{model.name}/")
        except Exception as e:
            logger.info(f"  ERROR: {e}")

            traceback.print_exc()

    logger.info(f"\n✓ Generated {total_plots} plots total")
    logger.info("  Plots saved in plots/ directory (organized by model)")


if __name__ == "__main__":
    main()
