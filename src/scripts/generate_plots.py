#!/usr/bin/env python3
"""
Generate state-level plots for all models

Usage:
    election-plot                                  # Default: plot key swing states (for 2016)
    election-plot --all                            # Plot all states with sufficient data
    election-plot --states FL PA MI WI             # Plot specific states
    election-plot --year 2020 --all                # Plot all states for 2020
    election-plot --year 2020 --polls-file data/polls/2020_president_polls.csv --all
"""

import importlib
import inspect
import argparse
import traceback
from pathlib import Path
from importlib import resources

import pandas as pd  # type: ignore[import-untyped]

import src.models as models_package
from src.models.base_model import ElectionForecastModel
from src.utils.logging_config import setup_logging, get_logger
from src.utils.data_utils import load_polling_data, set_election_config

logger = get_logger(__name__)


def discover_models():
    """Auto-discover all model classes using importlib.resources."""
    models = []

    try:
        for item in resources.files(models_package).iterdir():
            if not item.is_file():
                continue
            if not item.name.endswith(".py"):
                continue
            if item.name.startswith("_") or item.name == "base_model.py":
                continue

            module_name = f"src.models.{item.name[:-3]}"
            try:
                module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, ElectionForecastModel)
                        and obj is not ElectionForecastModel
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
  election-plot                                      # Plot key swing states (default year 2016)
  election-plot --all                                # Plot all states
  election-plot --states FL PA MI                    # Plot specific states
  election-plot --year 2020 --all                    # Plot all states (2020)
  election-plot --year 2020 --polls-file data/polls/2020_president_polls.csv --all
        """,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate plots for all states with sufficient polling data",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        help="Specific state codes to plot (e.g., FL PA MI WI)",
    )
    parser.add_argument(
        "--year",
        "-y",
        type=int,
        default=2016,
        help="Election year to plot (e.g. 2012, 2016, 2020). Default: 2016.",
    )
    parser.add_argument(
        "--polls-file",
        type=str,
        default=None,
        help=(
            "Optional path to a FiveThirtyEight-style polls CSV. "
            "If omitted, the loader uses the built-in 2016 timeseries for year=2016 "
            "or data/polls/{year}_president_polls.csv for other years."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(__name__, level="DEBUG" if args.verbose else "INFO")

    # IMPORTANT: make plotting use the same election year / polls file
    # as the forecasting step.
    set_election_config(year=args.year, polls_file=args.polls_file)

    # Load polls for the configured election; used only to decide which states exist
    polls = load_polling_data()

    # Determine which states to plot
    if args.states:
        states_to_plot = [s.upper() for s in args.states]
        logger.info(f"Plotting {len(states_to_plot)} specified states")
    elif args.all:
        # All states with polling data (drop NaNs to avoid str/float sorting error)
        states_to_plot = sorted(
            s for s in polls["state_code"].dropna().unique().tolist()
        )
        logger.info(f"Plotting all {len(states_to_plot)} states with polling data")
    else:
        # Default: key swing states
        default_states = ["FL", "PA", "MI", "WI", "NC", "AZ", "NV", "GA", "OH", "VA"]
        # Only keep those that actually appear in the current polls
        available = set(polls["state_code"].dropna().unique().tolist())
        states_to_plot = [s for s in default_states if s in available]
        logger.info(f"Plotting {len(states_to_plot)} key swing states")

    if not states_to_plot:
        logger.info("No states to plot (none found in polling data).")
        return

    logger.info(f"States: {', '.join(states_to_plot)}\n")

    # Discover models
    logger.info("Discovering models...")
    model_classes = discover_models()

    if not model_classes:
        logger.info("No models found in src.models")
        return

    logger.info(f"Found {len(model_classes)} model(s):")
    for name, _ in model_classes:
        logger.info(f"  - {name}")

    # Ensure base plots directory exists
    Path("plots").mkdir(parents=True, exist_ok=True)

    # Generate plots for each model
    total_plots = 0
    for model_name, ModelClass in model_classes:
        logger.info(f"\nGenerating plots for {model_name}...")
        try:
            model = ModelClass()

            # Load predictions from CSV if they exist
            pred_file = Path(f"predictions/{model.name}.csv")
            if pred_file.exists():
                pred_df = pd.read_csv(pred_file)

                if pred_df.empty:
                    logger.info(f"  Warning: Predictions file {pred_file} is empty")
                    continue

                # Convert forecast_date to datetime for correct plotting
                if "forecast_date" in pred_df.columns:
                    pred_df["forecast_date"] = pd.to_datetime(
                        pred_df["forecast_date"]
                    )

                # Attach predictions in the same structure run_forecast() uses
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
