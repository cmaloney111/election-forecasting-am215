#!/usr/bin/env python3
import importlib
import inspect
import argparse
import traceback
import cProfile
import pstats
import pandas as pd
from datetime import timedelta
from importlib import resources

import election_forecasting.models as models_package
from election_forecasting.models.base_model import ElectionForecastModel
from election_forecasting.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def discover_models():
    """
    Auto-discover all model classes using importlib.resources

    Returns:
        List of tuples (model_class_name, model_class) sorted by name
    """
    models = []

    try:
        # Get all module names in the models package
        for item in resources.files(models_package).iterdir():
            if not item.is_file():
                continue
            if not item.name.endswith(".py"):
                continue
            if item.name.startswith("_") or item.name == "base_model.py":
                continue

            # Import the module
            module_name = (
                f"election_forecasting.models.{item.name[:-3]}"  # gets rid of .py
            )
            try:
                module = importlib.import_module(module_name)

                # Find all classes that inherit from ElectionForecastModel
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, ElectionForecastModel)
                        and obj != ElectionForecastModel
                        and obj.__module__ == module_name
                    ):
                        models.append((name, obj))
            except Exception as e:
                logger.warning(f"Could not import {module_name}: {e}")

    except Exception as e:
        logger.error(f"Error discovering models: {e}")

    return sorted(models, key=lambda x: x[0])  # sort by name


def generate_forecast_dates(
    n_dates, election_date="2016-11-08", start_date="2016-09-01"
):
    """
    Generate n evenly-spaced forecast dates between start_date and election_date

    Args:
        n_dates: Number of forecast dates to generate
        election_date: Election day
        start_date: Earliest date to start forecasting from

    Returns:
        List of pd.Timestamp forecast dates
    """
    election = pd.to_datetime(election_date)
    start = pd.to_datetime(start_date)

    # Calculate total days available (end 1 day before election)
    last_date = election - timedelta(days=1)
    total_days = (last_date - start).days

    # Generate n evenly-spaced dates (work backwards from election)
    dates = []
    for i in range(n_dates):
        days_from_end = (
            int(total_days * (n_dates - 1 - i) / (n_dates - 1)) if n_dates > 1 else 0
        )
        forecast_date = last_date - timedelta(days=days_from_end)
        dates.append(forecast_date)

    return dates


def main():
    parser = argparse.ArgumentParser(
        description="Run all election forecasting models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  election-forecast              # Default: 4 forecast dates
  election-forecast --dates 8    # Use 8 forecast dates
  election-forecast -n 16        # Use 16 forecast dates
  election-forecast -v           # Verbose output
        """,
    )
    parser.add_argument(
        "--dates",
        "-n",
        type=int,
        default=4,
        help="Number of forecast dates to use (default: 4)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        metavar="FILE",
        help="Enable profiling and save to FILE (e.g., forecast.prof)",
    )

    args = parser.parse_args()

    # Setup profiling if requested
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    # Setup logging
    setup_logging(__name__, level="DEBUG" if args.verbose else "INFO")

    forecast_dates = generate_forecast_dates(args.dates)

    logger.info(f"Using {len(forecast_dates)} forecast dates")
    if args.verbose:
        for date in forecast_dates:
            days_to_election = (pd.to_datetime("2016-11-08") - date).days
            logger.info(f"  - {date.date()} ({days_to_election} days before election)")

    logger.info("Looking for models...")
    model_classes = discover_models()

    if not model_classes:
        logger.warning("No models found in election_forecasting.models")
        return

    logger.info(f"Found {len(model_classes)} model(s)")
    if args.verbose:
        for name, _ in model_classes:
            logger.info(f"  - {name}")

    for model_name, ModelClass in model_classes:
        logger.info(f"\nRunning: {model_name}")

        try:
            model = ModelClass()
            pred_df = model.run_forecast(
                forecast_dates=forecast_dates, verbose=args.verbose
            )
            metrics_df = model.save_results()

            if args.verbose:
                logger.info(f"Total predictions: {len(pred_df)}")
            logger.info(f"Metrics:\n{metrics_df.to_string(index=False)}")
        except Exception as e:
            logger.error(f"ERROR running {model_name}: {e}")
            traceback.print_exc()

    # Save profiling data if enabled
    if args.profile:
        profiler.disable()
        profiler.dump_stats(args.profile)
        logger.info(f"\nProfiling data saved to {args.profile}")
        logger.info(f"View with: snakeviz {args.profile}")


if __name__ == "__main__":
    main()
