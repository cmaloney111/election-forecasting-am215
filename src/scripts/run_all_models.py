#!/usr/bin/env python3
import importlib
import inspect
import argparse
import traceback
import cProfile
import pandas as pd  # type: ignore[import-untyped]
from datetime import timedelta
from importlib import resources

import src.models as models_package
from src.models.base_model import ElectionForecastModel
from src.utils.logging_config import setup_logging, get_logger
from src.utils.data_utils import set_election_config
from src.utils.data_utils import get_current_election_date
from typing import List, Optional


logger = get_logger(__name__)


def discover_models():
    """
    Auto-discover all model classes using importlib.resources.

    Returns:
        List of tuples (model_class_name, model_class) sorted by name.
    """
    models = []

    try:
        for item in resources.files(models_package).iterdir():
            if not item.is_file():
                continue
            if not item.name.endswith(".py"):
                continue
            if item.name.startswith("_") or item.name == "base_model.py":
                continue

            module_name = f"src.models.{item.name[:-3]}"  # strip .py
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
                logger.warning(f"Could not import {module_name}: {e}")

    except Exception as e:
        logger.error(f"Error discovering models: {e}")

    return sorted(models, key=lambda x: x[0])


def _default_election_and_start_dates(year: int) -> tuple[str, str]:
    """
    Helper to provide sensible default election / start dates per cycle.
    """
    election_dates = {
        2012: "2012-11-06",
        2016: "2016-11-08",
        2020: "2020-11-03",
    }
    election_date = election_dates.get(year, f"{year}-11-01")
    start_date = f"{year}-09-01"
    return election_date, start_date


def generate_forecast_dates(
    n_dates: int,
    election_date: Optional[str] = None,
    start_date: Optional[str] = None,
) -> List[pd.Timestamp]:
    """
    Generate n evenly-spaced forecast dates between start_date and election_date.

    Args:
        n_dates: Number of forecast dates to generate.
        election_date: Election day as a string (YYYY-MM-DD). If None,
            use the currently configured election date.
        start_date: Earliest date to start forecasting from. If None,
            default to September 1 of the election year.

    Returns:
        List of pd.Timestamp forecast dates.
    """
    # Default election_date to the configured election (2016 in tests)
    if election_date is None:
        election_date = get_current_election_date()

    # Default start_date to Sept 1 of the election year
    if start_date is None:
        year = int(pd.to_datetime(election_date).year)
        start_date = f"{year}-09-01"

    election = pd.to_datetime(election_date)
    start = pd.to_datetime(start_date)

    # Calculate total days available (end 1 day before election)
    last_date = election - timedelta(days=1)
    total_days = (last_date - start).days

    # Generate n evenly-spaced dates (work backwards from election)
    dates: List[pd.Timestamp] = []
    for i in range(n_dates):
        if n_dates > 1:
            days_from_end = int(total_days * (n_dates - 1 - i) / (n_dates - 1))
        else:
            days_from_end = 0
        forecast_date = last_date - timedelta(days=days_from_end)
        dates.append(forecast_date)

    return dates


def main():
    parser = argparse.ArgumentParser(
        description="Run all election forecasting models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  election-forecast                        # Default: 4 forecast dates on 2016
  election-forecast --dates 8              # Use 8 forecast dates
  election-forecast --year 2020            # Run on 2020 data (expects 2020_president_polls.csv)
  election-forecast --year 2012 --polls-file data/polls/2012_president_polls.csv
  election-forecast -n 16                  # Use 16 forecast dates
  election-forecast -v                     # Verbose output
  election-forecast --parallel 4           # Use 4 parallel workers
  election-forecast -w 8                   # Use 8 parallel workers
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
        "--year",
        "-y",
        type=int,
        default=2016,
        help="Election year to run (e.g. 2012, 2016, 2020). Default: 2016.",
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
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        metavar="FILE",
        help="Enable profiling and save to FILE (e.g., forecast.prof)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        metavar="SEED",
        help="Random seed for reproducibility (default: None for non-deterministic)",
    )
    parser.add_argument(
        "--parallel",
        "-w",
        type=int,
        metavar="WORKERS",
        help="Number of parallel workers for state-level parallelization (default: None for sequential)",
    )

    args = parser.parse_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    # Configure global election year / polls file for all downstream loaders
    set_election_config(year=args.year, polls_file=args.polls_file)

    setup_logging(__name__, level="DEBUG" if args.verbose else "INFO")

    election_date, start_date = _default_election_and_start_dates(args.year)
    forecast_dates = generate_forecast_dates(
        n_dates=args.dates,
        election_date=election_date,
        start_date=start_date,
    )

    logger.info(f"Using {len(forecast_dates)} forecast dates for year {args.year}")
    if args.verbose:
        for date in forecast_dates:
            days_to_election = (pd.to_datetime(election_date) - date).days
            logger.info(f"  - {date.date()} ({days_to_election} days before election)")

    logger.info("Looking for models...")
    model_classes = discover_models()

    if not model_classes:
        logger.warning("No models found in src.models")
        return

    logger.info(f"Found {len(model_classes)} model(s)")
    if args.verbose:
        for name, _ in model_classes:
            logger.info(f"  - {name}")

    for model_name, ModelClass in model_classes:
        logger.info(f"\nRunning: {model_name}")

        try:
            model = ModelClass(seed=args.seed)
            pred_df = model.run_forecast(
                forecast_dates=forecast_dates,
                verbose=args.verbose,
                n_workers=args.parallel,
            )
            metrics_df = model.save_results()

            if args.verbose:
                logger.info(f"Total predictions: {len(pred_df)}")
            logger.info(f"Metrics:\n{metrics_df.to_string(index=False)}")
        except Exception as e:
            logger.error(f"ERROR running {model_name}: {e}")
            traceback.print_exc()

    if args.profile:
        profiler.disable()
        profiler.dump_stats(args.profile)
        logger.info(f"\nProfiling data saved to {args.profile}")
        logger.info("View with: snakeviz {args.profile}")


if __name__ == "__main__":
    main()
