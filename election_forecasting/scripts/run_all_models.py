#!/usr/bin/env python3
import importlib
import inspect
import argparse
import pandas as pd
from datetime import timedelta
from importlib import resources

from election_forecasting.models.base_model import ElectionForecastModel

def discover_models():
    """
    Auto-discover all model classes using importlib.resources

    Returns:
        List of tuples (model_class_name, model_class) sorted by name
    """
    models = []

    try:
        import election_forecasting.models as models_package

        # Get all module names in the models package
        for item in resources.files(models_package).iterdir():
            if not item.is_file():
                continue
            if not item.name.endswith('.py'):
                continue
            if item.name.startswith('_') or item.name == 'base_model.py':
                continue

            # Import the module
            module_name = f'election_forecasting.models.{item.name[:-3]}' # gets rid of .py
            try:
                module = importlib.import_module(module_name)

                # Find all classes that inherit from ElectionForecastModel
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, ElectionForecastModel) and
                        obj != ElectionForecastModel and
                        obj.__module__ == module_name):
                        models.append((name, obj))
            except Exception as e:
                print(f"Warning: Could not import {module_name}: {e}")

    except Exception as e:
        print(f"Error discovering models: {e}")

    return sorted(models, key=lambda x: x[0]) # sort by name

def generate_forecast_dates(n_dates, election_date='2016-11-08', start_date='2016-09-01'):
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
        days_from_end = int(total_days * (n_dates - 1 - i) / (n_dates - 1)) if n_dates > 1 else 0
        forecast_date = last_date - timedelta(days=days_from_end)
        dates.append(forecast_date)

    return dates

def main():
    parser = argparse.ArgumentParser(
        description='Run all election forecasting models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  election-forecast              # Default: 4 forecast dates
  election-forecast --dates 8    # Use 8 forecast dates
  election-forecast -n 16        # Use 16 forecast dates
  election-forecast -v           # Verbose output
        """
    )
    parser.add_argument(
        '--dates', '-n',
        type=int,
        default=4,
        help='Number of forecast dates to use (default: 4)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action="store_true",
        help='Enable verbose output'
    )

    args = parser.parse_args()

    forecast_dates = generate_forecast_dates(args.dates)

    print(f"Using {len(forecast_dates)} forecast dates")
    if args.verbose:
        for date in forecast_dates:
            days_to_election = (pd.to_datetime('2016-11-08') - date).days
            print(f"  - {date.date()} ({days_to_election} days before election)")
    print()

    print("Looking for models...")
    model_classes = discover_models()

    if not model_classes:
        print("No models found in election_forecasting.models")
        return

    print(f"Found {len(model_classes)} model(s)")
    if args.verbose:
        for name, _ in model_classes:
            print(f"  - {name}")
    print()

    for model_name, ModelClass in model_classes:
        print(f"\nRunning: {model_name}")

        try:
            model = ModelClass()
            pred_df = model.run_forecast(forecast_dates=forecast_dates, verbose=args.verbose)
            metrics_df = model.save_results()

            if args.verbose:
                print(f"\nTotal predictions: {len(pred_df)}\n")
            print("Metrics:")
            print(metrics_df.to_string(index=False))
        except Exception as e:
            print(f"ERROR running {model_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
