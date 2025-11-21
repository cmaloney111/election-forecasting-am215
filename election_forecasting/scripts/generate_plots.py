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
from importlib import resources
from election_forecasting.models.base_model import ElectionForecastModel

def discover_models():
    """Auto-discover all model classes using importlib.resources"""
    models = []

    try:
        import election_forecasting.models as models_package

        for item in resources.files(models_package).iterdir():
            if not item.is_file():
                continue
            if not item.name.endswith('.py'):
                continue
            if item.name.startswith('_') or item.name == 'base_model.py':
                continue

            module_name = f'election_forecasting.models.{item.name[:-3]}'
            try:
                module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, ElectionForecastModel) and
                        obj != ElectionForecastModel and
                        obj.__module__ == module_name):
                        models.append((name, obj))
            except Exception as e:
                print(f"Warning: Could not import {module_name}: {e}")

    except Exception as e:
        print(f"Error discovering models: {e}")

    return sorted(models, key=lambda x: x[0])

def main():
    parser = argparse.ArgumentParser(
        description='Generate state-level forecast plots for all models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  election-plot                    # Plot key swing states
  election-plot --all              # Plot all states
  election-plot --states FL PA MI  # Plot specific states
        """
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate plots for all states with sufficient polling data'
    )
    parser.add_argument(
        '--states',
        nargs='+',
        help='Specific state codes to plot (e.g., FL PA MI WI)'
    )

    args = parser.parse_args()

    # Determine which states to plot
    if args.states:
        states_to_plot = [s.upper() for s in args.states]
        print(f"Plotting {len(states_to_plot)} specified states")
    elif args.all:
        # Get all states with polling data
        from election_forecasting.utils.data_utils import load_polling_data
        polls = load_polling_data()
        states_to_plot = sorted([s for s in polls['state_code'].unique() if s])
        print(f"Plotting all {len(states_to_plot)} states with polling data")
    else:
        # Default: key swing states
        states_to_plot = ['FL', 'PA', 'MI', 'WI', 'NC', 'AZ', 'NV', 'GA', 'OH', 'VA']
        print(f"Plotting {len(states_to_plot)} key swing states")

    print(f"States: {', '.join(states_to_plot)}\n")

    # Discover models
    print("Discovering models...")
    model_classes = discover_models()

    if not model_classes:
        print("No models found in election_forecasting.models")
        return

    print(f"Found {len(model_classes)} model(s):")
    for name, _ in model_classes:
        print(f"  - {name}")
    print()

    # Generate plots for each model
    total_plots = 0
    for model_name, ModelClass in model_classes:
        print(f"Generating plots for {model_name}...")
        try:
            model = ModelClass()

            # Load predictions from CSV if they exist
            import pandas as pd
            from pathlib import Path
            pred_file = Path(f'predictions/{model.name}.csv')
            if pred_file.exists():
                pred_df = pd.read_csv(pred_file)
                # Convert forecast_date to datetime
                pred_df['forecast_date'] = pd.to_datetime(pred_df['forecast_date'])
                model.predictions = pred_df.to_dict('records')
            else:
                print(f"  Warning: No predictions found at {pred_file}")
                print(f"  Run 'election-forecast' first to generate predictions")
                continue

            for state in states_to_plot:
                try:
                    model.plot_state(state)
                    total_plots += 1
                except Exception as e:
                    print(f"  Warning: Could not plot {state}: {e}")
            print(f"  ✓ Saved to plots/{model.name}/")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✓ Generated {total_plots} plots total")
    print(f"  Plots saved in plots/ directory (organized by model)")

if __name__ == '__main__':
    main()
