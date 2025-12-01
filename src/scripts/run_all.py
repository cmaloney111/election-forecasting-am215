#!/usr/bin/env python3
import argparse
import sys
import time

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.scripts.run_all_models import main as forecast_main
from src.scripts.compare_models import main as compare_main
from src.scripts.generate_plots import main as plot_main

console = Console()


def run_with_temp_argv(argv, func):
    """Temporarily override sys.argv to call a subcommand."""
    original = sys.argv
    sys.argv = argv
    try:
        func()
    except SystemExit:
        pass
    finally:
        sys.argv = original


def run_step(step_number, title, func, argv=None):
    """Run a step with a spinner, timing, and pretty output."""
    console.rule(f"[bold cyan]Step {step_number}/3 • {title}")
    start = time.time()

    with console.status(f"[cyan]{title}...", spinner="dots"):
        if argv:
            run_with_temp_argv(argv, func)
        else:
            try:
                func()
            except SystemExit:
                pass

    end = time.time()
    duration = end - start

    console.print(f"[green] Completed[/green] ({duration:.2f}s)")
    return duration


def main():
    parser = argparse.ArgumentParser(description="Run election forecasting pipeline")
    parser.add_argument("--dates", "-n", type=int, default=4)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        metavar="FILE",
        help="Enable profiling and save to FILE (e.g., pipeline.prof)",
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

    timings = {}

    argv = ["election-forecast", "--dates", str(args.dates)]
    if args.verbose:
        argv.append("--verbose")
    if args.profile:
        argv.extend(["--profile", args.profile])
    if args.seed is not None:
        argv.extend(["--seed", str(args.seed)])
    if args.parallel is not None:
        argv.extend(["--parallel", str(args.parallel)])

    timings["Forecasts"] = run_step(
        1,
        "Running forecasts",
        forecast_main,
        argv=argv,
    )

    timings["Model Comparison"] = run_step(
        2,
        "Comparing models",
        compare_main,
    )

    timings["Plot Generation"] = run_step(
        3,
        "Generating plots",
        plot_main,
    )

    table = Table(
        title="Pipeline Summary",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Step")
    table.add_column("Duration (s)", justify="right")

    for step, duration in timings.items():
        table.add_row(step, f"{duration:.2f}")

    console.print()
    console.print(table)
    console.print()
    console.print(
        Panel.fit("[bold green]All steps completed successfully![/bold green]")
    )


if __name__ == "__main__":
    main()
