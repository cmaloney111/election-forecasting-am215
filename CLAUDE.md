# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

State-level presidential election forecasting system using polling time-series data from the 2016 U.S. election. Implements four different forecasting models that predict state-level Democratic margins and win probabilities.

## Installation & Setup

```bash
# Install package and dependencies
uv pip install -e .
```

## Core Commands

### Running Forecasts

```bash
# Complete pipeline: run all models, compare, and generate plots
election-run-all

# With custom number of forecast dates (default: 4)
election-run-all --dates 8

# Run forecasts only (without comparison or plots)
election-forecast
election-forecast --dates 6
election-forecast -v  # verbose output

# Compare model performance (generates CSV and PNG)
election-compare

# Generate state-level forecast plots
election-plot              # key swing states only
election-plot --all        # all states with polling data
election-plot --states FL PA MI WI  # specific states
```

### Development Commands

```bash
# Code quality
make lint           # Check formatting and linting with ruff
make format         # Auto-format code with ruff
make test           # Run tests (pytest)
make quality-check  # Run both lint and test

# Equivalent uv commands
uv run ruff check .
uv run ruff format .
uv run pytest tests/

# Documentation (if using Sphinx)
make docs
```

## Architecture

### Model Hierarchy

All forecasting models inherit from `ElectionForecastModel` (election_forecasting/models/base_model.py), which provides:
- Standard interface: `fit_and_forecast()` method that takes state polls, forecast date, election date, and returns predictions
- Data loading: `load_data()` method that loads polling data and election results
- Forecast execution: `run_forecast()` method that runs the model across all states and forecast dates
- Result persistence: `save_results()` method that generates predictions/*.csv and metrics/*.txt files
- Visualization: `plot_state()` method that creates time-series plots for individual states

### Four Implemented Models

1. **Hierarchical Bayes** (`hierarchical_bayes.py`) - Best performer
   - Combines fundamentals prior (weighted average of 2012/2008 results) with Kalman-filtered polls
   - Estimates pollster house effects using hierarchical shrinkage
   - Applies systematic bias correction (calibrated to 2016 Republican underestimation)
   - Uses Rauch-Tung-Striebel (RTS) backward smoother for optimal state estimation

2. **Poll Average** (`poll_average.py`) - Simple baseline
   - Weighted poll-of-polls average (weighted by sample size)
   - Uses 14-day window for recent polls
   - Empirical uncertainty estimation from poll variance

3. **Improved Kalman** (`improved_kalman.py`)
   - Brownian motion with drift model
   - EM algorithm for parameter estimation (drift μ, diffusion σ²)
   - Stronger regularization to prevent overfitting

4. **Kalman Diffusion** (`kalman_diffusion.py`)
   - Basic diffusion model (no drift term)
   - EM algorithm for variance parameter estimation

### Data Pipeline

**Data loading** (utils/data_utils.py):
- `load_polling_data()`: Reads FiveThirtyEight 2016 polls (4,209 polls)
  - Computes middate, Democratic margin, and two-party proportions
  - Maps state names to two-letter codes
- `load_election_results()`: Reads MIT Election Lab results for 2016
  - Returns dict mapping state code to actual Democratic margin
- `load_fundamentals()`: Computes prior from historical results
  - Weighted average: 70% from 2012 + 30% from 2008

**Metric computation** (utils/data_utils.py):
- `compute_metrics()`: Calculates Brier Score, Log Loss, and MAE for each forecast date

### Script Pipeline

The `election-run-all` command orchestrates three sequential steps:
1. `run_all_models.py`: Runs all four models for specified forecast dates
2. `compare_models.py`: Aggregates metrics, ranks models, generates comparison CSV/PNG
3. `generate_plots.py`: Creates state-level time-series plots for each model

## Data Sources

- **Polls**: `data/polls/fivethirtyeight_2016_polls_timeseries.csv`
  - 4,209 state-level polls across 50 states
  - Columns: state, startdate, enddate, rawpoll_clinton, rawpoll_trump, samplesize, pollster

- **Election Results**: `data/election_results/mit_president_state_1976_2020.csv`
  - Historical presidential results 1976-2020 (tab-separated)
  - Used for: 2016 actual results (evaluation) and fundamentals prior (2008/2012)

## Output Structure

```
predictions/           # Model predictions as CSV
  hierarchical_bayes.csv
  poll_average.csv
  improved_kalman.csv
  kalman_diffusion.csv

metrics/              # Evaluation metrics as text
  hierarchical_bayes.txt
  poll_average.txt
  improved_kalman.txt
  kalman_diffusion.txt

plots/                # State-level visualizations (organized by model)
  hierarchical_bayes/
    FL.png
    PA.png
    ...
  poll_average/
    FL.png
    ...
  ...

model_comparison.csv  # Cross-model performance table
model_comparison.png  # Visual comparison of all models
```

## Key Implementation Details

### Poll Data Processing
- Polls are filtered by `middate` (average of start and end dates)
- Two-party margin computed as: `(dem - rep) / (dem + rep)`
- Sample sizes used for variance weighting: `1.0 / samplesize + polling_error²`

### Forecast Dates
- Default: 4 dates (Oct 1, Oct 15, Nov 1, Nov 7, 2016)
- Configurable via `--dates` parameter
- Only forecasts states with `min_polls=10` polls by forecast date

### Kalman Filtering Pattern
Models using Kalman filters follow this structure:
- State equation: `x[t+1] = x[t] + μ*dt + ε` where ε ~ N(0, σ²*dt)
- Observation equation: `y[t] = x[t] + v[t]` where v[t] ~ N(0, R[t])
- Observation variance: `R[t] = 1/samplesize[t] + polling_error²`
- EM algorithm: Iteratively fit (μ, σ²) parameters to maximize likelihood

### House Effects Estimation (Hierarchical Bayes only)
- For each pollster, compute residuals vs. state-time average
- Use shrinkage estimator: `house_effect = (n_polls / (n_polls + λ)) * mean_residual`
- Default shrinkage: λ = 10 (prevents overfitting to small pollsters)

## Development Guidelines

When adding a new model:
1. Create new file in `election_forecasting/models/`
2. Inherit from `ElectionForecastModel`
3. Implement `fit_and_forecast()` method returning dict with keys:
   - `win_probability`: float in [0,1]
   - `predicted_margin`: float (Democratic margin)
   - `margin_std`: float (standard deviation)
4. Add model instantiation to `run_all_models.py`
5. Run `election-run-all` to generate results for the new model

When modifying data loading:
- All data utilities live in `utils/data_utils.py`
- Data paths are relative to repository root
- State codes must match between polls and election results

When debugging a specific state:
- Use `model.plot_state('FL')` to visualize forecasts vs. polls
- Check `predictions/{model}.csv` for numerical predictions
- Compare with other models using `election-compare`
