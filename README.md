# Election Forecasting Models

[![CI](https://github.com/cmaloney111/election-forecasting-am215/actions/workflows/ci.yml/badge.svg)](https://github.com/cmaloney111/election-forecasting-am215/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cmaloney111/election-forecasting-am215/branch/main/graph/badge.svg)](https://codecov.io/gh/cmaloney111/election-forecasting-am215)
[![PyPI](https://img.shields.io/pypi/v/election-forecasting.svg)](https://pypi.org/project/election-forecasting/)
[![Python](https://img.shields.io/pypi/pyversions/election-forecasting.svg)](https://pypi.org/project/election-forecasting/)

State-level presidential election forecasting using polling time-series data from the 2016 U.S. presidential election.

## Installation

### Local Installation
```bash
# Install with uv
uv pip install -e .
```

### Docker
```bash
# Build the Docker image
docker build -t election-forecasting .

# Run forecasts in container
docker run -v $(pwd)/predictions:/app/predictions \
           -v $(pwd)/metrics:/app/metrics \
           election-forecasting election-forecast --dates 8

# Run with parallel execution (utilize host CPU cores)
docker run -v $(pwd)/predictions:/app/predictions \
           -v $(pwd)/metrics:/app/metrics \
           election-forecasting election-forecast --dates 16 --parallel 4
```

The Docker setup automatically mounts volumes for `predictions/` and `metrics/` so results persist on your host machine.

## Usage

### Quick Start: Run Everything
```bash
# Run complete pipeline: forecast, compare, and plot
election-run-all

# With custom number of forecast dates
election-run-all --dates 8
```

### Individual Commands

#### Run All Models
```bash
# Run with default 4 forecast dates
election-forecast

# Run with custom number (n) of forecast dates
election-forecast --dates n

# Run with verbose output
election-forecast -v

# Run with parallel execution (recommended for many dates)
election-forecast --dates 16 --parallel 4

# Set random seed for reproducibility
election-forecast --seed 42
```

**Parallel Execution:** Use `--parallel N` (or `-w N`) to enable multi-core processing. The workload is parallelized by forecast date, so this is most beneficial when using many dates (e.g., 8+). With 4 workers and 8+ dates, you can see significant speedup on multi-core machines.

#### Compare Model Performance
```bash
election-compare
```

This generates:
- `model_comparison.csv` - Detailed metrics table
- `model_comparison.png` - Performance visualization
- Console output with rankings

#### Generate State-Level Plots
```bash
# Plot key swing states (default)
election-plot

# Plot all states with polling data
election-plot --all

# Plot specific states
election-plot --states FL PA MI WI
```

## Models

### 1. Hierarchical Bayes (Best Overall)
Advanced Bayesian model combining fundamentals prior with Kalman-filtered polls and systematic bias correction.

**File:** `election_forecasting/models/hierarchical_bayes.py`

### 2. Poll Average
Simple weighted poll-of-polls average with empirical uncertainty estimation.

**File:** `election_forecasting/models/poll_average.py`

### 3. Improved Kalman
Brownian motion with drift using Kalman filter/RTS smoother and stronger regularization.

**File:** `election_forecasting/models/improved_kalman.py`

### 4. Kalman Diffusion
Basic diffusion model with EM algorithm for parameter estimation.

**File:** `election_forecasting/models/kalman_diffusion.py`

## Data Sources

- **Polls:** FiveThirtyEight 2016 state-level polling data (4,209 polls across 50 states)
- **Election Results:** MIT Election Lab 1976-2020 presidential election results (we use 2016)

## Outputs

All results are saved to:
- `predictions/` - Model predictions in CSV format
- `metrics/` - Evaluation metrics (Brier Score, Log Loss, MAE)
- `plots/` - State-level forecast visualizations (organized by model)

## License

MIT

````md
## Diagnostics and tests

This repository includes a small diagnostics suite to check both the
calibration utilities and the end-to-end forecasting pipeline.

### 1. Generate predictions

Most diagnostics expect that model predictions have already been generated:

```bash
# From the repository root
source .venv/bin/activate
election-run-all
```

This runs all configured models and writes prediction files under
`predictions/` (and corresponding metrics/summary files).

---

### 2. Global calibration diagnostics

The script `src/scripts/calibration_diagnostics.py` loads the prediction
files and computes **overall calibration statistics** and **reliability
curves** for each model, aggregating across all states and dates.

It writes CSV summaries and plots to disk.

```bash
source .venv/bin/activate
python src/scripts/calibration_diagnostics.py
```

Inspect the outputs (e.g. CSVs and PNGs) in the `diagnostics/` directory
as configured inside the script.

---

### 3. Per-state calibration and error diagnostics

To understand how models behave in individual states, we provide a
per-state diagnostics script:

```bash
source .venv/bin/activate
python src/scripts/per_state_calibration.py
```

This script:

* Loads all prediction CSVs from `predictions/`.
* Aggregates predictions **by state and model**.
* Computes:
  * Mean predicted Democratic win probability by state.
  * Mean empirical Democratic win rate by state.
  * Average predicted margin vs. actual margin by state.
  * Simple binned calibration summaries within each state (optional).

The outputs are written to:

* `diagnostics/per_state/per_state_metrics.csv`: per-state error and
  summary metrics for each model (e.g. average margin error by state).
* `diagnostics/per_state/per_state_calibration.csv`: optional per-state
  binned calibration statistics, if enough data are available.
* `diagnostics/per_state/calibration_<model>_<state>.png`: reliability
  curves for specific (model, state) combinations.

These diagnostics are useful for identifying states where a model tends
to **under-** or **over-predict** the Democratic margin or win
probability.

---

### 4. Diagnostics by forecast horizon (days until election)

To study how model performance changes as Election Day approaches, we
include horizon-based diagnostics:

```bash
source .venv/bin/activate
python src/scripts/horizon_diagnostics.py
```

This script:

* Stacks all prediction CSVs from `predictions/` into a single table.
* Computes the **forecast horizon** for each prediction:
  * `days_until_election = (Election Day – forecast_date)`.
* Groups by `(model, days_until_election)` and computes:
  * Brier score for win probabilities.
  * Log loss for win probabilities.
  * Mean absolute error (MAE) of predicted margins.
  * Mean predicted win probability vs. empirical win rate.
  * Mean predicted margin vs. mean actual margin.

Outputs are written to:

* `diagnostics/horizon/horizon_metrics.csv`: one row per
  `(model, days_until_election)` with the metrics above.
* `diagnostics/horizon/horizon_brier_score_<model>.png`:
  Brier score vs. days until election.
* `diagnostics/horizon/horizon_mae_margin_<model>.png`:
  margin MAE vs. days until election.
* `diagnostics/horizon/horizon_log_loss_<model>.png`:
  log loss vs. days until election.

These plots and tables summarize whether a model becomes more accurate
and better calibrated as the election gets closer, and how far in
advance its predictions are reliable.

---

### 5. Tests

We provide several test files:

* `tests/test_calibration.py` tests the low-level calibration helper
  functions in `src/utils/calibration.py` against small synthetic
  examples with known answers.
* `tests/test_smoke_models.py` is a “smoke test” that discovers the
  first forecasting model, runs it on a small number of forecast dates,
  and checks that the predictions and metrics are non-empty and contain
  only finite numeric values.
* `tests/test_horizon_diagnostics.py` checks that the horizon
  diagnostics (`src/diagnostics/horizon.py`) behave as expected on a
  small synthetic dataset and that the main summary columns are present.

To run these tests:

```bash
source .venv/bin/activate

# Run only the calibration tests
pytest tests/test_calibration.py

# Run only the smoke test
pytest tests/test_smoke_models.py

# Run only the horizon diagnostics tests
pytest tests/test_horizon_diagnostics.py

# Or run the full test suite
pytest
```
