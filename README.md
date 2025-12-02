# Election Forecasting Models

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
