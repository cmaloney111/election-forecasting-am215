# Concurrency Analysis Report

## Executive Summary

This report analyzes concurrency opportunities in the election forecasting codebase, focusing on parallelization potential for improved performance. The analysis identifies several high-value opportunities, particularly in the hierarchical Bayes model's state-level processing.

## Current Architecture

The forecasting system processes predictions sequentially:
1. For each model
2. For each state (with sufficient polling data)
3. For each forecast date

**Current Performance**: ~2-3 seconds for 4 forecast dates across 51 states and 4 models

## Concurrency Opportunities

### 1. State-Level Parallelization (HIGH VALUE)

**Location**: `src/models/base_model.py:run_forecast()`

**Current Implementation**:
```python
for state in states:
    state_polls = polls[polls["state_code"] == state].copy()
    # ... process state sequentially
```

**Opportunity**: Each state's forecast is independent and can be computed in parallel.

**Expected Speedup**: 4-8x on multi-core machines (51 states, typical 4-8 cores)

**Implementation Strategy**:
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def run_forecast(self, forecast_dates=None, min_polls=10, verbose=False, n_workers=None):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(self._forecast_state, state, ...): state
            for state in states
        }
        for future in as_completed(futures):
            state = futures[future]
            result = future.result()
            # collect results
```

**Challenges**:
- Models must be pickleable (serialize/deserialize for multiprocessing)
- RNG state management for reproducibility with parallelization
- Memory overhead (multiple model copies)

**Alternative**: Use `ThreadPoolExecutor` for lighter-weight parallelism if I/O bound

### 2. Model-Level Parallelization (MEDIUM VALUE)

**Location**: `src/scripts/run_all_models.py:main()`

**Current Implementation**:
```python
for model_name, ModelClass in model_classes:
    model = ModelClass(seed=args.seed)
    pred_df = model.run_forecast(...)
```

**Opportunity**: Different models can run in parallel

**Expected Speedup**: 2-4x (4 models)

**Implementation Strategy**:
```python
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(run_model, ModelClass, args.seed, forecast_dates): model_name
        for model_name, ModelClass in model_classes
    }
```

**Challenges**:
- Shared file I/O (data loading happens in each model)
- Less granular progress reporting
- Memory: 4 complete data copies

### 3. Hierarchical Bayes: Monte Carlo Simulations (LOW-MEDIUM VALUE)

**Location**: `src/models/hierarchical_bayes.py` (hypothetical - current models don't have explicit MC loops)

**Current Pattern** (if it existed):
```python
for i in range(n_samples):
    sample = draw_sample()
    results.append(sample)
```

**Opportunity**: Monte Carlo draws can be parallelized

**Expected Speedup**: Limited by number of samples and overhead

**Note**: Current Kalman models use vectorized numpy operations which are already optimized. Parallelizing `simulate_forward()` would likely be counterproductive due to overhead.

### 4. Forecast Date Parallelization (LOW VALUE)

**Location**: `src/models/base_model.py:run_forecast()`

**Opportunity**: Different forecast dates could run in parallel

**Expected Speedup**: Minimal (only 2-4 forecast dates typically)

**Recommendation**: Not worth the complexity - better to parallelize at state level

## Recommended Implementation Priority

### Phase 1: State-Level Parallelization (Implement First)
- **Impact**: Highest speedup potential
- **Complexity**: Medium
- **Files to modify**: `src/models/base_model.py`
- **Add command-line flag**: `--workers/-w N` to control parallelism

### Phase 2: Model-Level Parallelization (Optional)
- **Impact**: Additional speedup when running all models
- **Complexity**: Low
- **Files to modify**: `src/scripts/run_all_models.py`
- **Can combine with Phase 1**: Run multiple models in parallel, each using multiple workers for states

### Phase 3: Advanced Optimizations (Future Work)
- **Caching**: Cache expensive EM algorithm results
- **JIT Compilation**: Use `numba` for Kalman filter loops
- **GPU**: Batch multiple states for GPU processing (overkill for this problem size)

## Implementation Considerations

### Reproducibility with Parallelization

Challenge: Maintaining reproducible results with parallel RNG

**Solution**: Use independent RNG streams per worker
```python
def _forecast_state(self, state, seed_base):
    # Create state-specific seed
    state_seed = seed_base + hash(state) % (2**31) if seed_base else None
    local_rng = np.random.default_rng(state_seed)
    # Use local_rng for this state
```

### Memory Management

- **Process pools**: Higher memory overhead (full data copy per worker)
- **Thread pools**: Shared memory but GIL limitations (Python's Global Interpreter Lock)
- **Recommendation**: Start with process pools for CPU-bound work, monitor memory

### Error Handling

```python
for future in as_completed(futures):
    try:
        result = future.result()
    except Exception as e:
        logger.error(f"State {futures[future]} failed: {e}")
        # Continue with other states
```

### Progress Reporting

Use `tqdm` with `as_completed()` for progress bar:
```python
from tqdm import tqdm

with tqdm(total=len(futures)) as pbar:
    for future in as_completed(futures):
        result = future.result()
        pbar.update(1)
```

## Benchmark Results (Estimated)

Based on typical multi-core systems (8 cores):

| Configuration | Runtime | Speedup |
|--------------|---------|---------|
| Current (sequential) | 2.5s | 1x |
| State-level parallel (8 workers) | 0.5s | 5x |
| Model + State parallel | 0.15s | 16x |

*Note: Actual speedup depends on CPU cores, memory bandwidth, and I/O performance*

## Code Example: Complete Implementation

```python
# src/models/base_model.py

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def _forecast_single_state(self, state, state_polls, forecast_dates,
                          election_date, actual_margin, min_polls, seed_base):
    """Helper method for parallel state processing"""
    results = []

    # Create state-specific RNG
    if seed_base is not None:
        state_seed = seed_base + abs(hash(state)) % (2**31)
        local_rng = np.random.default_rng(state_seed)
    else:
        local_rng = None

    for forecast_date in forecast_dates:
        train_polls = state_polls[state_polls["middate"] <= forecast_date].copy()
        if len(train_polls) < min_polls:
            continue

        try:
            result = self.fit_and_forecast(
                train_polls, forecast_date, election_date,
                actual_margin, rng=local_rng
            )
            results.append({
                "state": state,
                "forecast_date": forecast_date,
                **result,
                "actual_margin": actual_margin,
            })
        except Exception as e:
            self.logger.error(f"Error forecasting {state} on {forecast_date}: {e}")

    return results

def run_forecast(self, forecast_dates=None, min_polls=10,
                verbose=False, n_workers=None):
    """Run forecast with optional parallelization"""

    # ... existing setup code ...

    if n_workers == 1 or n_workers == 0:
        # Sequential execution (current behavior)
        for state in states:
            # ... existing loop ...
    else:
        # Parallel execution
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for state in states:
                state_polls = polls[polls["state_code"] == state].copy()
                if len(state_polls) < min_polls:
                    continue

                future = executor.submit(
                    self._forecast_single_state,
                    state, state_polls, forecast_dates,
                    election_date, actual_margin.get(state),
                    min_polls, self.rng.bit_generator.state if self.rng else None
                )
                futures[future] = state

            for future in as_completed(futures):
                state = futures[future]
                try:
                    state_results = future.result()
                    self.predictions.extend(state_results)
                    if verbose:
                        self.logger.info(f"Completed {state}")
                except Exception as e:
                    self.logger.error(f"Failed to process {state}: {e}")

    return pd.DataFrame(self.predictions)
```

## Conclusion

**Highest ROI**: Implement state-level parallelization in `base_model.py:run_forecast()`

**Expected Impact**: 5-8x speedup on typical hardware

**Implementation Effort**: 2-3 hours for core functionality + testing

**Recommendation**: Add `--workers` flag to control parallelism, default to `cpu_count() - 1`, with option to disable via `--workers 1` for debugging.

## Not Recommended

1. **Parallelizing within Kalman filter iterations**: Current numpy vectorization is already optimal
2. **Parallelizing `simulate_forward()`**: Overhead exceeds benefits for N=2000 samples
3. **GPU acceleration**: Problem size too small, data transfer overhead too high
4. **Distributed computing**: Unnecessary for this problem scale

## References

- Python `concurrent.futures` documentation
- NumPy parallel random number generation: https://numpy.org/doc/stable/reference/random/parallel.html
- Process vs Thread pools: https://docs.python.org/3/library/concurrent.futures.html
