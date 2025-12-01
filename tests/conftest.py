"""Pytest configuration and fixtures"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_polls():
    """Generate sample polling data for testing"""
    dates = pd.date_range("2016-09-01", "2016-11-01", freq="3D")
    n_polls = len(dates)

    return pd.DataFrame(
        {
            "middate": dates,
            "state_code": ["FL"] * n_polls,
            "dem": np.random.uniform(0.45, 0.50, n_polls),
            "rep": np.random.uniform(0.45, 0.50, n_polls),
            "margin": np.random.uniform(-0.05, 0.05, n_polls),
            "dem_proportion": np.random.uniform(0.48, 0.52, n_polls),
            "samplesize": np.random.randint(500, 1500, n_polls),
            "pollster": [f"Pollster{i % 3}" for i in range(n_polls)],
            "state": ["Florida"] * n_polls,
        }
    )


@pytest.fixture
def sample_actual_results():
    """Generate sample actual election results"""
    return {"FL": 0.012, "PA": -0.007, "MI": -0.002, "WI": -0.008}


@pytest.fixture
def forecast_dates():
    """Standard forecast dates for testing"""
    return [pd.to_datetime("2016-10-15"), pd.to_datetime("2016-11-01")]


@pytest.fixture
def election_date():
    """Election date"""
    return pd.to_datetime("2016-11-08")


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directories"""
    (tmp_path / "predictions").mkdir()
    (tmp_path / "metrics").mkdir()
    (tmp_path / "plots").mkdir()
    return tmp_path
