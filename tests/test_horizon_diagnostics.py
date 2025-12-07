import pandas as pd

from src.diagnostics.horizon import compute_horizon_metrics


def test_compute_horizon_metrics_basic():
    # Simple synthetic dataset for a single model
    df = pd.DataFrame(
        {
            "state": ["AA", "BB", "AA", "BB"],
            "forecast_date": [
                "2016-11-01",  # 7 days before election
                "2016-11-01",
                "2016-10-15",  # 24 days before election
                "2016-10-15",
            ],
            "win_probability": [0.8, 0.3, 0.6, 0.4],
            "predicted_margin": [5.0, -2.0, 3.0, -1.0],
            "margin_std": [1.0, 1.0, 1.0, 1.0],
            "actual_margin": [6.0, -3.0, 4.0, -2.0],
            "model": ["toy_model"] * 4,
        }
    )

    metrics = compute_horizon_metrics(df)

    # We expect two horizons: 7 and 24 days
    horizons = sorted(metrics["days_until_election"].unique().tolist())
    assert horizons == [7, 24]

    # Check required columns exist
    required_cols = {
        "model",
        "days_until_election",
        "n_obs",
        "brier_score",
        "log_loss",
        "mae_margin",
        "mean_pred_prob",
        "empirical_win_rate",
        "mean_predicted_margin",
        "mean_actual_margin",
    }
    assert required_cols.issubset(set(metrics.columns))

    # Both horizons should have 2 observations in this toy example
    assert metrics["n_obs"].tolist() == [2, 2]
