from src.scripts.run_all_models import (
    discover_models,
    generate_forecast_dates,
)


def test_first_model_smoke():
    """
    Run the first discovered model on a small set of forecast dates
    and check that predictions and metrics look sane.
    """
    models = discover_models()
    assert models, "No models discovered in src.models"

    # Use only the first model to keep the test reasonably fast
    model_name, ModelClass = models[0]

    # Only a couple of forecast dates so this doesn't take forever
    forecast_dates = generate_forecast_dates(2)

    # Most models in this project accept a seed kwarg; if not, just ModelClass() works.
    model = ModelClass(seed=123)

    # Run a tiny forecast
    preds = model.run_forecast(
        forecast_dates=forecast_dates,
        verbose=False,
        n_workers=None,
    )

    # Let the model compute & save its usual metrics
    metrics = model.save_results()

    # 1. Predictions should not be empty
    assert not preds.empty, f"{model_name} produced no predictions"

    # 2. Metrics should not be empty
    assert not metrics.empty, f"{model_name} produced no metrics"

    # 3. All numeric prediction values should be finite
    pred_num = preds.select_dtypes(include="number")
    assert not pred_num.isna().any().any(), "NaNs in prediction numeric columns"
    assert not (pred_num == float("inf")).any().any(), (
        "inf in prediction numeric columns"
    )
    assert not (pred_num == float("-inf")).any().any(), (
        "-inf in prediction numeric columns"
    )

    # 4. All numeric metric values should be finite
    metrics_num = metrics.select_dtypes(include="number")
    assert not metrics_num.isna().any().any(), "NaNs in metric numeric columns"
    assert not (metrics_num == float("inf")).any().any(), (
        "inf in metric numeric columns"
    )
    assert not (metrics_num == float("-inf")).any().any(), (
        "-inf in metric numeric columns"
    )
