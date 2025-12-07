import pandas as pd

from src.utils.calibration import compute_reliability_curve, compute_brier_score


def test_compute_reliability_curve_perfect_calibration():
    """
    Construct synthetic data that is perfectly calibrated:
    - 10 cases with p=0.1 and 1 Democratic win
    - 10 cases with p=0.5 and 5 Democratic wins
    - 10 cases with p=0.9 and 9 Democratic wins
    """
    probs = [0.1] * 10 + [0.5] * 10 + [0.9] * 10

    # actual_margin > 0 => Dem win
    margins = (
        [1.0]
        + [-1.0] * 9  # 1/10 wins at p=0.1
        + [1.0] * 5
        + [-1.0] * 5  # 5/10 wins at p=0.5
        + [1.0] * 9
        + [-1.0]  # 9/10 wins at p=0.9
    )

    df = pd.DataFrame(
        {
            "win_probability": probs,
            "actual_margin": margins,
        }
    )

    reliability = compute_reliability_curve(df, n_bins=3)

    # We expect three non-empty bins
    assert len(reliability) == 3

    # In each bin, mean predicted prob should equal empirical win rate
    diff = (
        (reliability["mean_predicted"] - reliability["empirical_win_rate"]).abs().max()
    )
    assert diff < 1e-6


def test_compute_reliability_curve_no_empty_bins_returned():
    """
    If some probability bins have no observations, they should
    be dropped from the reliability curve (no NaNs).
    """
    df = pd.DataFrame(
        {
            "win_probability": [0.7, 0.8, 0.9],
            "actual_margin": [1.0, 1.0, -1.0],
        }
    )

    reliability = compute_reliability_curve(df, n_bins=5)

    # All rows correspond to bins with at least one observation
    assert (reliability["count"] >= 1).all()
    assert not reliability["count"].isna().any()


def test_probabilities_are_clipped_before_calculation():
    """
    Probabilities outside [0, 1] should be clipped so that
    diagnostics don't break if a model slightly overshoots.
    """
    df = pd.DataFrame(
        {
            "win_probability": [-0.2, 1.2],
            "actual_margin": [-1.0, 1.0],
        }
    )

    reliability = compute_reliability_curve(df, n_bins=2)

    # After clipping we effectively have probabilities 0.0 and 1.0
    means = sorted(reliability["mean_predicted"].tolist())
    assert means[0] == 0.0
    assert means[1] == 1.0

    # Brier score should be 0 for perfectly correct extreme predictions
    brier = compute_brier_score(df)
    assert brier == 0.0
