import numpy as np
import pytest

from src.metrics.evaluator import MetricsEvaluator


def test_wape_is_zero_when_predictions_are_perfect():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([100, 200, 300])

    result = MetricsEvaluator.calculate_wape(
        y_true, y_pred
    )

    assert result == pytest.approx(0.0)


def test_wape_calculation():
    y_true = np.array([100, 200])
    y_pred = np.array([110, 180])

    result = MetricsEvaluator.calculate_wape(
        y_true, y_pred
    )

    # Error absoluto total: 10 + 20 = 30.
    # Demanda real total: 100 + 200 = 300.
    assert result == pytest.approx(0.10)


def test_bias_is_positive_for_overforecast():
    y_true = np.array([100, 200])
    y_pred = np.array([110, 220])

    result = MetricsEvaluator.calculate_bias(
        y_true, y_pred
    )

    assert result > 0
    assert result == pytest.approx(0.10)


def test_bias_is_negative_for_underforecast():
    y_true = np.array([100, 200])
    y_pred = np.array([90, 180])

    result = MetricsEvaluator.calculate_bias(
        y_true, y_pred
    )

    assert result < 0
    assert result == pytest.approx(-0.10)


def test_wape_returns_nan_when_actuals_sum_to_zero():
    y_true = np.array([0, 0])
    y_pred = np.array([0, 0])

    result = MetricsEvaluator.calculate_wape(
        y_true, y_pred
    )

    assert np.isnan(result)
