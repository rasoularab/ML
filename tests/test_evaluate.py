import numpy as np

from quantml.evaluate import oos_r2


def test_perfect_prediction_gives_one():
    y = np.array([0.01, -0.02, 0.03, 0.0])
    assert oos_r2(y, y) == 1.0


def test_zero_forecast_against_zero_benchmark_gives_zero():
    y = np.array([0.01, -0.02, 0.03, 0.0])
    assert oos_r2(y, np.zeros_like(y)) == 0.0


def test_benchmark_is_used():
    y = np.array([1.0, 2.0, 3.0])
    bench = np.full(3, y.mean())
    assert abs(oos_r2(y, bench, y_bench=bench)) < 1e-12  # predicting the benchmark scores 0
