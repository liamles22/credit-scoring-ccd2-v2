import numpy as np

from scripts.paired_maturity_backtest import paired_intervals


def test_paired_intervals_include_direction_and_preserve_identical_scores():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    reversed_scores = -perfect

    intervals = paired_intervals(y, perfect, reversed_scores, perfect, n_boot=20)

    np.testing.assert_array_equal(intervals, [[2, 2], [0, 0], [-2, -2]])


def test_identical_imperfect_scores_have_zero_paired_difference():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.8, 0.3, 0.9])

    intervals = paired_intervals(y, score, score, score, n_boot=20)

    np.testing.assert_array_equal(intervals, np.zeros((3, 2)))
