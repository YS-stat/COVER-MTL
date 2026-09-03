"""Tests for the convergence-checked vanilla ARMUL solver."""

from __future__ import annotations

import numpy as np

from cover_mtl.simulations.baselines.armul import prepare_armul, tune_armul


def _data(seed: int = 1):
    generator = np.random.default_rng(seed)
    beta = np.asarray([[-1.0, 0.5], [-0.8, 0.5], [1.0, 0.5], [1.2, 0.5]])
    x = [generator.normal(size=(120, 2)) for _ in range(4)]
    y = [
        x_task @ beta[task] + generator.normal(scale=0.1, size=120)
        for task, x_task in enumerate(x)
    ]
    return x, y


def test_zero_penalty_matches_independent_least_squares():
    x, y = _data(2)
    problem = prepare_armul(x, y, initial_ridge=0.0)
    fit = problem.fit(0.0)
    expected = np.asarray(
        [np.linalg.lstsq(x_task, y_task, rcond=None)[0] for x_task, y_task in zip(x, y)]
    )
    np.testing.assert_allclose(fit.coefficients, expected, atol=1e-10)


def test_lambda_above_threshold_recovers_pooled_solution():
    x, y = _data(3)
    problem = prepare_armul(x, y)
    fit = problem.fit(1.01 * problem.lambda_max)
    assert fit.converged
    np.testing.assert_allclose(fit.corrections, 0.0, atol=1e-7)
    np.testing.assert_allclose(
        fit.coefficients, np.repeat(fit.common[None, :], 4, axis=0), atol=1e-7,
    )


def test_validation_path_returns_a_finite_selected_fit():
    train_x, train_y = _data(4)
    validation_x, validation_y = _data(5)
    result = tune_armul(
        train_x,
        train_y,
        validation_x,
        validation_y,
        lambda_ratios=(0.0, 0.03, 0.3, 1.0),
    )
    assert np.isfinite(result.validation_mse)
    assert len(result.tuning_rows) == 4
