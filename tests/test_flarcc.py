"""Mathematical and behavioral tests for the Python FLARCC baseline."""

from __future__ import annotations

import numpy as np

from cover_mtl.simulations.baselines.flarcc import prepare_flarcc, tune_flarcc


def _linear_tasks(seed: int = 1):
    generator = np.random.default_rng(seed)
    coefficients = np.asarray(
        [[-1.0, 0.4], [-0.9, 0.4], [1.2, 0.4], [1.3, 0.4]], dtype=np.float64,
    )
    x_by_task = [generator.normal(size=(160, 2)) for _ in range(4)]
    y_by_task = [
        x @ coefficients[task] + generator.normal(scale=0.1, size=x.shape[0])
        for task, x in enumerate(x_by_task)
    ]
    return x_by_task, y_by_task, coefficients


def test_transformed_prediction_matches_reconstructed_coefficients():
    x_by_task, y_by_task, _ = _linear_tasks()
    problem = prepare_flarcc(x_by_task, y_by_task, initial_ridge=1e-6)
    fit = problem.fit(lambda_fuse=0.2 * problem.lambda_max)

    transformed_prediction = (
        problem.unpenalized_design
        @ np.concatenate(
            [fit.intercepts, fit.coefficients[problem.orders[:, 0], np.arange(2)]]
        )
        + problem.penalized_design @ fit.ordered_differences.ravel()
    )
    direct_prediction = np.concatenate(fit.predict_by_task(x_by_task))
    sample_counts = np.asarray([x.shape[0] for x in x_by_task])
    row_weights = np.sqrt(
        sum(sample_counts) / (len(x_by_task) * np.repeat(sample_counts, sample_counts))
    )
    np.testing.assert_allclose(transformed_prediction, direct_prediction * row_weights)


def test_zero_penalty_matches_independent_least_squares():
    x_by_task, y_by_task, _ = _linear_tasks(seed=2)
    problem = prepare_flarcc(x_by_task, y_by_task, initial_ridge=1e-8)
    fit = problem.fit(lambda_fuse=0.0)
    for task, (x_task, y_task) in enumerate(zip(x_by_task, y_by_task)):
        design = np.column_stack([np.ones(x_task.shape[0]), x_task])
        expected = np.linalg.lstsq(design, y_task, rcond=None)[0]
        np.testing.assert_allclose(fit.intercepts[task], expected[0], atol=1e-8)
        np.testing.assert_allclose(fit.coefficients[task], expected[1:], atol=1e-8)


def test_lambda_max_fuses_all_adjacent_differences():
    x_by_task, y_by_task, _ = _linear_tasks(seed=3)
    problem = prepare_flarcc(x_by_task, y_by_task)
    fit = problem.fit(lambda_fuse=1.01 * problem.lambda_max)
    assert fit.adjacent_fusion_fraction(tolerance=1e-7) == 1.0
    np.testing.assert_allclose(
        fit.coefficients,
        np.repeat(fit.coefficients[:1], fit.coefficients.shape[0], axis=0),
        atol=1e-7,
    )


def test_fitted_point_improves_its_objective_over_zero_differences():
    x_by_task, y_by_task, _ = _linear_tasks(seed=5)
    problem = prepare_flarcc(x_by_task, y_by_task)
    lambda_fuse = 0.1 * problem.lambda_max
    fitted = problem.fit(lambda_fuse=lambda_fuse)
    fully_fused = problem.fit(lambda_fuse=1.01 * problem.lambda_max)
    fused_at_same_lambda = type(fully_fused)(
        lambda_fuse=lambda_fuse,
        intercepts=fully_fused.intercepts,
        coefficients=fully_fused.coefficients,
        ordered_differences=fully_fused.ordered_differences,
        converged=fully_fused.converged,
        iterations=fully_fused.iterations,
    )
    assert problem.objective(fitted) <= problem.objective(fused_at_same_lambda) + 1e-8


def test_unequal_sample_sizes_use_task_balanced_validation():
    generator = np.random.default_rng(4)
    train_x = [generator.normal(size=(40, 2)), generator.normal(size=(180, 2))]
    train_y = [train_x[0][:, 0], -train_x[1][:, 0]]
    validation_x = [generator.normal(size=(25, 2)), generator.normal(size=(75, 2))]
    validation_y = [validation_x[0][:, 0], -validation_x[1][:, 0]]
    result = tune_flarcc(
        train_x, train_y, validation_x, validation_y, lambda_ratios=(0.0, 0.1, 1.0),
    )
    manual = np.mean(
        [
            np.mean((prediction - response) ** 2)
            for prediction, response in zip(
                result.fit.predict_by_task(validation_x), validation_y
            )
        ]
    )
    np.testing.assert_allclose(result.validation_mse, manual)


def test_intercept_free_fit_keeps_task_intercepts_at_zero():
    x_by_task, y_by_task, _ = _linear_tasks(seed=6)
    problem = prepare_flarcc(x_by_task, y_by_task, include_task_intercepts=False,)
    fit = problem.fit(lambda_fuse=0.1 * problem.lambda_max)
    np.testing.assert_allclose(fit.intercepts, 0.0)
