"""Python implementation of FLARCC for Gaussian multi-study regression.

The implementation follows Tang and Song (2016): initial study-specific
coefficient estimates determine a separate task ordering for each feature,
and the fusion penalty is applied only to adjacent coefficients in that
ordering.  The ordered differences are reparameterized as weighted lasso
coefficients.  Task contributions to the squared-error loss are balanced,
including when sample sizes differ.

This module implements the fusion-only version used as a prediction baseline.
The optional variable-selection term from the original paper is deliberately
excluded so that tuning changes cross-task fusion rather than feature sparsity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import Lasso


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _as_float_matrix(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _as_float_vector(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _validate_task_data(
    x_by_task: Sequence[ArrayLike], y_by_task: Sequence[ArrayLike],
) -> tuple[list[FloatArray], list[FloatArray]]:
    if len(x_by_task) != len(y_by_task) or len(x_by_task) < 2:
        raise ValueError("x_by_task and y_by_task must describe at least two tasks.")
    x_arrays = [
        _as_float_matrix(value, f"x_by_task[{index}]")
        for index, value in enumerate(x_by_task)
    ]
    y_arrays = [
        _as_float_vector(value, f"y_by_task[{index}]")
        for index, value in enumerate(y_by_task)
    ]
    feature_count = x_arrays[0].shape[1]
    if feature_count == 0:
        raise ValueError("At least one feature is required.")
    for index, (x_task, y_task) in enumerate(zip(x_arrays, y_arrays)):
        if x_task.shape[1] != feature_count:
            raise ValueError("All tasks must use the same number of features.")
        if x_task.shape[0] != y_task.shape[0] or x_task.shape[0] == 0:
            raise ValueError(f"Task {index} has incompatible or empty data.")
    return x_arrays, y_arrays


def _ridge_initial_estimates(
    x_by_task: Sequence[FloatArray],
    y_by_task: Sequence[FloatArray],
    ridge: float,
    include_task_intercepts: bool,
) -> tuple[FloatArray, FloatArray]:
    """Compute stable task-specific initial estimates used only for ordering."""
    if ridge < 0:
        raise ValueError("initial_ridge must be nonnegative.")
    task_count = len(x_by_task)
    feature_count = x_by_task[0].shape[1]
    intercepts = np.empty(task_count, dtype=np.float64)
    coefficients = np.empty((task_count, feature_count), dtype=np.float64)
    identity = np.eye(feature_count, dtype=np.float64)
    for task_index, (x_task, y_task) in enumerate(zip(x_by_task, y_by_task)):
        x_mean = (
            x_task.mean(axis=0) if include_task_intercepts else np.zeros(feature_count)
        )
        y_mean = float(y_task.mean()) if include_task_intercepts else 0.0
        x_centered = x_task - x_mean
        y_centered = y_task - y_mean
        if ridge == 0:
            coefficient = np.linalg.lstsq(x_centered, y_centered, rcond=None)[0]
        else:
            gram = x_centered.T @ x_centered / x_task.shape[0]
            score = x_centered.T @ y_centered / x_task.shape[0]
            coefficient = np.linalg.solve(gram + ridge * identity, score)
        coefficients[task_index] = coefficient
        intercepts[task_index] = y_mean - float(x_mean @ coefficient)
    return intercepts, coefficients


def _feature_orders(initial_coefficients: FloatArray) -> tuple[IntArray, IntArray]:
    """Return deterministic ascending task orders and their inverse ranks."""
    task_count, feature_count = initial_coefficients.shape
    orders = np.empty((feature_count, task_count), dtype=np.int64)
    ranks = np.empty_like(orders)
    task_indices = np.arange(task_count, dtype=np.int64)
    for feature_index in range(feature_count):
        order = np.lexsort((task_indices, initial_coefficients[:, feature_index]))
        orders[feature_index] = order
        ranks[feature_index, order] = np.arange(task_count, dtype=np.int64)
    return orders, ranks


def _adaptive_weights(
    initial_coefficients: FloatArray,
    orders: IntArray,
    difference_power: float,
    range_power: float,
    weight_floor: float,
    weight_cap: float,
) -> FloatArray:
    """Construct Tang--Song adaptive weights for adjacent ordered differences."""
    if difference_power < 0 or range_power < 0:
        raise ValueError("Adaptive-weight powers must be nonnegative.")
    if weight_floor <= 0 or weight_cap < 1:
        raise ValueError("weight_floor must be positive and weight_cap at least one.")
    task_count, feature_count = initial_coefficients.shape
    weights = np.empty((feature_count, task_count - 1), dtype=np.float64)
    for feature_index in range(feature_count):
        ordered = initial_coefficients[orders[feature_index], feature_index]
        adjacent = np.maximum(np.abs(np.diff(ordered)), weight_floor)
        coefficient_range = max(float(ordered[-1] - ordered[0]), weight_floor)
        weights[feature_index] = adjacent ** (
            -difference_power
        ) * coefficient_range ** (-range_power)
    return np.clip(weights, 1.0 / weight_cap, weight_cap)


def _orthonormal_column_space(matrix: FloatArray) -> FloatArray:
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * singular_values[0]
    return left[:, singular_values > tolerance]


def _build_transformed_design(
    x_by_task: Sequence[FloatArray], ranks: IntArray, include_task_intercepts: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray]:
    """Build balanced design matrices for unpenalized and fused parameters."""
    task_count = len(x_by_task)
    feature_count = x_by_task[0].shape[1]
    sample_counts = np.asarray([x.shape[0] for x in x_by_task], dtype=np.int64)
    total_count = int(sample_counts.sum())
    tasks = np.repeat(np.arange(task_count, dtype=np.int64), sample_counts)
    x_stacked = np.concatenate(x_by_task, axis=0)

    # Under the ordered-difference parameterization, every task contains the
    # first coefficient for each feature.  These feature bases and all task
    # intercepts are unpenalized.
    intercept_count = task_count if include_task_intercepts else 0
    unpenalized = np.zeros(
        (total_count, intercept_count + feature_count), dtype=np.float64
    )
    if include_task_intercepts:
        unpenalized[np.arange(total_count), tasks] = 1.0
    unpenalized[:, intercept_count:] = x_stacked

    penalized = np.zeros(
        (total_count, feature_count * (task_count - 1)), dtype=np.float64
    )
    for feature_index in range(feature_count):
        feature = x_stacked[:, feature_index]
        for difference_index in range(1, task_count):
            column = feature_index * (task_count - 1) + difference_index - 1
            active = ranks[feature_index, tasks] >= difference_index
            penalized[active, column] = feature[active]

    # Multiplying rows by sqrt(N / (T n_t)) makes sklearn's 1/(2N)
    # squared-error term equal to the task-balanced loss.
    row_weights = np.sqrt(total_count / (task_count * sample_counts[tasks]))
    return (
        unpenalized * row_weights[:, None],
        penalized * row_weights[:, None],
        row_weights,
        tasks,
    )


@dataclass(frozen=True)
class FLARCCFit:
    """One fitted point on the FLARCC fusion path."""

    lambda_fuse: float
    intercepts: FloatArray
    coefficients: FloatArray
    ordered_differences: FloatArray
    converged: bool
    iterations: int

    def predict_by_task(self, x_by_task: Sequence[ArrayLike]) -> list[FloatArray]:
        if len(x_by_task) != self.coefficients.shape[0]:
            raise ValueError("Prediction data must contain every fitted task.")
        predictions = []
        for task_index, value in enumerate(x_by_task):
            x_task = _as_float_matrix(value, f"x_by_task[{task_index}]")
            if x_task.shape[1] != self.coefficients.shape[1]:
                raise ValueError("Prediction features do not match the fitted model.")
            predictions.append(
                self.intercepts[task_index] + x_task @ self.coefficients[task_index]
            )
        return predictions

    def adjacent_fusion_fraction(self, tolerance: float = 1e-8) -> float:
        """Return the fraction of ordered adjacent differences estimated as zero."""
        return float(np.mean(np.abs(self.ordered_differences) <= tolerance))


@dataclass
class FLARCCPreparedProblem:
    """Fixed ordering and transformed design shared by a FLARCC tuning path."""

    x_by_task: list[FloatArray]
    y_by_task: list[FloatArray]
    initial_intercepts: FloatArray
    initial_coefficients: FloatArray
    orders: IntArray
    ranks: IntArray
    adaptive_weights: FloatArray
    unpenalized_design: FloatArray
    penalized_design: FloatArray
    weighted_response: FloatArray
    column_space: FloatArray
    residual_response: FloatArray
    residual_penalized_design: FloatArray
    include_task_intercepts: bool

    @property
    def task_count(self) -> int:
        return len(self.x_by_task)

    @property
    def feature_count(self) -> int:
        return self.x_by_task[0].shape[1]

    @property
    def lambda_max(self) -> float:
        """Smallest lasso value that sets all ordered differences to zero."""
        if self.residual_penalized_design.shape[1] == 0:
            return 0.0
        scaled = self.residual_penalized_design / self.adaptive_weights.ravel()
        score = np.abs(scaled.T @ self.residual_response) / scaled.shape[0]
        return float(score.max(initial=0.0))

    def objective(self, fit: FLARCCFit) -> float:
        """Evaluate the task-balanced Gaussian FLARCC training objective."""
        predictions = np.concatenate(fit.predict_by_task(self.x_by_task))
        responses = np.concatenate(self.y_by_task)
        task_losses = []
        offset = 0
        for y_task in self.y_by_task:
            count = y_task.shape[0]
            task_losses.append(
                np.mean(
                    (
                        responses[offset : offset + count]
                        - predictions[offset : offset + count]
                    )
                    ** 2
                )
            )
            offset += count
        loss = 0.5 * float(np.mean(task_losses))
        penalty = fit.lambda_fuse * float(
            np.sum(self.adaptive_weights * np.abs(fit.ordered_differences))
        )
        return loss + penalty

    def fit(
        self,
        lambda_fuse: float,
        max_iter: int = 50_000,
        tolerance: float = 1e-8,
        selection: str = "cyclic",
    ) -> FLARCCFit:
        """Fit one value of the fusion penalty with the ordering held fixed."""
        if lambda_fuse < 0:
            raise ValueError("lambda_fuse must be nonnegative.")
        if max_iter <= 0 or tolerance <= 0:
            raise ValueError("max_iter and tolerance must be positive.")

        weights = self.adaptive_weights.ravel()
        if lambda_fuse == 0:
            full_design = np.concatenate(
                [self.unpenalized_design, self.penalized_design], axis=1
            )
            full_solution = np.linalg.lstsq(
                full_design, self.weighted_response, rcond=None
            )[0]
            unpenalized_count = self.unpenalized_design.shape[1]
            unpenalized_solution = full_solution[:unpenalized_count]
            differences = full_solution[unpenalized_count:]
            converged = True
            iterations = 1
        else:
            scaled_design = self.residual_penalized_design / weights
            estimator = Lasso(
                alpha=lambda_fuse,
                fit_intercept=False,
                max_iter=max_iter,
                tol=tolerance,
                selection=selection,
            )
            estimator.fit(scaled_design, self.residual_response)
            differences = estimator.coef_ / weights
            unpenalized_solution = np.linalg.lstsq(
                self.unpenalized_design,
                self.weighted_response - self.penalized_design @ differences,
                rcond=None,
            )[0]
            iterations = int(estimator.n_iter_)
            converged = iterations < max_iter

        if self.include_task_intercepts:
            intercepts = unpenalized_solution[: self.task_count]
            bases = unpenalized_solution[self.task_count :]
        else:
            intercepts = np.zeros(self.task_count, dtype=np.float64)
            bases = unpenalized_solution
        difference_matrix = differences.reshape(self.feature_count, self.task_count - 1)
        coefficients = np.empty((self.task_count, self.feature_count), dtype=np.float64)
        for feature_index in range(self.feature_count):
            ordered_coefficients = np.concatenate(
                [
                    np.asarray([bases[feature_index]], dtype=np.float64),
                    bases[feature_index] + np.cumsum(difference_matrix[feature_index]),
                ]
            )
            coefficients[
                self.orders[feature_index], feature_index
            ] = ordered_coefficients

        return FLARCCFit(
            lambda_fuse=float(lambda_fuse),
            intercepts=intercepts,
            coefficients=coefficients,
            ordered_differences=difference_matrix,
            converged=converged,
            iterations=iterations,
        )


@dataclass(frozen=True)
class FLARCCPathResult:
    """Validation-selected FLARCC fit and the complete tuning table."""

    fit: FLARCCFit
    selected_ratio: float
    selected_lambda: float
    validation_mse: float
    tuning_rows: tuple[dict[str, float | bool], ...]


def prepare_flarcc(
    x_by_task: Sequence[ArrayLike],
    y_by_task: Sequence[ArrayLike],
    *,
    initial_ridge: float = 1e-4,
    difference_power: float = 1.0,
    range_power: float = 0.0,
    weight_floor: float = 1e-4,
    weight_cap: float = 1e4,
    include_task_intercepts: bool = True,
) -> FLARCCPreparedProblem:
    """Prepare the ordering, weights, and transformed Gaussian FLARCC design."""
    x_arrays, y_arrays = _validate_task_data(x_by_task, y_by_task)
    initial_intercepts, initial_coefficients = _ridge_initial_estimates(
        x_arrays, y_arrays, initial_ridge, include_task_intercepts
    )
    orders, ranks = _feature_orders(initial_coefficients)
    adaptive_weights = _adaptive_weights(
        initial_coefficients,
        orders,
        difference_power,
        range_power,
        weight_floor,
        weight_cap,
    )
    unpenalized, penalized, row_weights, _ = _build_transformed_design(
        x_arrays, ranks, include_task_intercepts
    )
    response = np.concatenate(y_arrays) * row_weights
    column_space = _orthonormal_column_space(unpenalized)
    residual_response = response - column_space @ (column_space.T @ response)
    residual_penalized = penalized - column_space @ (column_space.T @ penalized)
    return FLARCCPreparedProblem(
        x_by_task=x_arrays,
        y_by_task=y_arrays,
        initial_intercepts=initial_intercepts,
        initial_coefficients=initial_coefficients,
        orders=orders,
        ranks=ranks,
        adaptive_weights=adaptive_weights,
        unpenalized_design=unpenalized,
        penalized_design=penalized,
        weighted_response=response,
        column_space=column_space,
        residual_response=residual_response,
        residual_penalized_design=residual_penalized,
        include_task_intercepts=include_task_intercepts,
    )


def _balanced_mse(
    prediction_by_task: Sequence[FloatArray], response_by_task: Sequence[FloatArray],
) -> float:
    losses = [
        float(np.mean((prediction - response) ** 2))
        for prediction, response in zip(prediction_by_task, response_by_task)
    ]
    return float(np.mean(losses))


def tune_flarcc(
    train_x_by_task: Sequence[ArrayLike],
    train_y_by_task: Sequence[ArrayLike],
    validation_x_by_task: Sequence[ArrayLike],
    validation_y_by_task: Sequence[ArrayLike],
    *,
    lambda_ratios: Iterable[float] = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0),
    initial_ridge: float = 1e-4,
    difference_power: float = 1.0,
    range_power: float = 0.0,
    weight_floor: float = 1e-4,
    weight_cap: float = 1e4,
    include_task_intercepts: bool = True,
    max_iter: int = 50_000,
    tolerance: float = 1e-8,
) -> FLARCCPathResult:
    """Select the FLARCC fusion level by task-balanced validation MSE.

    Ratios are expressed relative to the data-dependent lasso value at which
    every ordered difference is zero.  This keeps one tuning grid meaningful
    across simulation replicates with different response and feature scales.
    """
    validation_x, validation_y = _validate_task_data(
        validation_x_by_task, validation_y_by_task
    )
    problem = prepare_flarcc(
        train_x_by_task,
        train_y_by_task,
        initial_ridge=initial_ridge,
        difference_power=difference_power,
        range_power=range_power,
        weight_floor=weight_floor,
        weight_cap=weight_cap,
        include_task_intercepts=include_task_intercepts,
    )
    if len(validation_x) != problem.task_count:
        raise ValueError("Training and validation data must contain the same tasks.")
    if validation_x[0].shape[1] != problem.feature_count:
        raise ValueError("Training and validation feature dimensions differ.")

    ratios = sorted({float(value) for value in lambda_ratios})
    if not ratios or ratios[0] < 0:
        raise ValueError("lambda_ratios must contain nonnegative values.")
    scale = problem.lambda_max
    rows: list[dict[str, float | bool]] = []
    fitted: list[tuple[float, float, FLARCCFit]] = []
    for ratio in ratios:
        lambda_fuse = ratio * scale
        fit = problem.fit(
            lambda_fuse=lambda_fuse, max_iter=max_iter, tolerance=tolerance,
        )
        validation_mse = _balanced_mse(fit.predict_by_task(validation_x), validation_y)
        rows.append(
            {
                "lambda_ratio": ratio,
                "lambda_fuse": lambda_fuse,
                "validation_mse": validation_mse,
                "fusion_fraction": fit.adjacent_fusion_fraction(),
                "iterations": float(fit.iterations),
                "converged": fit.converged,
            }
        )
        fitted.append((validation_mse, -ratio, fit))

    validation_mse, negative_ratio, selected = min(
        fitted, key=lambda item: (item[0], item[1])
    )
    return FLARCCPathResult(
        fit=selected,
        selected_ratio=-negative_ratio,
        selected_lambda=selected.lambda_fuse,
        validation_mse=validation_mse,
        tuning_rows=tuple(rows),
    )
