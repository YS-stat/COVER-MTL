"""Convex vanilla ARMUL baseline for a fixed nonlinear representation.

The Gaussian objective is

    sum_t ||y_t - Z_t (gamma - v_t)||^2 / (2 n_t)
        + lambda sum_t ||v_t||_2.

It is optimized by proximal gradient with backtracking.  This reproduces the
vanilla ARMUL structure while replacing the legacy fixed-step implementation
with a convergence-checked solver suitable for the revision experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _validate(
    x_by_task: Sequence[ArrayLike], y_by_task: Sequence[ArrayLike],
) -> tuple[list[FloatArray], list[FloatArray]]:
    if len(x_by_task) != len(y_by_task) or len(x_by_task) < 2:
        raise ValueError("ARMUL requires matching data from at least two tasks.")
    x_arrays = [np.asarray(value, dtype=np.float64) for value in x_by_task]
    y_arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in y_by_task]
    dimension = x_arrays[0].shape[1]
    for x_task, y_task in zip(x_arrays, y_arrays):
        if x_task.ndim != 2 or x_task.shape != (y_task.shape[0], dimension):
            raise ValueError("ARMUL task arrays have incompatible shapes.")
        if (
            x_task.shape[0] == 0
            or not np.all(np.isfinite(x_task))
            or not np.all(np.isfinite(y_task))
        ):
            raise ValueError("ARMUL task arrays must be nonempty and finite.")
    return x_arrays, y_arrays


def _ridge_coefficients(
    x_by_task: Sequence[FloatArray], y_by_task: Sequence[FloatArray], ridge: float,
) -> FloatArray:
    dimension = x_by_task[0].shape[1]
    identity = np.eye(dimension)
    coefficients = []
    for x_task, y_task in zip(x_by_task, y_by_task):
        gram = x_task.T @ x_task / x_task.shape[0]
        score = x_task.T @ y_task / x_task.shape[0]
        if ridge == 0:
            coefficients.append(np.linalg.lstsq(x_task, y_task, rcond=None)[0])
        else:
            coefficients.append(np.linalg.solve(gram + ridge * identity, score))
    return np.asarray(coefficients)


def _pooled_coefficient(
    x_by_task: Sequence[FloatArray], y_by_task: Sequence[FloatArray],
) -> FloatArray:
    dimension = x_by_task[0].shape[1]
    gram = np.zeros((dimension, dimension), dtype=np.float64)
    score = np.zeros(dimension, dtype=np.float64)
    for x_task, y_task in zip(x_by_task, y_by_task):
        gram += x_task.T @ x_task / x_task.shape[0]
        score += x_task.T @ y_task / x_task.shape[0]
    return np.linalg.lstsq(gram, score, rcond=None)[0]


def _smooth_value_and_gradient(
    x_by_task: Sequence[FloatArray],
    y_by_task: Sequence[FloatArray],
    common: FloatArray,
    corrections: FloatArray,
) -> tuple[float, FloatArray, FloatArray]:
    sample_sizes = {x_task.shape[0] for x_task in x_by_task}
    if len(sample_sizes) == 1:
        x_array = np.stack(x_by_task)
        y_array = np.stack(y_by_task)
        coefficients = common[None, :] - corrections
        residual = np.einsum("tnd,td->tn", x_array, coefficients) - y_array
        task_scores = np.einsum("tnd,tn->td", x_array, residual) / x_array.shape[1]
        return (
            0.5 * float(np.sum(np.mean(residual ** 2, axis=1))),
            task_scores.sum(axis=0),
            -task_scores,
        )
    common_gradient = np.zeros_like(common)
    correction_gradient = np.empty_like(corrections)
    value = 0.0
    for task, (x_task, y_task) in enumerate(zip(x_by_task, y_by_task)):
        residual = x_task @ (common - corrections[task]) - y_task
        score = x_task.T @ residual / x_task.shape[0]
        value += 0.5 * float(np.mean(residual ** 2))
        common_gradient += score
        correction_gradient[task] = -score
    return value, common_gradient, correction_gradient


def _group_soft_threshold(value: FloatArray, threshold: float) -> FloatArray:
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    scale = np.maximum(0.0, 1.0 - threshold / np.maximum(norms, 1e-300))
    return scale * value


@dataclass(frozen=True)
class ARMULFit:
    lambda_fuse: float
    common: FloatArray
    corrections: FloatArray
    coefficients: FloatArray
    converged: bool
    iterations: int
    objective: float

    def predict_by_task(self, x_by_task: Sequence[ArrayLike]) -> list[FloatArray]:
        if len(x_by_task) != self.coefficients.shape[0]:
            raise ValueError("Prediction data must contain every fitted task.")
        return [
            np.asarray(x_task, dtype=np.float64) @ self.coefficients[task]
            for task, x_task in enumerate(x_by_task)
        ]


@dataclass(frozen=True)
class ARMULPathResult:
    fit: ARMULFit
    selected_ratio: float
    selected_lambda: float
    validation_mse: float
    tuning_rows: tuple[dict[str, float | bool], ...]


@dataclass
class ARMULPreparedProblem:
    x_by_task: list[FloatArray]
    y_by_task: list[FloatArray]
    initial_coefficients: FloatArray

    @property
    def lambda_max(self) -> float:
        common = _pooled_coefficient(self.x_by_task, self.y_by_task)
        scores = []
        for x_task, y_task in zip(self.x_by_task, self.y_by_task):
            residual = x_task @ common - y_task
            scores.append(np.linalg.norm(x_task.T @ residual / x_task.shape[0]))
        return float(max(scores))

    def fit(
        self, lambda_fuse: float, *, max_iter: int = 10_000, tolerance: float = 1e-8,
    ) -> ARMULFit:
        if lambda_fuse < 0 or max_iter <= 0 or tolerance <= 0:
            raise ValueError("Invalid ARMUL optimization control.")
        if lambda_fuse == 0:
            coefficients = self.initial_coefficients.copy()
            common = coefficients.mean(axis=0)
            corrections = common[None, :] - coefficients
            value, _, _ = _smooth_value_and_gradient(
                self.x_by_task, self.y_by_task, common, corrections
            )
            return ARMULFit(
                lambda_fuse=0.0,
                common=common,
                corrections=corrections,
                coefficients=coefficients,
                converged=True,
                iterations=0,
                objective=value,
            )

        if lambda_fuse >= self.lambda_max * (1.0 - 1e-12):
            common = _pooled_coefficient(self.x_by_task, self.y_by_task)
            corrections = np.zeros_like(self.initial_coefficients)
            value, _, _ = _smooth_value_and_gradient(
                self.x_by_task, self.y_by_task, common, corrections
            )
            return ARMULFit(
                lambda_fuse=float(lambda_fuse),
                common=common,
                corrections=corrections,
                coefficients=np.repeat(common[None, :], len(self.x_by_task), axis=0),
                converged=True,
                iterations=1,
                objective=value,
            )

        # The smooth Hessian is bounded by (T + 1) times the largest
        # task-specific Gram eigenvalue.  This gives a fixed valid step and
        # permits accelerated proximal gradient without line-search loops.
        largest_gram_eigenvalue = max(
            float(np.linalg.eigvalsh(x_task.T @ x_task / x_task.shape[0]).max())
            for x_task in self.x_by_task
        )
        step = 1.0 / ((len(self.x_by_task) + 1.0) * largest_gram_eigenvalue)
        common = self.initial_coefficients.mean(axis=0)
        corrections = common[None, :] - self.initial_coefficients
        accelerated_common = common.copy()
        accelerated_corrections = corrections.copy()
        acceleration = 1.0
        converged = False
        objective = float("inf")
        for iteration in range(1, max_iter + 1):
            _, common_gradient, correction_gradient = _smooth_value_and_gradient(
                self.x_by_task,
                self.y_by_task,
                accelerated_common,
                accelerated_corrections,
            )
            new_common = accelerated_common - step * common_gradient
            new_corrections = _group_soft_threshold(
                accelerated_corrections - step * correction_gradient,
                step * lambda_fuse,
            )
            change = np.sqrt(
                float(np.sum((new_common - common) ** 2))
                + float(np.sum((new_corrections - corrections) ** 2))
            )
            scale = 1.0 + np.sqrt(
                float(common @ common) + float(np.sum(corrections ** 2))
            )
            new_acceleration = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * acceleration ** 2))
            momentum = (acceleration - 1.0) / new_acceleration
            accelerated_common = new_common + momentum * (new_common - common)
            accelerated_corrections = new_corrections + momentum * (
                new_corrections - corrections
            )
            common, corrections = new_common, new_corrections
            acceleration = new_acceleration
            new_smooth, _, _ = _smooth_value_and_gradient(
                self.x_by_task, self.y_by_task, common, corrections
            )
            objective = new_smooth + lambda_fuse * float(
                np.linalg.norm(corrections, axis=1).sum()
            )
            if change <= tolerance * scale:
                converged = True
                break
        coefficients = common[None, :] - corrections
        return ARMULFit(
            lambda_fuse=float(lambda_fuse),
            common=common,
            corrections=corrections,
            coefficients=coefficients,
            converged=converged,
            iterations=iteration,
            objective=objective,
        )


def prepare_armul(
    x_by_task: Sequence[ArrayLike],
    y_by_task: Sequence[ArrayLike],
    *,
    initial_ridge: float = 1e-4,
) -> ARMULPreparedProblem:
    x_arrays, y_arrays = _validate(x_by_task, y_by_task)
    if initial_ridge < 0:
        raise ValueError("initial_ridge must be nonnegative.")
    return ARMULPreparedProblem(
        x_by_task=x_arrays,
        y_by_task=y_arrays,
        initial_coefficients=_ridge_coefficients(x_arrays, y_arrays, initial_ridge),
    )


def tune_armul(
    train_x_by_task: Sequence[ArrayLike],
    train_y_by_task: Sequence[ArrayLike],
    validation_x_by_task: Sequence[ArrayLike],
    validation_y_by_task: Sequence[ArrayLike],
    *,
    lambda_ratios: Iterable[float] = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0),
    initial_ridge: float = 1e-4,
    max_iter: int = 5_000,
    tolerance: float = 1e-5,
) -> ARMULPathResult:
    validation_x, validation_y = _validate(validation_x_by_task, validation_y_by_task)
    problem = prepare_armul(
        train_x_by_task, train_y_by_task, initial_ridge=initial_ridge
    )
    ratios = sorted({float(value) for value in lambda_ratios})
    if not ratios or ratios[0] < 0:
        raise ValueError("lambda_ratios must contain nonnegative values.")
    rows: list[dict[str, float | bool]] = []
    candidates: list[tuple[float, float, ARMULFit]] = []
    for ratio in ratios:
        fit = problem.fit(
            ratio * problem.lambda_max, max_iter=max_iter, tolerance=tolerance,
        )
        losses = [
            np.mean((prediction - response) ** 2)
            for prediction, response in zip(
                fit.predict_by_task(validation_x), validation_y
            )
        ]
        validation_mse = float(np.mean(losses))
        rows.append(
            {
                "lambda_ratio": ratio,
                "lambda_fuse": fit.lambda_fuse,
                "validation_mse": validation_mse,
                "mean_correction_norm": float(
                    np.linalg.norm(fit.corrections, axis=1).mean()
                ),
                "iterations": float(fit.iterations),
                "converged": fit.converged,
            }
        )
        candidates.append((validation_mse, -ratio, fit))
    validation_mse, negative_ratio, selected = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return ARMULPathResult(
        fit=selected,
        selected_ratio=-negative_ratio,
        selected_lambda=selected.lambda_fuse,
        validation_mse=validation_mse,
        tuning_rows=tuple(rows),
    )
