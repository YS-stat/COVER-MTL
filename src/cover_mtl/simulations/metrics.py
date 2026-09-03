"""Method-agnostic prediction and function-level evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .data import SimulationSplit


FloatArray = NDArray[np.float64]
PredictionFunction = Callable[[FloatArray], FloatArray]


def evaluate_functions(
    predict_all_tasks: PredictionFunction, test: SimulationSplit,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Evaluate every estimator through its induced task-balanced decomposition."""
    task_rows: list[dict[str, float]] = []
    for task_index in range(test.num_tasks):
        task_x = test.x[task_index]
        task_y = test.y[task_index]
        task_mean = test.conditional_mean[task_index]
        task_common = test.common[task_index]
        task_deviation = test.deviation[task_index]
        all_predictions = np.asarray(predict_all_tasks(task_x), dtype=np.float64)
        expected_shape = (task_x.shape[0], test.num_tasks)
        if all_predictions.shape != expected_shape:
            raise ValueError(
                f"predict_all_tasks returned {all_predictions.shape}, expected {expected_shape}."
            )
        prediction = all_predictions[:, task_index]
        estimated_common = all_predictions.mean(axis=1)
        estimated_deviation = prediction - estimated_common
        task_rows.append(
            {
                "task": float(task_index),
                "prediction_mse": float(np.mean((task_y - prediction) ** 2)),
                "excess_mse": float(np.mean((task_mean - prediction) ** 2)),
                "common_mse": float(np.mean((task_common - estimated_common) ** 2)),
                "deviation_mse": float(
                    np.mean((task_deviation - estimated_deviation) ** 2)
                ),
                "estimated_deviation_energy": float(np.mean(estimated_deviation ** 2)),
            }
        )
    summary = {
        "prediction_mse": float(np.mean([row["prediction_mse"] for row in task_rows])),
        "excess_mse": float(np.mean([row["excess_mse"] for row in task_rows])),
        "worst_task_excess_mse": float(
            np.max([row["excess_mse"] for row in task_rows])
        ),
        "common_mse": float(np.mean([row["common_mse"] for row in task_rows])),
        "deviation_mse": float(np.mean([row["deviation_mse"] for row in task_rows])),
        "estimated_deviation_energy": float(
            np.mean([row["estimated_deviation_energy"] for row in task_rows])
        ),
    }
    return summary, task_rows
