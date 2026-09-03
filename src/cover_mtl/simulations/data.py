"""Task-indexed data containers shared by the formal simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
TaskArrays = tuple[FloatArray, ...]


class SimulationConfig(Protocol):
    """Fields required by the common simulation runner."""

    scenario: str
    num_tasks: int
    input_dim: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        ...


@dataclass(frozen=True)
class SimulationSplit:
    """One task-indexed split, with equal or task-specific sample sizes."""

    x: FloatArray | TaskArrays
    y: FloatArray | TaskArrays
    conditional_mean: FloatArray | TaskArrays
    common: FloatArray | TaskArrays
    deviation: FloatArray | TaskArrays

    @property
    def num_tasks(self) -> int:
        return len(self.x)

    @property
    def sample_sizes(self) -> tuple[int, ...]:
        return tuple(array.shape[0] for array in self.x_by_task())

    @property
    def sample_size(self) -> int:
        """Return the smallest task size for balanced mini-batches."""
        return min(self.sample_sizes)

    def stacked(self) -> tuple[FloatArray, FloatArray, IntArray]:
        x_by_task = self.x_by_task()
        y_by_task = self.y_by_task()
        tasks = np.concatenate(
            [
                np.full(array.shape[0], task, dtype=np.int64)
                for task, array in enumerate(x_by_task)
            ]
        )
        return (
            np.concatenate(x_by_task, axis=0),
            np.concatenate(y_by_task, axis=0),
            tasks,
        )

    def x_by_task(self) -> list[FloatArray]:
        return [np.asarray(self.x[task]) for task in range(self.num_tasks)]

    def y_by_task(self) -> list[FloatArray]:
        return [np.asarray(self.y[task]) for task in range(self.num_tasks)]


@dataclass(frozen=True)
class SimulationReplicate:
    """Independent training, validation, and test splits with population truth."""

    config: SimulationConfig
    train: SimulationSplit
    validation: SimulationSplit
    test: SimulationSplit
    truth: Any
