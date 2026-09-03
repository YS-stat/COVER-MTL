"""Donor-grouped splits and leakage-free preprocessing for the GTEx experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

TARGET_BRAIN_TISSUES = (
    "brain_amygdala",
    "brain_anterior_cingulate_cortex_ba24",
    "brain_caudate_basal_ganglia",
    "brain_cerebellar_hemisphere",
    "brain_cerebellum",
    "brain_cortex",
    "brain_frontal_cortex_ba9",
    "brain_hypothalamus",
    "brain_nucleus_accumbens_basal_ganglia",
    "brain_putamen_basal_ganglia",
    "brain_spinal_cord_cervical_c-1",
)


@dataclass(frozen=True)
class RawTaskData:
    tissue: str
    donor: NDArray[np.str_]
    x: FloatArray
    y: FloatArray


@dataclass(frozen=True)
class TransformedFold:
    tissues: tuple[str, ...]
    train_x: tuple[FloatArray, ...]
    train_y: tuple[FloatArray, ...]
    validation_x: tuple[FloatArray, ...]
    validation_y: tuple[FloatArray, ...]
    test_x: tuple[FloatArray, ...]
    test_y: tuple[FloatArray, ...]
    test_y_raw: tuple[FloatArray, ...]
    response_mean: FloatArray
    response_scale: FloatArray
    selected_genes: tuple[str, ...]
    pca_eigenvalues: FloatArray
    fold_counts: tuple[dict[str, int], ...]


def load_tasks(path: Path, response_gene: str) -> tuple[list[RawTaskData], list[str]]:
    frame = pd.read_parquet(path)
    metadata = {"sample_id", "donor_id", "tissue"}
    gene_columns = [column for column in frame.columns if column not in metadata]
    if response_gene not in gene_columns:
        raise ValueError(f"Response gene {response_gene!r} is not available.")
    predictors = [gene for gene in gene_columns if gene not in {"JAM2", "SH2D2A"}]
    tasks: list[RawTaskData] = []
    for tissue, group in frame.groupby("tissue", sort=True):
        tasks.append(
            RawTaskData(
                tissue=str(tissue),
                donor=group["donor_id"].astype(str).to_numpy(),
                x=group[predictors].to_numpy(dtype=np.float64),
                y=group[response_gene].to_numpy(dtype=np.float64),
            )
        )
    return tasks, predictors


def donor_fold_map(tasks: list[RawTaskData], seed: int, folds: int = 5) -> dict[str, int]:
    donors = sorted({str(donor) for task in tasks for donor in task.donor})
    generator = np.random.default_rng(seed)
    generator.shuffle(donors)
    return {donor: index % folds for index, donor in enumerate(donors)}


def _partition_indices(
    donors: NDArray[np.str_], mapping: dict[str, int], test_fold: int, folds: int
) -> tuple[IntArray, IntArray, IntArray]:
    assignment = np.asarray([mapping[str(donor)] for donor in donors], dtype=np.int64)
    validation_fold = (test_fold + 1) % folds
    test = np.flatnonzero(assignment == test_fold)
    validation = np.flatnonzero(assignment == validation_fold)
    train = np.flatnonzero((assignment != test_fold) & (assignment != validation_fold))
    if min(train.size, validation.size, test.size) == 0:
        raise ValueError("Every tissue must have nonempty train, validation, and test sets.")
    return train, validation, test


def _task_balanced_location_scale(arrays: list[FloatArray]) -> tuple[FloatArray, FloatArray]:
    mean = np.mean([array.mean(axis=0) for array in arrays], axis=0)
    second = np.mean([(array * array).mean(axis=0) for array in arrays], axis=0)
    variance = np.maximum(second - mean * mean, 1e-12)
    return mean, np.sqrt(variance)


def build_fold(
    tasks: list[RawTaskData],
    predictor_names: list[str],
    mapping: dict[str, int],
    test_fold: int,
    *,
    folds: int = 5,
    screened_features: int = 500,
    principal_components: int = 30,
) -> TransformedFold:
    partitions = [
        _partition_indices(task.donor, mapping, test_fold, folds) for task in tasks
    ]
    train_raw = [task.x[indices[0]] for task, indices in zip(tasks, partitions)]
    feature_mean, feature_scale = _task_balanced_location_scale(train_raw)
    task_balanced_variance = feature_scale * feature_scale
    order = np.argsort(-task_balanced_variance, kind="stable")
    selected = order[: min(screened_features, order.size)]
    selected_names = tuple(predictor_names[index] for index in selected)

    standardized_train = [
        (array[:, selected] - feature_mean[selected]) / feature_scale[selected]
        for array in train_raw
    ]
    covariance = np.mean(
        [array.T @ array / array.shape[0] for array in standardized_train], axis=0
    )
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    descending = np.argsort(eigenvalues)[::-1]
    count = min(principal_components, int(np.sum(eigenvalues > 1e-10)))
    directions = eigenvectors[:, descending[:count]]
    retained = np.maximum(eigenvalues[descending[:count]], 1e-10)

    def transform(array: FloatArray) -> FloatArray:
        standardized = (
            array[:, selected] - feature_mean[selected]
        ) / feature_scale[selected]
        return (standardized @ directions / np.sqrt(retained)).astype(
            np.float64, copy=False
        )

    train_x = tuple(transform(task.x[index[0]]) for task, index in zip(tasks, partitions))
    validation_x = tuple(
        transform(task.x[index[1]]) for task, index in zip(tasks, partitions)
    )
    test_x = tuple(transform(task.x[index[2]]) for task, index in zip(tasks, partitions))

    response_mean = np.asarray(
        [task.y[index[0]].mean() for task, index in zip(tasks, partitions)],
        dtype=np.float64,
    )
    response_scale = np.asarray(
        [max(task.y[index[0]].std(ddof=0), 1e-8) for task, index in zip(tasks, partitions)],
        dtype=np.float64,
    )

    def transform_response(task_index: int, values: FloatArray) -> FloatArray:
        return (values - response_mean[task_index]) / response_scale[task_index]

    train_y = tuple(
        transform_response(task_index, task.y[index[0]])
        for task_index, (task, index) in enumerate(zip(tasks, partitions))
    )
    validation_y = tuple(
        transform_response(task_index, task.y[index[1]])
        for task_index, (task, index) in enumerate(zip(tasks, partitions))
    )
    test_y = tuple(
        transform_response(task_index, task.y[index[2]])
        for task_index, (task, index) in enumerate(zip(tasks, partitions))
    )
    test_y_raw = tuple(task.y[index[2]] for task, index in zip(tasks, partitions))
    fold_counts = tuple(
        {
            "train": int(index[0].size),
            "validation": int(index[1].size),
            "test": int(index[2].size),
        }
        for index in partitions
    )
    return TransformedFold(
        tissues=tuple(task.tissue for task in tasks),
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        test_x=test_x,
        test_y=test_y,
        test_y_raw=test_y_raw,
        response_mean=response_mean,
        response_scale=response_scale,
        selected_genes=selected_names,
        pca_eigenvalues=retained,
        fold_counts=fold_counts,
    )
