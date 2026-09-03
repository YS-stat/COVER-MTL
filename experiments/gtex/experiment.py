"""Shared configuration and utilities for the frozen GTEx experiment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from data import TransformedFold

from cover_mtl.models import (
    MMoEPredictor,
    MultiTaskPredictor,
    PooledPredictor,
    SharedDecompositionPredictor,
    SingleTaskPredictor,
)
from cover_mtl.simulations.config import OptimizationConfig
from cover_mtl.simulations.data import SimulationSplit
from cover_mtl.simulations.training import (
    NeuralFit,
    fit_neural_model,
    predict_decomposition,
    seed_everything,
)


COUPLINGS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


@dataclass(frozen=True)
class ExperimentConfig:
    fold_seed: int = 20260829
    model_seed: int = 20260830
    folds: int = 5
    screened_features: int = 500
    principal_components: int = 30
    hidden_width: int = 32
    representation_dim: int = 8
    steps: int = 1800
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size_per_task: int = 32
    evaluation_interval: int = 25
    patience_evaluations: int = 24


def derive_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31 - 1)


def as_simulation_split(x: tuple[np.ndarray, ...], y: tuple[np.ndarray, ...]) -> SimulationSplit:
    zeros = tuple(np.zeros_like(values) for values in y)
    return SimulationSplit(x=x, y=y, conditional_mean=zeros, common=zeros, deviation=zeros)


def optimization_config(config: ExperimentConfig) -> OptimizationConfig:
    return OptimizationConfig(
        steps=config.steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size_per_task=config.batch_size_per_task,
        evaluation_interval=config.evaluation_interval,
        patience_evaluations=config.patience_evaluations,
        gradient_clip=5.0,
        minimum_improvement=1e-6,
    )


def shared_model(
    config: ExperimentConfig,
    num_tasks: int,
    integration: str,
) -> SharedDecompositionPredictor:
    return SharedDecompositionPredictor(
        input_dim=config.principal_components,
        num_tasks=num_tasks,
        representation_dim=config.representation_dim,
        common_hidden=(config.hidden_width,),
        representation_hidden=(config.hidden_width,),
        activation="relu",
        integration=integration,
        head_initialization_scale=0.02,
    )


def initialize_model(
    method: str,
    config: ExperimentConfig,
    num_tasks: int,
    seed: int,
) -> MultiTaskPredictor:
    """Initialize a neural baseline from a method-specific deterministic seed."""
    seed_everything(seed)
    if method == "Pool":
        return PooledPredictor(
            config.principal_components,
            num_tasks,
            (config.hidden_width,),
            "relu",
        )
    if method == "STL":
        return SingleTaskPredictor(
            config.principal_components,
            num_tasks,
            (config.hidden_width,),
            "relu",
        )
    if method == "MMoE":
        return MMoEPredictor(
            config.principal_components,
            num_tasks,
            (config.hidden_width,),
            expert_dim=config.representation_dim,
            num_experts=3,
            activation="relu",
        )
    if method == "HPS":
        return shared_model(config, num_tasks, "none")
    raise ValueError(f"Unknown method: {method}")


def fit_model(
    model: MultiTaskPredictor,
    fold: TransformedFold,
    config: ExperimentConfig,
    *,
    coupling: float,
    seed: int,
    device: str,
    validation_task_indices: tuple[int, ...] | None = None,
) -> NeuralFit:
    return fit_neural_model(
        model,
        as_simulation_split(fold.train_x, fold.train_y),
        as_simulation_split(fold.validation_x, fold.validation_y),
        optimization_config(config),
        coupling=coupling,
        seed=seed,
        device=device,
        validation_task_indices=validation_task_indices,
    )


@torch.no_grad()
def predict_by_task(
    model: MultiTaskPredictor, arrays: tuple[np.ndarray, ...], device: str
) -> tuple[np.ndarray, ...]:
    torch_device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = model.to(torch_device).eval()
    output = []
    for task_index, array in enumerate(arrays):
        values = torch.as_tensor(array, dtype=torch.float32, device=torch_device)
        output.append(model.predict_task(values, task_index).cpu().numpy().astype(np.float64))
    model.cpu()
    return tuple(output)


def task_mse(response: tuple[np.ndarray, ...], prediction: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.asarray(
        [np.mean((truth - estimate) ** 2) for truth, estimate in zip(response, prediction)],
        dtype=np.float64,
    )


def prediction_rows(
    method: str,
    predictions: tuple[np.ndarray, ...],
    fold: TransformedFold,
    selected_coupling: float,
    fit_seconds: float,
) -> list[dict[str, object]]:
    rows = []
    for task_index, tissue in enumerate(fold.tissues):
        raw_prediction = (
            fold.response_mean[task_index]
            + fold.response_scale[task_index] * predictions[task_index]
        )
        rows.append(
            {
                "method": method,
                "tissue": tissue,
                "standardized_mse": float(
                    np.mean((fold.test_y[task_index] - predictions[task_index]) ** 2)
                ),
                "raw_mse": float(
                    np.mean((fold.test_y_raw[task_index] - raw_prediction) ** 2)
                ),
                "selected_coupling": selected_coupling,
                "fit_seconds": fit_seconds,
                **fold.fold_counts[task_index],
            }
        )
    return rows


def overlap_rows(
    model: MultiTaskPredictor,
    fold: TransformedFold,
    device: str,
    selected_coupling: float = 0.0,
) -> list[dict[str, object]]:
    if not isinstance(model, SharedDecompositionPredictor):
        return []
    representations = []
    for task_index, values in enumerate(fold.train_x):
        _, representation = predict_decomposition(model, values, device=device)
        representations.append(representation)
    sigmas = [values.T @ values / values.shape[0] for values in representations]
    heads = model.beta.detach().cpu().to(torch.float64)
    rows = []
    for left in range(len(sigmas)):
        for right in range(left + 1, len(sigmas)):
            a = torch.as_tensor(sigmas[left], dtype=torch.float64)
            b = torch.as_tensor(sigmas[right], dtype=torch.float64)
            total = torch.linalg.pinv(a + b, hermitian=True)
            omega = 2.0 * a @ total @ b
            omega = 0.5 * (omega + omega.T)
            eigenvalues = torch.linalg.eigvalsh(omega).clamp_min(0)
            descending = torch.flip(eigenvalues, dims=(0,)).numpy()
            denominator = 0.5 * (np.trace(sigmas[left]) + np.trace(sigmas[right]))
            difference = heads[left] - heads[right]
            overlap_energy = float(difference @ omega @ difference)
            average = 0.5 * (a + b)
            average_energy = float(difference @ average @ difference)
            unweighted_distance = float(difference @ difference)
            trace = float(eigenvalues.sum())
            squared_trace = float((eigenvalues * eigenvalues).sum())
            row: dict[str, object] = {
                "left_tissue": fold.tissues[left],
                "right_tissue": fold.tissues[right],
                "overlap_trace": trace,
                "normalized_overlap_trace": float(
                    trace / max(denominator, 1e-12)
                ),
                "overlap_effective_rank": float(
                    trace * trace / max(squared_trace, 1e-12)
                ),
                "minimum_overlap_eigenvalue": float(eigenvalues.min()),
                "maximum_overlap_eigenvalue": float(eigenvalues.max()),
                "unweighted_head_distance": unweighted_distance,
                "average_moment_head_energy": average_energy,
                "overlap_weighted_head_energy": overlap_energy,
                "supported_contrast_fraction": float(
                    overlap_energy / max(average_energy, 1e-12)
                ),
                "selected_coupling": selected_coupling,
                "pairwise_penalty_contribution": float(
                    selected_coupling
                    * overlap_energy
                    / (len(sigmas) * (len(sigmas) - 1))
                ),
            }
            row.update(
                {
                    f"overlap_eigenvalue_{index + 1}": float(value)
                    for index, value in enumerate(descending)
                }
            )
            rows.append(row)
    return rows
