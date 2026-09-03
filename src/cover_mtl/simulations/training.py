"""Deterministic, task-balanced neural training and prediction utilities."""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .config import OptimizationConfig
from .data import SimulationSplit
from ..models import MultiTaskPredictor


FloatArray = NDArray[np.float64]


def seed_everything(seed: int) -> None:
    """Seed every random source used by the experiment package."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class NeuralFit:
    model: MultiTaskPredictor
    best_validation_mse: float
    best_step: int
    history: tuple[dict[str, float], ...]


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _split_tensors(
    split: SimulationSplit, device: torch.device
) -> tuple[Tensor | list[Tensor], Tensor | list[Tensor]]:
    if isinstance(split.x, np.ndarray) and isinstance(split.y, np.ndarray):
        return (
            torch.as_tensor(split.x, dtype=torch.float32, device=device),
            torch.as_tensor(split.y, dtype=torch.float32, device=device),
        )
    return (
        [
            torch.as_tensor(array, dtype=torch.float32, device=device)
            for array in split.x_by_task()
        ],
        [
            torch.as_tensor(array, dtype=torch.float32, device=device)
            for array in split.y_by_task()
        ],
    )


def _balanced_validation_mse(
    model: MultiTaskPredictor,
    x: Tensor | list[Tensor],
    response: Tensor | list[Tensor],
    task_indices: tuple[int, ...] | None = None,
) -> float:
    selected = range(model.num_tasks) if task_indices is None else task_indices
    losses = []
    for task_index in selected:
        predictions = model.predict_task(x[task_index], task_index)
        losses.append((response[task_index] - predictions).square().mean())
    return float(torch.stack(losses).mean().detach().cpu())


def _cpu_state(model: MultiTaskPredictor) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def fit_neural_model(
    model: MultiTaskPredictor,
    train: SimulationSplit,
    validation: SimulationSplit,
    optimization: OptimizationConfig,
    *,
    coupling: float,
    seed: int,
    device: str,
    validation_task_indices: tuple[int, ...] | None = None,
) -> NeuralFit:
    """Fit one neural method with balanced mini-batches and early stopping."""
    if train.num_tasks != model.num_tasks or validation.num_tasks != model.num_tasks:
        raise ValueError("Data and model task counts differ.")
    if validation_task_indices is not None:
        if not validation_task_indices:
            raise ValueError("At least one validation task must be selected.")
        if min(validation_task_indices) < 0 or max(validation_task_indices) >= model.num_tasks:
            raise ValueError("A validation task index is outside the available range.")
    seed_everything(seed)
    torch_device = _resolve_device(device)
    model = model.to(torch_device)
    train_x, train_y = _split_tensors(train, torch_device)
    validation_x, validation_y = _split_tensors(validation, torch_device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(optimization.weight_decay),
        lr=optimization.learning_rate,
    )
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(seed + 17)

    model.eval()
    with torch.no_grad():
        best_validation = _balanced_validation_mse(
            model, validation_x, validation_y, validation_task_indices
        )
    model.train()
    best_step = 0
    best_state = _cpu_state(model)
    stale = 0
    history: list[dict[str, float]] = [
        {
            "step": 0.0,
            "validation_mse": best_validation,
            "batch_prediction_loss": float("nan"),
            "batch_integration_loss": float("nan"),
        }
    ]
    batch_size = min(optimization.batch_size_per_task, train.sample_size)

    for step in range(1, optimization.steps + 1):
        if isinstance(train_x, Tensor) and isinstance(train_y, Tensor):
            indices = torch.randint(
                train.sample_size,
                size=(model.num_tasks, batch_size),
                generator=generator,
                device=torch_device,
            )
            task_grid = torch.arange(model.num_tasks, device=torch_device)[:, None]
            batch_x = train_x[task_grid, indices].reshape(-1, train_x.shape[-1])
            batch_y = train_y[task_grid, indices].reshape(-1)
            batch_task = task_grid.expand_as(indices).reshape(-1)
        else:
            if isinstance(train_x, Tensor) or isinstance(train_y, Tensor):
                raise TypeError(
                    "Training features and responses must use matching storage."
                )
            sampled_x, sampled_y, sampled_tasks = [], [], []
            for task_index, (task_x, task_y) in enumerate(zip(train_x, train_y)):
                indices = torch.randint(
                    task_x.shape[0],
                    size=(batch_size,),
                    generator=generator,
                    device=torch_device,
                )
                sampled_x.append(task_x[indices])
                sampled_y.append(task_y[indices])
                sampled_tasks.append(
                    torch.full(
                        (batch_size,), task_index, dtype=torch.long, device=torch_device
                    )
                )
            batch_x = torch.cat(sampled_x, dim=0)
            batch_y = torch.cat(sampled_y, dim=0)
            batch_task = torch.cat(sampled_tasks, dim=0)

        optimizer.zero_grad(set_to_none=True)
        parts = model.objective(batch_x, batch_y, batch_task, coupling)
        parts.total.backward()
        if optimization.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), optimization.gradient_clip
            )
        optimizer.step()
        model.post_step()

        evaluate = (
            step == 1
            or step % optimization.evaluation_interval == 0
            or step == optimization.steps
        )
        if not evaluate:
            continue
        model.eval()
        with torch.no_grad():
            validation_mse = _balanced_validation_mse(
                model, validation_x, validation_y, validation_task_indices
            )
            monitor_parts = model.objective(batch_x, batch_y, batch_task, coupling)
        model.train()
        history.append(
            {
                "step": float(step),
                "validation_mse": validation_mse,
                "batch_prediction_loss": float(monitor_parts.prediction.cpu()),
                "batch_integration_loss": float(monitor_parts.integration.cpu()),
            }
        )
        if validation_mse < best_validation - optimization.minimum_improvement:
            best_validation = validation_mse
            best_step = step
            best_state = _cpu_state(model)
            stale = 0
        else:
            stale += 1
        if stale >= optimization.patience_evaluations:
            break

    model.load_state_dict(best_state)
    model.post_step()
    return NeuralFit(
        model=model.cpu(),
        best_validation_mse=best_validation,
        best_step=best_step,
        history=tuple(history),
    )


def clone_model(model: MultiTaskPredictor) -> MultiTaskPredictor:
    """Deep-copy a fitted initialization before method-specific fine-tuning."""
    return copy.deepcopy(model)


@torch.no_grad()
def predict_all_tasks(
    model: MultiTaskPredictor, x: FloatArray, *, device: str, batch_size: int = 8192,
) -> FloatArray:
    """Return an n by T prediction matrix using bounded device memory."""
    torch_device = _resolve_device(device)
    model = model.to(torch_device).eval()
    predictions = []
    for start in range(0, x.shape[0], batch_size):
        batch = torch.as_tensor(
            x[start : start + batch_size], dtype=torch.float32, device=torch_device
        )
        predictions.append(model.predict_all_tasks(batch).cpu().numpy())
    model.cpu()
    return np.concatenate(predictions, axis=0).astype(np.float64, copy=False)


@torch.no_grad()
def predict_decomposition(
    model: MultiTaskPredictor, x: FloatArray, *, device: str, batch_size: int = 8192,
) -> tuple[FloatArray, FloatArray]:
    """Return common-function and representation values for a shared model."""
    if not hasattr(model, "common_and_representation"):
        raise TypeError(
            "The model does not expose a common/representation decomposition."
        )
    torch_device = _resolve_device(device)
    model = model.to(torch_device).eval()
    common_values = []
    representation_values = []
    for start in range(0, x.shape[0], batch_size):
        batch = torch.as_tensor(
            x[start : start + batch_size], dtype=torch.float32, device=torch_device
        )
        common, representation = model.common_and_representation(batch)
        common_values.append(common.cpu().numpy())
        representation_values.append(representation.cpu().numpy())
    model.cpu()
    return (
        np.concatenate(common_values).astype(np.float64, copy=False),
        np.concatenate(representation_values).astype(np.float64, copy=False),
    )
