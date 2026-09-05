"""Run one paired replicate of the primary nonlinear experiment."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines.armul import tune_armul
from .baselines.flarcc import tune_flarcc
from .config import NetworkConfig, OptimizationConfig
from .data import SimulationConfig, SimulationReplicate, SimulationSplit
from .metrics import evaluate_functions
from ..models import (
    FixedHeadCommonPredictor,
    MMoEPredictor,
    MultiTaskPredictor,
    PooledPredictor,
    SharedDecompositionPredictor,
    SingleTaskPredictor,
)
from .randomness import derive_seed
from .training import (
    NeuralFit,
    clone_model,
    fit_neural_model,
    predict_all_tasks,
    predict_decomposition,
)


DEFAULT_METHODS = (
    "Pool",
    "STL",
    "HPS",
    "MMoE",
    "COVER",
    "ARMUL",
    "FLARCC",
)
SUPPORTED_METHODS = (
    "Pool",
    "STL",
    "HPS",
    "Continued-HPS",
    "MMoE",
    "Average-Moment",
    "COVER",
    "ARMUL",
    "FLARCC",
)
DEFAULT_COUPLINGS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
CLASSICAL_RATIOS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)


def _shared_model(
    dgp: SimulationConfig, network: NetworkConfig, integration: str,
) -> SharedDecompositionPredictor:
    return SharedDecompositionPredictor(
        input_dim=dgp.input_dim,
        num_tasks=dgp.num_tasks,
        representation_dim=network.representation_dim,
        common_hidden=network.common_hidden,
        representation_hidden=network.representation_hidden,
        activation=network.activation,
        integration=integration,
        head_initialization_scale=network.head_initialization_scale,
        representation_identity=network.representation_identity,
    )


def _parameter_count(model: MultiTaskPredictor) -> int:
    # Pairwise consensus vectors are optimization auxiliaries and are not
    # required by the fitted prediction rule.
    count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name != "consensus"
    )
    # ARMUL and FLARCC return a predictor whose fitted task coefficients are
    # stored as a non-trainable buffer. They are still fitted model parameters
    # and therefore belong in the reported model size.
    if isinstance(model, FixedHeadCommonPredictor):
        count += model.coefficients.numel()
    return count


def _cuda_device(device: str) -> torch.device | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return torch.device(device)


def _evaluate_neural(
    model: MultiTaskPredictor, test: SimulationSplit, device: str,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    return evaluate_functions(
        lambda x: predict_all_tasks(model, x, device=device), test,
    )


def _fit_one_neural(
    method: str,
    model: MultiTaskPredictor,
    data: SimulationReplicate,
    optimization: OptimizationConfig,
    seed: int,
    device: str,
    coupling: float = 0.0,
) -> tuple[NeuralFit, dict[str, object], list[dict[str, object]]]:
    cuda_device = _cuda_device(device)
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
    start = time.perf_counter()
    fit = fit_neural_model(
        model,
        data.train,
        data.validation,
        optimization,
        coupling=coupling,
        seed=seed,
        device=device,
    )
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
    elapsed = time.perf_counter() - start
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(cuda_device) / 2 ** 20)
        if cuda_device is not None
        else float("nan")
    )
    summary, task_rows = _evaluate_neural(fit.model, data.test, device)
    row: dict[str, object] = {
        "method": method,
        "selected_coupling": coupling,
        "validation_mse": fit.best_validation_mse,
        "best_step": fit.best_step,
        "fit_seconds": elapsed,
        "peak_device_memory_mb": peak_memory_mb,
        "parameter_count": _parameter_count(fit.model),
        **summary,
    }
    for task_row in task_rows:
        task_row["method"] = method
    return fit, row, task_rows


def _load_hps_initialization(
    model: SharedDecompositionPredictor, hps: SharedDecompositionPredictor,
) -> None:
    missing, unexpected = model.load_state_dict(hps.state_dict(), strict=False)
    if unexpected or any(name != "consensus" for name in missing):
        raise RuntimeError(
            f"Unexpected HPS initialization mismatch: missing={missing}, unexpected={unexpected}."
        )
    model.initialize_consensus_from_heads()


def _fit_coupled_path(
    method: str,
    integration: str,
    hps_fit: NeuralFit,
    data: SimulationReplicate,
    network: NetworkConfig,
    optimization: OptimizationConfig,
    couplings: tuple[float, ...],
    base_seed: int,
    device: str,
    selection_rule: str = "mean",
) -> tuple[
    NeuralFit, dict[str, object], list[dict[str, object]], list[dict[str, object]]
]:
    if selection_rule not in {"mean", "relative_mean", "worst", "safe"}:
        raise ValueError("selection_rule must be mean, relative_mean, worst, or safe.")
    hps_model = hps_fit.model
    if not isinstance(hps_model, SharedDecompositionPredictor):
        raise TypeError("Coupled paths require a shared-decomposition HPS fit.")
    _, hps_validation_rows = _evaluate_neural(hps_model, data.validation, device)
    hps_task_mse = np.asarray([row["prediction_mse"] for row in hps_validation_rows])
    hps_worst_mse = float(hps_task_mse.max())
    candidates: list[dict[str, object]] = [
        {
            "mean": hps_fit.best_validation_mse,
            "relative_mean": 1.0,
            "worst": hps_worst_mse,
            "max_increase": 0.0,
            "max_relative_increase": 0.0,
            "fit": hps_fit,
            "coupling": 0.0,
        }
    ]
    tuning_rows: list[dict[str, object]] = [
        {
            "method": method,
            "coupling": 0.0,
            "validation_mse": hps_fit.best_validation_mse,
            "best_step": hps_fit.best_step,
            "fit_seconds": 0.0,
            "validation_worst_task_mse": hps_worst_mse,
            "validation_relative_mse_vs_hps": 1.0,
            "validation_max_increase_vs_hps": 0.0,
            "validation_max_relative_increase_vs_hps": 0.0,
            "selection_rule": selection_rule,
        }
    ]
    cuda_device = _cuda_device(device)
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
    total_seconds = 0.0
    for coupling in couplings:
        if coupling <= 0:
            continue
        model = _shared_model(data.config, network, integration)
        _load_hps_initialization(model, hps_model)
        seed = derive_seed(base_seed, method, coupling)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        start = time.perf_counter()
        fit = fit_neural_model(
            model,
            data.train,
            data.validation,
            optimization,
            coupling=coupling,
            seed=seed,
            device=device,
        )
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        elapsed = time.perf_counter() - start
        total_seconds += elapsed
        _, validation_rows = _evaluate_neural(fit.model, data.validation, device)
        task_mse = np.asarray([row["prediction_mse"] for row in validation_rows])
        worst_mse = float(task_mse.max())
        relative_mean = float(np.mean(task_mse / np.maximum(hps_task_mse, 1e-8)))
        max_increase = float((task_mse - hps_task_mse).max())
        max_relative_increase = float(
            np.max((task_mse - hps_task_mse) / np.maximum(hps_task_mse, 1e-8))
        )
        tuning_rows.append(
            {
                "method": method,
                "coupling": coupling,
                "validation_mse": fit.best_validation_mse,
                "best_step": fit.best_step,
                "fit_seconds": elapsed,
                "validation_worst_task_mse": worst_mse,
                "validation_relative_mse_vs_hps": relative_mean,
                "validation_max_increase_vs_hps": max_increase,
                "validation_max_relative_increase_vs_hps": max_relative_increase,
                "selection_rule": selection_rule,
            }
        )
        candidates.append(
            {
                "mean": fit.best_validation_mse,
                "relative_mean": relative_mean,
                "worst": worst_mse,
                "max_increase": max_increase,
                "max_relative_increase": max_relative_increase,
                "fit": fit,
                "coupling": coupling,
            }
        )
    if selection_rule == "safe":
        eligible = [
            candidate
            for candidate in candidates
            if float(candidate["max_relative_increase"]) <= 0.05 + 1e-10
        ]
        selected = min(
            eligible, key=lambda item: (float(item["mean"]), float(item["coupling"])),
        )
    elif selection_rule == "relative_mean":
        selected = min(
            candidates,
            key=lambda item: (
                float(item["relative_mean"]),
                float(item["mean"]),
                float(item["coupling"]),
            ),
        )
    elif selection_rule == "worst":
        selected = min(
            candidates,
            key=lambda item: (
                float(item["worst"]),
                float(item["mean"]),
                float(item["coupling"]),
            ),
        )
    else:
        selected = min(
            candidates, key=lambda item: (float(item["mean"]), float(item["coupling"])),
        )
    validation_mse = float(selected["mean"])
    selected_fit = selected["fit"]
    selected_coupling = float(selected["coupling"])
    if not isinstance(selected_fit, NeuralFit):
        raise TypeError("The selected coupled candidate must be a NeuralFit.")
    summary, task_rows = _evaluate_neural(selected_fit.model, data.test, device)
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(cuda_device) / 2 ** 20)
        if cuda_device is not None
        else float("nan")
    )
    row: dict[str, object] = {
        "method": method,
        "selected_coupling": selected_coupling,
        "validation_mse": validation_mse,
        "best_step": selected_fit.best_step,
        "fit_seconds": total_seconds,
        "peak_device_memory_mb": peak_memory_mb,
        "parameter_count": _parameter_count(selected_fit.model),
        **summary,
    }
    for task_row in task_rows:
        task_row["method"] = method
    return selected_fit, row, task_rows, tuning_rows


def _fixed_features(
    model: SharedDecompositionPredictor, split: SimulationSplit, device: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    common_values, representations, residuals = [], [], []
    for task in range(split.num_tasks):
        common, representation = predict_decomposition(
            model, split.x[task], device=device
        )
        common_values.append(common)
        representations.append(representation)
        residuals.append(split.y[task] - common)
    return common_values, representations, residuals


def _refit_common_component(
    hps: SharedDecompositionPredictor,
    coefficients: np.ndarray,
    data: SimulationReplicate,
    optimization: OptimizationConfig,
    seed: int,
    device: str,
) -> NeuralFit:
    """Refit the shared nonlinear component around fixed baseline heads."""
    model = FixedHeadCommonPredictor(
        hps.common_network,
        hps.representation_network,
        torch.as_tensor(coefficients, dtype=torch.float32),
    )
    return fit_neural_model(
        model,
        data.train,
        data.validation,
        optimization,
        coupling=0.0,
        seed=seed,
        device=device,
    )


def _fit_armul(
    hps: SharedDecompositionPredictor,
    data: SimulationReplicate,
    optimization: OptimizationConfig,
    seed: int,
    device: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    _, train_z, train_residual = _fixed_features(hps, data.train, device)
    _, validation_z, validation_residual = _fixed_features(hps, data.validation, device)
    start = time.perf_counter()
    path = tune_armul(
        train_z,
        train_residual,
        validation_z,
        validation_residual,
        lambda_ratios=CLASSICAL_RATIOS,
    )
    refit = _refit_common_component(
        hps, path.fit.coefficients, data, optimization, seed, device,
    )
    elapsed = time.perf_counter() - start
    summary, task_rows = _evaluate_neural(refit.model, data.test, device)
    for task_row in task_rows:
        task_row["method"] = "ARMUL"
    row: dict[str, object] = {
        "method": "ARMUL",
        "selected_coupling": path.selected_ratio,
        "validation_mse": refit.best_validation_mse,
        "best_step": refit.best_step,
        "fit_seconds": elapsed,
        "parameter_count": _parameter_count(refit.model),
        **summary,
    }
    tuning = [{"method": "ARMUL", **item} for item in path.tuning_rows]
    return row, task_rows, tuning


def _fit_flarcc(
    hps: SharedDecompositionPredictor,
    data: SimulationReplicate,
    optimization: OptimizationConfig,
    seed: int,
    device: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    _, train_z, train_residual = _fixed_features(hps, data.train, device)
    _, validation_z, validation_residual = _fixed_features(hps, data.validation, device)
    candidates = []
    tuning_rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for range_power in (0.0, 1.0):
        path = tune_flarcc(
            train_z,
            train_residual,
            validation_z,
            validation_residual,
            lambda_ratios=CLASSICAL_RATIOS,
            range_power=range_power,
            include_task_intercepts=False,
        )
        candidates.append(
            (path.validation_mse, -path.selected_ratio, range_power, path)
        )
        tuning_rows.extend(
            {"method": "FLARCC", "range_power": range_power, **item,}
            for item in path.tuning_rows
        )
    _, _, selected_range_power, selected = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    refit = _refit_common_component(
        hps, selected.fit.coefficients, data, optimization, seed, device,
    )
    elapsed = time.perf_counter() - start
    summary, task_rows = _evaluate_neural(refit.model, data.test, device)
    for task_row in task_rows:
        task_row["method"] = "FLARCC"
    row: dict[str, object] = {
        "method": "FLARCC",
        "selected_coupling": selected.selected_ratio,
        "selected_range_power": selected_range_power,
        "validation_mse": refit.best_validation_mse,
        "best_step": refit.best_step,
        "fit_seconds": elapsed,
        "parameter_count": _parameter_count(refit.model),
        **summary,
    }
    return row, task_rows, tuning_rows


def run_replicate(
    *,
    scenario: str,
    replicate: int,
    output_dir: Path,
    network_name: str,
    methods: tuple[str, ...],
    couplings: tuple[float, ...],
    device: str,
    prepared_data: SimulationReplicate,
    network_config: NetworkConfig,
    optimization_config: OptimizationConfig,
    coupling_optimization_config: OptimizationConfig,
    coupling_selection: str = "mean",
) -> None:
    unknown = set(methods) - set(SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    data = prepared_data
    dgp = data.config
    if dgp.scenario != scenario:
        raise ValueError("Prepared data and requested scenario differ.")
    data_seed = dgp.seed
    network = network_config
    optimization = optimization_config
    coupling_optimization = coupling_optimization_config
    metric_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []

    needs_hps = bool(
        {
            "HPS",
            "Continued-HPS",
            "Average-Moment",
            "COVER",
            "ARMUL",
            "FLARCC",
        }
        & set(methods)
    )
    hps_fit: NeuralFit | None = None
    if needs_hps:
        seed = derive_seed(20260825, "main_model", scenario, replicate, "HPS")
        hps_fit, row, rows = _fit_one_neural(
            "HPS",
            _shared_model(dgp, network, "none"),
            data,
            optimization,
            seed,
            device,
        )
        if "HPS" in methods:
            metric_rows.append(row)
            task_rows.extend(rows)

        if "Continued-HPS" in methods:
            continuation_seed = derive_seed(
                derive_seed(20260825, "main_model", scenario, replicate),
                "COVER",
                0.0,
            )
            _, continued_row, continued_rows = _fit_one_neural(
                "Continued-HPS",
                clone_model(hps_fit.model),
                data,
                coupling_optimization,
                continuation_seed,
                device,
                coupling=0.0,
            )
            continued_row["warm_start"] = "HPS"
            continued_row["additional_budget"] = coupling_optimization.steps
            metric_rows.append(continued_row)
            task_rows.extend(continued_rows)

    if "Pool" in methods:
        seed = derive_seed(20260825, "main_model", scenario, replicate, "Pool")
        _, row, rows = _fit_one_neural(
            "Pool",
            PooledPredictor(
                dgp.input_dim, dgp.num_tasks, network.common_hidden, network.activation,
            ),
            data,
            optimization,
            seed,
            device,
        )
        metric_rows.append(row)
        task_rows.extend(rows)

    if "STL" in methods:
        seed = derive_seed(20260825, "main_model", scenario, replicate, "STL")
        _, row, rows = _fit_one_neural(
            "STL",
            SingleTaskPredictor(
                dgp.input_dim, dgp.num_tasks, network.common_hidden, network.activation,
            ),
            data,
            optimization,
            seed,
            device,
        )
        metric_rows.append(row)
        task_rows.extend(rows)

    if "MMoE" in methods:
        seed = derive_seed(20260825, "main_model", scenario, replicate, "MMoE")
        _, row, rows = _fit_one_neural(
            "MMoE",
            MMoEPredictor(
                dgp.input_dim,
                dgp.num_tasks,
                hidden_dims=(64, 64),
                expert_dim=32,
                num_experts=4,
                activation=network.activation,
            ),
            data,
            optimization,
            seed,
            device,
        )
        metric_rows.append(row)
        task_rows.extend(rows)

    if hps_fit is not None:
        for method, integration in (
            ("Average-Moment", "average_moment"),
            ("COVER", "cover"),
        ):
            if method not in methods:
                continue
            selected, row, rows, path_rows = _fit_coupled_path(
                method,
                integration,
                hps_fit,
                data,
                network,
                coupling_optimization,
                couplings,
                derive_seed(20260825, "main_model", scenario, replicate),
                device,
                selection_rule=coupling_selection,
            )
            del selected
            metric_rows.append(row)
            task_rows.extend(rows)
            tuning_rows.extend(path_rows)
        hps_model = hps_fit.model
        if not isinstance(hps_model, SharedDecompositionPredictor):
            raise TypeError("Fixed-head baselines require a shared HPS model.")
        if "ARMUL" in methods:
            row, rows, path_rows = _fit_armul(
                hps_model,
                data,
                coupling_optimization,
                derive_seed(20260825, "main_model", scenario, replicate, "ARMUL_refit"),
                device,
            )
            metric_rows.append(row)
            task_rows.extend(rows)
            tuning_rows.extend(path_rows)
        if "FLARCC" in methods:
            row, rows, path_rows = _fit_flarcc(
                hps_model,
                data,
                coupling_optimization,
                derive_seed(
                    20260825, "main_model", scenario, replicate, "FLARCC_refit"
                ),
                device,
            )
            metric_rows.append(row)
            task_rows.extend(rows)
            tuning_rows.extend(path_rows)

    task_frame = pd.DataFrame(task_rows)
    if "STL" in set(task_frame.get("method", [])):
        stl = task_frame[task_frame["method"] == "STL"].set_index("task")["excess_mse"]
        for row in metric_rows:
            current = task_frame[task_frame["method"] == row["method"]].set_index(
                "task"
            )["excess_mse"]
            row["fraction_tasks_improved_vs_stl"] = float((current < stl).mean())
    for row in metric_rows:
        row.update(
            {
                "scenario": scenario,
                "replicate": replicate,
                "data_seed": data_seed,
                "network": network_name,
            }
        )
    task_frame["scenario"] = scenario
    task_frame["replicate"] = replicate
    job_dir = output_dir / scenario / f"rep_{replicate:03d}"
    job_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(job_dir / "metrics.csv", index=False)
    task_frame.to_csv(job_dir / "task_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(job_dir / "tuning.csv", index=False)
    with (job_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "output_schema_version": 2,
                "dgp": dgp.to_dict(),
                "network_name": network_name,
                "network": asdict(network),
                "optimization": asdict(optimization),
                "coupling_optimization": asdict(coupling_optimization),
                "methods": methods,
                "couplings": couplings,
                "coupling_selection": coupling_selection,
            },
            stream,
            indent=2,
        )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
