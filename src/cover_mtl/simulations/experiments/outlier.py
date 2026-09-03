"""COVER-only experiment for covariate-overlap outlier decoupling."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import NetworkConfig, OptimizationConfig
from ..data import SimulationReplicate, SimulationSplit
from ..runner import _fit_coupled_path, _fit_one_neural, _shared_model
from ..randomness import derive_seed
from ..dgp import SpectralNeuralConfig, SpectralNeuralTruth
from ..training import predict_decomposition


NUM_TASKS = 24
REPRESENTATION_DIMENSION = 24
COMMON_DIMENSION = 5
INPUT_DIMENSION = REPRESENTATION_DIMENSION + COMMON_DIMENSION
OUTLIER_TASK = 0
OUTLIER_SIGNAL_SCALE = 1.5

NETWORK = NetworkConfig(
    common_hidden=(32,),
    representation_hidden=(48,),
    representation_dim=REPRESENTATION_DIMENSION,
)
OPTIMIZATION = OptimizationConfig(
    steps=2500, evaluation_interval=25, patience_evaluations=28, batch_size_per_task=64,
)
COUPLING_OPTIMIZATION = OptimizationConfig(
    steps=1000, evaluation_interval=25, patience_evaluations=16, batch_size_per_task=64,
)


def _orthogonal(generator: np.random.Generator, dimension: int) -> np.ndarray:
    basis, triangular = np.linalg.qr(generator.normal(size=(dimension, dimension)))
    signs = np.where(np.diag(triangular) < 0, -1.0, 1.0)
    return basis * signs


def _tanh_second_moment(variance: float, quadrature_order: int = 80) -> float:
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    values = np.tanh(np.sqrt(2.0 * variance) * nodes) ** 2
    return float(weights @ values / np.sqrt(np.pi))


def _overlap(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    value = 2.0 * left @ np.linalg.pinv(left + right, rcond=1e-10) @ right
    return 0.5 * (value + value.T)


def _degree_diagnostics(moments: np.ndarray) -> dict[str, float]:
    pooled = moments.mean(axis=0)
    pooled_inverse = np.linalg.pinv(pooled, rcond=1e-10)
    rank = max(int(np.linalg.matrix_rank(pooled, tol=1e-10)), 1)
    edge_strengths = np.zeros((NUM_TASKS, NUM_TASKS), dtype=np.float64)
    for left, right in combinations(range(NUM_TASKS), 2):
        omega = _overlap(moments[left], moments[right])
        strength = float(np.trace(pooled_inverse @ omega) / rank)
        edge_strengths[left, right] = strength
        edge_strengths[right, left] = strength
    degrees = edge_strengths.sum(axis=1) / (NUM_TASKS - 1)
    inlier_degrees = degrees[1:]
    inlier_edges = edge_strengths[1:, 1:]
    upper = inlier_edges[np.triu_indices(NUM_TASKS - 1, k=1)]
    outlier_degree = float(degrees[OUTLIER_TASK])
    inlier_degree = float(inlier_degrees.mean())
    return {
        "outlier_overlap_degree": outlier_degree,
        "mean_inlier_overlap_degree": inlier_degree,
        "mean_inlier_edge_strength": float(upper.mean()),
        "outlier_isolation_ratio": outlier_degree / max(inlier_degree, 1e-12),
        "outlier_degree_rank": float(1 + np.sum(degrees < outlier_degree)),
        "outlier_detected_as_minimum": float(
            outlier_degree < float(inlier_degrees.min()) - 1e-8
        ),
    }


def _make_split(
    generator: np.random.Generator,
    sample_size: int,
    truth: SpectralNeuralTruth,
    noise_scale: float,
) -> SimulationSplit:
    x = np.empty((NUM_TASKS, sample_size, INPUT_DIMENSION), dtype=np.float64)
    common = np.empty((NUM_TASKS, sample_size), dtype=np.float64)
    deviation = np.empty_like(common)
    for task in range(NUM_TASKS):
        latent = generator.normal(size=(sample_size, INPUT_DIMENSION))
        latent *= np.sqrt(truth.covariance_diagonals[task])
        task_x = latent @ truth.rotation.T
        x[task] = task_x
        common[task] = truth.common_function(task_x)
        deviation[task] = truth.representation(task_x) @ truth.coefficients[task]
    conditional_mean = common + deviation
    response = conditional_mean + generator.normal(
        scale=noise_scale, size=conditional_mean.shape,
    )
    return SimulationSplit(
        x=x,
        y=response,
        conditional_mean=conditional_mean,
        common=common,
        deviation=deviation,
    )


def _generate_replicate(overlap: float, replicate: int) -> SimulationReplicate:
    seed = derive_seed(20260828, "outlier_decoupling_data", overlap, replicate)
    generator = np.random.default_rng(seed)
    representation_variances = np.empty(
        (NUM_TASKS, REPRESENTATION_DIMENSION), dtype=np.float64
    )
    representation_variances[1:, : REPRESENTATION_DIMENSION // 2] = 1.0
    representation_variances[1:, REPRESENTATION_DIMENSION // 2 :] = overlap
    representation_variances[OUTLIER_TASK, : REPRESENTATION_DIMENSION // 2] = overlap
    representation_variances[OUTLIER_TASK, REPRESENTATION_DIMENSION // 2 :] = 1.0

    transformed_values = {value: _tanh_second_moment(value) for value in (overlap, 1.0)}
    representation_moments = np.asarray(
        [
            np.diag([transformed_values[float(value)] for value in diagonal])
            for diagonal in representation_variances
        ]
    )

    direction = np.zeros(REPRESENTATION_DIMENSION, dtype=np.float64)
    direction[REPRESENTATION_DIMENSION // 2 :] = generator.normal(
        size=REPRESENTATION_DIMENSION // 2
    )
    direction /= np.sqrt(
        max(float(direction @ representation_moments[OUTLIER_TASK] @ direction), 1e-12,)
    )
    raw_coefficients = np.zeros((NUM_TASKS, REPRESENTATION_DIMENSION))
    raw_coefficients[OUTLIER_TASK] = OUTLIER_SIGNAL_SCALE * direction
    coefficients = raw_coefficients - raw_coefficients.mean(axis=0, keepdims=True)

    covariance_diagonals = np.ones((NUM_TASKS, INPUT_DIMENSION))
    covariance_diagonals[:, :REPRESENTATION_DIMENSION] = representation_variances
    latent_covariances = np.asarray(
        [np.diag(diagonal) for diagonal in covariance_diagonals]
    )
    rotation = _orthogonal(generator, INPUT_DIMENSION)
    truth = SpectralNeuralTruth(
        rotation=rotation,
        covariance_diagonals=covariance_diagonals,
        latent_covariances=latent_covariances,
        representation_moments=representation_moments,
        coefficients=coefficients,
        profile_labels=np.concatenate(
            [np.ones(1, dtype=np.int64), np.zeros(NUM_TASKS - 1, dtype=np.int64)]
        ),
        generalized_eigenvalues=np.empty(0, dtype=np.float64),
        signal_overlap_ratio=float("nan"),
        coefficient_rank=int(np.linalg.matrix_rank(coefficients, tol=1e-8)),
        representation_dim=REPRESENTATION_DIMENSION,
        representation_transform="tanh",
        common_coordinate_start=REPRESENTATION_DIMENSION,
        diagonal_latent_covariance=True,
    )
    config = SpectralNeuralConfig(
        scenario="both_overlap_aligned",
        num_tasks=NUM_TASKS,
        num_profiles=2,
        input_dim=INPUT_DIMENSION,
        representation_dim=REPRESENTATION_DIMENSION,
        active_per_profile=12,
        train_size=100,
        validation_size=200,
        test_size=3000,
        weak_variance=overlap,
        posterior_scale=OUTLIER_SIGNAL_SCALE,
        within_profile_scale=0.0,
        covariance_geometry="diagonal",
        posterior_geometry="clustered",
        representation_transform="tanh",
        rotation_structure="full",
        common_coordinate_mode="separate",
        subspace_rank=4,
        low_mode_count=24,
        seed=seed,
    )
    arguments = {"generator": generator, "truth": truth, "noise_scale": 1.0}
    return SimulationReplicate(
        config=config,
        train=_make_split(sample_size=config.train_size, **arguments),
        validation=_make_split(sample_size=config.validation_size, **arguments),
        test=_make_split(sample_size=config.test_size, **arguments),
        truth=truth,
    )


def _learned_moments(model, data: SimulationReplicate, device: str) -> np.ndarray:
    moments = []
    for task, features in enumerate(data.train.x_by_task()):
        del task
        _, representation = predict_decomposition(model, features, device=device)
        moments.append(representation.T @ representation / representation.shape[0])
    return np.asarray(moments)


def _learned_outlier_energy(model, moments: np.ndarray) -> float:
    if not hasattr(model, "beta"):
        raise TypeError("The fitted COVER model does not expose task heads.")
    coefficients = model.beta.detach().cpu().numpy().astype(np.float64)
    inlier_center = coefficients[1:].mean(axis=0)
    contrast = coefficients[OUTLIER_TASK] - inlier_center
    energies = []
    for task in range(1, NUM_TASKS):
        omega = _overlap(moments[OUTLIER_TASK], moments[task])
        energies.append(float(contrast @ omega @ contrast))
    return float(np.mean(energies))


def run_job(
    *,
    overlap: float,
    replicate: int,
    output_dir: Path,
    device: str,
    couplings: tuple[float, ...],
) -> None:
    data = _generate_replicate(overlap, replicate)
    hps_seed = derive_seed(20260828, "outlier_decoupling_model", overlap, replicate)
    hps_fit, _, _ = _fit_one_neural(
        "HPS",
        _shared_model(data.config, NETWORK, "none"),
        data,
        OPTIMIZATION,
        hps_seed,
        device,
    )
    cover_fit, cover_row, cover_tasks, tuning = _fit_coupled_path(
        "COVER",
        "cover",
        hps_fit,
        data,
        NETWORK,
        COUPLING_OPTIMIZATION,
        couplings,
        hps_seed,
        device,
    )

    population_diagnostics = {
        f"population_{name}": value
        for name, value in _degree_diagnostics(
            np.asarray(data.truth.representation_moments)
        ).items()
    }
    learned_moments = _learned_moments(cover_fit.model, data, device)
    learned_diagnostics = {
        f"learned_{name}": value
        for name, value in _degree_diagnostics(learned_moments).items()
    }
    task_frame = pd.DataFrame(cover_tasks)
    outlier_error = float(
        task_frame.loc[task_frame["task"] == OUTLIER_TASK, "excess_mse"].iloc[0]
    )
    inlier_error = float(
        task_frame.loc[task_frame["task"] != OUTLIER_TASK, "excess_mse"].mean()
    )
    cover_row.update(
        {
            "overlap": overlap,
            "replicate": replicate,
            "outlier_task": OUTLIER_TASK,
            "outlier_excess_mse": outlier_error,
            "mean_inlier_excess_mse": inlier_error,
            "learned_outlier_contrast_energy": _learned_outlier_energy(
                cover_fit.model, learned_moments
            ),
            **population_diagnostics,
            **learned_diagnostics,
        }
    )
    task_frame["overlap"] = overlap
    task_frame["replicate"] = replicate
    task_frame["is_outlier"] = task_frame["task"] == OUTLIER_TASK
    tuning_frame = pd.DataFrame(tuning)
    tuning_frame["overlap"] = overlap
    tuning_frame["replicate"] = replicate

    tag = f"{overlap:.4g}".replace(".", "p")
    job_dir = output_dir / f"overlap_{tag}" / f"rep_{replicate:03d}"
    job_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([cover_row]).to_csv(job_dir / "metrics.csv", index=False)
    task_frame.to_csv(job_dir / "task_metrics.csv", index=False)
    tuning_frame.to_csv(job_dir / "tuning.csv", index=False)
    with (job_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "output_schema_version": 2,
                "dgp": data.config.to_dict(),
                "network": asdict(NETWORK),
                "optimization": asdict(OPTIMIZATION),
                "coupling_optimization": asdict(COUPLING_OPTIMIZATION),
                "couplings": couplings,
                "methods": ("COVER",),
                "outlier_task": OUTLIER_TASK,
                "outlier_signal_scale": OUTLIER_SIGNAL_SCALE,
            },
            stream,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap", required=True, type=float)
    parser.add_argument("--replicate", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--couplings", default="0,0.01,0.03,0.1,0.3,1,3,10,30")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0 < args.overlap <= 1:
        raise ValueError("overlap must lie in (0, 1].")
    run_job(
        overlap=args.overlap,
        replicate=args.replicate,
        output_dir=args.output_dir,
        device=args.device,
        couplings=tuple(float(value) for value in args.couplings.split(",")),
    )


if __name__ == "__main__":
    main()
