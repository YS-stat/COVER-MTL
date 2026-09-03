"""Run one replicate of a formal primary, control, or ablation experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import NetworkConfig, OptimizationConfig
from ..dgp import (
    SpectralNeuralConfig,
    SpectralNeuralTruth,
    generate_spectral_neural_replicate,
)
from ..randomness import derive_seed
from ..runner import DEFAULT_COUPLINGS, DEFAULT_METHODS, run_replicate
from ..training import seed_everything


SCENARIOS = (
    "homogeneous",
    "covariate_only",
    "posterior_only",
    "both_overlap_aligned",
    "both_random",
)


def _bayes_profile_accuracy(x: np.ndarray, truth: SpectralNeuralTruth) -> float:
    """Compute the Bayes covariance-profile classification accuracy."""
    profile_count = np.unique(truth.profile_labels).size
    representatives = [
        int(np.flatnonzero(truth.profile_labels == profile)[0])
        for profile in range(profile_count)
    ]
    covariances = np.asarray(
        [
            truth.rotation @ truth.latent_covariances[task] @ truth.rotation.T
            for task in representatives
        ]
    )
    inverses = np.linalg.inv(covariances)
    log_determinants = np.linalg.slogdet(covariances)[1]
    correct = 0
    total = 0
    for task in range(x.shape[0]):
        quadratic = np.einsum("nd,kde,ne->nk", x[task], inverses, x[task])
        prediction = np.argmin(quadratic + log_determinants[None, :], axis=1)
        correct += int(np.sum(prediction == truth.profile_labels[task]))
        total += x.shape[1]
    return correct / total


def run_job(
    *,
    scenario: str,
    replicate: int,
    output_dir: Path,
    device: str,
    within_profile_scale: float,
    methods: tuple[str, ...],
    couplings: tuple[float, ...],
) -> None:
    """Fit every requested method to one paired data replicate."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown formal scenario {scenario!r}.")
    data_seed = derive_seed(20260825, "spectral_neural_data", scenario, replicate)
    config = SpectralNeuralConfig(
        scenario=scenario,
        num_tasks=48,
        num_profiles=6,
        input_dim=24,
        representation_dim=24,
        active_per_profile=4,
        train_size=100,
        validation_size=200,
        test_size=3000,
        weak_variance=0.001,
        posterior_scale=1.0,
        within_profile_scale=within_profile_scale,
        low_mode_count=48,
        noise_scale=1.0,
        covariance_geometry="diagonal",
        posterior_geometry="clustered",
        representation_transform="tanh",
        subspace_rank=4,
        moment_sample_size=10_000,
        seed=data_seed,
    )
    data = generate_spectral_neural_replicate(config)
    network = NetworkConfig(
        common_hidden=(32,),
        representation_hidden=(48,),
        representation_dim=24,
        activation="relu",
    )
    optimization = OptimizationConfig(
        steps=2500,
        evaluation_interval=25,
        patience_evaluations=28,
        batch_size_per_task=64,
    )
    coupling_optimization = OptimizationConfig(
        steps=1000,
        evaluation_interval=25,
        patience_evaluations=16,
        batch_size_per_task=64,
    )
    seed_everything(
        derive_seed(20260825, "spectral_neural_process", scenario, replicate)
    )
    run_replicate(
        scenario=scenario,
        replicate=replicate,
        output_dir=output_dir,
        network_name="clustered_diagonal_tanh",
        methods=methods,
        couplings=couplings,
        device=device,
        prepared_data=data,
        network_config=network,
        optimization_config=optimization,
        coupling_optimization_config=coupling_optimization,
    )

    metrics_path = output_dir / scenario / f"rep_{replicate:03d}" / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    profile_accuracy = _bayes_profile_accuracy(data.test.x, data.truth)
    metrics["bayes_profile_accuracy"] = profile_accuracy
    metrics["task_accuracy_ceiling"] = profile_accuracy / config.tasks_per_profile
    metrics["common_energy"] = float(np.mean(data.test.common ** 2))
    metrics["deviation_energy"] = float(np.mean(data.test.deviation ** 2))
    metrics["signal_overlap_ratio"] = data.truth.signal_overlap_ratio
    metrics["true_head_rank"] = data.truth.coefficient_rank
    metrics.to_csv(metrics_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--replicate", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--within-profile-scale", type=float, default=0.20)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument(
        "--couplings", default=",".join(map(str, (0.0,) + DEFAULT_COUPLINGS)),
    )
    args = parser.parse_args()
    run_job(
        scenario=args.scenario,
        replicate=args.replicate,
        output_dir=args.output_dir,
        device=args.device,
        within_profile_scale=args.within_profile_scale,
        methods=tuple(value for value in args.methods.split(",") if value),
        couplings=tuple(float(value) for value in args.couplings.split(",") if value),
    )


if __name__ == "__main__":
    main()
