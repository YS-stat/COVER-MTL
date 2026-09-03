"""Initialization and architecture sensitivity for the selected nonlinear DGP."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import NetworkConfig, OptimizationConfig
from ..runner import run_replicate
from ..randomness import derive_seed
from ..dgp import (
    SpectralNeuralConfig,
    generate_spectral_neural_replicate,
)


NETWORKS: dict[str, NetworkConfig] = {
    "base": NetworkConfig(
        common_hidden=(32,), representation_hidden=(48,), representation_dim=24,
    ),
    "d12": NetworkConfig(
        common_hidden=(32,), representation_hidden=(48,), representation_dim=12,
    ),
    "d36": NetworkConfig(
        common_hidden=(32,), representation_hidden=(48,), representation_dim=36,
    ),
    "narrow": NetworkConfig(
        common_hidden=(24,), representation_hidden=(32,), representation_dim=24,
    ),
    "wide": NetworkConfig(
        common_hidden=(64,), representation_hidden=(96,), representation_dim=24,
    ),
    "deep": NetworkConfig(
        common_hidden=(32, 32), representation_hidden=(48, 48), representation_dim=24,
    ),
}


def run_job(
    *, variant: str, initialization: int, output_dir: Path, device: str,
) -> None:
    network = NETWORKS[variant]
    scenario = "both_overlap_aligned"
    data_seed = derive_seed(20260826, "sensitivity_fixed_data")
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
        within_profile_scale=0.20,
        covariance_geometry="diagonal",
        posterior_geometry="clustered",
        representation_transform="tanh",
        subspace_rank=4,
        seed=data_seed,
    )
    data = generate_spectral_neural_replicate(config)
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
    variant_dir = output_dir / variant
    run_replicate(
        scenario=scenario,
        replicate=initialization,
        output_dir=variant_dir,
        network_name=variant,
        methods=("HPS", "COVER"),
        couplings=(0.0, 0.1, 0.3, 1.0, 3.0, 10.0),
        device=device,
        prepared_data=data,
        network_config=network,
        optimization_config=optimization,
        coupling_optimization_config=coupling_optimization,
    )
    metrics_path = variant_dir / scenario / f"rep_{initialization:03d}" / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics["variant"] = variant
    metrics["initialization"] = initialization
    metrics["data_seed"] = data_seed
    metrics.to_csv(metrics_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=tuple(NETWORKS))
    parser.add_argument("--initialization", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_job(
        variant=args.variant,
        initialization=args.initialization,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
