"""One-axis scaling experiment matched to the main strong setting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import NetworkConfig, OptimizationConfig
from ..runner import DEFAULT_METHODS, run_replicate
from ..randomness import derive_seed
from ..dgp import (
    SpectralNeuralConfig,
    generate_spectral_neural_replicate,
)


def run_job(
    *,
    axis: str,
    value: int,
    replicate: int,
    output_dir: Path,
    device: str,
    methods: tuple[str, ...],
    couplings: tuple[float, ...],
    posterior_scale: float,
    within_profile_scale: float,
    weak_variance: float,
    tasks_per_profile: int | None,
    task_train_size: int,
    dimension_train_size: int,
    posterior_geometry: str,
    rotation_structure: str,
    common_coordinate_mode: str,
) -> None:
    representation_dim = 24
    if axis == "tasks":
        num_tasks = value
        input_dim = 24
        train_size = task_train_size
    elif axis == "dimension":
        num_tasks = 48
        input_dim = value
        train_size = dimension_train_size
    else:
        raise ValueError("axis must be tasks or dimension.")
    num_profiles = 6 if tasks_per_profile is None else num_tasks // tasks_per_profile
    if tasks_per_profile is not None and num_tasks % tasks_per_profile:
        raise ValueError("The task count must be divisible by tasks_per_profile.")
    if num_tasks % num_profiles:
        raise ValueError("The task count must be divisible by the profile count.")
    if input_dim < representation_dim:
        raise ValueError("The input dimension must be at least 24.")

    scenario = "both_overlap_aligned"
    data_seed = derive_seed(20260827, "final_scaling_data", axis, value, replicate)
    dgp = SpectralNeuralConfig(
        scenario=scenario,
        num_tasks=num_tasks,
        num_profiles=num_profiles,
        input_dim=input_dim,
        representation_dim=representation_dim,
        active_per_profile=4,
        train_size=train_size,
        validation_size=200,
        test_size=3000,
        weak_variance=weak_variance,
        posterior_scale=posterior_scale,
        within_profile_scale=within_profile_scale,
        covariance_geometry="diagonal",
        posterior_geometry=posterior_geometry,
        representation_transform="tanh",
        rotation_structure=rotation_structure,
        common_coordinate_mode=common_coordinate_mode,
        subspace_rank=4,
        low_mode_count=48,
        moment_sample_size=10_000,
        seed=data_seed,
    )
    data = generate_spectral_neural_replicate(dgp)
    network = NetworkConfig(
        common_hidden=(32,),
        representation_hidden=(48,),
        representation_dim=representation_dim,
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
    setting_dir = output_dir / f"{axis}_{value:04d}"
    run_replicate(
        scenario=scenario,
        replicate=replicate,
        output_dir=setting_dir,
        network_name=(
            f"final_strong_scaling_d24_signal{posterior_scale:g}"
            f"_within{within_profile_scale:g}_profiles{num_profiles}"
            f"_{posterior_geometry}_{rotation_structure}"
        ),
        methods=methods,
        couplings=couplings,
        device=device,
        prepared_data=data,
        network_config=network,
        optimization_config=optimization,
        coupling_optimization_config=coupling_optimization,
    )
    metrics_path = setting_dir / scenario / f"rep_{replicate:03d}" / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    hps_rows = metrics.loc[metrics["method"] == "HPS"]
    hps_seconds = float(hps_rows["fit_seconds"].iloc[0]) if not hps_rows.empty else 0.0
    hps_memory = (
        float(hps_rows["peak_device_memory_mb"].iloc[0]) if not hps_rows.empty else 0.0
    )
    metrics["workflow_seconds"] = metrics["fit_seconds"]
    hps_dependent = metrics["method"].isin(
        ("Average-Moment", "COVER", "ARMUL", "FLARCC")
    )
    metrics.loc[hps_dependent, "workflow_seconds"] += hps_seconds
    metrics.loc[hps_dependent, "peak_device_memory_mb"] = (
        metrics.loc[hps_dependent, "peak_device_memory_mb"]
        .fillna(hps_memory)
        .clip(lower=hps_memory)
    )
    metrics["scaling_axis"] = axis
    metrics["scaling_value"] = value
    metrics["num_tasks"] = num_tasks
    metrics["input_dim"] = input_dim
    metrics["representation_dim"] = representation_dim
    metrics["num_profiles"] = num_profiles
    metrics["tasks_per_profile"] = num_tasks // num_profiles
    metrics["train_size"] = train_size
    metrics["posterior_scale"] = posterior_scale
    metrics["within_profile_scale"] = within_profile_scale
    metrics["weak_variance"] = weak_variance
    metrics["posterior_geometry"] = posterior_geometry
    metrics["rotation_structure"] = rotation_structure
    metrics["common_coordinate_mode"] = common_coordinate_mode
    metrics.to_csv(metrics_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", required=True, choices=("tasks", "dimension"))
    parser.add_argument("--value", required=True, type=int)
    parser.add_argument("--replicate", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--couplings", required=True)
    parser.add_argument("--posterior-scale", type=float, default=1.0)
    parser.add_argument("--within-profile-scale", type=float, default=0.30)
    parser.add_argument("--weak-variance", type=float, default=0.001)
    parser.add_argument("--tasks-per-profile", type=int, default=0)
    parser.add_argument("--task-train-size", type=int, default=100)
    parser.add_argument("--dimension-train-size", type=int, default=100)
    parser.add_argument(
        "--posterior-geometry",
        choices=("spectral", "clustered", "projected"),
        default="clustered",
    )
    parser.add_argument(
        "--rotation-structure", choices=("full", "signal_block"), default="full",
    )
    parser.add_argument(
        "--common-coordinate-mode",
        choices=("auto", "representation", "separate"),
        default="auto",
    )
    args = parser.parse_args()
    run_job(
        axis=args.axis,
        value=args.value,
        replicate=args.replicate,
        output_dir=args.output_dir,
        device=args.device,
        methods=tuple(value for value in args.methods.split(",") if value),
        couplings=tuple(float(value) for value in args.couplings.split(",") if value),
        posterior_scale=args.posterior_scale,
        within_profile_scale=args.within_profile_scale,
        weak_variance=args.weak_variance,
        tasks_per_profile=(
            args.tasks_per_profile if args.tasks_per_profile > 0 else None
        ),
        task_train_size=args.task_train_size,
        dimension_train_size=args.dimension_train_size,
        posterior_geometry=args.posterior_geometry,
        rotation_structure=args.rotation_structure,
        common_coordinate_mode=args.common_coordinate_mode,
    )


if __name__ == "__main__":
    main()
