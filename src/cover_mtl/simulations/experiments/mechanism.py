"""One-dimensional mechanism experiments under the final overlap DGP."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import NetworkConfig, OptimizationConfig
from ..runner import (
    DEFAULT_COUPLINGS,
    _fit_coupled_path,
    _fit_one_neural,
    _shared_model,
)
from ..randomness import derive_seed
from ..dgp import (
    SpectralNeuralConfig,
    generate_spectral_neural_replicate,
)


MECHANISM_NETWORK = NetworkConfig(
    common_hidden=(32,), representation_hidden=(48,), representation_dim=24,
)


def _population_overlap_diagnostics(data) -> dict[str, float]:
    """Return scale-normalized overlap diagnostics available in simulation."""
    moments = np.asarray(data.truth.representation_moments, dtype=np.float64)
    coefficients = np.asarray(data.truth.coefficients, dtype=np.float64)
    pooled = moments.mean(axis=0)
    pooled_inverse = np.linalg.pinv(pooled, rcond=1e-10)
    rank = max(int(np.linalg.matrix_rank(pooled, tol=1e-10)), 1)
    strengths = []
    overlap_energy = 0.0
    contrast_energy = 0.0
    for left, right in combinations(range(moments.shape[0]), 2):
        left_moment, right_moment = moments[left], moments[right]
        overlap = (
            2.0
            * left_moment
            @ np.linalg.pinv(left_moment + right_moment, rcond=1e-10)
            @ right_moment
        )
        overlap = 0.5 * (overlap + overlap.T)
        difference = coefficients[left] - coefficients[right]
        strengths.append(float(np.trace(pooled_inverse @ overlap) / rank))
        overlap_energy += float(difference @ overlap @ difference)
        average_moment = 0.5 * (left_moment + right_moment)
        contrast_energy += float(difference @ average_moment @ difference)
    edge_count = max(len(strengths), 1)
    return {
        "normalized_overlap_strength": float(np.mean(strengths)),
        "overlap_energy": overlap_energy / edge_count,
        "contrast_energy": contrast_energy / edge_count,
        "supported_contrast_fraction": (
            overlap_energy / contrast_energy
            if contrast_energy > 1e-12
            else float("nan")
        ),
    }


def run_job(
    *,
    axis: str,
    value: float,
    replicate: int,
    output_dir: Path,
    device: str,
    couplings: tuple[float, ...],
) -> None:
    if axis not in {"covariate", "posterior"}:
        raise ValueError("axis must be covariate or posterior.")
    weak_variance = value if axis == "covariate" else 0.001
    posterior_scale = 1.0 if axis == "covariate" else value
    data_seed = derive_seed(20260827, "spectral_mechanism_data", axis, replicate)
    dgp = SpectralNeuralConfig(
        scenario="both_overlap_aligned",
        num_tasks=48,
        num_profiles=6,
        input_dim=24,
        representation_dim=24,
        active_per_profile=4,
        train_size=100,
        validation_size=200,
        test_size=3000,
        weak_variance=weak_variance,
        posterior_scale=posterior_scale,
        within_profile_scale=0.30,
        covariance_geometry="diagonal",
        posterior_geometry="clustered",
        representation_transform="tanh",
        subspace_rank=4,
        low_mode_count=48,
        seed=data_seed,
    )
    network = MECHANISM_NETWORK
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
    data = generate_spectral_neural_replicate(dgp)
    overlap_diagnostics = _population_overlap_diagnostics(data)
    hps_seed = derive_seed(20260827, "spectral_mechanism_model", axis, replicate, "HPS")
    hps_fit, hps_row, hps_tasks = _fit_one_neural(
        "HPS",
        _shared_model(dgp, network, "none"),
        data,
        optimization,
        hps_seed,
        device,
    )
    metric_rows = [hps_row]
    task_rows = hps_tasks
    tuning_rows: list[dict[str, object]] = []
    for method, integration in (
        ("Average-Moment", "average_moment"),
        ("COVER", "cover"),
    ):
        _, row, rows, path = _fit_coupled_path(
            method,
            integration,
            hps_fit,
            data,
            network,
            coupling_optimization,
            couplings,
            derive_seed(20260827, "spectral_mechanism_model", axis, replicate),
            device,
        )
        metric_rows.append(row)
        task_rows.extend(rows)
        tuning_rows.extend(path)
    for row in metric_rows:
        row.update(
            {
                "axis": axis,
                "value": value,
                "weak_variance": weak_variance,
                "posterior_scale": posterior_scale,
                "within_profile_scale": 0.30,
                "replicate": replicate,
                "data_seed": data_seed,
                **overlap_diagnostics,
            }
        )
    task_frame = pd.DataFrame(task_rows)
    task_frame["axis"] = axis
    task_frame["value"] = value
    task_frame["replicate"] = replicate
    value_tag = f"{value:.4g}".replace(".", "p")
    job_dir = output_dir / axis / f"value_{value_tag}" / f"rep_{replicate:03d}"
    job_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(job_dir / "metrics.csv", index=False)
    task_frame.to_csv(job_dir / "task_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(job_dir / "tuning.csv", index=False)
    with (job_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "output_schema_version": 2,
                "dgp": dgp.to_dict(),
                "network_name": "spectral_base_d24",
                "network": asdict(network),
                "optimization": asdict(optimization),
                "coupling_optimization": asdict(coupling_optimization),
                "couplings": couplings,
                "methods": ("HPS", "Average-Moment", "COVER"),
            },
            stream,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", required=True, choices=("covariate", "posterior"))
    parser.add_argument("--value", required=True, type=float)
    parser.add_argument("--replicate", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--couplings", default=",".join(map(str, DEFAULT_COUPLINGS)))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_job(
        axis=args.axis,
        value=args.value,
        replicate=args.replicate,
        output_dir=args.output_dir,
        device=args.device,
        couplings=tuple(float(item) for item in args.couplings.split(",")),
    )


if __name__ == "__main__":
    main()
