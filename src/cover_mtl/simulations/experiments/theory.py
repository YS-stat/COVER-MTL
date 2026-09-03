"""Monte Carlo verification of the fixed-representation bias--variance identity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import block_diag, eigh, null_space


def _harmonic_overlap(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    overlap = 2.0 * left @ np.linalg.pinv(left + right) @ right
    return 0.5 * (overlap + overlap.T)


def _laplacian(second_moments: list[np.ndarray]) -> np.ndarray:
    task_count = len(second_moments)
    dimension = second_moments[0].shape[0]
    laplacian = np.zeros((task_count * dimension, task_count * dimension))
    for left in range(task_count):
        for right in range(left + 1, task_count):
            overlap = _harmonic_overlap(second_moments[left], second_moments[right])
            left_slice = slice(left * dimension, (left + 1) * dimension)
            right_slice = slice(right * dimension, (right + 1) * dimension)
            laplacian[left_slice, left_slice] += overlap
            laplacian[right_slice, right_slice] += overlap
            laplacian[left_slice, right_slice] -= overlap
            laplacian[right_slice, left_slice] -= overlap
    return laplacian


def _population_moments(
    design: str, task_count: int, dimension: int, generator: np.random.Generator,
) -> list[np.ndarray]:
    if design == "equal":
        return [np.eye(dimension) for _ in range(task_count)]
    code = np.empty((task_count, dimension), dtype=float)
    for coordinate in range(dimension):
        code[:, coordinate] = np.roll(
            np.r_[np.ones(task_count // 2), np.zeros(task_count - task_count // 2)],
            coordinate,
        )
    if design in {"diagonal", "near_singular"}:
        low = 0.2 if design == "diagonal" else 0.03
        return [np.diag(low + (1.0 - low) * code[task]) for task in range(task_count)]
    if design == "rotated":
        moments = []
        for task in range(task_count):
            rotation, _ = np.linalg.qr(generator.normal(size=(dimension, dimension)))
            eigenvalues = 0.15 + 0.85 * code[task]
            moments.append(rotation @ np.diag(eigenvalues) @ rotation.T)
        return moments
    if design == "random_spd":
        moments = []
        for _ in range(task_count):
            rotation, _ = np.linalg.qr(generator.normal(size=(dimension, dimension)))
            eigenvalues = generator.uniform(0.08, 1.2, size=dimension)
            moments.append(rotation @ np.diag(eigenvalues) @ rotation.T)
        return moments
    raise ValueError(f"Unknown fixed design {design!r}.")


def _one_design(
    design: str,
    task_count: int,
    dimension: int,
    sample_size: int,
    draws: int,
    lambdas: np.ndarray,
    noise_scale: float,
    seed: int,
) -> list[dict[str, float | str]]:
    generator = np.random.default_rng(seed)
    population_moments = _population_moments(design, task_count, dimension, generator)
    features = np.stack(
        [
            generator.multivariate_normal(np.zeros(dimension), moment, size=sample_size)
            for moment in population_moments
        ]
    )
    empirical_moments = [
        features[task].T @ features[task] / sample_size for task in range(task_count)
    ]
    block_moment = block_diag(*empirical_moments)
    laplacian = _laplacian(empirical_moments)
    task_contrast = null_space(np.ones((1, task_count)))
    centered_basis = np.kron(task_contrast, np.eye(dimension))
    centered_moment = centered_basis.T @ block_moment @ centered_basis
    centered_laplacian = centered_basis.T @ laplacian @ centered_basis
    eigenvalues, eigenvectors = eigh(centered_laplacian, centered_moment)
    generalized_vectors = centered_basis @ eigenvectors

    coefficients = generator.normal(size=(task_count, dimension))
    coefficients -= coefficients.mean(axis=0, keepdims=True)
    coefficients /= np.sqrt(np.mean(coefficients ** 2))
    coefficient_vector = coefficients.reshape(-1)
    signal_coordinates = generalized_vectors.T @ block_moment @ coefficient_vector

    errors = generator.normal(scale=noise_scale, size=(draws, task_count, sample_size))
    scores = np.einsum("tnd,mtn->mtd", features, errors) / sample_size
    coefficient_noise = np.stack(
        [
            scores[:, task] @ np.linalg.inv(empirical_moments[task]).T
            for task in range(task_count)
        ],
        axis=1,
    )
    independent_estimates = coefficient_vector[None, :] + coefficient_noise.reshape(
        draws, -1
    )
    right_hand_side = independent_estimates @ block_moment @ centered_basis
    rows: list[dict[str, float | str]] = []
    for coupling in lambdas:
        kappa = 2.0 * coupling / (task_count - 1)
        system = centered_moment + kappa * centered_laplacian
        centered_estimates = right_hand_side @ np.linalg.inv(system)
        estimates = centered_estimates @ centered_basis.T
        errors_beta = estimates - coefficient_vector[None, :]
        empirical_risk = float(
            np.mean(np.einsum("mi,ij,mj->m", errors_beta, block_moment, errors_beta))
            / task_count
        )
        theoretical_bias = (
            kappa * eigenvalues / (1.0 + kappa * eigenvalues)
        ) ** 2 * signal_coordinates ** 2
        theoretical_variance = (
            noise_scale ** 2 / sample_size / (1.0 + kappa * eigenvalues) ** 2
        )
        theoretical_risk = float(
            np.sum(theoretical_bias + theoretical_variance) / task_count
        )
        rows.append(
            {
                "design": design,
                "coupling": float(coupling),
                "empirical_risk": empirical_risk,
                "theoretical_risk": theoretical_risk,
                "relative_error": abs(empirical_risk - theoretical_risk)
                / max(theoretical_risk, 1e-14),
                "theoretical_bias": float(theoretical_bias.sum() / task_count),
                "theoretical_variance": float(theoretical_variance.sum() / task_count),
                "effective_dimension": float(
                    np.sum(1.0 / (1.0 + kappa * eigenvalues) ** 2)
                ),
                "minimum_eigenvalue": float(eigenvalues.min()),
                "maximum_eigenvalue": float(eigenvalues.max()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--lambda-count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    lambdas = np.r_[0.0, np.geomspace(1e-3, 10.0, args.lambda_count - 1)]
    rows = []
    for index, design in enumerate(
        ("equal", "diagonal", "rotated", "near_singular", "random_spd")
    ):
        rows.extend(
            _one_design(
                design=design,
                task_count=6,
                dimension=6,
                sample_size=100,
                draws=args.draws,
                lambdas=lambdas,
                noise_scale=1.0,
                seed=args.seed + 1009 * index,
            )
        )
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    audit = table.groupby("design")["relative_error"].agg(["mean", "max"])
    print(audit.to_string())


if __name__ == "__main__":
    main()
