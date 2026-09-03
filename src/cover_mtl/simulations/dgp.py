"""Nonlinear DGP whose posterior contrasts follow covariate-overlap modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .data import SimulationReplicate, SimulationSplit


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SpectralScenario = Literal[
    "homogeneous",
    "covariate_only",
    "posterior_only",
    "both_overlap_aligned",
    "both_misaligned",
    "both_random",
    "strong_task_specific",
]


@dataclass(frozen=True)
class SpectralNeuralConfig:
    """Configuration for the overlap-mode nonlinear experiment."""

    scenario: SpectralScenario
    num_tasks: int = 24
    num_profiles: int = 12
    input_dim: int = 26
    representation_dim: int = 18
    active_per_profile: int = 9
    train_size: int = 100
    validation_size: int = 200
    test_size: int = 3000
    weak_variance: float = 0.20
    posterior_scale: float = 1.25
    within_profile_scale: float = 0.10
    strong_posterior_scale: float = 3.0
    low_mode_count: int = 36
    noise_scale: float = 1.0
    covariance_geometry: Literal["diagonal", "subspace"] = "diagonal"
    posterior_geometry: Literal["spectral", "clustered", "projected"] = "spectral"
    representation_transform: Literal["linear", "tanh"] = "linear"
    rotation_structure: Literal["full", "signal_block"] = "full"
    common_coordinate_mode: Literal["auto", "representation", "separate"] = "auto"
    subspace_rank: int = 5
    moment_sample_size: int = 10_000
    seed: int = 1

    def __post_init__(self) -> None:
        allowed = {
            "homogeneous",
            "covariate_only",
            "posterior_only",
            "both_overlap_aligned",
            "both_misaligned",
            "both_random",
            "strong_task_specific",
        }
        if self.scenario not in allowed:
            raise ValueError(f"Unknown spectral scenario {self.scenario!r}.")
        if self.num_tasks % self.num_profiles != 0:
            raise ValueError("Tasks must divide evenly across covariance profiles.")
        if self.input_dim < self.representation_dim:
            raise ValueError("input_dim must be at least representation_dim.")
        if self.input_dim < 5:
            raise ValueError("The input must include five common-function coordinates.")
        if self.covariance_geometry not in {"diagonal", "subspace"}:
            raise ValueError("Unknown covariance geometry.")
        if self.posterior_geometry not in {
            "spectral",
            "clustered",
            "projected",
        }:
            raise ValueError("Unknown posterior geometry.")
        if self.representation_transform not in {"linear", "tanh"}:
            raise ValueError("Unknown representation transform.")
        if self.rotation_structure not in {"full", "signal_block"}:
            raise ValueError("Unknown rotation structure.")
        if self.common_coordinate_mode not in {
            "auto",
            "representation",
            "separate",
        }:
            raise ValueError("Unknown common-coordinate mode.")
        if (
            self.common_coordinate_mode == "separate"
            and self.input_dim < self.representation_dim + 5
        ):
            raise ValueError(
                "Separate common coordinates require representation_dim + 5 inputs."
            )
        if self.covariance_geometry == "diagonal":
            total_degree = self.num_profiles * self.active_per_profile
            if total_degree % self.representation_dim:
                raise ValueError(
                    "Profile supports must have a balanced integer degree."
                )
        if not 0 < self.subspace_rank <= self.representation_dim:
            raise ValueError("subspace_rank must lie in [1, representation_dim].")
        if not 0 < self.weak_variance <= 1:
            raise ValueError("weak_variance must lie in (0, 1].")
        if min(self.train_size, self.validation_size, self.test_size) <= 0:
            raise ValueError("All split sizes must be positive.")
        if self.moment_sample_size <= 0:
            raise ValueError("moment_sample_size must be positive.")
        if self.low_mode_count < self.representation_dim:
            raise ValueError("Use at least representation_dim low-overlap modes.")

    @property
    def tasks_per_profile(self) -> int:
        return self.num_tasks // self.num_profiles

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SpectralNeuralTruth:
    rotation: FloatArray
    covariance_diagonals: FloatArray
    latent_covariances: FloatArray
    representation_moments: FloatArray
    coefficients: FloatArray
    profile_labels: IntArray
    generalized_eigenvalues: FloatArray
    signal_overlap_ratio: float
    coefficient_rank: int
    representation_dim: int
    representation_transform: str
    common_coordinate_start: int
    diagonal_latent_covariance: bool

    def latent_coordinates(self, x: FloatArray) -> FloatArray:
        return x @ self.rotation

    def representation(self, x: FloatArray) -> FloatArray:
        value = self.latent_coordinates(x)[..., : self.representation_dim]
        if self.representation_transform == "tanh":
            return np.tanh(value)
        return value

    def common_function(self, x: FloatArray) -> FloatArray:
        latent = self.latent_coordinates(x)
        start = self.common_coordinate_start
        u = latent[..., start : start + 5]
        return (
            0.8 * np.sin(u[..., 0])
            + 0.5 * np.maximum(u[..., 1], 0.0)
            - 0.4 * np.maximum(-u[..., 2], 0.0)
            + 0.3 * u[..., 3] * u[..., 4]
        )


def _orthogonal(generator: np.random.Generator, dimension: int) -> FloatArray:
    basis, triangular = np.linalg.qr(generator.normal(size=(dimension, dimension)))
    signs = np.where(np.diag(triangular) < 0, -1.0, 1.0)
    return basis * signs


def _balanced_supports(
    generator: np.random.Generator,
    num_profiles: int,
    dimension: int,
    active_per_profile: int,
) -> FloatArray:
    """Construct distinct support patterns with balanced coordinate degrees."""
    target_degree = num_profiles * active_per_profile // dimension
    supports = np.zeros((num_profiles, dimension), dtype=np.float64)
    for _ in range(20_000):
        supports.fill(0.0)
        remaining = np.full(dimension, target_degree, dtype=np.int64)
        success = True
        for profile in generator.permutation(num_profiles):
            available = np.flatnonzero(remaining > 0)
            if available.size < active_per_profile:
                success = False
                break
            weights = remaining[available].astype(np.float64)
            selected = generator.choice(
                available,
                size=active_per_profile,
                replace=False,
                p=weights / weights.sum(),
            )
            supports[profile, selected] = 1.0
            remaining[selected] -= 1
        if (
            success
            and np.all(remaining == 0)
            and np.unique(supports, axis=0).shape[0] == num_profiles
        ):
            return supports.copy()
    raise RuntimeError("Failed to construct balanced covariance profiles.")


def _overlap(left: FloatArray, right: FloatArray) -> FloatArray:
    value = 2.0 * left @ np.linalg.pinv(left + right) @ right
    return 0.5 * (value + value.T)


def _block_geometry(
    moments: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    task_count, dimension, _ = moments.shape
    size = task_count * dimension
    design = np.zeros((size, size), dtype=np.float64)
    laplacian = np.zeros_like(design)
    for task, moment in enumerate(moments):
        block = slice(task * dimension, (task + 1) * dimension)
        design[block, block] = moment
    for left in range(task_count):
        left_block = slice(left * dimension, (left + 1) * dimension)
        for right in range(left + 1, task_count):
            right_block = slice(right * dimension, (right + 1) * dimension)
            omega = _overlap(moments[left], moments[right])
            laplacian[left_block, left_block] += omega
            laplacian[right_block, right_block] += omega
            laplacian[left_block, right_block] -= omega
            laplacian[right_block, left_block] -= omega
    values, vectors = np.linalg.eigh(design)
    inverse_root = (vectors * np.maximum(values, 1e-12) ** -0.5) @ vectors.T
    whitened = inverse_root @ laplacian @ inverse_root
    eigenvalues, whitened_vectors = np.linalg.eigh(0.5 * (whitened + whitened.T))
    eigenvectors = inverse_root @ whitened_vectors
    return design, laplacian, eigenvalues, eigenvectors


def _center_rows(coefficients: FloatArray) -> FloatArray:
    return coefficients - coefficients.mean(axis=0, keepdims=True)


def _prediction_rms(coefficients: FloatArray, moments: FloatArray) -> float:
    energies = np.einsum("td,tdk,tk->t", coefficients, moments, coefficients)
    return float(np.sqrt(np.mean(energies)))


def _rescale(
    coefficients: FloatArray, moments: FloatArray, target_scale: float,
) -> FloatArray:
    if target_scale == 0:
        return np.zeros_like(coefficients)
    current = _prediction_rms(coefficients, moments)
    if current <= 1e-12:
        raise RuntimeError("Cannot rescale a zero posterior component.")
    return coefficients * (target_scale / current)


def _direct_overlap_ratio(coefficients: FloatArray, moments: FloatArray,) -> float:
    """Compute the overlap energy without forming the block Laplacian."""
    signal_energy = float(np.einsum("td,tdk,tk->", coefficients, moments, coefficients))
    if signal_energy <= 1e-12:
        return 0.0
    overlap_energy = 0.0
    for left in range(coefficients.shape[0]):
        for right in range(left + 1, coefficients.shape[0]):
            difference = coefficients[left] - coefficients[right]
            omega = _overlap(moments[left], moments[right])
            overlap_energy += float(difference @ omega @ difference)
    return overlap_energy / signal_energy


def _full_rank_low_mode_signal(
    generator: np.random.Generator,
    eigenvalues: FloatArray,
    eigenvectors: FloatArray,
    task_count: int,
    dimension: int,
    mode_count: int,
) -> FloatArray:
    positive = np.flatnonzero(eigenvalues > 1e-8)
    selected = positive[: min(mode_count, positive.size)]
    if selected.size < dimension:
        raise RuntimeError("The overlap geometry has too few positive modes.")
    for _ in range(500):
        signal = eigenvectors[:, selected] @ generator.normal(size=selected.size)
        coefficients = _center_rows(signal.reshape(task_count, dimension))
        if np.linalg.matrix_rank(coefficients, tol=1e-8) == dimension:
            return coefficients
    raise RuntimeError("Failed to generate a full-rank low-overlap signal.")


def _full_rank_high_mode_signal(
    generator: np.random.Generator,
    eigenvalues: FloatArray,
    eigenvectors: FloatArray,
    task_count: int,
    dimension: int,
    mode_count: int,
) -> FloatArray:
    positive = np.flatnonzero(eigenvalues > 1e-8)
    selected = positive[-min(mode_count, positive.size) :]
    if selected.size < dimension:
        raise RuntimeError("The overlap geometry has too few positive modes.")
    for _ in range(500):
        signal = eigenvectors[:, selected] @ generator.normal(size=selected.size)
        coefficients = _center_rows(signal.reshape(task_count, dimension))
        if np.linalg.matrix_rank(coefficients, tol=1e-8) == dimension:
            return coefficients
    raise RuntimeError("Failed to generate a full-rank high-overlap signal.")


def _posterior_coefficients(
    generator: np.random.Generator,
    config: SpectralNeuralConfig,
    moments: FloatArray,
    profile_labels: IntArray,
) -> tuple[FloatArray, FloatArray, float]:
    if config.scenario in {"homogeneous", "covariate_only"}:
        coefficients = np.zeros((config.num_tasks, config.representation_dim))
        return coefficients, np.empty(0, dtype=np.float64), 0.0
    if config.scenario == "both_overlap_aligned" and config.posterior_geometry in {
        "clustered",
        "projected",
    }:
        prototypes = []
        global_direction = generator.normal(size=config.representation_dim)
        for profile in range(config.num_profiles):
            representative = int(np.flatnonzero(profile_labels == profile)[0])
            _, vectors = np.linalg.eigh(moments[representative])
            active = vectors[:, -config.subspace_rank :]
            if config.posterior_geometry == "projected":
                prototypes.append(active @ (active.T @ global_direction))
            else:
                prototypes.append(active @ generator.normal(size=config.subspace_rank))
        between = np.asarray(prototypes)[profile_labels]
        between = _rescale(_center_rows(between), moments, config.posterior_scale)
        within = np.zeros_like(between)
        for profile in range(config.num_profiles):
            indices = np.flatnonzero(profile_labels == profile)
            _, vectors = np.linalg.eigh(moments[indices[0]])
            active = vectors[:, -config.subspace_rank :]
            local = generator.normal(size=(indices.size, config.subspace_rank))
            local -= local.mean(axis=0, keepdims=True)
            within[indices] = local @ active.T
        within = _rescale(within, moments, config.within_profile_scale)
        coefficients = _center_rows(between + within)
        return (
            coefficients,
            np.empty(0, dtype=np.float64),
            _direct_overlap_ratio(coefficients, moments),
        )
    if config.scenario in {
        "posterior_only",
        "both_random",
        "strong_task_specific",
    }:
        coefficients = _center_rows(
            generator.normal(size=(config.num_tasks, config.representation_dim))
        )
        scale = (
            config.strong_posterior_scale
            if config.scenario == "strong_task_specific"
            else config.posterior_scale
        )
        coefficients = _rescale(coefficients, moments, scale)
        return (
            coefficients,
            np.empty(0, dtype=np.float64),
            _direct_overlap_ratio(coefficients, moments),
        )
    design, laplacian, eigenvalues, eigenvectors = _block_geometry(moments)
    if config.scenario == "both_overlap_aligned":
        coefficients = _full_rank_low_mode_signal(
            generator,
            eigenvalues,
            eigenvectors,
            config.num_tasks,
            config.representation_dim,
            config.low_mode_count,
        )
        coefficients = _rescale(coefficients, moments, config.posterior_scale)
    elif config.scenario == "both_misaligned":
        coefficients = _full_rank_high_mode_signal(
            generator,
            eigenvalues,
            eigenvectors,
            config.num_tasks,
            config.representation_dim,
            config.low_mode_count,
        )
        coefficients = _rescale(coefficients, moments, config.posterior_scale)
    else:
        raise RuntimeError(f"Unhandled scenario {config.scenario!r}.")
    flattened = coefficients.ravel()
    signal_energy = float(flattened @ design @ flattened)
    overlap_ratio = float(flattened @ laplacian @ flattened / max(signal_energy, 1e-12))
    return coefficients, eigenvalues, overlap_ratio


def _make_split(
    generator: np.random.Generator,
    sample_size: int,
    truth: SpectralNeuralTruth,
    noise_scale: float,
) -> SimulationSplit:
    task_count, input_dim = truth.covariance_diagonals.shape
    x = np.empty((task_count, sample_size, input_dim), dtype=np.float64)
    common = np.empty((task_count, sample_size), dtype=np.float64)
    deviation = np.empty_like(common)
    for task in range(task_count):
        if truth.diagonal_latent_covariance:
            latent = generator.normal(size=(sample_size, input_dim))
            latent *= np.sqrt(truth.covariance_diagonals[task])
        else:
            latent = generator.multivariate_normal(
                np.zeros(input_dim), truth.latent_covariances[task], size=sample_size,
            )
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


def _transformed_moments(
    generator: np.random.Generator,
    raw_moments: FloatArray,
    transform: str,
    sample_size: int,
) -> FloatArray:
    """Approximate representation second moments under each covariate profile.

    The returned matrices are Monte Carlo approximations used to construct and
    diagnose the posterior signal. The exact covariate laws are stored in
    ``latent_covariances`` and generate every train, validation, and test split.
    """
    if transform == "linear":
        return raw_moments.copy()
    moments = []
    dimension = raw_moments.shape[1]
    for covariance in raw_moments:
        samples = generator.multivariate_normal(
            mean=np.zeros(dimension), cov=covariance, size=sample_size,
        )
        representation = np.tanh(samples)
        moments.append(representation.T @ representation / sample_size)
    return np.asarray(moments)


def generate_spectral_neural_replicate(
    config: SpectralNeuralConfig,
) -> SimulationReplicate:
    generator = np.random.default_rng(config.seed)
    profile_labels = np.repeat(
        np.arange(config.num_profiles, dtype=np.int64), config.tasks_per_profile,
    )
    if config.covariance_geometry == "diagonal":
        profile_supports = _balanced_supports(
            generator,
            config.num_profiles,
            config.representation_dim,
            config.active_per_profile,
        )
        profile_variances = (
            config.weak_variance + (1.0 - config.weak_variance) * profile_supports
        )
        raw_profile_moments = np.asarray(
            [np.diag(value) for value in profile_variances]
        )
    else:
        identity = np.eye(config.representation_dim)
        raw_profile_moments = []
        for _ in range(config.num_profiles):
            basis, _ = np.linalg.qr(
                generator.normal(size=(config.representation_dim, config.subspace_rank))
            )
            raw_profile_moments.append(
                config.weak_variance * identity
                + (1.0 - config.weak_variance) * basis @ basis.T
            )
        raw_profile_moments = np.asarray(raw_profile_moments)
    if config.scenario in {"homogeneous", "posterior_only"}:
        raw_profile_moments = np.repeat(
            np.eye(config.representation_dim)[None, :, :], config.num_profiles, axis=0,
        )
    moment_generator = np.random.default_rng(config.seed + 1_000_003)
    profile_moments = _transformed_moments(
        moment_generator,
        raw_profile_moments,
        config.representation_transform,
        config.moment_sample_size,
    )
    task_representation_moments = profile_moments[profile_labels]
    latent_covariances = np.repeat(
        np.eye(config.input_dim)[None, :, :], config.num_tasks, axis=0,
    )
    latent_covariances[
        :, : config.representation_dim, : config.representation_dim
    ] = raw_profile_moments[profile_labels]
    covariance_diagonals = np.diagonal(latent_covariances, axis1=1, axis2=2).copy()
    coefficients, eigenvalues, overlap_ratio = _posterior_coefficients(
        generator, config, task_representation_moments, profile_labels,
    )
    if config.common_coordinate_mode == "representation":
        common_coordinate_start = 0
    elif config.common_coordinate_mode == "separate":
        common_coordinate_start = config.representation_dim
    else:
        common_coordinate_start = (
            config.representation_dim
            if config.input_dim >= config.representation_dim + 5
            else 0
        )
    if config.rotation_structure == "full":
        rotation = _orthogonal(generator, config.input_dim)
    else:
        signal_dim = max(config.representation_dim, common_coordinate_start + 5,)
        rotation = np.eye(config.input_dim)
        rotation[:signal_dim, :signal_dim] = _orthogonal(generator, signal_dim)
    truth = SpectralNeuralTruth(
        rotation=rotation,
        covariance_diagonals=covariance_diagonals,
        latent_covariances=latent_covariances,
        representation_moments=task_representation_moments,
        coefficients=coefficients,
        profile_labels=profile_labels,
        generalized_eigenvalues=eigenvalues,
        signal_overlap_ratio=overlap_ratio,
        coefficient_rank=int(np.linalg.matrix_rank(coefficients, tol=1e-8)),
        representation_dim=config.representation_dim,
        representation_transform=config.representation_transform,
        common_coordinate_start=common_coordinate_start,
        diagonal_latent_covariance=config.covariance_geometry == "diagonal",
    )
    arguments = {
        "generator": generator,
        "truth": truth,
        "noise_scale": config.noise_scale,
    }
    return SimulationReplicate(
        config=config,
        train=_make_split(sample_size=config.train_size, **arguments),
        validation=_make_split(sample_size=config.validation_size, **arguments),
        test=_make_split(sample_size=config.test_size, **arguments),
        truth=truth,
    )
