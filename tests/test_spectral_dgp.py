"""Tests for the paper-facing factorial simulation design."""

from __future__ import annotations

import numpy as np

from cover_mtl.simulations.dgp import (
    SpectralNeuralConfig,
    generate_spectral_neural_replicate,
)


def _config(scenario: str) -> SpectralNeuralConfig:
    return SpectralNeuralConfig(
        scenario=scenario,
        num_tasks=6,
        num_profiles=3,
        input_dim=6,
        representation_dim=6,
        active_per_profile=2,
        train_size=8,
        validation_size=7,
        test_size=9,
        weak_variance=0.05,
        posterior_scale=0.8,
        within_profile_scale=0.1,
        low_mode_count=6,
        covariance_geometry="diagonal",
        posterior_geometry="clustered",
        representation_transform="tanh",
        subspace_rank=2,
        moment_sample_size=2_000,
        seed=2026,
    )


def test_factorial_scenarios_switch_only_intended_heterogeneity() -> None:
    generated = {
        scenario: generate_spectral_neural_replicate(_config(scenario))
        for scenario in (
            "homogeneous",
            "covariate_only",
            "posterior_only",
            "both_overlap_aligned",
        )
    }
    homogeneous = generated["homogeneous"].truth
    covariate_only = generated["covariate_only"].truth
    posterior_only = generated["posterior_only"].truth
    both = generated["both_overlap_aligned"].truth
    assert np.allclose(
        homogeneous.latent_covariances, homogeneous.latent_covariances[0]
    )
    assert np.allclose(homogeneous.coefficients, 0.0)
    assert not np.allclose(
        covariate_only.representation_moments, covariate_only.representation_moments[0]
    )
    assert np.allclose(covariate_only.coefficients, 0.0)
    assert np.allclose(
        posterior_only.latent_covariances, posterior_only.latent_covariances[0]
    )
    assert not np.allclose(posterior_only.coefficients, 0.0)
    assert not np.allclose(both.representation_moments, both.representation_moments[0])
    assert not np.allclose(both.coefficients, 0.0)


def test_truth_is_task_centered_and_reconstructs_each_conditional_mean() -> None:
    replicate = generate_spectral_neural_replicate(_config("both_overlap_aligned"))
    np.testing.assert_allclose(
        replicate.truth.coefficients.mean(axis=0), 0.0, atol=1e-12
    )
    for split in (replicate.train, replicate.validation, replicate.test):
        np.testing.assert_allclose(
            split.common + split.deviation, split.conditional_mean
        )
        for task in range(split.num_tasks):
            truth = replicate.truth
            expected = (
                truth.common_function(split.x[task])
                + truth.representation(split.x[task]) @ truth.coefficients[task]
            )
            np.testing.assert_allclose(
                expected, split.conditional_mean[task], atol=1e-12
            )
