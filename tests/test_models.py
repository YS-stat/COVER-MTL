"""Tests for neural model constraints, losses, and generic metrics."""

from __future__ import annotations

import numpy as np
import torch

from cover_mtl.models import (
    FixedHeadCommonPredictor,
    MMoEPredictor,
    SharedDecompositionPredictor,
)
from cover_mtl.simulations.dgp import (
    SpectralNeuralConfig,
    generate_spectral_neural_replicate,
)
from cover_mtl.simulations.metrics import evaluate_functions
from cover_mtl.simulations.runner import _parameter_count


def _shared_model(integration: str) -> SharedDecompositionPredictor:
    return SharedDecompositionPredictor(
        input_dim=6,
        num_tasks=4,
        representation_dim=3,
        common_hidden=(8,),
        representation_hidden=(8,),
        integration=integration,
    )


def test_centered_heads_make_common_output_the_task_average():
    torch.manual_seed(1)
    model = _shared_model("none")
    x = torch.randn(17, 6)
    common, _ = model.common_and_representation(x)
    predictions = model.predict_all_tasks(x)
    torch.testing.assert_close(model.beta.mean(dim=0), torch.zeros(3))
    torch.testing.assert_close(predictions.mean(dim=1), common)


def test_cover_auxiliary_cost_profiles_to_overlap_penalty():
    torch.manual_seed(2)
    model = _shared_model("cover")
    x = torch.randn(40, 6)
    task = torch.arange(4).repeat_interleave(10)
    _, representation = model.common_and_representation(x)
    moments = model._task_second_moments(representation, task)
    profiled = 0.0
    with torch.no_grad():
        for edge_index, (left, right) in enumerate(
            zip(model.edge_left.tolist(), model.edge_right.tolist())
        ):
            total = moments[left] + moments[right]
            consensus = torch.linalg.pinv(total) @ (
                moments[left] @ model.beta[left] + moments[right] @ model.beta[right]
            )
            model.consensus[edge_index] = consensus
            difference = model.beta[left] - model.beta[right]
            omega = 2 * moments[left] @ torch.linalg.pinv(total) @ moments[right]
            profiled = profiled + difference @ omega @ difference
    auxiliary = model._integration_cost(representation, task)
    torch.testing.assert_close(2.0 * auxiliary, profiled, rtol=2e-4, atol=2e-6)


def test_average_moment_uses_the_pairwise_arithmetic_geometry():
    torch.manual_seed(3)
    model = _shared_model("average_moment")
    x = torch.randn(40, 6)
    task = torch.arange(4).repeat_interleave(10)
    _, representation = model.common_and_representation(x)
    moments = model._task_second_moments(representation, task)
    expected = 0.0
    for left, right in zip(model.edge_left.tolist(), model.edge_right.tolist()):
        difference = model.beta[left] - model.beta[right]
        pair_average = 0.5 * (moments[left] + moments[right])
        expected = expected + difference @ pair_average @ difference
    actual = model._integration_cost(representation, task)
    torch.testing.assert_close(actual, expected)


def test_cover_objective_uses_the_paper_scaling():
    torch.manual_seed(31)
    model = _shared_model("cover")
    x = torch.randn(40, 6)
    task = torch.arange(4).repeat_interleave(10)
    response = torch.randn(40)
    coupling = 0.7
    _, representation = model.common_and_representation(x)
    raw_cost = model._integration_cost(representation, task)
    objective = model.objective(x, response, task, coupling)
    expected = 2.0 * coupling * raw_cost / (4 * 3)
    torch.testing.assert_close(objective.integration, expected)


def test_fixed_head_parameter_count_includes_fitted_coefficients():
    base = _shared_model("none")
    coefficients = torch.randn(4, 3)
    model = FixedHeadCommonPredictor(
        base.common_network, base.representation_network, coefficients,
    )
    neural_count = sum(parameter.numel() for parameter in model.parameters())
    assert _parameter_count(model) == neural_count + coefficients.numel()


def test_cover_parameter_count_excludes_consensus_auxiliaries():
    model = _shared_model("cover")
    expected = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name != "consensus"
    )
    assert _parameter_count(model) == expected


def test_mmoe_task_specific_paths_match_full_predictions_and_keep_gradients():
    torch.manual_seed(4)
    model = MMoEPredictor(
        input_dim=6, num_tasks=4, hidden_dims=(8,), expert_dim=5, num_experts=3,
    )
    x = torch.randn(20, 6)
    task = torch.arange(4).repeat_interleave(5)
    all_predictions = model.predict_all_tasks(x)
    selected = model.selected_prediction(x, task)
    expected = all_predictions.gather(1, task[:, None]).squeeze(1)
    torch.testing.assert_close(selected, expected)
    for task_index in range(4):
        torch.testing.assert_close(
            model.predict_task(x, task_index), all_predictions[:, task_index],
        )
    selected.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.experts.parameters())
    assert all(parameter.grad is not None for parameter in model.gates.parameters())
    assert all(parameter.grad is not None for parameter in model.towers.parameters())


def test_oracle_predictor_has_zero_function_error():
    replicate = generate_spectral_neural_replicate(
        SpectralNeuralConfig(
            scenario="both_random",
            num_tasks=4,
            num_profiles=2,
            input_dim=9,
            representation_dim=4,
            active_per_profile=2,
            train_size=20,
            validation_size=20,
            test_size=30,
            low_mode_count=4,
            subspace_rank=2,
            seed=8,
        )
    )

    def oracle(x: np.ndarray) -> np.ndarray:
        common = replicate.truth.common_function(x)
        representation = replicate.truth.representation(x)
        return common[:, None] + representation @ replicate.truth.coefficients.T

    summary, _ = evaluate_functions(oracle, replicate.test)
    assert summary["excess_mse"] < 1e-24
    assert summary["common_mse"] < 1e-24
    assert summary["deviation_mse"] < 1e-24
