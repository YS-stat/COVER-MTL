"""Algebraic tests for the covariate-overlap operator."""

from __future__ import annotations

import torch

from cover_mtl.geometry import (
    auxiliary_pair_cost,
    harmonic_overlap,
    profiled_pair_cost,
)


def test_overlap_is_symmetric_psd_for_noncommuting_moments() -> None:
    torch.manual_seed(11)
    left_root = torch.randn(5, 5, dtype=torch.float64)
    right_root = torch.randn(5, 5, dtype=torch.float64)
    left = left_root @ left_root.T + 0.2 * torch.eye(5, dtype=torch.float64)
    right = right_root @ right_root.T + 0.2 * torch.eye(5, dtype=torch.float64)
    overlap = harmonic_overlap(left, right)
    torch.testing.assert_close(overlap, overlap.T)
    assert float(torch.linalg.eigvalsh(overlap).min()) > -1e-10


def test_profiled_cost_matches_direct_minimization_for_noncommuting_moments() -> None:
    torch.manual_seed(12)
    left_root = torch.randn(4, 4, dtype=torch.float64)
    right_root = torch.randn(4, 4, dtype=torch.float64)
    left = left_root @ left_root.T + 0.3 * torch.eye(4, dtype=torch.float64)
    right = right_root @ right_root.T + 0.3 * torch.eye(4, dtype=torch.float64)
    beta_left = torch.randn(4, dtype=torch.float64)
    beta_right = torch.randn(4, dtype=torch.float64)
    consensus = torch.linalg.solve(left + right, left @ beta_left + right @ beta_right,)
    auxiliary = auxiliary_pair_cost(beta_left, beta_right, consensus, left, right,)
    profiled = profiled_pair_cost(beta_left, beta_right, left, right)
    torch.testing.assert_close(auxiliary, 0.5 * profiled)


def test_profiled_cost_matches_direct_minimization_in_singular_case() -> None:
    left = torch.diag(torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64))
    right = torch.diag(torch.tensor([1.0, 0.0, 3.0], dtype=torch.float64))
    beta_left = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    beta_right = torch.tensor([-1.0, 4.0, 2.0], dtype=torch.float64)
    total = left + right
    consensus = torch.linalg.pinv(total) @ (left @ beta_left + right @ beta_right)
    left_difference = beta_left - consensus
    right_difference = beta_right - consensus
    auxiliary = (
        left_difference @ left @ left_difference
        + right_difference @ right @ right_difference
    )
    profiled = profiled_pair_cost(beta_left, beta_right, left, right)
    torch.testing.assert_close(auxiliary, 0.5 * profiled)
