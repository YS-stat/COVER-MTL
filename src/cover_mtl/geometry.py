"""Representation-space geometry for COVER-MTL.

Source training uses the exact auxiliary objective and therefore does not call
the matrix routines in this module. These routines support diagnostics,
profiled-objective checks, and new-task adaptation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import torch
from torch import Tensor


def _check_square(matrix: Tensor, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square 2D tensor; got {matrix.shape}.")


def symmetrize(matrix: Tensor) -> Tensor:
    """Return the symmetric part of a square matrix."""
    _check_square(matrix, "matrix")
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def psd_pinv(matrix: Tensor, rcond: float = 1e-10) -> Tensor:
    """Return the Moore--Penrose inverse of a symmetric PSD matrix."""
    _check_square(matrix, "matrix")
    if rcond <= 0:
        raise ValueError("rcond must be positive.")
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetrize(matrix))
    scale = eigenvalues.abs().max().clamp_min(torch.finfo(matrix.dtype).tiny)
    threshold = rcond * scale
    inverse = torch.zeros_like(eigenvalues)
    retained = eigenvalues > threshold
    inverse[retained] = eigenvalues[retained].reciprocal()
    return (eigenvectors * inverse.unsqueeze(0)) @ eigenvectors.transpose(-1, -2)


def project_psd(matrix: Tensor, rcond: float = 1e-12) -> Tensor:
    """Remove floating-point asymmetry and negligible negative eigenvalues."""
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetrize(matrix))
    scale = eigenvalues.abs().max().clamp_min(torch.finfo(matrix.dtype).tiny)
    tolerance = rcond * scale
    if torch.any(eigenvalues < -tolerance):
        minimum = float(eigenvalues.min().detach().cpu())
        raise ValueError(f"Matrix is not PSD; minimum eigenvalue is {minimum:.3e}.")
    clipped = eigenvalues.clamp_min(0.0)
    return (eigenvectors * clipped.unsqueeze(0)) @ eigenvectors.transpose(-1, -2)


def second_moment(features: Tensor) -> Tensor:
    """Compute n^{-1} sum_i z_i z_i^T without centering."""
    if features.ndim != 2:
        raise ValueError("features must have shape (n, d).")
    if features.shape[0] == 0:
        raise ValueError("features must contain at least one observation.")
    return features.transpose(0, 1) @ features / features.shape[0]


def harmonic_overlap(
    sigma_left: Tensor, sigma_right: Tensor, rcond: float = 1e-10,
) -> Tensor:
    """Compute the generalized harmonic overlap 2 A(A+B)^dagger B."""
    _check_square(sigma_left, "sigma_left")
    _check_square(sigma_right, "sigma_right")
    if sigma_left.shape != sigma_right.shape:
        raise ValueError("sigma_left and sigma_right must have the same shape.")
    total_pinv = psd_pinv(sigma_left + sigma_right, rcond=rcond)
    overlap = 2.0 * sigma_left @ total_pinv @ sigma_right
    return project_psd(overlap)


def pairwise_edges(num_tasks: int) -> list[tuple[int, int]]:
    """Return all undirected task pairs in lexicographic order."""
    if num_tasks < 2:
        raise ValueError("COVER-MTL requires at least two source tasks.")
    return list(combinations(range(num_tasks), 2))


def quadratic_norm(vector: Tensor, matrix: Tensor) -> Tensor:
    """Compute vector^T matrix vector."""
    return vector @ matrix @ vector


def profiled_pair_cost(
    beta_left: Tensor,
    beta_right: Tensor,
    sigma_left: Tensor,
    sigma_right: Tensor,
    rcond: float = 1e-10,
) -> Tensor:
    """Compute the profiled pair cost ||beta_left-beta_right||^2_Omega."""
    difference = beta_left - beta_right
    overlap = harmonic_overlap(sigma_left, sigma_right, rcond=rcond)
    return quadratic_norm(difference, overlap)


def auxiliary_pair_cost(
    beta_left: Tensor,
    beta_right: Tensor,
    consensus: Tensor,
    sigma_left: Tensor,
    sigma_right: Tensor,
) -> Tensor:
    """Compute the unprofiled pairwise reconciliation cost."""
    left = beta_left - consensus
    right = beta_right - consensus
    return quadratic_norm(left, sigma_left) + quadratic_norm(right, sigma_right)


def target_overlap_summary(
    sigma_target: Tensor,
    source_sigmas: Iterable[Tensor],
    source_heads: Tensor,
    rcond: float = 1e-10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return target-source overlaps, aggregate precision, and barycenter."""
    sigmas = list(source_sigmas)
    if source_heads.ndim != 2:
        raise ValueError("source_heads must have shape (T, d).")
    if len(sigmas) != source_heads.shape[0]:
        raise ValueError("The numbers of source sigmas and heads must agree.")
    overlaps = torch.stack(
        [harmonic_overlap(sigma_target, sigma, rcond=rcond) for sigma in sigmas]
    )
    precision = overlaps.sum(dim=0)
    weighted_sum = torch.einsum("tij,tj->i", overlaps, source_heads)
    barycenter = psd_pinv(precision, rcond=rcond) @ weighted_sum
    return overlaps, precision, barycenter
