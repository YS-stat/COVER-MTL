"""Frozen neural-network and optimization configurations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkConfig:
    """Architecture shared by HPS, Average-Moment, and COVER-MTL."""

    common_hidden: tuple[int, ...] = (128, 128)
    representation_hidden: tuple[int, ...] = (128, 128)
    representation_dim: int = 8
    activation: str = "relu"
    head_initialization_scale: float = 0.02
    representation_identity: bool = False


@dataclass(frozen=True)
class OptimizationConfig:
    """Common optimization controls used by the formal experiments."""

    steps: int = 3000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size_per_task: int = 64
    evaluation_interval: int = 25
    patience_evaluations: int = 30
    gradient_clip: float = 5.0
    minimum_improvement: float = 1e-6
