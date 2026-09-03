"""Neural predictors and integration objectives used in the experiments."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import torch
from torch import Tensor, nn


def _activation(name: str) -> type[nn.Module]:
    choices: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if name.lower() not in choices:
        raise ValueError(f"Unknown activation {name!r}.")
    return choices[name.lower()]


class MLP(nn.Module):
    """A small multilayer perceptron with a linear output layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Input and output dimensions must be positive.")
        activation_class = _activation(activation)
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            if width <= 0:
                raise ValueError("Every hidden width must be positive.")
            layers.extend([nn.Linear(previous, width), activation_class()])
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


@dataclass(frozen=True)
class ObjectiveParts:
    total: Tensor
    prediction: Tensor
    integration: Tensor


class MultiTaskPredictor(nn.Module):
    """Interface shared by all neural multi-task comparison methods."""

    num_tasks: int

    def predict_all_tasks(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def predict_task(self, x: Tensor, task_index: int) -> Tensor:
        """Predict one task without requiring a model-specific fast path."""
        if not 0 <= task_index < self.num_tasks:
            raise IndexError("task_index is outside the fitted task range.")
        return self.predict_all_tasks(x)[:, task_index]

    def selected_prediction(self, x: Tensor, task: Tensor) -> Tensor:
        predictions = self.predict_all_tasks(x)
        return predictions.gather(1, task[:, None]).squeeze(1)

    def objective(
        self, x: Tensor, response: Tensor, task: Tensor, coupling: float,
    ) -> ObjectiveParts:
        prediction = self.selected_prediction(x, task)
        prediction_loss = 0.5 * (response - prediction).square().mean()
        zero = prediction_loss.new_zeros(())
        return ObjectiveParts(prediction_loss, prediction_loss, zero)

    def post_step(self) -> None:
        return None

    def parameter_groups(self, weight_decay: float) -> list[dict[str, object]]:
        return [{"params": self.parameters(), "weight_decay": weight_decay}]


class PooledPredictor(MultiTaskPredictor):
    """One regression network shared without task-specific parameters."""

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.network = MLP(input_dim, hidden_dims, 1, activation)

    def predict_all_tasks(self, x: Tensor) -> Tensor:
        prediction = self.network(x)
        return prediction.repeat(1, self.num_tasks)

    def predict_task(self, x: Tensor, task_index: int) -> Tensor:
        if not 0 <= task_index < self.num_tasks:
            raise IndexError("task_index is outside the fitted task range.")
        return self.network(x).squeeze(1)


class SingleTaskPredictor(MultiTaskPredictor):
    """Independent neural regression networks, one for each task."""

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.networks = nn.ModuleList(
            [MLP(input_dim, hidden_dims, 1, activation) for _ in range(num_tasks)]
        )

    def predict_all_tasks(self, x: Tensor) -> Tensor:
        return torch.cat([network(x) for network in self.networks], dim=1)

    def predict_task(self, x: Tensor, task_index: int) -> Tensor:
        if not 0 <= task_index < self.num_tasks:
            raise IndexError("task_index is outside the fitted task range.")
        return self.networks[task_index](x).squeeze(1)

    def objective(
        self, x: Tensor, response: Tensor, task: Tensor, coupling: float,
    ) -> ObjectiveParts:
        del coupling
        losses = []
        for task_index, network in enumerate(self.networks):
            mask = task == task_index
            if not torch.any(mask):
                raise ValueError("Every optimization batch must contain every task.")
            prediction = network(x[mask]).squeeze(1)
            losses.append(0.5 * (response[mask] - prediction).square().mean())
        prediction_loss = torch.stack(losses).mean()
        zero = prediction_loss.new_zeros(())
        return ObjectiveParts(prediction_loss, prediction_loss, zero)


class SharedDecompositionPredictor(MultiTaskPredictor):
    """Shared common function and representation with centered task heads."""

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        representation_dim: int,
        common_hidden: Sequence[int],
        representation_hidden: Sequence[int],
        activation: str = "relu",
        integration: str = "none",
        head_initialization_scale: float = 0.02,
        representation_identity: bool = False,
    ) -> None:
        super().__init__()
        if integration not in {"none", "average_moment", "cover"}:
            raise ValueError("integration must be none, average_moment, or cover.")
        self.num_tasks = num_tasks
        self.representation_dim = representation_dim
        self.integration = integration
        self.common_network = MLP(input_dim, common_hidden, 1, activation)
        if representation_identity:
            if input_dim != representation_dim:
                raise ValueError(
                    "An identity representation requires input_dim = representation_dim."
                )
            self.representation_network = nn.Identity()
        else:
            self.representation_network = MLP(
                input_dim, representation_hidden, representation_dim, activation
            )
        self.beta = nn.Parameter(torch.empty(num_tasks, representation_dim))
        if head_initialization_scale <= 0:
            raise ValueError("head_initialization_scale must be positive.")
        nn.init.normal_(self.beta, mean=0.0, std=head_initialization_scale)
        edge_array = torch.tensor(
            list(combinations(range(num_tasks), 2)), dtype=torch.long
        )
        self.register_buffer("edge_left", edge_array[:, 0])
        self.register_buffer("edge_right", edge_array[:, 1])
        if integration == "cover":
            self.consensus = nn.Parameter(
                torch.zeros(edge_array.shape[0], representation_dim)
            )
        else:
            self.register_parameter("consensus", None)
        self.center_heads_()

    def common_and_representation(self, x: Tensor) -> tuple[Tensor, Tensor]:
        common = self.common_network(x).squeeze(1)
        representation = self.representation_network(x)
        return common, representation

    def predict_all_tasks(self, x: Tensor) -> Tensor:
        common, representation = self.common_and_representation(x)
        return common[:, None] + representation @ self.beta.T

    def predict_task(self, x: Tensor, task_index: int) -> Tensor:
        if not 0 <= task_index < self.num_tasks:
            raise IndexError("task_index is outside the fitted task range.")
        common, representation = self.common_and_representation(x)
        return common + representation @ self.beta[task_index]

    @torch.no_grad()
    def center_heads_(self) -> None:
        self.beta.sub_(self.beta.mean(dim=0, keepdim=True))

    def post_step(self) -> None:
        self.center_heads_()

    def parameter_groups(self, weight_decay: float) -> list[dict[str, object]]:
        integration_parameters: list[nn.Parameter] = [self.beta]
        if self.consensus is not None:
            integration_parameters.append(self.consensus)
        return [
            {
                "params": list(self.common_network.parameters())
                + list(self.representation_network.parameters()),
                "weight_decay": weight_decay,
            },
            {"params": integration_parameters, "weight_decay": 0.0},
        ]

    def _task_second_moments(self, representation: Tensor, task: Tensor) -> Tensor:
        moments = []
        for task_index in range(self.num_tasks):
            task_features = representation[task == task_index]
            if task_features.shape[0] == 0:
                raise ValueError("Every optimization batch must contain every task.")
            moments.append(task_features.T @ task_features / task_features.shape[0])
        return torch.stack(moments)

    def _integration_cost(self, representation: Tensor, task: Tensor) -> Tensor:
        moments = self._task_second_moments(representation, task)
        if self.integration == "cover":
            if self.consensus is None:
                raise RuntimeError("COVER integration requires consensus parameters.")
            left_difference = self.beta[self.edge_left] - self.consensus
            right_difference = self.beta[self.edge_right] - self.consensus
            left_cost = torch.einsum(
                "ei,eij,ej->e",
                left_difference,
                moments[self.edge_left],
                left_difference,
            )
            right_cost = torch.einsum(
                "ei,eij,ej->e",
                right_difference,
                moments[self.edge_right],
                right_difference,
            )
            return left_cost.sum() + right_cost.sum()
        if self.integration == "average_moment":
            difference = self.beta[self.edge_left] - self.beta[self.edge_right]
            average_moment = 0.5 * (moments[self.edge_left] + moments[self.edge_right])
            return torch.einsum("ei,eij,ej->", difference, average_moment, difference)
        return representation.new_zeros(())

    def objective(
        self, x: Tensor, response: Tensor, task: Tensor, coupling: float,
    ) -> ObjectiveParts:
        if coupling < 0:
            raise ValueError("coupling must be nonnegative.")
        common, representation = self.common_and_representation(x)
        prediction = common + (representation * self.beta[task]).sum(dim=1)
        prediction_loss = 0.5 * (response - prediction).square().mean()
        raw_cost = self._integration_cost(representation, task)
        if self.integration == "cover":
            scale = 2.0 * coupling / (self.num_tasks * (self.num_tasks - 1))
        elif self.integration == "average_moment":
            scale = coupling / (self.num_tasks * (self.num_tasks - 1))
        else:
            scale = 0.0
        integration_loss = scale * raw_cost
        return ObjectiveParts(
            total=prediction_loss + integration_loss,
            prediction=prediction_loss,
            integration=integration_loss,
        )

    @torch.no_grad()
    def initialize_consensus_from_heads(self) -> None:
        if self.consensus is not None:
            self.consensus.copy_(
                0.5 * (self.beta[self.edge_left] + self.beta[self.edge_right])
            )

    @torch.no_grad()
    def initialize_consensus(self, values: Tensor) -> None:
        """Set profiled pairwise consensus values before coupled fine-tuning."""
        if self.consensus is None:
            raise RuntimeError("Only COVER models contain consensus parameters.")
        if values.shape != self.consensus.shape:
            raise ValueError("Consensus initialization has an incompatible shape.")
        self.consensus.copy_(
            values.to(device=self.consensus.device, dtype=self.consensus.dtype)
        )

    @torch.no_grad()
    def initialize_heads(self, values: Tensor) -> None:
        """Set centered task heads before coupled fine-tuning."""
        if values.shape != self.beta.shape:
            raise ValueError("Head initialization has an incompatible shape.")
        self.beta.copy_(values.to(device=self.beta.device, dtype=self.beta.dtype))
        self.center_heads_()


class FixedHeadCommonPredictor(MultiTaskPredictor):
    """Refit a shared nonlinear component around fixed task coefficients."""

    def __init__(
        self,
        common_network: nn.Module,
        representation_network: nn.Module,
        coefficients: Tensor,
    ) -> None:
        super().__init__()
        if coefficients.ndim != 2:
            raise ValueError("Fixed task coefficients must be a matrix.")
        self.num_tasks = coefficients.shape[0]
        self.common_network = copy.deepcopy(common_network)
        self.representation_network = copy.deepcopy(representation_network)
        self.representation_network.requires_grad_(False)
        self.register_buffer("coefficients", coefficients.detach().clone())

    def predict_all_tasks(self, x: Tensor) -> Tensor:
        common = self.common_network(x)
        representation = self.representation_network(x)
        return common + representation @ self.coefficients.T

    def predict_task(self, x: Tensor, task_index: int) -> Tensor:
        if not 0 <= task_index < self.num_tasks:
            raise IndexError("task_index is outside the fitted task range.")
        common = self.common_network(x).squeeze(1)
        representation = self.representation_network(x)
        return common + representation @ self.coefficients[task_index]

    def parameter_groups(self, weight_decay: float) -> list[dict[str, object]]:
        return [
            {"params": self.common_network.parameters(), "weight_decay": weight_decay,}
        ]


class MMoEPredictor(MultiTaskPredictor):
    """Multi-gate mixture-of-experts baseline with task-specific towers."""

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        hidden_dims: Sequence[int],
        expert_dim: int = 32,
        num_experts: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if num_experts < 2 or expert_dim <= 0:
            raise ValueError("MMoE requires at least two positive-dimensional experts.")
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                MLP(input_dim, hidden_dims, expert_dim, activation)
                for _ in range(num_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(input_dim, num_experts) for _ in range(num_tasks)]
        )
        self.towers = nn.ModuleList(
            [nn.Linear(expert_dim, 1) for _ in range(num_tasks)]
        )

    def predict_all_tasks(self, x: Tensor) -> Tensor:
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        predictions = []
        for task_index in range(self.num_tasks):
            weights = torch.softmax(self.gates[task_index](x), dim=1)
            mixture = torch.einsum("ne,ned->nd", weights, expert_outputs)
            predictions.append(self.towers[task_index](mixture))
        return torch.cat(predictions, dim=1)

    def predict_task(self, x: Tensor, task_index: int) -> Tensor:
        if not 0 <= task_index < self.num_tasks:
            raise IndexError("task_index is outside the fitted task range.")
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        weights = torch.softmax(self.gates[task_index](x), dim=1)
        mixture = torch.einsum("ne,ned->nd", weights, expert_outputs)
        return self.towers[task_index](mixture).squeeze(1)

    def selected_prediction(self, x: Tensor, task: Tensor) -> Tensor:
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        prediction = x.new_zeros(x.shape[0])
        for task_index in range(self.num_tasks):
            indices = torch.nonzero(task == task_index, as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                raise ValueError("Every optimization batch must contain every task.")
            task_x = x.index_select(0, indices)
            task_experts = expert_outputs.index_select(0, indices)
            weights = torch.softmax(self.gates[task_index](task_x), dim=1)
            mixture = torch.einsum("ne,ned->nd", weights, task_experts)
            values = self.towers[task_index](mixture).squeeze(1)
            prediction = prediction.index_copy(0, indices, values)
        return prediction
