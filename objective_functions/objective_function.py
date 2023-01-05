"""
Defines a

self.problem = Problem(
    "max",
    objective_func=self.objective_function,
    initial_bounds=limits,
    solution_length=solution_length,
    vectorized=True,
)
"""
from typing import Callable, Tuple, Union

import torch


class ObjectiveFunction:
    def __init__(
        self,
        name: str,
        n_dims: int,
        maximize: bool = True,
        limits: Tuple[float, float] = None,
        model: Callable[[torch.Tensor], torch.Tensor] = None,
        device: torch.device = "cpu",
    ) -> None:
        self.name = name
        self.function = None
        self.ndims = n_dims
        self.maximize = maximize
        self.limits = limits

        if isinstance(model, torch.nn.Module):
            self.model = model.to(device)
        else:
            self.model = model

        self.device = device

    def evaluate_objective(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
