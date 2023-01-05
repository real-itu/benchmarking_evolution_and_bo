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
import numpy as np

from .artificial_landscapes.artificial_landscape import (
    ArtificialLandscape,
    POSSIBLE_FUNCTIONS,
)


class ObjectiveFunction(ArtificialLandscape):
    def __init__(
        self,
        name: str,
        n_dims: int = None,
        limits: Tuple[float, float] = None,
        maximize: bool = True,
        model: Callable[[torch.Tensor], torch.Tensor] = None,
        device: torch.device = "cpu",
    ) -> None:
        if name in POSSIBLE_FUNCTIONS:
            # We are defining an artificial landscape
            assert (
                maximize
            ), "We have only implemented the maximization of artificial landscapes :("

            super().__init__(name=name, n_dims=n_dims)
            self.known_optima: bool = True
            self.maximize: bool = True

        else:
            # We are defining an RL task
            if isinstance(model, torch.nn.Module):
                assert (
                    n_dims is None
                ), "You are defining an RL task. Why are you providing n_dims?"

                self.model = model.to(device)
                self.n_dims = sum(p.numel() for p in model.parameters())
            else:
                self.model = model
                self.n_dims = n_dims

            self.known_optima: bool = False
            self.maximize: bool = True
            self.limits = limits
