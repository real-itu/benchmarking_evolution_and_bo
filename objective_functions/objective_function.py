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

from .gym.RL_objective import ObjectiveRLGym

class ObjectiveFunction(ObjectiveRLGym, ArtificialLandscape): # FIXME 🙈

    def __init__(
        self,
        name: str,
        seed: int,
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
            ArtificialLandscape.__init__(self, name=name, n_dims=n_dims)
            self.known_optima: bool = True
            self.maximize: bool = True
            
        else:
            # We are defining an RL task
            if model is not None:
                assert (n_dims is None), "You are defining an RL task. Why are you providing n_dims?"
            
            ObjectiveRLGym.__init__(self, environment = name, model = model, seed = seed, maximize = maximize, limits = limits)

    def evaluate_objective(self, x: torch.Tensor) -> torch.Tensor: # HACK 🙈
        if vars(self)['known_optima']:
            return ArtificialLandscape.evaluate_objective(self, x)
        else:
            return ObjectiveRLGym.evaluate_objective(self, x)