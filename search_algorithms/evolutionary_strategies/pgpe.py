"""
Implements Policy Gradient Parameter Exploring (PGPE)
in our common interface. For more details check

[1] Evotorch's docs: https://docs.evotorch.ai/v0.3.0/reference/evotorch/algorithms/distributed/gaussian/#evotorch.algorithms.distributed.gaussian.PGPE
"""
from typing import Callable, Tuple

import torch
from torch.distributions import Normal

from evotorch import Problem
from evotorch.algorithms import PGPE as PGPE_from_evotorch

from .evolutionary_strategy import EvolutionaryStrategy

from objective_functions.objective_function import ObjectiveFunction


class PGPE(EvolutionaryStrategy):
    """
    TODO: add docs.
    """

    def __init__(
        self,
        objective_function: ObjectiveFunction,
        population_size: int,
        exploration: float,
        solution_length: int,
        limits: Tuple[float, float] = None,
        center_learning_rate: float = 0.1,
        stdev_learning_rate: float = 0.1,
    ):
        """
        TODO: add docs.
        """
        super().__init__(objective_function, population_size)
        self.exploration = exploration
        self.solution_length = solution_length
        self.limits = limits
        self.center_learning_rate = center_learning_rate
        self.stdev_learning_rate = stdev_learning_rate

        self.problem = Problem(
            "max",
            objective_func=self.objective_function,
            initial_bounds=limits,
            solution_length=solution_length,
            vectorized=True,
            dtype=torch.get_default_dtype(),
        )

        self._pgpe_searcher = PGPE_from_evotorch(
            self.problem,
            popsize=self.population_size,
            stdev_init=exploration,
            center_learning_rate=center_learning_rate,
            stdev_learning_rate=stdev_learning_rate,
        )

    def get_current_best(self):
        return self._pgpe_searcher._get_mu()

    def get_population(self):
        mu = self._pgpe_searcher._get_mu()
        stdev = self._pgpe_searcher._get_sigma()
        return Normal(mu, stdev).sample((self.population_size,))

    def step(self):
        self._pgpe_searcher.step()
