"""
Implements a basic evolution strategy using torch,
and tests it on the objective functions.
"""
from typing import Callable

import torch
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np

from objective_functions import cross_in_tray, easom, egg_holder, shifted_sphere
from vis_utils import plot_algorithm


class SimpleEvolutionStrategy:
    def __init__(
        self,
        objective_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_population: int = 100,
        exploration: float = 1.0,
        initial_best: torch.Tensor = torch.Tensor([0.0, 0.0]),
    ) -> None:
        self.objective_function = objective_function
        self.n_population = n_population

        # Setting up the initial mean and fixed covariance
        self.best = initial_best
        self.covar = exploration * torch.eye(2)

        # Setting up the population's distribution
        self.population_distribution = MultivariateNormal(
            loc=self.best, covariance_matrix=self.covar
        )

    def step(self) -> torch.Tensor:
        """
        Samples from the current population distribution,
        and saves the best candidate in the mean. It also returns
        the samples.
        """
        # Sample and evaluate
        samples = self.population_distribution.sample((self.n_population,))
        fitnesses = self.objective_function(samples[:, 0], samples[:, 1])

        # next one's the best in fitness
        self.best = samples[torch.argmax(fitnesses)]

        # Re-defining the population distribution
        self.population_distribution = MultivariateNormal(
            loc=self.best, covariance_matrix=self.covar
        )

        return samples


if __name__ == "__main__":
    obj_function = shifted_sphere
    limits = [-4.0, 4.0]
    exploration = 0.1

    # obj_function = easom
    # limits = [np.pi - 4, np.pi + 4]
    # exploration = 0.1

    # obj_function = cross_in_tray
    # limits = [-10, 10]
    # exploration = 10.0

    # obj_function = egg_holder
    # limits = [-512, 512]
    # exploration = 100

    simple_evo = SimpleEvolutionStrategy(
        objective_function=obj_function,
        exploration=exploration,
        initial_best=torch.Tensor([-7.0, -7.0]),
    )
    fig, ax = plt.subplots(1, 1)

    for _ in range(100):
        # Save the current mean for plotting
        current_mean = simple_evo.best

        # Run a step
        samples = simple_evo.step()

        # Visualize
        plot_algorithm(
            ax=ax,
            obj_function=obj_function,
            limits=limits,
            current_best=current_mean,
            population=samples,
            next_best=simple_evo.best,
        )

        plt.pause(0.01)
        ax.clear()
        # plt.close(fig)

        print(f"Best fitness: {simple_evo.objective_function(*simple_evo.best)}")
