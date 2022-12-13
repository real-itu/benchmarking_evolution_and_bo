"""
Implements a basic evolution strategy using torch,
and tests it on the objective functions.
"""
from typing import Callable

import torch
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np

from experiments.toy_examples.toy_objective_functions import ObjectiveFunction, counted
from utils.visualization.evolutionary_strategies import plot_algorithm


class SimpleEvolutionStrategy:
    def __init__(
        self,
        objective_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        population_size: int = 100,
        exploration: float = 1.0,
        initial_best: torch.Tensor = torch.Tensor([0.0, 0.0]),
    ) -> None:
        self.objective_function = objective_function
        self.population_size = population_size

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
        samples = self.population_distribution.sample((self.population_size,))
        fitnesses = self.objective_function(samples[:, 0], samples[:, 1])

        # next one's the best in fitness
        self.best = samples[torch.argmax(fitnesses)]

        # Re-defining the population distribution
        self.population_distribution = MultivariateNormal(
            loc=self.best, covariance_matrix=self.covar
        )

        return samples


if __name__ == "__main__":
    # Defining the function to optimize
    name = "easom"  # "shifted_sphere", "easom", "cross_in_tray", "egg_holder"

    # Hyperparameters for the search
    # Num. of generations and population size
    n_generations = 100
    population_size = 100

    # Breaking as soon as the best fitness is this close to the actual optima
    # in absolute terms.
    tolerance_for_optima = 1e-3

    # Do we actually want to break?
    break_when_close_to_optima = True

    # Initial covariance, needs to be modified depending on {name}
    exploration = 0.1

    # Defining the objective function, limits, and so on...
    # They are all contained in the ObjectiveFunction class
    objective = ObjectiveFunction(name)
    obj_function = objective.function
    limits = objective.limits

    # Counting the evaluations of the objective function
    @counted
    def obj_function_counted(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return obj_function(x, y)

    simple_evo = SimpleEvolutionStrategy(
        objective_function=obj_function_counted,
        exploration=exploration,
        population_size=population_size,
    )
    fig, ax = plt.subplots(1, 1)

    for _ in range(n_generations):
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

        # (uncounted) best fitness evaluation
        best_fitness = obj_function(*simple_evo.best)
        print(f"Best fitness: {best_fitness}")

        if (
            torch.isclose(best_fitness, objective.optima, atol=tolerance_for_optima)
            and break_when_close_to_optima
        ):
            print(f"Found a good-enough optima, breaking.")
            break

    print(
        f"The obj. function was evaluated in {obj_function_counted.n_points} points ({obj_function_counted.calls} calls)"
    )
