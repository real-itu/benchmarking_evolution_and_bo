"""
Uses evotorch to run CMA-ES on the test objective functions.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.logging import StdOutLogger

from objective_functions import easom, cross_in_tray, egg_holder, shifted_sphere
from vis_utils import plot_algorithm


if __name__ == "__main__":
    # Defining the bounds for the specific obj. functions
    obj_function = shifted_sphere
    limits = [-4.0, 4.0]
    exploration = 0.1

    # obj_function = easom
    # limits = [np.pi - 4, np.pi + 4]
    # exploration = 0.1

    # obj_function = cross_in_tray
    # limits = [-10, 10]
    # exploration = 0.1

    # obj_function = egg_holder
    # limits = [-512, 512]
    # exploration = 10.0

    def wrapped_obj_function(inputs: torch.Tensor) -> torch.Tensor:
        """
        A wrapper that makes obj_function have only one argument.
        """
        return obj_function(inputs[:, 0], inputs[:, 1])

    # Defining the problem in evotorch
    problem = Problem(
        "max",
        objective_func=wrapped_obj_function,
        initial_bounds=limits,
        solution_length=2,
        vectorized=True,
    )

    cmaes_searcher = CMAES(problem, stdev_init=exploration, popsize=100)
    _ = StdOutLogger(cmaes_searcher, interval=10)

    fig, ax = plt.subplots(1, 1)
    for _ in range(100):
        # Get the current best and population
        current_best = cmaes_searcher.get_status_value("pop_best").access_values()
        population = cmaes_searcher.population.access_values()

        # Run a step
        cmaes_searcher.step()

        # Get the next best
        next_best = cmaes_searcher.get_status_value("pop_best").access_values()

        # Visualize
        plot_algorithm(
            ax=ax,
            obj_function=obj_function,
            limits=limits,
            current_best=current_best,
            population=population,
            next_best=next_best,
        )

        plt.pause(0.01)
        ax.clear()
