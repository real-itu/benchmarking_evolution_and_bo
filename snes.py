"""
Tests evotorch's implementation of SNES in the objective
functions provided.
"""
import torch
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger

from objective_functions import easom, cross_in_tray, egg_holder
from vis_utils import plot_algorithm


if __name__ == "__main__":
    # Defining the bounds for the specific obj. functions
    # obj_function = easom
    # limits = [np.pi - 4, np.pi + 4]
    # exploration = 1.0

    obj_function = cross_in_tray
    limits = [-10, 10]
    exploration = 1.0

    # obj_function = egg_holder
    # limits = [-512, 512]
    # exploration = 100

    def wrapped_obj_function(inputs: torch.Tensor) -> torch.Tensor:
        """
        A wrapper that makes obj_function only have one argument.
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

    snes_searcher = SNES(
        problem,
        popsize=100,
        stdev_init=exploration,
        center_learning_rate=0.1,
        stdev_learning_rate=0.1,
    )
    _ = StdOutLogger(snes_searcher, interval=10)

    fig, ax = plt.subplots(1, 1)
    for _ in range(100):
        # Get the current best and population
        current_best = snes_searcher._get_mu()
        current_std = snes_searcher._get_sigma()
        population = Normal(loc=current_best, scale=current_std).sample((100,))

        # Run a step
        snes_searcher.step()

        # Get the next best
        next_best = snes_searcher._get_mu()

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
