"""
Uses evotorch to run CMA-ES on the test objective functions.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.logging import StdOutLogger

from objective_functions import ObjectiveFunction, counted
from vis_utils import plot_algorithm


if __name__ == "__main__":
    # Defining the function to optimize
    name = "easom"  # "shifted_sphere", "easom", "cross_in_tray", "egg_holder"

    # Hyperparameters for the search
    # Num. of generations
    n_generations = 100
    population_size = 10

    # Breaking as soon as the best fitness is this close to the actual optima
    # in absolute terms
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

    def wrapped_obj_function(inputs: torch.Tensor) -> torch.Tensor:
        """
        A wrapper that makes obj_function have only one argument, because that's what evotorch expects.
        """
        return obj_function_counted(inputs[:, 0], inputs[:, 1])

    # Defining the problem in evotorch
    problem = Problem(
        "max",
        objective_func=wrapped_obj_function,
        initial_bounds=limits,
        solution_length=2,
        vectorized=True,
    )

    cmaes_searcher = CMAES(problem, stdev_init=exploration, popsize=population_size)

    fig, ax = plt.subplots(1, 1)
    for _ in range(n_generations):
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

        # (uncounted) best fitness evaluation
        best_fitness = obj_function(*next_best)
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
