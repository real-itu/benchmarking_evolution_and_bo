"""
Uses evotorch to run PGPE on the test objective functions.
"""
import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt

from evotorch import Problem
from evotorch.algorithms import PGPE

from objective_functions import ObjectiveFunction, counted
from vis_utils import plot_algorithm


if __name__ == "__main__":
    # Defining the function to optimize
    name = "easom"  # "shifted_sphere", "easom", "cross_in_tray", "egg_holder"

    # Hyperparameters for the search
    # Num. of generations
    n_generations = 100
    population_size = 100

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

    # Wrapping the function for evotorch.
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

    pgpe_searcher = PGPE(
        problem,
        popsize=population_size,
        stdev_init=exploration,
        center_learning_rate=0.1,
        stdev_learning_rate=0.1,
    )

    fig, ax = plt.subplots(1, 1)
    for _ in range(n_generations):
        # Get the current best and population
        current_best = pgpe_searcher._get_mu()
        current_std = pgpe_searcher._get_sigma()
        population = Normal(loc=current_best, scale=current_std).sample(
            (population_size,)
        )

        # Run a step
        pgpe_searcher.step()

        # Get the next best
        next_best = pgpe_searcher._get_mu()

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
