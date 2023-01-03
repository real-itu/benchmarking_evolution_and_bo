"""
Uses evotorch to run CMA-ES on the test objective functions.
"""
import torch
import matplotlib.pyplot as plt

from search_algorithms.evolutionary_strategies.cma_es import CMA_ES

from objective_functions.artificial_landscapes.artificial_landscape import (
    ArtificialLandscape,
)
from utils.visualization.evolutionary_strategies import plot_algorithm

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
    exploration = 0.2

    # Defining the objective function, limits, and so on...
    # They are all contained in the ObjectiveFunction class
    objective = ArtificialLandscape(name)
    obj_function = objective.function
    limits = objective.limits
    solution_length = objective.solution_length

    cma_es = CMA_ES(
        objective_function=obj_function,
        population_size=population_size,
        exploration=exploration,
        solution_length=solution_length,
        limits=limits,
    )

    fig, ax = plt.subplots(1, 1)
    for _ in range(n_generations):
        # Get the current best and population
        current_best = cma_es.get_current_best()
        population = cma_es.get_population()

        # Run a step
        cma_es.step()

        # Get the next best
        next_best = cma_es.get_current_best()

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
        best_fitness = obj_function(next_best)
        print(f"Best fitness: {best_fitness}")

        if (
            torch.isclose(best_fitness, objective.optima, atol=tolerance_for_optima)
            and break_when_close_to_optima
        ):
            print(f"Found a good-enough optima, breaking.")
            break

    print(
        f"The obj. function was evaluated in {cma_es.objective_function.n_points} points ({cma_es.objective_function.calls} calls)"
    )
