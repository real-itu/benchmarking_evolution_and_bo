"""
Tests evotorch's implementation of SNES in the objective
functions provided.
"""
import torch
import matplotlib.pyplot as plt

from search_algorithms.evolutionary_strategies.snes import SNES
from objective_functions.objective_function import ObjectiveFunction

from utils.visualization.evolutionary_strategies import plot_algorithm


if __name__ == "__main__":
    # Defining the function to optimize
    name = "easom"  # "shifted_sphere", "easom", "cross_in_tray", "egg_holder"
    model = None  # None, "small", "medium", "large"
    n_dims = 2  # Whatever int your mind desires (in most cases)

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

    # Learning rates
    center_learning_rate = 0.1
    std_learning_rate = 0.1

    # Defining the objective function, limits, and so on...
    # They are all contained in the ObjectiveFunction class
    objective_function = ObjectiveFunction(name, n_dims=n_dims, model=model)
    limits = objective_function.limits
    solution_length = objective_function.solution_length

    snes = SNES(
        objective_function=objective_function,
        population_size=population_size,
        exploration=exploration,
        solution_length=solution_length,
        limits=limits,
    )

    fig, ax = plt.subplots(1, 1)
    for _ in range(n_generations):
        # Get the current best and population
        current_best = snes.get_current_best()
        population = snes.get_population()

        # Run a step
        snes.step()

        # Get the next best
        next_best = snes.get_current_best()

        # Visualize
        plot_algorithm(
            ax=ax,
            obj_function=objective_function,
            limits=limits,
            current_best=current_best,
            population=population,
            next_best=next_best,
        )

        plt.pause(0.01)
        ax.clear()

        # (uncounted) best fitness evaluation
        best_fitness = objective_function(next_best)
        print(f"Best fitness: {best_fitness}")

        if (
            torch.isclose(
                best_fitness, objective_function.optima, atol=tolerance_for_optima
            )
            and break_when_close_to_optima
        ):
            print(f"Found a good-enough optima, breaking.")
            break

    print(
        f"The obj. function was evaluated in {snes.objective_function.n_points} points ({snes.objective_function.calls} calls)"
    )
