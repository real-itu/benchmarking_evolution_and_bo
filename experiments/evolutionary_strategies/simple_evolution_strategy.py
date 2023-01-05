"""
Implements a basic evolution strategy using torch,
and tests it on the objective functions.
"""
import torch
import matplotlib.pyplot as plt

from search_algorithms.evolutionary_strategies.simple_evolution_strategy import (
    SimpleEvolutionStrategy,
)
from objective_functions.objective_function import ObjectiveFunction
from utils.visualization.evolutionary_strategies import plot_algorithm


if __name__ == "__main__":
    # Defining the function to optimize
    name = "easom"  # "shifted_sphere", "easom", "cross_in_tray", "egg_holder"
    model = None  # None, "small", "medium", "large"
    n_dims = 2  # Whatever int your mind desires (in most cases)

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
    objective_function = ObjectiveFunction(name=name, n_dims=n_dims, model=model)
    limits = objective_function.limits

    simple_evo = SimpleEvolutionStrategy(
        objective_function=objective_function,
        population_size=population_size,
        exploration=exploration,
    )

    fig, ax = plt.subplots(1, 1)
    for _ in range(n_generations):
        # Save the current mean for plotting
        current_mean = simple_evo.get_current_best()

        # Run a step
        samples = simple_evo.step()

        # Visualize
        plot_algorithm(
            ax=ax,
            obj_function=objective_function,
            limits=limits,
            current_best=current_mean,
            population=samples,
            next_best=simple_evo.best,
        )

        plt.pause(0.01)
        ax.clear()

        # (uncounted) best fitness evaluation
        best_fitness = objective_function(simple_evo.get_current_best())
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
        f"The obj. function was evaluated in {simple_evo.objective_function.n_points} points ({simple_evo.objective_function.calls} calls)"
    )
