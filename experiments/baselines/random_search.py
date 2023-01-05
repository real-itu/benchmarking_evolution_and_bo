"""
Implements the toy examples for random search
"""
import matplotlib.pyplot as plt
import torch

from search_algorithms import RandomSearch

from objective_functions.artificial_landscapes.test_functions import ObjectiveFunction

from utils.visualization.evolutionary_strategies import plot_algorithm

if __name__ == "__main__":
    # Defining the function to optimize
    name = "easom"  # "shifted_sphere", "easom", "cross_in_tray", "egg_holder"

    # Hyperparameters for the search
    # Num. of generations/iterations
    n_iterations = 1000

    # Breaking as soon as the best fitness is this close to the actual optima
    # in absolute terms
    tolerance_for_optima = 1e-3

    # Do we actually want to break?
    break_when_close_to_optima = True

    # Defining the objective function, limits, and so on...
    # They are all contained in the ObjectiveFunction class
    objective = ObjectiveFunction(name)
    obj_function = objective.function
    limits = objective.limits
    solution_length = objective.solution_length

    random_search = RandomSearch(
        objective_function=obj_function,
        max_iterations=n_iterations,
        limits=limits,
    )

    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    for _ in range(n_iterations):
        random_search.step()

        plot_algorithm(
            ax=ax,
            obj_function=obj_function,
            limits=limits,
            current_best=random_search.trace[-2]
            if len(random_search.trace) > 2
            else random_search.trace[-1],
            population=random_search.trace,
            next_best=random_search.trace[-1],
        )

        ax.set_title("Obj. function")
        plt.pause(0.01)
        ax.clear()

        # (uncounted) best fitness evaluation
        current_fitness = obj_function(random_search.trace[-1])
        print(f"Current fitness: {current_fitness}")
        if (
            torch.isclose(current_fitness, objective.optima, atol=tolerance_for_optima)
            and break_when_close_to_optima
        ):
            print(f"Found a good-enough optima, breaking.")
            break

    print(
        f"The obj. function was evaluated in {random_search.objective_function.n_points} points ({random_search.objective_function.calls} calls)"
    )
