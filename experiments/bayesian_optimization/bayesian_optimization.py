"""
Implements the toy examples for Bayesian Optimization.
"""

from matplotlib import pyplot as plt

import torch

import gpytorch

from search_algorithms.bayesian_optimization import BayesianOptimization

from objective_functions.objective_function import ObjectiveFunction

from utils.visualization.evolutionary_strategies import plot_algorithm


if __name__ == "__main__":
    # Defining the function to optimize
    name = "easom"  # "shifted_sphere", "easom", "cross_in_tray", "egg_holder"
    model = None  # None, "small", "medium", "large"
    n_dims = 2  # Whatever int your mind desires (in most cases)

    # Hyperparameters for the search
    # Num. of generations/iterations
    n_iterations = 100

    # Breaking as soon as the best fitness is this close to the actual optima
    # in absolute terms
    tolerance_for_optima = 1e-3

    # Do we actually want to break?
    break_when_close_to_optima = True

    # Defining the objective function, limits, and so on...
    # They are all contained in the ObjectiveFunction class
    objective_function = ObjectiveFunction(name, n_dims=n_dims, model=model)
    limits = objective_function.limits
    solution_length = objective_function.solution_length

    bayes_opt = BayesianOptimization(
        objective_function=objective_function,
        max_iterations=n_iterations,
        limits=limits,
        kernel=gpytorch.kernels.MaternKernel,
    )

    _, (ax_obj_function, ax_prediction, ax_acquisition) = plt.subplots(
        1, 3, figsize=(3 * 6, 6)
    )
    for _ in range(n_iterations):
        bayes_opt.step(
            ax_for_prediction=ax_prediction, ax_for_acquisition=ax_acquisition
        )

        plot_algorithm(
            ax=ax_obj_function,
            obj_function=objective_function,
            limits=limits,
            current_best=bayes_opt.trace[-2],
            population=bayes_opt.trace,
            next_best=bayes_opt.trace[-1],
        )

        ax_obj_function.set_title("Obj. function")
        ax_prediction.set_title("GP prediction")
        ax_acquisition.set_title("Acq. function")
        plt.pause(0.01)
        for ax in [ax_obj_function, ax_prediction, ax_acquisition]:
            ax.clear()

        # (uncounted) best fitness evaluation
        # TODO: We shouldn't be calling the obj. function again
        # since it would imply re-simulating. This info is already
        # inside bayes_opt.
        current_fitness = objective_function(bayes_opt.trace[-1])
        print(f"Current fitness: {current_fitness}")
        if (
            torch.isclose(
                current_fitness, objective_function.optima, atol=tolerance_for_optima
            )
            and break_when_close_to_optima
        ):
            print(f"Found a good-enough optima, breaking.")
            break

    print(
        f"The obj. function was evaluated in {bayes_opt.objective_function.n_points} points ({bayes_opt.objective_function.calls} calls)"
    )
