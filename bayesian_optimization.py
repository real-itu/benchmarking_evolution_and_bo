"""
Let's benchmark against simple B.O.
"""

from typing import Tuple, Callable
from matplotlib import pyplot as plt

import torch
import numpy as np

import gpytorch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement

from gpytorch.mlls import ExactMarginalLogLikelihood

from objective_functions import ObjectiveFunction, counted
from vis_utils import plot_algorithm, plot_prediction, plot_acquisition


def bayesian_optimization_iteration(
    z: torch.Tensor,
    obj_values: torch.Tensor,
    limits: Tuple[float, float],
    ax_prediction: plt.Axes,
    ax_acquisition: plt.Axes,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs a B.O. iteration and returns the next candidate and its value.
    """
    # Defining the Gaussian Process
    kernel = gpytorch.kernels.MaternKernel()
    model = SingleTaskGP(z, obj_values, covar_module=kernel)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    model.eval()
    acq_function = ExpectedImprovement(model, obj_values.max())

    # Optimizing the acq. function by hand on a discrete grid.
    zs = torch.Tensor(
        [
            [x, y]
            for x in torch.linspace(*limits, 100)
            for y in reversed(torch.linspace(*limits, 100))
        ]
    )
    acq_values = acq_function(zs.unsqueeze(1))
    candidate = zs[acq_values.argmax()]

    # Visualizing
    plot_prediction(model, ax_prediction, limits, z, candidate)
    plot_acquisition(acq_function, ax_acquisition, limits, z, candidate)

    return candidate


def run_experiment(
    objective: ObjectiveFunction,
    n_iterations: int = 100,
    tolerance_for_optima: float = 1e-3,
    break_when_close_to_optima: bool = True,
):
    """
    Runs Bayesian Optimization over the given acquisition function.
    Starts with an initializaton at the origin.

    It returns the pair (best_z, best_obj_value).
    """
    obj_function = objective.function
    limits = objective.limits

    # Counting the evaluations of the objective function
    @counted
    def obj_function_counted(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return obj_function(x, y)

    # Initialize with the origin
    z = torch.Tensor([[0.0, 0.0]])
    obj_values = obj_function_counted(z[:, 0], z[:, 1]).unsqueeze(1)

    _, (ax_obj_function, ax_prediction, ax_acquisition) = plt.subplots(
        1, 3, figsize=(3 * 6, 6)
    )
    for i in range(n_iterations):
        candidate = bayesian_optimization_iteration(
            z=z,
            obj_values=obj_values,
            limits=limits,
            ax_prediction=ax_prediction,
            ax_acquisition=ax_acquisition,
        )
        obj_value_at_candidate = obj_function_counted(*candidate)
        obj_values = torch.vstack((obj_values, obj_value_at_candidate))

        print(f"(Iteration {i+1}) tested {candidate} and got {obj_value_at_candidate}")

        z = torch.vstack((z, candidate))

        plot_algorithm(
            ax=ax_obj_function,
            obj_function=obj_function,
            limits=limits,
            current_best=z[-2],
            population=z,
            next_best=z[-1],
        )

        ax_obj_function.set_title("Obj. function")
        ax_prediction.set_title("GP prediction")
        ax_acquisition.set_title("Acq. function")
        plt.pause(0.01)
        for ax in [ax_obj_function, ax_prediction, ax_acquisition]:
            ax.clear()

        if (
            torch.isclose(
                obj_value_at_candidate, objective.optima, atol=tolerance_for_optima
            )
            and break_when_close_to_optima
        ):
            print(f"Found a good-enough optima, breaking.")
            break

    print(
        f"The obj. function was evaluated in {obj_function_counted.n_points} points ({obj_function_counted.calls} calls)"
    )


if __name__ == "__main__":
    # Defining the function to optimize
    name = "shifted_sphere"  # "shifted_sphere", "easom", "cross_in_tray", "egg_holder"

    # Hyperparameters for the search
    # Num. of generations/iterations
    n_generations = 100

    # Breaking as soon as the best fitness is this close to the actual optima
    # in absolute terms
    tolerance_for_optima = 1e-3

    # Do we actually want to break?
    break_when_close_to_optima = True

    # Defining the objective function, limits, and so on...
    # They are all contained in the ObjectiveFunction class
    objective = ObjectiveFunction(name)

    run_experiment(
        objective=objective,
        n_iterations=n_generations,
        tolerance_for_optima=tolerance_for_optima,
        break_when_close_to_optima=break_when_close_to_optima,
    )
