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

from objective_functions import easom, cross_in_tray, egg_holder, shifted_sphere
from vis_utils import plot_algorithm, plot_prediction, plot_acquisition


def bayesian_optimization_iteration(
    z: torch.Tensor,
    objective_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    limits: Tuple[float, float],
    ax_prediction: plt.Axes,
    ax_acquisition: plt.Axes,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs a B.O. iteration and returns the next candidate and its value.
    """
    # Calling the obj function on all values.
    obj_values = objective_function(z[:, 0], z[:, 1]).unsqueeze(1)

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
    obj_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    limits: Tuple[float, float],
    n_iterations: int = 100,
):
    """
    Runs Bayesian Optimization over the given acquisition function.
    Starts with an initializaton at the origin.

    It returns the pair (best_z, best_obj_value).
    """
    # Initialize with the origin
    z = torch.Tensor([[0.0, 0.0]])

    _, (ax_obj_function, ax_prediction, ax_acquisition) = plt.subplots(
        1, 3, figsize=(3 * 6, 6)
    )
    ax_obj_function.set_title("Obj. function")
    ax_prediction.set_title("GP prediction")
    ax_acquisition.set_title("Acq. function")
    for i in range(n_iterations):
        candidate = bayesian_optimization_iteration(
            z=z,
            objective_function=obj_function,
            limits=limits,
            ax_prediction=ax_prediction,
            ax_acquisition=ax_acquisition,
        )
        print(
            f"(Iteration {i+1}) tested {candidate} and got {obj_function(candidate[0], candidate[1])}"
        )

        z = torch.vstack((z, candidate))

        plot_algorithm(
            ax=ax_obj_function,
            obj_function=obj_function,
            limits=limits,
            current_best=z[-2],
            population=z,
            next_best=z[-1],
        )

        plt.pause(0.01)
        for ax in [ax_obj_function, ax_prediction, ax_acquisition]:
            ax.clear()


if __name__ == "__main__":
    # Defining the bounds for the specific obj. functions
    obj_function = shifted_sphere
    limits = [-4.0, 4.0]
    
    # obj_function = easom
    # limits = [np.pi - 4, np.pi + 4]

    # obj_function = cross_in_tray
    # limits = [-10, 10]

    # obj_function = egg_holder
    # limits = [-512, 512]

    run_experiment(obj_function=obj_function, limits=limits)
