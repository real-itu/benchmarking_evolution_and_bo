"""
Let's benchmark against simple B.O.
"""

from typing import Tuple, Callable
from matplotlib import pyplot as plt

import torch

import gpytorch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, AcquisitionFunction

from gpytorch.mlls import ExactMarginalLogLikelihood

from utils.visualization.evolutionary_strategies import plot_algorithm
from utils.visualization.bayesian_optimization import plot_prediction, plot_acquisition
from utils.wrappers.counters import counted

torch.set_default_dtype(torch.float64)


class BayesianOptimization:
    def __init__(
        self,
        objective_function: Callable[[torch.Tensor], torch.Tensor],
        max_iterations: int,
        limits: Tuple[float, float] = None,
        kernel: gpytorch.kernels.Kernel = None,
        acquisition_function: AcquisitionFunction = None,
    ) -> None:
        # Wrapping the objective function with the same counter
        # and storing it.
        @counted
        def counted_objective_function(inputs: torch.Tensor):
            return objective_function(inputs)

        self.objective_function = counted_objective_function
        self.max_iterations = max_iterations
        self.kernel = kernel
        self.acquisition_function = acquisition_function
        self.limits = limits

        # Initializing the variables.
        self.iteration_counter = 0
        self.trace = torch.randn((1, 2)).clip(*limits)
        self.objective_values = self.objective_function(self.trace).unsqueeze(0)

    def get_current_best(self) -> torch.Tensor:
        return self.trace[self.objective_values.argmax()]

    def get_trace(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.trace, self.objective_values

    def step(
        self,
        n_points_in_acq_grid: int = 400,
        ax_for_prediction: plt.Axes = None,
        ax_for_acquisition: plt.Axes = None,
    ):
        # Defining the Gaussian Process
        kernel = self.kernel()
        model = SingleTaskGP(self.trace, self.objective_values, covar_module=kernel)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        model.eval()

        # Instantiate the acquisition function
        acquisiton_funciton = ExpectedImprovement(model, self.objective_values.max())

        # Optimizing the acq. function by hand on a discrete grid.
        zs = torch.Tensor(
            [
                [x, y]
                for x in torch.linspace(*self.limits, n_points_in_acq_grid)
                for y in reversed(torch.linspace(*self.limits, n_points_in_acq_grid))
            ]
        )
        acq_values = acquisiton_funciton(zs.unsqueeze(1))
        candidate = zs[acq_values.argmax()]

        # Visualize the prediction
        if ax_for_prediction is not None:
            plot_prediction(
                model=model,
                ax=ax_for_prediction,
                limits=self.limits,
                z=self.trace,
                candidate=candidate,
            )

        if ax_for_acquisition is not None:
            plot_acquisition(
                acq_function=acquisiton_funciton,
                ax=ax_for_acquisition,
                limits=self.limits,
                z=self.trace,
                candidate=candidate,
            )

        # Evaluate the obj. function in this one, and append to
        # the trace.
        obj_value = self.objective_function(candidate)
        self.trace = torch.vstack((self.trace, candidate))
        self.objective_values = torch.vstack((self.objective_values, obj_value))
        self.iteration_counter += 1
