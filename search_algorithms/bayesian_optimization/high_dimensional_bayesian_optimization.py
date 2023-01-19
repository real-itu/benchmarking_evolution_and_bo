from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import gpytorch

from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition import ExpectedImprovement, AcquisitionFunction
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin

from torch.quasirandom import SobolEngine

from utils.wrappers.counters import counted
from utils.visualization.bayesian_optimization import plot_prediction, plot_acquisition


class HighDimensionalBayesianOptimization:
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

        self.__objective_function = objective_function
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
        model = SaasFullyBayesianSingleTaskGP(
            self.trace,
            self.objective_values,
        )
        fit_fully_bayesian_model_nuts(model)
        model.eval()

        # Instantiate the acquisition function
        acquisiton_funciton = ExpectedImprovement(model, self.objective_values.max())

        DIM = self.__objective_function.solution_length
        candidate, acq_values = optimize_acqf(
            acquisiton_funciton,
            bounds=torch.cat(
                (
                    self.limits[0] * torch.ones(1, DIM),
                    self.limits[1] * torch.ones(1, DIM),
                )
            ),
            q=1,
            num_restarts=10,
            raw_samples=1024,
        )

        # Optimizing the acq. function by hand on a discrete grid.
        # zs = torch.Tensor(
        #     [
        #         [x, y]
        #         for x in torch.linspace(*self.limits, n_points_in_acq_grid)
        #         for y in reversed(torch.linspace(*self.limits, n_points_in_acq_grid))
        #     ]
        # )
        # acq_values = acquisiton_funciton(zs.unsqueeze(1))
        # candidate = zs[acq_values.argmax()]

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
