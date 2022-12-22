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

from experiments.toy_examples.toy_objective_functions import ObjectiveFunction
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
        self, ax_for_prediction: plt.Axes = None, ax_for_acquisition: plt.Axes = None
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
                for x in torch.linspace(*self.limits, 100)
                for y in reversed(torch.linspace(*self.limits, 100))
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


if __name__ == "__main__":
    # Defining the function to optimize
    name = "shifted_sphere"  # "shifted_sphere", "easom", "cross_in_tray", "egg_holder"

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
    objective = ObjectiveFunction(name)
    obj_function = objective.function
    limits = objective.limits
    solution_length = objective.solution_length

    bayes_opt = BayesianOptimization(
        objective_function=obj_function,
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
            obj_function=obj_function,
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
        current_fitness = obj_function(bayes_opt.trace[-1])
        print(f"Current fitness: {current_fitness}")
        if (
            torch.isclose(current_fitness, objective.optima, atol=tolerance_for_optima)
            and break_when_close_to_optima
        ):
            print(f"Found a good-enough optima, breaking.")
            break

    print(
        f"The obj. function was evaluated in {bayes_opt.objective_function.n_points} points ({bayes_opt.objective_function.calls} calls)"
    )
