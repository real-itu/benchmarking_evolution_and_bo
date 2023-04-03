"""
In this script we compare ES and BO on two test functions,
saving enough visualizations for my dissertation.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd

from multiprocessing import Pool

import gpytorch

from search_algorithms.bayesian_optimization import BayesianOptimization
from search_algorithms.evolutionary_strategies.cma_es import CMA_ES

from objective_functions.artificial_landscapes import ArtificialLandscape
from objective_functions.objective_function import ObjectiveFunction

from utils.visualization.evolutionary_strategies import (
    plot_algorithm,
    plot_heatmap_in_ax,
)

TEST_FUNCTION_1 = "easom"
TEST_FUNCTION_2 = "cross_in_tray"

sns.set_style("whitegrid")
sns.set(font_scale=1.5)

FIG_PATH = Path("/Users/migd/Projects/dissertation/Figures/Chapter_3/bo_related_plots")


def plot_the_two_test_functions():
    """
    Saves two figures with the two test functions.
    """
    test_function_1 = ArtificialLandscape(TEST_FUNCTION_1)
    fig_1, ax_1 = plt.subplots(1, 1, figsize=(7, 7))
    plot = plot_heatmap_in_ax(
        ax_1, test_function_1.function, *test_function_1.limits, cmap="plasma"
    )
    plt.colorbar(plot, ax=ax_1, fraction=0.046, pad=0.04)
    ax_1.grid(False)

    test_function_2 = ArtificialLandscape(TEST_FUNCTION_2)
    fig_2, ax_2 = plt.subplots(1, 1, figsize=(7, 7))
    plot = plot_heatmap_in_ax(
        ax_2, test_function_2.function, *test_function_2.limits, cmap="plasma"
    )
    plt.colorbar(plot, ax=ax_2, fraction=0.046, pad=0.04)
    ax_2.grid(False)

    for i, fig in enumerate([fig_1, fig_2]):
        fig.savefig(FIG_PATH / f"test_function_{i+1}.jpg", dpi=120, bbox_inches="tight")


def run_BO(test_function_name: str, seed: int = 0, visualize: bool = True):
    n_dims = 2  # Whatever int your mind desires (in most cases)

    # Hyperparameters for the search
    # Num. of generations/iterations
    n_iterations = 100

    # Breaking as soon as the best fitness is this close to the actual optima
    # in absolute terms
    tolerance_for_optima = 1e-2

    # Do we actually want to break?
    break_when_close_to_optima = True

    # Defining the objective function, limits, and so on...
    # They are all contained in the ObjectiveFunction class
    objective_function = ObjectiveFunction(test_function_name, seed, n_dims=n_dims)
    limits = objective_function.limits
    solution_length = objective_function.solution_length

    bayes_opt = BayesianOptimization(
        objective_function=objective_function,
        max_iterations=n_iterations,
        limits=limits,
        kernel=gpytorch.kernels.MaternKernel,
    )

    if test_function_name == "easom":
        colorbar_limits = (0.0, 1.0)
    elif test_function_name == "alpine_02":
        colorbar_limits = (-6, 7.1)
    elif test_function_name == "cross_in_tray":
        colorbar_limits = (11 / 10, 25 / 10)
    else:
        raise ValueError(...)

    found_optima = False
    for step in range(n_iterations):
        if visualize:
            fig_step, (ax_prediction, ax_acquisition) = plt.subplots(
                1, 2, figsize=(2 * 7, 7), sharey=True
            )
            # fig_step.suptitle(f"Iteration {step+1}")
        else:
            ax_prediction = None
            ax_acquisition = None

        bayes_opt.step(
            ax_for_prediction=ax_prediction,
            ax_for_acquisition=ax_acquisition,
            cmap_prediction="plasma",
            cmap_acquisition="Blues",
            plot_colorbar_in_acquisition=True,
            colorbar_limits_for_prediction=colorbar_limits,
        )

        if visualize:
            ax_prediction.set_title("GP prediction")
            ax_acquisition.set_title("Acq. function")

            ax_prediction.grid(False)
            ax_acquisition.grid(False)

            FIG_PATH_FOR_TRACE = FIG_PATH / test_function_name
            FIG_PATH_FOR_TRACE.mkdir(exist_ok=True)
            fig_step.savefig(
                FIG_PATH_FOR_TRACE / f"{step:03d}.jpg", dpi=120, bbox_inches="tight"
            )
            plt.close(fig_step)

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
            found_optima = True
            break

    print(
        f"The obj. function was evaluated in {bayes_opt.objective_function.n_points} points ({bayes_opt.objective_function.calls} calls)"
    )

    print(f"Best fitness: {bayes_opt.objective_values.max()}")

    return (
        found_optima,
        bayes_opt.objective_function.calls,
        bayes_opt.objective_function.n_points,
    )


if __name__ == "__main__":
    # plot_the_two_test_functions()
    # plt.close()

    # run_BO(TEST_FUNCTION_1)
    # run_BO(TEST_FUNCTION_2)

    iterable = [["easom", i, False] for i in range(50)]
    with Pool(14) as pool:
        results = pool.starmap(run_BO, iterable)

    calls_and_points_easom = [
        (n_calls, n_points)
        for found_optima, n_calls, n_points in results
        if found_optima
    ]

    # for i in range(50):
    #     found_optima, n_calls, n_points = run_BO("easom", seed=i, visualize=False)
    #     if found_optima:
    #         calls_and_points_easom.append((n_calls, n_points))

    data_path = Path(__file__).parent.resolve()
    df_easom = pd.DataFrame(calls_and_points_easom, columns=["num_calls", "num_points"])
    df_easom.to_csv(data_path / "easom_bo.csv")

    # for i in range(50):
    #     found_optima, n_calls, n_points = run_BO(
    #         "cross_in_tray", seed=i, visualize=False
    #     )
    #     if found_optima:
    #         calls_and_points_cross.append((n_calls, n_points))

    iterable = [["cross_in_tray", i, False] for i in range(50)]
    with Pool(14) as pool:
        results = pool.starmap(run_BO, iterable)

    calls_and_points_cross = [
        (n_calls, n_points)
        for found_optima, n_calls, n_points in results
        if found_optima
    ]
    df_cross = pd.DataFrame(calls_and_points_cross, columns=["num_calls", "num_points"])
    df_cross.to_csv(data_path / "cross_bo.csv")
