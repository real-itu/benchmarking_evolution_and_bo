"""
Runs CMA-ES on easom for visualization and to compare the
number of objective function calls.
"""
from typing import Tuple
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from search_algorithms.evolutionary_strategies.cma_es import CMA_ES

from objective_functions.objective_function import ObjectiveFunction
from objective_functions.artificial_landscapes import ArtificialLandscape

from utils.visualization.evolutionary_strategies import plot_algorithm

sns.set_style("whitegrid")
sns.set(font_scale=1.5)

FIG_PATH = Path("/Users/migd/Projects/dissertation/Figures/Chapter_3/bo_related_plots")

FIG_PATH_FOR_CMA_ES = FIG_PATH / "cma_es_trace"
FIG_PATH_FOR_CMA_ES.mkdir(exist_ok=True)


def run_CMA_ES(name: str = "easom", visualize: bool = True) -> Tuple[bool, int, int]:
    """
    Returns (found_optima, n_calls, n_points)
    """
    # Hyperparameters for the search
    # Num. of generations
    n_generations = 100
    population_size = 10

    # Breaking as soon as the best fitness is this close to the actual optima
    # in absolute terms
    tolerance_for_optima = 1e-2

    # Do we actually want to break?
    break_when_close_to_optima = True

    # Initial covariance, needs to be modified depending on {name}
    exploration = 0.2

    # Defining the objective function, limits, and so on...
    # They are all contained in the ObjectiveFunction class
    objective_function = ArtificialLandscape(name)
    limits = objective_function.limits
    solution_length = objective_function.solution_length

    cma_es = CMA_ES(
        objective_function=objective_function,
        population_size=population_size,
        exploration=exploration,
        solution_length=solution_length,
        limits=limits,
    )

    found_optima = False
    for generation in range(n_generations):
        # Get the current best and population
        if visualize:
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        current_best = cma_es.get_current_best()
        population = cma_es.get_population()

        # Run a step
        cma_es.step()

        # Get the next best
        next_best = cma_es.get_current_best()

        # Visualize
        if visualize:
            plot_algorithm(
                ax=ax,
                obj_function=objective_function,
                limits=limits,
                current_best=current_best,
                population=population,
                next_best=next_best,
                cmap="plasma",
            )
            ax.grid(False)
            fig.savefig(
                FIG_PATH_FOR_CMA_ES / f"{generation:03d}.jpg",
                dpi=120,
                bbox_inches="tight",
            )
            plt.close(fig)

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
            found_optima = True
            break

    print(
        f"The obj. function was evaluated in {cma_es.objective_function.n_points} points ({cma_es.objective_function.calls} calls)"
    )

    return (
        found_optima,
        cma_es.objective_function.calls,
        cma_es.objective_function.n_points,
    )


if __name__ == "__main__":
    # run_CMA_ES()

    calls_and_points_easom = []
    calls_and_points_cross = []
    for _ in range(50):
        found_optima, calls, points = run_CMA_ES(name="easom", visualize=False)
        if found_optima:
            calls_and_points_easom.append((calls, points))

    for _ in range(50):
        found_optima, calls, points = run_CMA_ES(name="cross_in_tray", visualize=False)
        if found_optima:
            calls_and_points_cross.append((calls, points))

    # Saving them
    data_path = Path(__file__).parent.resolve()
    df_easom = pd.DataFrame(calls_and_points_easom, columns=["num_calls", "num_points"])
    df_cross = pd.DataFrame(calls_and_points_cross, columns=["num_calls", "num_points"])

    df_easom.to_csv(data_path / "easom.csv")
    df_cross.to_csv(data_path / "cross.csv")
