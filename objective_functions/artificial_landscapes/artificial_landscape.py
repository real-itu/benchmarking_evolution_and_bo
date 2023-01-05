"""
A series of testbed functions for optimization. FYI, we changed
the functions from testing maximization instead of minimization.

When run, this script plots the three example functions provided.

See for more examples:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
from typing import Callable, Literal

import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.visualization.evolutionary_strategies import plot_heatmap_in_ax

from .definitions import (
    easom,
    cross_in_tray,
    shifted_sphere,
    egg_holder,
    ackley_function_01,
    alpine_01,
    alpine_02,
    bent_cigar,
    brown,
    chung_reynolds,
    cosine_mixture,
)

POSSIBLE_FUNCTIONS = [
    "ackley_function_01",
    "alpine_01",
    "alpine_02",
    "bent_cigar",
    "brown",
    "chung_reynolds",
    "shifted_sphere",
    "easom",
    "cross_in_tray",
    "egg_holder",
]


class ArtificialLandscape:
    """
    This class will contain the toy objective functions, their limits,
    and the optima location.

    For more information, check definitions.py and [1].

    [1]:  https://al-roomi.org/benchmarks/unconstrained/n-dimensions
    """

    def __init__(
        self,
        name: Literal[
            "ackley_function_01",
            "alpine_01",
            "alpine_02",
            "bent_cigar",
            "brown",
            "chung_reynolds",
            "shifted_sphere",
            "easom",
            "cross_in_tray",
            "egg_holder",
        ],
        n_dims: int = 2,
    ) -> None:
        self.maximize = True
        self.known_optima = True

        if name == "ackley_function_01":
            self.function = ackley_function_01
            self.limits = [-32.0, 32.0]
            self.optima_location = torch.Tensor([0.0] * n_dims)
            self.solution_length = n_dims
        elif name == "alpine_01":
            self.function = alpine_01
            self.limits = [-10.0, 10.0]
            self.optima_location = torch.Tensor([0.0] * n_dims)
            self.solution_length = n_dims
        elif name == "alpine_02":
            self.function = alpine_02
            self.limits = [0.0, 10.0]
            self.optima_location = torch.Tensor([7.9170526982459462172] * n_dims)
            self.solution_length = n_dims
        elif name == "bent_cigar":
            self.function = bent_cigar
            self.limits = [-100.0, 100.0]
            self.optima_location = torch.Tensor([0.0] * n_dims)
            self.solution_length = n_dims
        elif name == "brown":
            self.function = brown
            self.limits = [-1.0, 4.0]
            self.optima_location = torch.Tensor([0.0] * n_dims)
            self.solution_length = n_dims
        elif name == "chung_reynolds":
            self.function = chung_reynolds
            self.limits = [-100.0, 100.0]
            self.optima_location = torch.Tensor([0.0] * n_dims)
            self.solution_length = n_dims
        elif name == "cosine_mixture":
            self.function = cosine_mixture
            self.limits = [-1.0, 1.0]
            self.optima_location = torch.Tensor([0.0] * n_dims)
            self.solution_length = n_dims
        elif name == "shifted_sphere":
            self.function = shifted_sphere
            self.limits = [-4.0, 4.0]
            self.optima_location = torch.Tensor([1.0, 1.0])
            self.solution_length = 2
        elif name == "easom":
            self.function = easom
            self.limits = [np.pi - 4, np.pi + 4]
            self.optima_location = torch.Tensor([np.pi, np.pi])
            self.solution_length = 2
        elif name == "cross_in_tray":
            self.function = cross_in_tray
            self.limits = [-10, 10]
            self.optima_location = torch.Tensor([1.34941, 1.34941])
            self.solution_length = 2
        elif name == "egg_holder":
            self.function = egg_holder
            self.limits = [-700, 700]
            self.optima_location = torch.Tensor([512, 404.2319])
            self.solution_length = 2
        else:
            raise ValueError(
                f'Expected {name} to be one of "shifted_sphere", "easom", "cross_in_tray", "egg_holder"'
            )

        self.optima = self.function(self.optima_location)

    def evaluate_objective(self, x: torch.Tensor) -> torch.Tensor:
        return self.function(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.function(x)


if __name__ == "__main__":
    fig = plt.figure()
    fig, axes = plt.subplots(4, 4)
    for ax, function in zip(axes.flatten(), POSSIBLE_FUNCTIONS):
        artificial_landscape = ArtificialLandscape(function)
        plot_heatmap_in_ax(
            ax, artificial_landscape.function, *artificial_landscape.limits
        )
        ax.set_title(function)

    fig.tight_layout()
    plt.show()
