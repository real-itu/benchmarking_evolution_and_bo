"""
Contains a series of visualization utils for the evolutionary algorithms
as well as for Bayesian Optimization.
"""
from typing import Callable, Tuple
from itertools import product

import matplotlib.pyplot as plt
import torch

from .common import _image_from_values


def plot_surface_in_ax(
    ax: plt.Axes,
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    min_: float,
    max_: float,
):
    """
    Plots the provided function in the square [min_, max_]^2 as a surface.
    """
    X = torch.linspace(min_, max_, 100)
    Y = torch.linspace(min_, max_, 100)
    X, Y = torch.meshgrid(X, Y)
    res = function(X, Y)

    ax.plot_surface(X, Y, res)


def plot_heatmap_in_ax(
    ax: plt.Axes,
    function: Callable[[torch.Tensor], torch.Tensor],
    min_: float,
    max_: float,
    cmap: str = None,
):
    """
    Plots the provided function in the square [min_, max_]^2 as a heatmap.
    """
    X = torch.linspace(min_, max_, 100)
    Y = torch.linspace(min_, max_, 100)

    fine_grid = torch.Tensor([[x, y] for x, y in product(X, Y)])

    res_img = _image_from_values(function(fine_grid), [min_, max_], 100)

    plot = ax.imshow(res_img, extent=[min_, max_, min_, max_], cmap=cmap)
    return plot


def plot_algorithm(
    ax: plt.Axes,
    obj_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    limits: Tuple[float, float],
    current_best: torch.Tensor,
    population: torch.Tensor,
    next_best: torch.Tensor,
    cmap: str = None,
):
    """
    First plots the objective function as a heatmap in the axis,
    then plots the population using white circles, the current best
    using a red circle, and the next best using blue circles. These
    are then visualized only on the [*limits]^2 square.
    """
    plot_heatmap_in_ax(ax, obj_function, *limits, cmap=cmap)
    ax.scatter(population[:, 0], population[:, 1], c="white", edgecolors="black")
    ax.scatter([current_best[0]], [current_best[1]], c="red", edgecolors="black")
    ax.scatter([next_best[0]], [next_best[1]], c="blue", edgecolors="black")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
