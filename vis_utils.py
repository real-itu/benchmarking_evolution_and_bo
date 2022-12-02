"""
Contains a series of visualization utils for the evolutionary algorithms
as well as for Bayesian Optimization.
"""
from typing import Callable, Tuple
from itertools import product

import matplotlib.pyplot as plt

import torch
import numpy as np

from gpytorch.models import ExactGP
from botorch.acquisition import AcquisitionFunction


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
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    min_: float,
    max_: float,
):
    """
    Plots the provided function in the square [min_, max_]^2 as a heatmap.
    """
    X = torch.linspace(min_, max_, 100)
    Y = torch.linspace(min_, max_, 100)
    X, Y = torch.meshgrid(X, Y)
    res = function(X, Y)

    ax.imshow(res, extent=[min_, max_, min_, max_])


def plot_algorithm(
    ax: plt.Axes,
    obj_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    limits: Tuple[float, float],
    current_best: torch.Tensor,
    population: torch.Tensor,
    next_best: torch.Tensor,
):
    """
    First plots the objective function as a heatmap in the axis,
    then plots the population using white circles, the current best
    using a red circle, and the next best using blue circles. These
    are then visualized only on the [*limits]^2 square.
    """
    plot_heatmap_in_ax(ax, obj_function, *limits)
    ax.scatter(population[:, 0], population[:, 1], c="white", edgecolors="black")
    ax.scatter([current_best[0]], [current_best[1]], c="red", edgecolors="black")
    ax.scatter([next_best[0]], [next_best[1]], c="blue", edgecolors="black")
    ax.set_xlim(limits)
    ax.set_ylim(limits)


def _image_from_values(
    values: torch.Tensor,
    limits: Tuple[float, float],
    n_points_in_grid: int,
):
    """
    Transforms a tensor of values into an
    {n_points_in_grid}x{n_points_in_grid} image.
    """
    z1s = torch.linspace(*limits, n_points_in_grid)
    z2s = torch.linspace(*limits, n_points_in_grid)

    fine_grid = torch.Tensor([[x, y] for x, y in product(z1s, z2s)])
    p_dict = {(x.item(), y.item()): v.item() for (x, y), v in zip(fine_grid, values)}

    positions = {
        (x.item(), y.item()): (i, j)
        for j, x in enumerate(z1s)
        for i, y in enumerate(reversed(z2s))
    }

    p_img = np.zeros((len(z2s), len(z1s)))
    for z, (i, j) in positions.items():
        p_img[i, j] = p_dict[z]

    return p_img


def plot_prediction(
    model: ExactGP,
    ax: plt.Axes,
    limits: Tuple[float, float],
    z: torch.Tensor,
    candidate: torch.Tensor,
):
    """
    Plots mean of the GP in the axes in a fine grid
    in latent space. Assumes that the latent space
    is of size 2.
    """
    n_points_in_grid = 75

    fine_grid_in_latent_space = torch.Tensor(
        [
            [x, y]
            for x, y in product(
                torch.linspace(*limits, n_points_in_grid),
                torch.linspace(*limits, n_points_in_grid),
            )
        ]
    )

    predicted_distribution = model(fine_grid_in_latent_space)
    means = predicted_distribution.mean
    means_as_img = _image_from_values(means, limits, n_points_in_grid)

    plot = ax.imshow(means_as_img, extent=[*limits, *limits], cmap="Blues")
    ax.scatter(z[:, 0], z[:, 1], c="white", edgecolors="black")
    ax.scatter([z[-1, 0]], [z[-1, 1]], c="red", edgecolors="black")
    ax.scatter([candidate[0]], [candidate[1]], c="blue", edgecolors="black")


def plot_acquisition(
    acq_function: AcquisitionFunction,
    ax: plt.Axes,
    limits: Tuple[float, float],
    z: torch.Tensor,
    candidate: torch.Tensor,
):
    n_points_in_grid = 75

    fine_grid_in_latent_space = torch.Tensor(
        [
            [x, y]
            for x, y in product(
                torch.linspace(*limits, n_points_in_grid),
                torch.linspace(*limits, n_points_in_grid),
            )
        ]
    ).unsqueeze(1)

    acq_values = acq_function(fine_grid_in_latent_space)
    acq_values_as_img = _image_from_values(acq_values, limits, n_points_in_grid)

    plot = ax.imshow(acq_values_as_img, extent=[*limits, *limits], cmap="Blues")
    ax.scatter(z[:, 0], z[:, 1], c="white", edgecolors="black")
    ax.scatter([z[-1, 0]], [z[-1, 1]], c="red", edgecolors="black")
    ax.scatter([candidate[0]], [candidate[1]], c="blue", edgecolors="black")
