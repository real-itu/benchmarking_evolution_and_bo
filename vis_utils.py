from typing import Callable, Tuple

import matplotlib.pyplot as plt

import torch


def plot_surface_in_ax(
    ax: plt.Axes,
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    min_: float,
    max_: float,
):
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
    plot_heatmap_in_ax(ax, obj_function, *limits)
    ax.scatter(population[:, 0], population[:, 1], c="white", edgecolors="black")
    ax.scatter([current_best[0]], [current_best[1]], c="red", edgecolors="black")
    ax.scatter([next_best[0]], [next_best[1]], c="blue", edgecolors="black")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
