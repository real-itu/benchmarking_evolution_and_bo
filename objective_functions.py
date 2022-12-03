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

from vis_utils import plot_heatmap_in_ax


def counted(obj_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    """
    Counts on how many points the obj function was evaluated
    """

    def wrapped(x: torch.Tensor, y: torch.Tensor):
        wrapped.calls += 1

        if len(x.shape) == 0:
            wrapped.n_points += 1
        else:
            wrapped.n_points += len(x)

        return obj_function(x, y)

    wrapped.calls = 0
    wrapped.n_points = 0
    return wrapped


def easom(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Easom is very flat, with a maxima at (pi, pi).
    """
    return (
        torch.cos(x) * torch.cos(y) * torch.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))
    )


def cross_in_tray(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Cross-in-tray has several local maxima in a quilt-like pattern.
    """
    quotient = torch.sqrt(x**2 + y**2) / np.pi
    return (
        1e-4
        * (
            torch.abs(torch.sin(x) * torch.sin(y) * torch.exp(torch.abs(10 - quotient)))
            + 1
        )
        ** 0.1
    )


def egg_holder(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    The egg holder is especially difficult.
    """
    return (y + 47) * torch.sin(torch.sqrt(torch.abs(x / 2 + (y + 47)))) + (
        x * torch.sin(torch.sqrt(torch.abs(x - (y + 47))))
    )


def shifted_sphere(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    The usual squared norm, but shifted away from the origin by a bit.
    Maximized at (1, 1)
    """
    return -((x - 1) ** 2 + (y - 1) ** 2)


class ObjectiveFunction:
    """
    This class will contain the objective function, the limits,
    and the optima location.
    """

    def __init__(
        self, name: Literal["shifted_sphere", "easom", "cross_in_tray", "egg_holder"]
    ) -> None:
        if name == "shifted_sphere":
            self.function = shifted_sphere
            self.limits = [-4.0, 4.0]
            self.optima_location = torch.Tensor([1.0, 1.0])
        elif name == "easom":
            self.function = easom
            self.limits = [np.pi - 4, np.pi + 4]
            self.optima_location = torch.Tensor([np.pi, np.pi])
        elif name == "cross_in_tray":
            self.function = cross_in_tray
            self.limits = [-10, 10]
            self.optima_location = torch.Tensor([1.34941, 1.34941])
        elif name == "egg_holder":
            self.function = egg_holder
            self.limits = [-700, 700]
            self.optima_location = torch.Tensor([512, 404.2319])
        else:
            raise ValueError(
                f'Expected {name} to be one of "shifted_sphere", "easom", "cross_in_tray", "egg_holder"'
            )

        self.optima = self.function(*self.optima_location)


if __name__ == "__main__":
    fig = plt.figure()
    ax_shifted_sphere = fig.add_subplot(141)
    plot_heatmap_in_ax(ax_shifted_sphere, shifted_sphere, -4.0, 4.0)

    ax_easom = fig.add_subplot(142)
    plot_heatmap_in_ax(ax_easom, easom, np.pi - 4, np.pi + 4)

    ax_cross = fig.add_subplot(143)
    plot_heatmap_in_ax(ax_cross, cross_in_tray, -10.0, 10.0)

    ax_egg = fig.add_subplot(144)
    plot_heatmap_in_ax(ax_egg, egg_holder, -512.0, 512.0)

    fig.tight_layout()
    plt.show()
