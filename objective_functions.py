"""
A series of testbed functions for optimization. FYI, we changed
the functions from testing maximization instead of minimization.

When run, this script plots the three example functions provided.

See for more examples:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from vis_utils import plot_heatmap_in_ax


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


if __name__ == "__main__":

    fig = plt.figure()
    ax_easom = fig.add_subplot(131)
    plot_heatmap_in_ax(ax_easom, easom, np.pi - 4, np.pi + 4)

    ax_cross = fig.add_subplot(132)
    plot_heatmap_in_ax(ax_cross, cross_in_tray, -10, 10)

    ax_egg = fig.add_subplot(133)
    plot_heatmap_in_ax(ax_egg, egg_holder, -512, 512)

    fig.tight_layout()
    plt.show()
