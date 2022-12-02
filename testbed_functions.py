"""
A series of testbed functions for optimization.
See: https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

import torch
import numpy as np


def easom(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        torch.cos(x) * torch.cos(y) * torch.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))
    )


def cross_in_tray(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
    return (y + 47) * torch.sin(torch.sqrt(torch.abs(x / 2 + (y + 47)))) + (
        x * torch.sin(torch.sqrt(torch.abs(x - (y + 47))))
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from vis_utils import plot_heatmap_in_ax

    fig = plt.figure()
    ax_easom = fig.add_subplot(131)
    plot_heatmap_in_ax(ax_easom, easom, np.pi - 4, np.pi + 4)

    ax_cross = fig.add_subplot(132)
    plot_heatmap_in_ax(ax_cross, cross_in_tray, -10, 10)

    ax_egg = fig.add_subplot(133)
    plot_heatmap_in_ax(ax_egg, egg_holder, -512, 512)

    fig.tight_layout()
    plt.show()
