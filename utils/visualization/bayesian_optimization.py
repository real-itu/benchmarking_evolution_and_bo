from typing import Tuple
from itertools import product

import matplotlib.pyplot as plt
import torch

from gpytorch.models import ExactGP
from botorch.acquisition import AcquisitionFunction

from .common import _image_from_values


def plot_prediction(
    model: ExactGP,
    ax: plt.Axes,
    limits: Tuple[float, float],
    z: torch.Tensor,
    candidate: torch.Tensor,
    cmap: str = None,
    colorbar_limits: Tuple[float, float] = (None, None),
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

    lower, upper = colorbar_limits

    plot = ax.imshow(
        means_as_img, extent=[*limits, *limits], cmap=cmap, vmin=lower, vmax=upper
    )
    ax.scatter(z[:, 0], z[:, 1], c="white", edgecolors="black")
    ax.scatter([z[-1, 0]], [z[-1, 1]], c="red", edgecolors="black")
    ax.scatter([candidate[0]], [candidate[1]], c="blue", edgecolors="black")

    if colorbar_limits != (None, None):
        plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)


def plot_acquisition(
    acq_function: AcquisitionFunction,
    ax: plt.Axes,
    limits: Tuple[float, float],
    z: torch.Tensor,
    candidate: torch.Tensor,
    cmap: str = "Blues",
    plot_colorbar: bool = False,
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

    plot = ax.imshow(acq_values_as_img, extent=[*limits, *limits], cmap=cmap)
    ax.scatter(z[:, 0], z[:, 1], c="white", edgecolors="black")
    ax.scatter([z[-1, 0]], [z[-1, 1]], c="red", edgecolors="black")
    ax.scatter([candidate[0]], [candidate[1]], c="blue", edgecolors="black")

    if plot_colorbar:
        plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
