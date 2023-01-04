"""
This script defines all the artificial landscapes with
signature [torch.Tensor] -> torch.Tensor. You might
see that the signs have been flipped from [1]. This is
because we're dealing with maximizations instead of
minimizations.

In what follows, x is a tensor of arbitrary dimension
(either (b, d), or (d,), where d is the dimension of
the design space).

[1] Ali R. Al-Roomi (2015). Unconstrained Single-Objective Benchmark
    Functions Repository [https://www.al-roomi.org/benchmarks/unconstrained].
    Halifax, Nova Scotia, Canada: Dalhousie University, Electrical and Computer
    Engineering.
"""
import torch
import numpy as np


def ackley_function_01(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    _, d = x.shape

    first = torch.exp(-0.2 * torch.sqrt((1 / d) * torch.sum(x**2, dim=1)))
    second = torch.exp((1 / d) * torch.sum(torch.cos(2 * np.pi * x), dim=1))
    res = 20 * first + second - np.exp(1.0) - 20

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def alpine_01(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    res = -torch.sum(torch.abs(x * torch.sin(x) + 0.1 * x), dim=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def alpine_02(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    res = torch.prod(torch.sin(x) * torch.sqrt(x), dim=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def bent_cigar(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    first = x[..., 0] ** 2
    second = 1e6 * torch.sum(x[..., 1:] ** 1, dim=1)
    res = -(first + second)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def brown(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    first = x[..., :-1] ** 2
    second = x[..., 1:] ** 2

    res = -torch.sum(first ** (second + 1) + second ** (first + 1), dim=1)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def chung_reynolds(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    res = -(torch.sum(x**2, dim=1) ** 2)

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def cosine_mixture(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 1:
        # Add a batch dimension if it's missing
        x = x.unsqueeze(0)
        batched = False
    else:
        batched = True

    first = 0.1 * torch.sum(torch.cos(5 * np.pi * x), dim=1)
    second = torch.sum(x**2, dim=1)

    res = first - second

    # Remove the batch dim if it wasn't there in the beginning
    if not batched:
        res = res.squeeze(0)

    return res


def easom(xy: torch.Tensor) -> torch.Tensor:
    """
    Easom is very flat, with a maxima at (pi, pi).
    """
    x = xy[..., 0]
    y = xy[..., 1]
    return (
        torch.cos(x) * torch.cos(y) * torch.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))
    )


def cross_in_tray(xy: torch.Tensor) -> torch.Tensor:
    """
    Cross-in-tray has several local maxima in a quilt-like pattern.
    """
    x = xy[..., 0]
    y = xy[..., 1]
    quotient = torch.sqrt(x**2 + y**2) / np.pi
    return (
        1e-4
        * (
            torch.abs(torch.sin(x) * torch.sin(y) * torch.exp(torch.abs(10 - quotient)))
            + 1
        )
        ** 0.1
    )


def egg_holder(xy: torch.Tensor) -> torch.Tensor:
    """
    The egg holder is especially difficult.
    """
    x = xy[..., 0]
    y = xy[..., 1]
    return (y + 47) * torch.sin(torch.sqrt(torch.abs(x / 2 + (y + 47)))) + (
        x * torch.sin(torch.sqrt(torch.abs(x - (y + 47))))
    )


def shifted_sphere(xy: torch.Tensor) -> torch.Tensor:
    """
    The usual squared norm, but shifted away from the origin by a bit.
    Maximized at (1, 1)
    """
    x = xy[..., 0]
    y = xy[..., 1]
    return -((x - 1) ** 2 + (y - 1) ** 2)


if __name__ == "__main__":
    x = torch.randn((100, 2))
    print(x)
    res = ackley_function_01(x)
    print(res)
    print(res.shape)
