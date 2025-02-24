import torch
import torch.nn as nn
import math
from botorch.test_functions import Hartmann, Branin, Beale, Ackley

__all__ = [
    "forrester1D",
    "branin2D",
    "beale2D",
    "hartmann6D",
    "ackley6D",
]


def forrester1D(x: torch.Tensor, negate: bool = True, add_dim: bool = True):
    """Forrester synthetic test function.

    Args:
        x, (B, N, 1)
        negate, bool: whether to negate the function.

    Returns:
        y, (B, N, 1) if add_dim else (B, N)
    """
    y = ((6 * x - 2) ** 2) * torch.sin(12 * x - 4)
    if negate:
        y = -y
    return y if add_dim else y.squeeze(-1)


def branin2D(x: torch.Tensor, negate: bool = True, add_dim: bool = True):
    """Branin synthetic test function.

    Args:
        x, (B, N, 2)
        negate, bool: whether to negate the function.
    Returns:
        y, (B, N, 1) if add_dim else (B, N)
    """
    branin2d = Branin(negate=negate).to(x)
    y = branin2d(x)
    return y.unsqueeze(-1) if add_dim else y


def beale2D(x: torch.Tensor, negate: bool = True, add_dim: bool = True):
    """Beale synthetic test function.

    Args:
        x, (B, N, 2)
        negate, bool: whether to negate the function.
    Returns:
        y, (B, N, 1) if add_dim else (B, N)
    """
    beale2d = Beale(negate=negate).to(x)
    y = beale2d(x)
    return y.unsqueeze(-1) if add_dim else y


def hartmann6D(x: torch.Tensor, negate: bool = True, add_dim: bool = True):
    """Hartmann synthetic test function.

    Args:
        x, (B, N, 6)
        negate, bool: whether to negate the function.

    Returns:
        y, (B, N, 1) if add_dim else (B, N)
    """
    hart6 = Hartmann(dim=6, negate=negate).to(x)
    y = hart6(x)
    return y.unsqueeze(-1) if add_dim else y


def ackley6D(x: torch.Tensor, negate: bool = True, add_dim: bool = True):
    """Ackley synthetic test function.

    Args:
        x, (B, N, 6)
        negate, bool: whether to negate the function.

    Returns:
        y, (B, N, 1) if add_dim else (B, N)
    """
    ack6D = Ackley(dim=6, negate=negate).to(x)
    y = ack6D(x)
    return y.unsqueeze(-1) if add_dim else y
