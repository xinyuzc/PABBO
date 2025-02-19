import torch
import math


def rbf(
    x1: torch.Tensor,
    x2: torch.Tensor,
    length_scale: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """rbf kernel for N samples of d_x dimension, return the covariance matrix of shape [N, M]
    Args:
        x1, [N, D]: input 1.
        x2, [M, D]: input 2.
        length_scale, [d_x]: length scales over each input dimension.
        std, [1]: function scale.
    Returns:
        cov, [N, M]: covariance matrix.
    """
    dist = (x1.unsqueeze(-2) - x2.unsqueeze(-3)) / length_scale
    cov = std.pow(2) * torch.exp(-0.5 * dist.pow(2).sum(-1))
    return cov


def matern52(
    x1: torch.Tensor,
    x2: torch.Tensor,
    length_scale: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """matern52 kernel for N samples of d_x dimension, return the covariance matrix of shape [N, M]
    Args:
        x1, [N, D]: input 1.
        x2, [M, D]: input 2.
        length_scale, [d_x]: length scales over each input dimension.
        std, [1]: function scale.
    Returns:
        cov, [N, M]: covariance matrix.
    """
    dist = torch.norm((x1.unsqueeze(-2) - x2.unsqueeze(-3)) / length_scale, dim=-1)
    cov = (
        std.pow(2)
        * (1 + math.sqrt(5.0) * dist + 5.0 * dist.pow(2) / 3.0)
        * torch.exp(-math.sqrt(5.0) * dist)
    )
    return cov


def matern32(
    x1: torch.Tensor,
    x2: torch.Tensor,
    length_scale: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """matern32 kernel for N samples of d_x dimension, return the covariance matrix of shape [N, M]
    Args:
        x1, [N, D]: input 1.
        x2, [M, D]: input 2.
        length_scale, [d_x]: length scales over each input dimension.
        std, [1]: function scale.
    Returns:
        cov, [N, M]: covariance matrix.
    """
    dist = torch.norm((x1.unsqueeze(-2) - x2.unsqueeze(-3)) / length_scale, dim=-1)
    cov = std.pow(2) * (1 + math.sqrt(3.0) * dist) * torch.exp(-math.sqrt(3.0) * dist)
    return cov


def matern12(
    x1: torch.Tensor,
    x2: torch.Tensor,
    length_scale: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """matern32 kernel for N samples of d_x dimension, return the covariance matrix of shape [N, M]
    Args:
        x1, [N, D]: input 1.
        x2, [M, D]: input 2.
        length_scale, [d_x]: length scales over each input dimension.
        std, [1]: function scale.
    Returns:
        cov, [N, M]: covariance matrix.
    """
    dist = torch.norm((x1.unsqueeze(-2) - x2.unsqueeze(-3)) / length_scale, dim=-1)
    cov = std.pow(2) * torch.exp(-1.0 * dist)
    return cov
