import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
import numpy as np


def preference_cls_loss(
    f: torch.Tensor, c: torch.Tensor, reduction="mean"
) -> torch.Tensor:
    """Compute the BCE loss between predicted preference and the ground truch. NOTE maximization.

    Args:
       f, (B, 2N, 1): predicted function values.
       c, (B, N, 1): ground truth preference label c_i for function value pair (f_2i, f_2i+1)
       reduction, str: reduction to apply to the output
    Returns:
        loss, (1)
    """
    # compute the probability of x_1 being preferred than x_0. NOTE c=1 if x_1 is preferred.
    r = f[:, 1::2] - f[:, 0::2]
    r_sigmoid = F.sigmoid(r)

    # compute the BCE loss between predicted and the ground truth
    loss = F.binary_cross_entropy(
        r_sigmoid.flatten(), c.float().flatten(), reduction=reduction
    )
    return loss.squeeze()


def accuracy(f: torch.Tensor, c: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Compute the accuracy between predicted preference and the ground truth. NOTE maximization.

    Args:
       f, (B, 2N, 1): predicted function values.
       c, (B, N, 1): ground truth preference label c_i for function value pair (f_2i, f_2i+1)
       reduce, bool: whether to average across all samples
    Returns:
        acc, (1) if reduce else (B, N)
    """
    # compute the probability of x_1 being preferred than x_0
    r = f[:, 1::2] - f[:, 0::2]
    r_sigmoid = F.sigmoid(r)

    # compute the accuracy
    acc = ((r_sigmoid > 0.5) == c).squeeze(-1).float()

    # whether to compute the average
    if reduce:
        acc = acc.mean()
    return acc


def kendalltau_correlation(
    pred: torch.Tensor, y: torch.Tensor, reduce: bool = True
) -> np.ndarray:
    """Compute the Kendall-tau correlation between predicted function values and the ground truth.

    Args:
        pred, (B, N, 1): predicted function values.
        y, (B, N, 1): the ground truth.
        reduce, bool: whether to average across all batches

    Returns:
        cor, (1) if reduce else (B,)"""
    B = pred.shape[0]
    # NOTE perform over each dataset as `kendalltau` function will flatten non-1D arrays.
    cor = np.stack(
        [
            kendalltau(
                pred[b].squeeze().detach().cpu().numpy(), y[b].detach().cpu().numpy()
            ).correlation
            for b in range(B)
        ]
    )

    if reduce:
        cor = np.mean(cor)
    return cor
