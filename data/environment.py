import torch
import numpy as np
from typing import Tuple, Union
from data.utils import sample_from_full_combination


def sample_random_pairs_from_Q(
    pair: torch.Tensor,
    pair_y: torch.Tensor,
    c: torch.Tensor,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """sample random query pairs from the candidate query pair set Q.

    Args:
        pair, (B, num_query_pair, 2 d_x): pair set.
        pair_y, (B, num_query_pair, 2): corresponding function values.
        c, (B, num_query_pair, 1): corresponding preference.
        num_samples, int: number of samples

    Returns:
        sample_pair, (B, num_samples, 2 d_x): sampled pairs.
        sample_pair_y, (B, num_samples, 2)
        sample_c, (B, num_samples, 1)
        idx, (num_samples): the indices of sampled pairs.
    """
    idx = np.random.choice(a=len(pair), size=(num_samples), replace=False)

    sample_pair = pair[:, idx]  # (B, num_samples, 2 d_x)
    sample_pair_y = pair_y[:, idx]  # (B, num_samples, 2)
    sample_c = c[:, idx]  # (B, num_samples, 1)
    return sample_pair, sample_pair_y, sample_c, idx


def gather_data_at_index(data: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather elements along the sequence dimension.

    Args:
        data, (B, N, d): batch of sequences, with d-dimensional elements.
        idx, (B, 1, 1): batch of indice to be gathered.

    Returns:
        ele, (B, 1, d): gathered elements.
    """
    _, _, d = data.shape
    tile_idx = idx.tile(1, 1, d)
    # data[b, idx[b].item()]
    ele = torch.gather(input=data, dim=1, index=tile_idx)
    return ele


def get_user_preference(
    pair_y: torch.Tensor, maximize: bool = True, p_noise: Union[float, None] = None
) -> torch.Tensor:
    """generates preference label given a pair of function values.

    Args:
        pairs_y, [..., 2]: function values for pairs.
        maximize, bool: maximize or minimize the utility function. default True.
        p_noise, float: observed Gaussian noise when deciding preference. Default as 0.1

    Returns:
        c, [..., 1]: the preference label.
    """
    p_noise = p_noise if p_noise is not None else 0.1
    pair_y_obs = pair_y + p_noise * torch.randn_like(pair_y, device=pair_y.device)
    c = (
        (pair_y_obs[..., 0] - pair_y_obs[..., 1] <= 0).float()
        if maximize
        else (pair_y_obs[..., 0] - pair_y_obs[..., 1] > 0).float()
    )
    return c.unsqueeze(-1)


def get_latent_value_ranking(
    y: torch.Tensor,
    pair_idx: torch.Tensor,
) -> torch.Tensor:
    """Replace the latent values with the ranking values.

    Args:
        y, (B, num_points, 1): latent values of all points.
        pair_idx, (num_pair, 2): the indices of points in Q.

    Returns:
        y, (B, num_points, 1): the latent values of points in Q are replaced by their rankings,
        which will serve as the reward signals during policy learning.
    """
    B = len(y)

    # a point might appear in multiple query pairs, thus we first find out M unique points in query pair set Q
    unique_pair_idx = np.unique(pair_idx.squeeze())
    M = len(unique_pair_idx)

    # their true latent values
    unique_y_Q = y[:, unique_pair_idx]  # (B, M, 1)

    # (B, M, 1), indices that sort y in acsending order. i.e., (y_{i,1} < y_{i,2} < ...)
    indices = torch.argsort(unique_y_Q, dim=1)

    # (1, 2, ..., M). NOTE 1 for the smallest latent value, vice versa.
    rank_array = torch.tile(torch.arange(M).reshape(1, M, 1), (B, 1, 1))

    rank_y = torch.zeros((B, M, 1), dtype=torch.int64)
    #  scatter rank value based on indices
    rank_y.scatter_(dim=1, index=indices, src=rank_array)
    rank_y = rank_y.to(torch.float32)  # NOTE
    y[:, unique_pair_idx] = rank_y
    return y


def generate_query_pair_set(
    X: torch.Tensor,
    y: torch.Tensor,
    num_total_points: int,
    n_random_pairs: Union[None, int],
    maximize: bool = True,
    p_noise: float = 0.0,
    rank_latent_value: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate the query pair set when given a set of points.

    Args:
        X, (B, num_points, d_x)
        y, (B, num_points, d_x)
        maximize, bool: whether to do maximization.
        p_noise, float: observed noise when deciding preference.
        rank_latent_value, bool: whether to replace the true latent value with its ranking in the pair set.

    Returns:
        pair_idx, (num_pair, 2): the indices of sampled pairs; shared between dataset in batches.
        pair, (B, num_pair, 2 d_x): the location of pairs
        pair_y, (B, num_pair, 2): the corresponding function values of pairs
        c, (B, num_pair, 1): the preference label of pairs.
    """
    # sample the indices of sampled pairs.
    # NOTE the pair set for a curve is fixed. Random seeds only control starting pairs.
    pair_idx = sample_from_full_combination(
        num_total_points=num_total_points,
        num_random_pairs=n_random_pairs,
        device=X.device,
    )

    # ablation study on reward model: substitute the true latent values with the ranking
    if rank_latent_value:
        y = get_latent_value_ranking(y=y, pair_idx=pair_idx)

    pair = X[:, pair_idx].float().flatten(start_dim=-2)
    pair_y = y[:, pair_idx].float().flatten(start_dim=-2)
    c = get_user_preference(pair_y=pair_y, maximize=maximize, p_noise=p_noise)
    return pair_idx, pair, pair_y, c
