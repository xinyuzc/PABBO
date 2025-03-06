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
        sample_pair_y, (B, num_samples, 2): associated utility values.
        sample_c, (B, num_samples, 1): associated preference.
        idx, (num_samples): the indices of sampled pair.
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


def get_ranking_reward(
    y: torch.Tensor,
    pair_idx: torch.Tensor,
) -> torch.Tensor:
    """Get reward based on the ranking of utility values: the larger the utility value, the higher the reward.
    For instance, y = [[[0.1], [0.3], [0.4]]], pair_idx=[[1, 2]], then return y=[[0.1], [0], [1]].

    Args:
        y, (B, num_points, 1): utility values at all datapoints.
        pair_idx, (num_pair, 2): the point indices for sampled pairs.

    Returns:
        y, (B, num_points, 1): utility values at all datapoints, with those appear in sampled pairs replaced by reward obtained from rankings.
        NOTE that those are not sampled keep unchanged, as it won't be accessed.
    """
    B = len(y)

    # first find M unique points in query pair set Q as a point might appear in multiple pairs
    unique_pair_idx = np.unique(pair_idx.squeeze())
    M = len(unique_pair_idx)

    # the true utility values
    unique_y_Q = y[:, unique_pair_idx]  # (B, M, 1)

    # (B, M, 1), indices that sort the unique utility values in ascending order. i.e., (y_{i,1} < y_{i,2} < ...)
    indices = torch.argsort(unique_y_Q, dim=1)

    # from 1 to M: reward=1 for the smallest utility value.
    reward_array = torch.tile(torch.arange(M).reshape(1, M, 1), (B, 1, 1))

    # scatter rewards according to the sorted indices
    reward_y = torch.zeros((B, M, 1), dtype=torch.int64)
    reward_y.scatter_(dim=1, index=indices, src=reward_array)
    reward_y = reward_y.to(torch.float32)  # NOTE

    # replace utility values of points that appear in sampled pairs with the reward
    y[:, unique_pair_idx] = reward_y
    return y


def generate_query_pair_set(
    X: torch.Tensor,
    y: torch.Tensor,
    num_total_points: int,
    n_random_pairs: Union[None, int] = None,
    maximize: bool = True,
    p_noise: float = 0.0,
    ranking_reward: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a set of pairs with preference labels by combining given datapoints.

    Args:
        X, (B, num_points, d_x): a batch of datapoints.
        y, (B, num_points, 1): associated utility values.
        maximize, bool: the point with larger utility value is preferred if true.
        p_noise, float: observed noise when giving preference.
        rank_latent_value, bool: whether to replace the true latent value with its ranking in the pair set.

    Returns:
        pair_idx, (num_pair, 2): the point indices of pairs. Shared between dataset in batches.
        pair, (B, num_pair, 2 * d_x): the pairs.
        pair_y, (B, num_pair, 2): the associated utility values of pairs
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
    if ranking_reward:
        y = get_ranking_reward(y=y, pair_idx=pair_idx)

    pair = X[:, pair_idx].float().flatten(start_dim=-2)
    pair_y = y[:, pair_idx].float().flatten(start_dim=-2)
    c = get_user_preference(pair_y=pair_y, maximize=maximize, p_noise=p_noise)
    return pair_idx, pair, pair_y, c
