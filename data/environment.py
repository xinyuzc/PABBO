import torch
from torch import Tensor
import numpy as np
from typing import Tuple, Union, Optional
from data.utils import sample_from_full_combination


def sample_random_pairs_from_Q(
    pair: Tensor,
    pair_y: Tensor,
    c: Tensor,
    num_samples: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """sample random pairs from the candidate query pair set Q.

    Args:
        pair, [B, num_pairs, 2 * d_x]: candidate query pair set Q.
        pair_y, [B, num_pairs, 2]: associated utility values.
        c, [B, num_pairs, 1]: associated preference.
        num_samples, scalar: number of samples.

    Returns:
        sample_pair, [B, num_samples, 2 * d_x]: sampled pairs.
        sample_pair_y, [B, num_samples, 2]: associated utility values.
        sample_c, [B, num_samples, 1]: preference for sampled pairs.
        idx, [num_samples]: indices of sampled pairs - shared across batch.
    """
    idx = np.random.choice(a=pair.shape[1], size=(num_samples), replace=False)

    sample_pair = pair[:, idx]
    sample_pair_y = pair_y[:, idx]
    sample_c = c[:, idx]
    return sample_pair, sample_pair_y, sample_c, idx


def gather_data_at_index(data: Tensor, idx: Tensor) -> Tensor:
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
    pair_y: Tensor, maximize: bool = True, p_noise: Optional[float] = None
) -> Tensor:
    """Generate user preference for sampled pairs based on the utility values.

    Args:
        pair_y, [B, num_pairs, 2]: utility values for sampled pairs.
        maximize, bool: the point with larger utility value is preferred if true.
        p_noise, float: observed noise when giving preference.

    Returns:
        c, [B, num_pairs, 1]: preference for sampled pairs.
        Preference is 0 if element 0 is preferred, and 1 if element 1 is preferred.
    """
    # add gaussian noise to the utility values
    p_noise = p_noise if p_noise is not None else 0.1
    pair_y_obs = pair_y + p_noise * torch.randn_like(pair_y, device=pair_y.device)

    # [B, num_pairs] -> [B, num_pairs, 1]
    c = (
        (pair_y_obs[..., 0] - pair_y_obs[..., 1] <= 0).float()
        if maximize
        else (pair_y_obs[..., 0] - pair_y_obs[..., 1] > 0).float()
    )
    return c.unsqueeze(-1)


def get_ranking_reward(
    y: Tensor,
    pair_idx: Tensor,
    maximize: bool = True,  # dummy - default maximize!
) -> Tensor:
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

    # from 1 to M: reward = 1 for the smallest utility value, 2 for the second smallest, and so on.
    # [1, M, 1] -> [B, M, 1]
    reward_array = torch.arange(M).unsqueeze(0).unsqueeze(2).to(y)
    reward_array = reward_array.expand(B, M, 1)

    # scatter rewards according to the sorted indices
    reward_y = torch.zeros((B, M, 1), dtype=torch.int64)
    reward_y.scatter_(dim=1, index=indices, src=reward_array)
    reward_y = reward_y.to(torch.float32)  # NOTE

    # replace utility values of points that appear in sampled pairs with the reward
    y[:, unique_pair_idx] = reward_y
    return y


def generate_pair_set(
    X: Tensor,  # [B, num_datapoints, d_x]
    y: Tensor,  # [B, num_datapoints, 1]
    num_total_points: int,
    n_random_pairs: Union[None, int] = None,
    maximize: bool = True,
    p_noise: float = 0.0,
    ranking_reward: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate a set of pairs from the batch of datapoints.

    Args:
        X, [B, num_points, d_x]: a batch of datapoints.
        y, [B, num_points, 1]: associated utility values.
        maximize, bool: the point with larger utility value is preferred if true.
        p_noise, float: observed noise when giving preference.
        ranking_reward, bool: whether to replace the true latent value with its ranking in the pair set.

    Returns:
        pair_idx, [num_pairs, 2]: indices of sampled pairs.
        pair, [B, num_pairs, 2 * d_x]: sampled pairs.
        pair_y, [B, num_pairs, 2]: associated utility values.
        c, [B, num_pairs, 1]: preference for sampled pairs.
    """
    # sample indices of pairs from the full combination of datapoints
    pair_idx = sample_from_full_combination(
        num_total_points=num_total_points,
        num_random_pairs=n_random_pairs,
        device=X.device,
    )

    # ablation study on reward model: replace the true latent utility values with the ranking in the query pair set
    if ranking_reward:
        y = get_ranking_reward(y=y, pair_idx=pair_idx)

    # gather the sampled pairs and their associated utility values: [B, num_pairs, 2 * d_x] and [B, num_pairs, 2]
    pair = X[:, pair_idx].float().flatten(start_dim=-2)
    pair_y = y[:, pair_idx].float().flatten(start_dim=-2)

    # get the preference for sampled pairs
    c = get_user_preference(pair_y=pair_y, maximize=maximize, p_noise=p_noise)
    return pair_idx, pair, pair_y, c
