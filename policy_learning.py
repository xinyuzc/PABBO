import torch
from torch import Tensor
import numpy as np
from data.environment import gather_data_at_index
from typing import Tuple, List
from policies.transformer import TransformerModel
from torchrl.modules import MaskedCategorical


def sample_from_masked_categorical(
    logits: Tensor, mask: Tensor, argmax: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    """sample from the masked Cateogrical distribution.

    Args:
        logits, (B, N, 1): logit values for all elements.
        mask, (B, N, 1): whether to mask out elements at certain indices.
        argmax, bool: wheather to take the argmax or sample from the entire distribution.

    Returns:
        action, (B, ): LongTensor, the indice of selected element.
        log_prob, (B, ): the log-prob of selected element.
        entropy, (B, ): reflects how certain the policy is.
    """
    # the last column should be the unnormalized probs
    logits, mask = logits.squeeze(-1), mask.squeeze(-1)

    # NOTE the total number of elements remain unchanged since we are masking them from being sampled rather than kicking them out.
    # when using parallel PABBO,
    # this ensures that acquisition values from all query set are comparable, no matter how many elements have been masked in any of them.

    m = MaskedCategorical(logits=logits, mask=mask)
    entropy = m.entropy()
    # choose one with highest logit values
    if argmax:
        # manually fill masked logits with negative inf values
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        action = masked_logits.argmax(dim=-1)  # (B)
    else:
        # sample from the masked distribution
        action = m.sample()

    # for debugging: p(a) approx 1 / number_of_elements when the distribution is uniform
    log_prob = m.log_prob(action)

    return action.long().detach(), log_prob, entropy.detach()


def action(
    model: TransformerModel,
    context_pairs: Tensor,
    context_preference: Tensor,
    t: int,
    T: int,
    X_pending: Tensor,
    pair_idx: Tensor,
    mask: Tensor,
    argmax: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """choose the next query pair from the candidate query pair set Q.

    Args:
        model, TransformerModel: transformer model.
        context_pairs, [B, num_ctx_pair, 2 * d_x]: context pairs, where each pair is a concatenation of two query points.
        context_preference, [B, num_ctx_pair, 1]: user preference for context pairs.
        t, int: current time step.
        T, int: total time steps.
        X_pending, [B, num_query_points, d_x]: query points in the candidate query pair set Q.
        pair_idx, [B, num_query_pairs, 2]: indices of query points in the candidate query pair set Q.
        mask, [B, num_query_pairs, 1]: whether to mask out elements at certain indices.
        argmax, bool: whether to take the argmax or sample from the entire distribution.

    Returns:
        acquisition values, [B, num_query_pairs, 1]: predicted acquisition values for all candidate query pairs.
        next_pair_idx, [B]: the index of the selected query pair.
        log_prob, [B]: the log-prob of selected query pair.
        entropy, [B]: reflects how certain the policy is.
    """
    # predict acquisition values for all query pairs
    acq_values = model(
        query_src=context_pairs,
        c_src=context_preference,
        eval_pos=X_pending,
        acquire=True,
        acquire_comb_idx=pair_idx.long(),  # NOTE dtype to long for indexing.
        t=t,
        T=T,
    )[0]

    # create a masked categorical distribution with predicted acq_values and sample the next query pair
    next_pair_idx, log_prob, entropy = sample_from_masked_categorical(
        logits=acq_values, mask=mask, argmax=argmax
    )

    # only need gradient from the log_prob
    return (
        acq_values.detach(),
        next_pair_idx.detach(),
        log_prob,
        entropy.detach(),
    )


def finish_episode(
    entropys: List[Tensor],
    rewards: List[Tensor],
    log_probs: List[Tensor],
    discount_factor: float = 0.98,
    eps: float = np.finfo(np.float32).eps.item(),
    sum_over_tra: bool = True,
):
    """Compute the policy learning loss.

    Args:
        entropys, H x [B]: entropy at each time step
        rewards, H x [B]: reward at each time step
        log_probs, H x [B]: log-prob of selected action at each time step
        discount_factor, float: discount factor for rewards at later time steps.
        eps, float: small value to avoid division by zero.
        sum_over_tra, bool: whether to sum losses over the trajectory length or take the mean.
        NOTE when using varying lengths of trajectories during training, it is recommended to take the mean.

    Returns:
        mean loss, [1]: mean loss across batch dim.
        mean entropy, [1]: mean entropy across batch dim.
    """
    entropy = torch.stack(entropys, dim=-1)  # (B, H)
    reward = torch.stack(rewards, dim=-1)  # (B, H)
    log_prob = torch.stack(log_probs, dim=-1)  # (B, H)

    B, H = reward.shape
    discounts = discount_factor ** torch.arange(H)
    
    # returns[b, t] = discount_factor ** (t - 1) * reward[b, t]
    returns = discounts * reward

    # standardize the returns: [B, H]
    returns = (returns - returns.mean(dim=-1, keepdim=True)) / (
        returns.std(dim=-1, keepdim=True) + eps
    )

    # sum up the negative expected returns
    # TODO should take mean when using varying lengths of trajectories
    loss = -returns * log_prob  # [B, H]
    loss = loss.sum(dim=-1) if sum_over_tra else loss.mean(dim=-1)  # [B]

    loss_mean, entropy_mean = torch.mean(loss), entropy.mean()
    return loss_mean, entropy_mean


def get_reward(
    context_pairs_y: Tensor,
    acq_values: Tensor,
    pair_y: Tensor,
    regret_option: str = "simple_regret",
    maximize: bool = True,  # dummy - default maximize!
) -> Tensor:
    """Compute the reward for each trajectory in the batch.

    Args:
        context_pairs_y, [B, num_ctx_pair, 2]: utility values for context pairs.
        acq_values, [B, num_query_pair, 1]: predicted acquisition values for all candidate query pairs.
        pair_y, [B, num_query_pair, 2]: utility values for all candidate query pairs.
        regret_option, str: the type of regret to compute.
            - "simple_regret": reward is the maximum utility value of context pairs.
            - "inference_regret": reward is the difference between the maximum utility value of context pairs and the maximum utility value of the selected query pair.

    Returns:
        reward, [B]: reward for each trajectory in the batch.
    """
    assert regret_option in [
        "simple_regret",
        "inference_regret",
    ], "regret_option should be either 'simple_regret' or 'inference_regret'."

    B = context_pairs_y.shape[0]
    if regret_option == "simple_regret":
        # [B, num_ctx_pair, 2] -> [B, num_ctx_pair * 2]
        context_pairs_y_flat = context_pairs_y.flatten(start_dim=1)

        # [B]
        reward = torch.max(context_pairs_y_flat, dim=-1).values
    elif regret_option == "inference_regret":
        # get the indices of the highest acquisition function value: [B, 1, 1]
        max_acq_idx = torch.max(acq_values, dim=1, keepdim=True).indices

        # get the pair with highest acq value: [B, 1, 2]
        max_acq_pair_y = gather_data_at_index(
            data=pair_y,  # [B, num_query_pair, 2]
            idx=max_acq_idx,  # [B, 1, 1]
        )
        max_acq_pair_y_flat = max_acq_pair_y.flatten(start_dim=1)  # [B, 2]

        # get the inferred optimal utility value: [B]
        y_best_acq = torch.max(
            max_acq_pair_y_flat,
            dim=-1,
        ).values

        # get the true maximum utility value from the sampled query pairs: [B]
        pair_y_flat = pair_y.flatten(start_dim=1)
        y_opt = torch.max(pair_y_flat, dim=-1).values

        reward = y_best_acq - y_opt

    assert reward.shape == (B,), f"reward should be of shape [{B}]."

    # NOTE no gradient from reward
    reward = reward.detach()
    return reward
