import torch
import numpy as np
from data.environment import gather_data_at_index
from typing import Tuple, List
from policies.transformer import TransformerModel
from torchrl.modules import MaskedCategorical


def sample_from_masked_categorical(
    logits: torch.Tensor, mask: torch.Tensor, argmax: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        max_indices = masked_logits.argmax(dim=-1)  # (B)
        action = masked_logits.gather(1, max_indices.unsqueeze(-1)).squeeze(-1)
    else:
        # sample from the masked distribution
        action = m.sample()

    # for debugging: p(a) approx 1 / number_of_elements when the distribution is uniform
    log_prob = m.log_prob(action)

    return action.long().detach(), log_prob, entropy.detach()


def action(
    model: TransformerModel,
    context_pairs: torch.Tensor,
    context_preference: torch.Tensor,
    t: int,
    T: int,
    X_pending: torch.Tensor,
    pair_idx: torch.Tensor,
    mask: torch.Tensor,
    argmax: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """select action.

    Args:
        model, TransformerModel.
        context_pairs, (B, num_ctx_pair, 2 d_x): previous queries.
        context_preference, (B, num_ctx_pair, 1): corresponding true preference labels.
        t, int: current time step.
        T, int: time budget.
        X_pending, (B, num_query_points, d_x): all the query points.
        pair_idx, (B, num_query_pairs, 2): indices to retrieve the candidate query pair set.
        mask, (B, num_query_pairs, 1): mask to indicate previous queries.
        argmax, bool: whether to return the query pair with highest acq_values if True or sample from the entire acq_value distribution. Default False.

    Returns:
        acquisition values, (B, num_query_pairs, 1): acq_values for all pairs in candidate query pair set Q.
        next_pair_idx, (B,): the indices of the next query pair in Q.
        log_prob, (B, ): log-prob of the next query pair.
        entropy, (B,): categorical distribution's entropy value.
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
    return (
        acq_values.detach(),  # NOTE
        next_pair_idx,
        log_prob,
        entropy,
    )


def finish_episode(
    entropys: List[torch.Tensor],
    rewards: List[torch.Tensor],
    log_probs: List[torch.Tensor],
    discount_factor: float = 0.98,
    eps: float = np.finfo(np.float32).eps.item(),
):
    """Compute the policy learning loss.

    Args:
        entropys, list of Tensor (B, ): policy head's entropy value at each time step
        rewards, list of Tensor (B, ): reward at each time step
        log_probs, list of Tensor (B, ): log-prob corresponding to query pair selected at each time step
        discount_factor, float: control the importance of future rewards.
        eps: to avoid divide-by-zero issue.

    Returns:
        mean loss, (1)
        mean entropy, (1)
    """
    entropy = torch.stack(entropys, dim=-1)  # (B, H)
    reward = torch.stack(rewards, dim=-1)  # (B, H)
    log_prob = torch.stack(log_probs, dim=-1)  # (B, H)

    H = reward.shape[1]
    discounts = discount_factor ** torch.arange(H)
    returns = discounts * reward  # gamma^{t-1}r_t
    returns = (returns - returns.mean(dim=-1, keepdim=True)) / (
        returns.std(dim=-1, keepdim=True) + eps
    )  # standardize along trajectories

    # sum up the negative expected returns
    loss = (-returns * log_prob).sum(dim=-1)

    loss, entropy = torch.mean(loss), entropy.mean()
    return loss, entropy


def get_reward(
    context_pairs_y: torch.Tensor,
    acq_values: torch.Tensor,
    pair_y: torch.Tensor,
    regret_option: str = "simple_regret",
) -> torch.Tensor:
    """Get reward.

    Args:
        context_pairs_y, (B, num_ctx, 2): function values for current observed query pairs.
        acq_values, (B, num_query_pair, 1): predicted acquisition values for all candidate query pairs.
        pair_y, (B, num_query_pair, 2): function values for all candidate query pairs.
        reward_option, list in ["simple_regret", "inference_regret"].

    Returns:
        reward, (B, ):
    """
    if regret_option == "simple_regret":
        best_y = torch.max(
            context_pairs_y.flatten(start_dim=1), dim=-1
        ).values.squeeze()  # (B, )
        reward = best_y
    elif regret_option == "inference_regret":
        infer_optima_idx = torch.max(acq_values, dim=1).indices[:, None, :]
        infer_best_y = torch.max(
            gather_data_at_index(data=pair_y, idx=infer_optima_idx).flatten(
                start_dim=1
            ),  # (B, 1, 2) -> (B, 2)
            dim=-1,
        ).values  # (B, )
        opt_y = torch.max(
            pair_y.flatten(start_dim=1), dim=-1
        ).values  # NOTE optimum over the query pair set
        reward = infer_best_y - opt_y

    # NOTE no gradient from reward
    reward = reward.detach()
    return reward
