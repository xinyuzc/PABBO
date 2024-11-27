import torch
from policies.transformer import TransformerModel
from typing import Tuple
from torchrl.modules import MaskedCategorical


def sample_action(
    logits: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """sample action from a masked Categorical distribution.
    Args:
        logits, [B, num_query_pairs, 1]: acquisition function values for all query pairs.
        mask, [B, num_query_pairs, 1]: a bool mask to indicate whether query pairs have been selected or not.

    Returns:
        action, [B]: the index of selected pair.
        log_prob, [B]: the associated log-probability.
        entropy, [B]: entropy of the Categorical distribution.
    """
    logits, mask = logits.squeeze(-1), mask.squeeze(-1)
    m = MaskedCategorical(logits=logits, mask=mask)
    entropy = m.entropy()
    action = m.sample()
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """select next query pair at step t by sampling from the Categorical distribution-based policy.

    Args:
        context_pairs, [B, N, 2 * d_x]: pairs of x observed so far.
        context_preference [B, N, 2 * d_x]: associated preference labels.
        t, scalar: optimization step.
        T, scalar: budget.
        X_pending, [B, S, d_x]: query points.
        pair_idx, [B, M, 2]: indices of query pairs.
        mask: [B, M, 1]: mask for the query pairs to record which have been queried.

    Returns:
        acquisition values, [B, M, 1]: acquisition function values for all query pairs.
        next_pair_idx, [B,]: the index of selected next query pair.
        log_prob, [B]: the associated log-probability.
        entropy, [B]: entropy of the Categorical distribution.
    """
    acquisition_values = model(
        query_src=context_pairs,
        c_src=context_preference,
        eval_pos=X_pending,
        acquire=True,
        acquire_comb_idx=pair_idx,
        t=t,
        T=T,
    )[0]
    next_pair_idx, log_prob, entropy = sample_action(
        logits=acquisition_values, mask=mask
    )
    return (
        acquisition_values.detach(),
        next_pair_idx,
        log_prob,
        entropy,
    )
