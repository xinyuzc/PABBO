import argparse
import torch
import os.path as osp
import os
from tqdm import tqdm
import random
from policies.transformer import TransformerModel
from policies.policy import *
from data.data import *


def tnp(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return data


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def evaluate(
    model_name="PABBO_RBF",
    result_dir="evaluation/results",
    device="cpu",
    P_NOISE=0.0,
    NUM_QUERY_POINTS=256,
    T=30,
):
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)
    print("Loading initialization")
    initializations = torch.load("datasets/GP2D_initial_pairs.pt")
    NUM_SEEDS = len(initializations)
    B = len(initializations["0"]["context_c"])
    d_x = 2
    print(f"Evaluate on {B} GP samples with {NUM_SEEDS} random seeds.")

    model = TransformerModel(
        d_x=2,
        d_f=1,
        d_model=64,
        nhead=4,
        dropout=0.0,
        n_layers=6,
        dim_feedforward=128,
        emb_depth=3,
    )
    if model_name == "PABBO_RBF":
        ckpt = torch.load(
            "evaluation/trained_models/PABBO_GP2D_RBF/ckpt.tar", map_location="cpu"
        )
    else:
        ckpt = torch.load(
            "evaluation/trained_models/PABBO_GP2D_Mixture/ckpt.tar", map_location="cpu"
        )
    model.load_state_dict(ckpt["model"])
    model.eval()
    SIMPLE_REGRET, STEP_SIMPLE_REGRET = [], []

    for seed in range(NUM_SEEDS):
        set_all_seeds(seed)
        seed_initializations = initializations[str(seed)]
        orn_X_bound = seed_initializations[
            "orn_X_bound"
        ]  # input bounds for the original problems
        tar_X_bound = seed_initializations[
            "tar_X_bound"
        ]  # PAABO is trained on GP samples with each dimension's input range as (-1, 1), thus that we need to map the test inputs to (-1, 1) if they are not in this range

        seed_simple_regret, seed_step_simple_regret = [], []
        for b in tqdm(range(B), f"Evaluating on {B} GP samples with seed {seed}"):
            context_pairs = seed_initializations["context_pairs"][[b]].to(device)
            print(f"Start optimization with initial pair: {context_pairs}")
            context_pairs_y = seed_initializations["context_pairs_y"][[b]].to(device)
            context_c = seed_initializations["context_c"][[b]].to(device)

            # create GP curve sampler
            sampler = SimpleGPSampler(
                kernel_function=globals()[
                    seed_initializations["sampler_kwargs"][b]["kernel_function"]
                ],
                mean=seed_initializations["sampler_kwargs"][b]["mean"],
                jitter=seed_initializations["sampler_kwargs"][b]["jitter"],
            )
            utility = OptimizationFunction(
                sampler=sampler, **seed_initializations["function_kwargs"][b]
            )
            sampler = UtilitySampler(
                d_x=d_x,
                utility_function=utility,
                Xopt=seed_initializations["Xopt"][b].to(device),
                yopt=seed_initializations["yopt"][b].to(device),
            )
            # sample query set
            X, y, Xopt, yopt = sampler.sample(
                batch_size=1,
                num_total_points=NUM_QUERY_POINTS,
                x_range=orn_X_bound.cpu().numpy().tolist(),
            )

            # record regrets
            max_y = torch.max(context_pairs_y.flatten(start_dim=1), dim=-1).values
            batch_simple_regret = [yopt.view(1) - max_y]
            batch_step_simple_regret = [
                yopt.view(1) - context_pairs_y[:, -1].max(dim=-1).values
            ]

            # map X to [-1, 1], on which PABBO was trained.
            X = scale_from_domain_1_to_domain_2(
                x=X, bound1=orn_X_bound, bound2=tar_X_bound
            )
            Xopt = scale_from_domain_1_to_domain_2(
                Xopt, bound1=orn_X_bound, bound2=tar_X_bound
            )

            # query pair set
            query_pair_indices = torch.combinations(torch.arange(0, NUM_QUERY_POINTS))

            # generate data
            pair = X[:, query_pair_indices].float().to(device).flatten(start_dim=-2)
            pair_y = y[:, query_pair_indices].float().to(device).flatten(start_dim=-2)
            c = get_user_preference(pair_y=pair_y, maximize=True, p_noise=P_NOISE)

            # mask to remove query pair that we have seen
            mask = torch.ones((pair.shape[0], pair.shape[1], 1)).bool()
            t = 1
            # optimization loop
            with torch.no_grad():
                while t <= T:  # one trajectory
                    # acquisition task: next query pair
                    _, next_pair_idx, _, _ = action(
                        model=model,
                        context_pairs=context_pairs,
                        context_preference=context_c,
                        t=t,
                        T=T,
                        X_pending=X,
                        pair_idx=query_pair_indices,
                        mask=mask,
                    )
                    # update optimization history
                    next_pair_idx = next_pair_idx[:, None, None]
                    mask.scatter_(
                        dim=1, index=next_pair_idx, src=torch.zeros_like(mask)
                    )
                    next_pair = gather_data_at_index(data=pair, idx=next_pair_idx)
                    next_pair_y = gather_data_at_index(data=pair_y, idx=next_pair_idx)
                    next_c = gather_data_at_index(data=c, idx=next_pair_idx)
                    context_pairs = torch.cat((context_pairs, next_pair), dim=1)
                    context_c = torch.cat((context_c, next_c), dim=1)
                    context_pairs_y = torch.cat((context_pairs_y, next_pair_y), dim=1)
                    t += 1

                    # record regrets
                    max_y = torch.max(
                        context_pairs_y.flatten(start_dim=1), dim=-1
                    ).values
                    batch_simple_regret.append(yopt.view(1) - max_y)
                    batch_step_simple_regret.append(
                        yopt.view(1) - context_pairs_y[:, -1].max(dim=-1).values
                    )

            seed_simple_regret.append(
                torch.stack(batch_simple_regret, dim=-1)
            )  # (1, T)
            seed_step_simple_regret.append(
                torch.stack(batch_step_simple_regret, dim=-1)
            )  # (1, T)

        seed_simple_regret = torch.cat(seed_simple_regret, dim=0).mean(dim=0)  # (T)
        seed_step_simple_regret = torch.cat(seed_step_simple_regret, dim=0).mean(
            dim=0
        )  # (T)
        SIMPLE_REGRET.append(seed_simple_regret)
        STEP_SIMPLE_REGRET.append(seed_step_simple_regret)

    SIMPLE_REGRET = torch.stack(SIMPLE_REGRET, dim=0).cpu().numpy()  # (num_seeds, T)
    STEP_SIMPLE_REGRET = torch.stack(STEP_SIMPLE_REGRET, dim=0).cpu().numpy()
    CUMULATIVE_REGRET = np.cumsum(STEP_SIMPLE_REGRET, axis=-1)

    final_simple_regret_mean, final_simple_regret_std = (
        SIMPLE_REGRET[:, -1].mean().item(),
        SIMPLE_REGRET[:, -1].std().item(),
    )

    print(
        f"final simple regret mean: {final_simple_regret_mean: .4f}, final_simple_regret_std: {final_simple_regret_std}"
    )
    final_cumulative_regret_mean, final_cumulative_regret_std = (
        CUMULATIVE_REGRET[:, -1].mean().item(),
        CUMULATIVE_REGRET[:, -1].std().item(),
    )

    print(
        f"final cumulative regret mean: {final_cumulative_regret_mean: .4f}, final_cumulative_regret_std: {final_cumulative_regret_std}"
    )

    save_dir = osp.join(result_dir, model_name)
    if not osp.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    torch.save(SIMPLE_REGRET, f"{save_dir}/SIMPLE_REGRET.pt")
    torch.save(STEP_SIMPLE_REGRET, f"{save_dir}/STEP_SIMPLE_REGRET.pt")
    torch.save(CUMULATIVE_REGRET, f"{save_dir}/CUMULATIVE_REGRET.pt")

    return SIMPLE_REGRET, STEP_SIMPLE_REGRET, CUMULATIVE_REGRET


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mn", "--model_name", choices=["PABBO_RBF", "PABBO_Mixture"])
    args = parser.parse_args()
    evaluate(model_name=args.model_name)  # change the model name
