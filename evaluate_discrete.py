import time
from wandb_wrapper import init as wandb_init
from omegaconf import DictConfig
import torch
import numpy as np
import os.path as osp
import os
import hydra
import wandb
from tqdm import tqdm
from utils.paths import DATASETS_PATH, RESULT_PATH
from policies.transformer import TransformerModel
from data.utils import set_all_seeds
from data.kernel import *
from data.utils import scale_from_domain_1_to_domain_2
from data.environment import generate_pair_set
from data.evaluation import *
from policy_learning import *
from utils.losses import kendalltau_correlation
from utils.plot import *
import logging


@hydra.main(version_base=None, config_path="configs")
def main(config: DictConfig):
    if config.experiment.wandb:
        wandb_init(config=config, **config.wandb)

    torch.set_default_dtype(torch.float32)
    torch.set_default_device(config.experiment.device)

    # load trained model
    model = TransformerModel(**config.model)
    root = osp.join(
        hydra.utils.get_original_cwd(),
        RESULT_PATH,
        config.experiment.model,
        config.experiment.expid,
    )
    if not osp.exists(root + "/ckpt.tar"):
        raise ValueError(f"Invalid path {root}.")
    ckpt = torch.load(
        os.path.join(root, "ckpt.tar"), map_location=config.experiment.device
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    res = evaluate(config, model)

    if config.experiment.override:
        save_dir = osp.join(
            hydra.utils.get_original_cwd(),
            RESULT_PATH,
            "evaluation",
            config.data.name,
            config.experiment.model,
            config.experiment.expid,
        )

        if not osp.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        print(f"Saving results in f{str(save_dir)}...")
        torch.save(
            res["simple_regret"],
            f"{save_dir}/SIMPLE_REGRET.pt",
        )
        torch.save(
            res["immediate_regret"],
            f"{save_dir}/IMMEDIATE_REGRET.pt",
        )
        torch.save(
            res["cumulative_regret"],
            f"{save_dir}/CUMULATIVE_REGRET.pt",
        )
        torch.save(res["entropy"], f"{save_dir}/ENTROPY.pt")
        torch.save(res["kt_cor"], f"{save_dir}/KT_COR.pt")
        torch.save(
            res["cumulative_time"],
            f"{save_dir}/CUMULATIVE_TIME.pt",
        )
        torch.save(
            res["inference_regret"],
            f"{save_dir}/INFERENCE_REGRET.pt",
        )

    # log to W&B
    if config.experiment.wandb:
        wandb.log(
            {
                f"simple_regret_vs_step_plot": wandb.plot.line_series(
                    xs=[
                        *range(res["simple_regret"].shape[-1])
                    ],  # list of x values: steps
                    ys=[
                        res["simple_regret"].mean(dim=1)[i]
                        for i in range(res["simple_regret"].shape[0])
                    ],  # list of y values: simple regret for each seed
                    keys=[
                        f"seed {i}" for i in range(res["simple_regret"].shape[0])
                    ],  # line labels: seed
                    title=f"Simple Regret vs Step Plot",
                    xname="Step",
                ),
                f"immediate_regret_vs_step_plot": wandb.plot.line_series(
                    xs=[
                        *range(res["immediate_regret"].shape[-1])
                    ],  # list of x values: steps
                    ys=[
                        res["immediate_regret"].mean(dim=1)[i]
                        for i in range(res["immediate_regret"].shape[0])
                    ],  # list of y values: immediate regret for each seed
                    keys=[
                        f"seed {i}" for i in range(res["immediate_regret"].shape[0])
                    ],  # line labels: seed
                    title=f"Immediate Regret vs Step Plot",
                    xname="Step",
                ),
                f"entropy_vs_step_plot": wandb.plot.line_series(
                    xs=[*range(res["entropy"].shape[-1])],  # list of x values: steps
                    ys=[
                        res["entropy"].mean(dim=1)[i]
                        for i in range(res["entropy"].shape[0])
                    ],  # list of y values: simple regret for each seed
                    keys=[
                        f"seed {i}" for i in range(res["entropy"].shape[0])
                    ],  # line labels: seed
                    title=f"Entropy vs Step Plot",
                    xname="Step",
                ),
                f"inference_regret_vs_step_plot": wandb.plot.line_series(
                    xs=[
                        *range(res["inference_regret"].shape[-1])
                    ],  # list of x values: steps
                    ys=[
                        res["inference_regret"].mean(dim=1)[i]
                        for i in range(res["inference_regret"].shape[0])
                    ],  # list of y values: simple regret for each seed
                    keys=[
                        f"seed {i}" for i in range(res["inference_regret"].shape[0])
                    ],  # line labels: seed
                    title=f"Inference Regret vs Step Plot",
                    xname="Step",
                ),
                f"kt_cor_vs_step_plot": wandb.plot.line_series(
                    xs=[*range(res["kt_cor"].shape[-1])],  # list of x values: steps
                    ys=[
                        res["kt_cor"].mean(dim=1)[i]
                        for i in range(res["kt_cor"].shape[0])
                    ],  # list of y values: simple regret for each seed
                    keys=[
                        f"seed {i}" for i in range(res["kt_cor"].shape[0])
                    ],  # line labels: seed
                    title=f"Kendall-tau Correlation vs Step Plot",
                    xname="Step",
                ),
            }
        )


def evaluate_on_a_discrete_dataset(
    model: TransformerModel,
    seed: int,
    X_pending: torch.Tensor,
    y_pending: torch.Tensor,
    Xopt: torch.Tensor,
    yopt: torch.Tensor,
    test_x_range: torch.Tensor,
    train_x_range: torch.Tensor,
    initial_pairs: torch.Tensor,
    initial_pairs_y: torch.Tensor,
    initial_c: torch.Tensor,
    p_noise: float,
    eval_max_T: int,
    argmax: bool,
):
    """Evaluate PABBO on one HPOB test dataset with specified random seed.

    Args:
        model: PABBO
        seed, int: random seed
        X_pending, (num_candidates, d_x): the candidates, i.e. the query set.
        y_pending, (num_candidates, 1): associated utility values.
        Xopt, (num_global_optima, d_x): the global optima locations.
        yopt, (num_global_optima, 1): the global optima.
        test_x_range, (d_x, 2): representing the range of test input values.
        train_x_range, (d_x, 2): representing the range of training input values.
        initial_pairs: (num_initial_pairs, 2*d_x): starting pairs.
        initial_pairs_y, (num_initial_pairs, 2): associated utility values.
        initial_c, (num_initial_pairs, 1): associated preference.
        p_noise, float: the observed noise when giving preference.
        eval_max_T, int: time budget.
        argmax, bool: whether to propose the query with the largest predicted acq_value or sample from the policy distribution.

    Returns: a dictionary containing various evaluation metrics:
        - "simple_regret", (1, T+1): A tensor containing simple regret values at each optimization step.
        - "immediate_regret", (1, T+1): A tensor containing immediate regret values at each optimization step.
        - "inference_regret", (1, T+1): A tensor containing inference regret values at each optimization step.
        - "cumulative_time", (1, T): A tensor containing cumulative inference at each optimization step.
        - "entropy", (1, T): A tensor containing entropy values at each optimization step.
        - "kt_cor", (1, T): A tensor containing kendall-tau correlation on the query set at each optimization step.
    """

    set_all_seeds(seed)

    # visualize the trajectory on 1- or 2-d function
    d_x = Xopt.shape[-1]

    # load initial data and test dataset; add batch dim
    context_pairs_y = initial_pairs_y[None, :, :]
    context_c = initial_c[None, :, :]
    context_pairs = scale_from_domain_1_to_domain_2(
        x=initial_pairs.view(-1, d_x),
        bound1=test_x_range,
        bound2=train_x_range,
    ).view(-1, 2 * d_x)[None, :, :]

    num_query_points = len(X_pending)
    X_pending = X_pending[None, :, :]
    y_pending = y_pending[None, :, :]

    # NOTE scale the input range of test samples into the range where PABBO was trained
    X_pending = scale_from_domain_1_to_domain_2(
        x=X_pending,
        bound1=test_x_range,
        bound2=train_x_range,
    )
    Xopt = scale_from_domain_1_to_domain_2(
        Xopt,
        bound1=test_x_range,
        bound2=train_x_range,
    )
    query_pair_idx, query_pair, query_pair_y, query_c = generate_pair_set(
        X=X_pending,
        y=y_pending,
        num_total_points=num_query_points,
        p_noise=p_noise,
    )
    mask = torch.ones(
        (query_pair.shape[0], query_pair.shape[1], 1)
    ).bool()  # (1, num_query_pairs, 1)

    # record metrics along the trajectory, [(1)]
    dataset_simple_regret = [
        yopt[0].view(1) - torch.max(context_pairs_y.flatten(start_dim=1), dim=-1).values
    ]
    dataset_immediate_regret = [
        yopt[0].view(1) - torch.max(context_pairs_y.flatten(start_dim=1), dim=-1).values
    ]
    dataset_entropy = list()
    dataset_kt_cor = list()
    dataset_cumulative_time = list()
    dataset_inference_regret = list()

    with torch.no_grad():
        # sequentially propose T queries
        for t in range(1, eval_max_T + 1):
            # evaluate kt-cor on the query set at each step
            pred_f = model(
                query_src=context_pairs,
                c_src=context_c,
                eval_pos=X_pending,
                acquire=False,
            )[1]
            kt_cor = kendalltau_correlation(
                pred=pred_f,
                y=y_pending,
                reduce=True,
            )  # (1)
            dataset_kt_cor.append(kt_cor)

            t0 = time.time()

            # predict acq_values for all points in the query set
            acq_values, next_pair_idx, log_prob, entropy = action(
                model=model,
                context_pairs=context_pairs,
                context_preference=context_c,
                t=t,
                T=eval_max_T,
                X_pending=X_pending,
                pair_idx=query_pair_idx,
                mask=mask,
                argmax=argmax,
            )
            t1 = time.time()
            dataset_cumulative_time.append(t1 - t0)

            # update observations and candidate set given the proposed query
            next_pair_idx = next_pair_idx[:, None, None]  # (B, 1, 1)
            # mask out the query pair from candidate query pair set
            mask.scatter_(
                dim=1,
                index=next_pair_idx,
                src=torch.zeros_like(mask, device=mask.device),
            )
            # update observations with the query
            next_pair = gather_data_at_index(data=query_pair, idx=next_pair_idx)
            next_pair_y = gather_data_at_index(data=query_pair_y, idx=next_pair_idx)
            next_c = gather_data_at_index(data=query_c, idx=next_pair_idx)
            context_pairs = torch.cat((context_pairs, next_pair), dim=1)
            context_c = torch.cat((context_c, next_c), dim=1)
            context_pairs_y = torch.cat((context_pairs_y, next_pair_y), dim=1)

            # record metrics
            dataset_simple_regret.append(
                yopt[0].view(1)
                - torch.max(context_pairs_y.flatten(start_dim=1), dim=-1).values,
            )  # (1)
            dataset_immediate_regret.append(
                yopt[0].view(1) - context_pairs_y[:, -1].max(dim=-1).values,
            )  # (1)
            dataset_entropy.append(entropy)
            inference_opt_idx = torch.max(
                acq_values.flatten(start_dim=1), dim=-1
            ).indices[
                :, None, None
            ]  # (B, 1, 1)
            infer_opt_pair_y = gather_data_at_index(
                data=query_pair_y, idx=inference_opt_idx
            )
            dataset_inference_regret.append(
                yopt[0].view(1)
                - torch.max(
                    infer_opt_pair_y.flatten(start_dim=1),
                    dim=-1,
                ).values
            )  # (1)

    dataset_cumulative_time = (
        torch.from_numpy(np.cumsum(dataset_cumulative_time)).reshape(1, -1).to(Xopt)
    )
    print(
        f"final simple regret: {dataset_simple_regret[-1].item(): .5f}, final inference regret: {dataset_inference_regret[-1].item(): .5f}, final kt-cor: {dataset_kt_cor[-1].item(): .5f}"
    )

    return {
        "simple_regret": torch.stack(dataset_simple_regret, dim=-1),
        "immediate_regret": torch.stack(dataset_immediate_regret, dim=-1),
        "inference_regret": torch.stack(dataset_inference_regret, dim=-1),
        "cumulative_time": dataset_cumulative_time,
        "entropy": torch.stack(dataset_entropy, dim=-1),
        "kt_cor": torch.from_numpy(np.stack(dataset_kt_cor)).reshape(1, -1).to(Xopt),
    }


def evaluate(config: DictConfig, model: TransformerModel):
    """Evaluate PABBO on HPOB tasks over discrete space.

    Args:
        config: Configuration object containing hyperparameters and settings.
        model: PABBO.

    Returns: a dictionary with the following keys and values:
            - "simple_regret", (num_seed, B, T+1): The simple regret metric.
            - "immediate_regret", (num_seed, B, T+1): The immediate regret metric.
            - "cumulative_regret", (num_seed, B, T+1): The cumulative regret metric.
            - "inference_regret", (num_seed, B, T): The inference regret metric.
            - "entropy", (num_seed, B, T): The entropy metric.
            - "kt_cor", (num_seed, B, T): The KT correlation metric.
            - "cumulative_time", (num_seed, B, T): The cumulative time metric.

    """
    datapath = get_evaluation_datapath(
        root=osp.join(hydra.utils.get_original_cwd(), DATASETS_PATH),
        dataname=config.data.name,
    )
    if not osp.exists(datapath):
        print(f"Generating PBBO evaluation task for {config.data.name}...")
        eval_task = generate_hpob_evaluation_task(
            num_seeds=config.eval.num_seeds,
            train_x_i_range=config.train.x_i_range,
            test_x_range=config.data.x_range,
            search_space_id=str(config.data.search_space_id),
            standardize=config.data.standardize,
            num_initial_pairs=config.eval.num_initial_pairs,
            p_noise=config.eval.p_noise,
            device=config.experiment.device,
        )
        torch.save(eval_task, datapath)
        print(
            f"Generated PBBO evaluation task for {config.data.name} is saved under {str(datapath)}."
        )
    else:
        print(
            f"Loading PBBO evaluation task for {config.data.name} from {str(datapath)}..."
        )
        eval_task = torch.load(
            datapath, map_location=config.experiment.device, weights_only=False
        )
    TEST_X_RANGE = torch.tensor(eval_task["test_x_range"])
    TRAIN_X_RANGE = torch.tensor(eval_task["train_x_range"])

    NUM_SEEDS = eval_task["num_seeds"]
    B = eval_task["num_datasets"]
    X = eval_task["X"].to(device=config.experiment.device, dtype=torch.float32)
    y = eval_task["y"].to(device=config.experiment.device, dtype=torch.float32)
    XOPT = eval_task["Xopt"].to(device=config.experiment.device, dtype=torch.float32)
    YOPT = eval_task["yopt"].to(
        device=config.experiment.device, dtype=torch.float32
    )  # (B, num_global_optima, 1)

    # record metrics averaged over all batches for each random seed, (num_seeds, T)
    SIMPLE_REGRET = list()
    IMMEDIATE_REGRET = list()
    ENTROPY = list()
    KT_COR = list()
    CUMULATIVE_TIME = list()
    INFERENCE_REGRET = list()

    # evaluate with each seed
    for seed in range(NUM_SEEDS):
        print(f"Loading initialization with random seed {seed}...")
        initialization = eval_task["initialization"][f"{seed}"]

        # record metrics averaged over all batches, [(1, T)]
        batch_simple_regret = list()
        batch_immediate_regret = list()
        batch_entropy = list()
        batch_kt_cor = list()
        batch_cumulative_time = list()
        batch_inference_regret = list()

        # evaluate on each dataset
        for b in tqdm(range(B), f"Evaluating on {B} datasets..."):
            # evaluate on each dataset
            dataset_metrics = evaluate_on_a_discrete_dataset(
                model=model,
                seed=seed,
                Xopt=XOPT[b],
                yopt=YOPT[b],
                X_pending=X[b],
                y_pending=y[b],
                test_x_range=TEST_X_RANGE,
                train_x_range=TRAIN_X_RANGE,
                initial_pairs=initialization["initial_pairs"][b].to(
                    device=config.experiment.device, dtype=torch.float32
                ),
                initial_pairs_y=initialization["initial_pairs_y"][b].to(
                    device=config.experiment.device, dtype=torch.float32
                ),
                initial_c=initialization["initial_c"][b].to(
                    device=config.experiment.device, dtype=torch.float32
                ),
                p_noise=config.eval.p_noise,
                eval_max_T=config.eval.eval_max_T,
                argmax=config.eval.argmax,
            )
            # record metrics averaged over datasets
            batch_simple_regret.append(dataset_metrics["simple_regret"])
            batch_immediate_regret.append(dataset_metrics["immediate_regret"])
            batch_inference_regret.append(dataset_metrics["inference_regret"])
            batch_cumulative_time.append(dataset_metrics["cumulative_time"])
            batch_entropy.append(dataset_metrics["entropy"])
            batch_kt_cor.append(dataset_metrics["kt_cor"])

        batch_simple_regret = torch.cat(batch_simple_regret, dim=0)  #  # (B, T+1)
        batch_immediate_regret = torch.cat(batch_immediate_regret, dim=0)  # (B, T+1)
        batch_entropy = torch.cat(batch_entropy, dim=0)  # (B, T)
        batch_kt_cor = torch.cat(batch_kt_cor, dim=0)  # (B, T)
        batch_cumulative_time = torch.cat(batch_cumulative_time, dim=0)  # (B, T)
        batch_inference_regret = torch.cat(batch_inference_regret, dim=0)  # (B, T)

        SIMPLE_REGRET.append(batch_simple_regret)
        IMMEDIATE_REGRET.append(batch_immediate_regret)
        ENTROPY.append(batch_entropy)
        KT_COR.append(batch_kt_cor)
        CUMULATIVE_TIME.append(batch_cumulative_time)
        INFERENCE_REGRET.append(batch_inference_regret)

    SIMPLE_REGRET = torch.stack(SIMPLE_REGRET, dim=0)  # (NUM_SEEDS, B, T+1)
    IMMEDIATE_REGRET = torch.stack(IMMEDIATE_REGRET, dim=0)
    CUMULATIVE_REGRET = torch.cumsum(IMMEDIATE_REGRET, dim=-1)  # (NUM_SEED, B, T+1)
    ENTROPY = torch.stack(ENTROPY, dim=0)  # (NUM_SEED, B, T)
    KT_COR = torch.stack(KT_COR, dim=0)  # (NUM_SEED, B, T)
    CUMULATIVE_TIME = torch.stack(CUMULATIVE_TIME, dim=0)  # (NUM_SEED, B, T)
    INFERENCE_REGRET = torch.stack(INFERENCE_REGRET, dim=0)  # (NUM_SEED, B, T)

    return {
        "simple_regret": SIMPLE_REGRET,
        "immediate_regret": IMMEDIATE_REGRET,
        "cumulative_regret": CUMULATIVE_REGRET,
        "inference_regret": INFERENCE_REGRET,
        "entropy": ENTROPY,
        "kt_cor": KT_COR,
        "cumulative_time": CUMULATIVE_TIME,
    }


if __name__ == "__main__":
    main()
