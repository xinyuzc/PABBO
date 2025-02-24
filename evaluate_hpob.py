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
from data.environment import generate_query_pair_set
from policy_learning import *
from utils.log import get_logger
from utils.losses import kendalltau_correlation
from utils.plot import *
import logging
import matplotlib.pyplot as plt
from types import SimpleNamespace


@hydra.main(version_base=None, config_path="configs")
def main(config: DictConfig):
    if config.experiment.wandb:
        wandb_init(config=config, **config.wandb)

    torch.set_default_dtype(torch.float32)
    torch.set_default_device(config.experiment.device)

    (
        SIMPLE_REGRET,
        IMMEDIATE_REGRET,
        CUMULATIVE_REGRET,
        ENTROPY,
        KT_COR,
        CUMULATIVE_TIME,
        INFERENCE_REGRET,
        fig,
    ) = evaluate(config)

    # log to W&B
    if config.experiment.wandb:
        srml = wandb.plot.line_series(
            xs=[*range(SIMPLE_REGRET.shape[-1])],  # (H)
            ys=SIMPLE_REGRET,  # (num_seed, H)
            keys=[f"seed_{i}" for i in range(SIMPLE_REGRET.shape[-1])],
            title="Simple Regret",
            xname="Steps",
        )
        imrml = wandb.plot.line_series(
            xs=[*range(IMMEDIATE_REGRET.shape[-1])],  # (H)
            ys=IMMEDIATE_REGRET,  # (num_seed, H)
            keys=[f"seed_{i}" for i in range(IMMEDIATE_REGRET.shape[-1])],
            title="Immediate Regret",
            xname="Steps",
        )
        crml = wandb.plot.line_series(
            xs=[*range(CUMULATIVE_REGRET.shape[-1])],  # (H)
            ys=CUMULATIVE_REGRET,  # (num_seed, H)
            keys=[f"seed_{i}" for i in range(CUMULATIVE_REGRET.shape[-1])],
            title="Cumulative Regret",
            xname="Steps",
        )
        eml = wandb.plot.line_series(
            xs=[*range(ENTROPY.shape[-1])],  # (H)
            ys=ENTROPY,  # (num_seed, H)
            keys=[f"seed_{i}" for i in range(ENTROPY.shape[-1])],
            title="Entropy",
            xname="Steps",
        )
        ktcml = wandb.plot.line_series(
            xs=[*range(KT_COR.shape[-1])],  # (H)
            ys=KT_COR,  # (num_seed, H)
            keys=[f"seed_{i}" for i in range(KT_COR.shape[-1])],
            title="Kendall-tau Correlation",
            xname="Steps",
        )
        ctml = wandb.plot.line_series(
            xs=[*range(CUMULATIVE_TIME.shape[-1])],  # (H)
            ys=CUMULATIVE_TIME,  # (num_seed, H)
            keys=[f"seed_{i}" for i in range(CUMULATIVE_TIME.shape[-1])],
            title="Cumulative Inference Time",
            xname="Steps",
        )
        irml = wandb.plot.line_series(
            xs=[*range(INFERENCE_REGRET.shape[-1])],  # (H)
            ys=INFERENCE_REGRET,  # (num_seed, H)
            keys=[f"seed_{i}" for i in range(INFERENCE_REGRET.shape[-1])],
            title="Inference Regret",
            xname="Steps",
        )

        log_ctx = {
            "simple_regret_plot": srml,
            "immediate_regret_plot": imrml,
            "cumulative_regret_plot": crml,
            "cumulative_inference_time_plot": ctml,
            "inference_regret_plot": irml,
            "entropy_plot": eml,
            "kendall-tau correlation plot": ktcml,
        }

        if fig is not None:
            handles, labels = fig.axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, ncol=len(labels), loc="lower center")
            fig.suptitle(
                f"[T={config.eval.eval_max_T}], r=({SIMPLE_REGRET[:, -1].mean().item(): .2f}, {SIMPLE_REGRET[:, -1].std().item(): .2f}), ri=({INFERENCE_REGRET[:, -1].mean().item(): .2f}, {INFERENCE_REGRET[:, -1].std().item(): .2f})"
            )
            log_ctx["optimization_trajectory_plot"] = wandb.Image(fig)

        wandb.log(log_ctx)


def evalate_on_a_dataset(
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
    logger: Union[logging.Logger, None] = None,
    plot_freq: float = -1,
):
    """Evaluate PABBO on one HPOB test dataset with specified random seed.

    Args:
        model: PABBO
        seed, int: random seed
        X_pending, (num_candidates, d_x): the candidates, i.e. the query set.
        y_pending, (num_candidates, 1): associated utility values.
        Xopt, (num_global_optima, d_x): the global optima locations.
        yopt, (num_global_optima, 1): the global optima.
        test_x_range, (d_x, 2)
        train_x_range, (d_x, 2)
        initial_pairs, (num_initial_pairs, 2 * d_x)
        initial_pairs_y, (num_initial_pairs, 2)
        initial_c, (num_initial_pairs, 2)
        p_noise, float: the observed noise when giving preference.
        eval_max_T, int: time budget.
        argmax, bool: whether to propose the query with the largest predicted acq_value or sample from the entire policy.
        plot_freq, int: the frequency of visualizing the trajectory on funcition.

    Returns:
        dataset_simple_regret, (1, T+1): yopt - max_{i=1}^t max {y_{i,1}, y_{i,2}}
        dataset_immediate_regret, (1, T+1): yopt - max {y_{t,1}, y_{t,2}}
        dataset_cumulative_time, (1, T)
        dataset_entropy, (1, T)
        dataset_kt_cor, (1, T)
        dataset_inference_regret, (1, T)

    """
    set_all_seeds(seed)

    # visualize the trajectory on 1- or 2-d function
    if plot_freq > 0:
        num_axes = eval_max_T // plot_freq + 1
        nrows = max(int(math.ceil(num_axes / 3)), 1)
        ncols = min(3, num_axes)
        fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
    else:
        fig = None

    d_x = Xopt.shape[-1]

    # load initial data and test dataset; add batch dim
    context_pairs = initial_pairs.reshape(1, -1, 2 * d_x)
    context_pairs_y = initial_pairs_y.reshape(1, -1, 2)
    context_c = initial_c.reshape(1, -1, 2)
    num_query_points = len(X_pending)

    X_pending = X_pending.reshape(1, -1, d_x)
    y_pending = y_pending.reshape(1, -1, 1)
    if logger is not None:
        logger.info(f"Original initial pairs: \n {context_pairs}")

    # NOTE scale the input range of test samples into the range where PABBO was trained
    X_pending = scale_from_domain_1_to_domain_2(
        x=X_pending,
        bound1=test_x_range,
        bound1=train_x_range,
    )
    if logger is not None:
        logger.info(f"Original Xopt: \n {Xopt}")

    Xopt = scale_from_domain_1_to_domain_2(
        Xopt,
        bound1=test_x_range,
        bound1=train_x_range,
    )
    query_pair_idx, query_pair, query_pair_y, query_c = generate_query_pair_set(
        X=X_pending,
        y=y_pending,
        num_total_points=num_query_points,
        p_noise=p_noise,
    )
    mask = (
        torch.ones((query_pair.shape[0], query_pair.shape[1], 1)).bool().to(Xopt)
    )  # (1, num_query_pairs, 1), mask to register queried pairs

    # NOTE scale the test input into the range where PABBO was trained
    context_pairs = scale_from_domain_1_to_domain_2(
        x=context_pairs.view(1, -1, d_x),
        bound1=test_x_range,
        bound2=train_x_range,
    ).view(1, -1, 2 * d_x)

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
        for t in range(1, eval_max_T):
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
                val=yopt[0].view(1) - context_pairs_y[:, -1].max(dim=-1).values,
            )  # (1)
            dataset_entropy.append(entropy)
            inference_opt_idx = torch.max(acq_values, dim=1).indices[:, None, :]
            infer_opt_pair = gather_data_at_index(
                data=query_pair, idx=inference_opt_idx
            )
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
            if plot_freq > 0 and (t % plot_freq == 0):
                # TODO plot
                if d_x == 1:
                    ax = fig.add_subplot(nrows, ncols, t // plot_freq + 1)
                    # TODO scale input range to test range
                    plot_prediction_on_1d_function(
                        X=scale_from_domain_1_to_domain_2(
                            x=X_pending[0].clone(),
                            bound1=test_x_range,
                            bound2=train_x_range,
                        ),
                        y_pred=pred_f[0].clone(),
                        y=y_pending[0].clone(),
                        Xopt=scale_from_domain_1_to_domain_2(
                            x=Xopt.clone(), bound1=test_x_range, bound2=train_x_range
                        ),
                        yopt=yopt.clone(),
                        x_range=test_x_range,
                        query_pair=scale_from_domain_1_to_domain_2(
                            x=context_pairs[0].clone().reshape(-1, d_x),
                            bound1=test_x_range,
                            bound2=train_x_range,
                        ),
                        query_pair_y=context_pairs_y[0].clone(),
                        infer_opt_pair=scale_from_domain_1_to_domain_2(
                            x=infer_opt_pair[0].clone().reshape(-1, d_x),
                            bound1=test_x_range,
                            bound2=train_x_range,
                        ),
                        infer_opt_pair_y=infer_opt_pair_y[0].clone(),
                        utility_function=None,
                        ax=ax,
                    )
                elif d_x == 2:
                    raise NotImplementedError
                else:
                    raise NotImplementedError(
                        "Visualization for data with more than 2 dimensions is not supported."
                    )
        return (
            torch.stack(dataset_simple_regret, dim=-1),
            torch.stack(dataset_immediate_regret, dim=-1),
            torch.from_numpy(np.cumsum(dataset_cumulative_time))
            .reshape(1, -1)
            .to(Xopt),  # cumulative time
            torch.stack(dataset_entropy, dim=-1),
            torch.stack(dataset_kt_cor, dim=-1),
            torch.stack(dataset_inference_regret, dim=-1),
            fig,
        )


def evaluate(config: DictConfig):
    """Evaluate PABBO on HPOB benchmark.

    Returns:
        SIMPLE_REGRET, (num_seed, T+1)
        IMMEDIATE_REGRET, (num_seed, T+1)
        CUMULATIVE_REGRET, (num_seed, T+1)
        ENTROPY, (num_seed, T)
        KT_COR, (num_seed, T)
        CUMULATIVE_TIME, (num_seed, T)
        INFERENCE_REGRET, (num_seed, T)
    """
    device = torch.device(config.experiment.device)

    # load model from saved checkpoint
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

    # logger
    logfilename = os.path.join(
        f'evaluate_{config.experiment.expid}_{time.strftime("%Y%m%d_%H%M%S")}.log',
    )
    logger = get_logger(file_name=logfilename, mode="w")

    # load preferential black-box optimization (PBBO) task for evaluation
    logger.info(
        f"Loading PBBO evaluation task for {config.data.name}: initial data, utility and optimum information..."
    )
    root = osp.join(hydra.utils.get_original_cwd(), DATASETS_PATH)
    if not osp.exists(root + f"{config.data.name}_initial_pairs.pt"):
        # create
        raise NotImplementedError
    else:
        eval_task = torch.load(
            osp.join(
                hydra.utils.get_original_cwd(),
                DATASETS_PATH,
                f"{config.data.name}_initial_pairs.pt",
            ),
            map_location=config.experiment.device,
        )

    # load HPOB test data
    logger.info(f"Loading test data for {config.data.name}...")
    test_data_filename = f"{config.data.name}.pt"
    test_datapath = osp.join(root, test_data_filename)
    if not osp.exists(test_datapath):
        raise FileExistsError(test_datapath)
        # TODO generate test data

    else:
        test_data = torch.load(test_datapath, map_location=config.experiment.device)
    test_data = SimpleNamespace(**test_data)

    NUM_SEEDS = eval_task["num_seeds"]
    B = eval_task["num_datasets"]
    TEST_X_RANGE = torch.tensor(eval_task["test_x_range"]).to(device)
    TRAIN_X_RANGE = torch.tensor(eval_task["train_x_range"]).to(device)
    SAMPLER_KWARGS = eval_task["sampler_kwargs"]
    FUNCTION_KWARGS = eval_task["function_kwargs"]
    XOPT = eval_task["Xopt"].to(device)
    YOPT = eval_task["yopt"].to(device)  # (B, num_global_optima, 1)

    # record metrics averaged over all batches for each random seed, (num_seeds, T)
    SIMPLE_REGRET = list()
    IMMEDIATE_REGRET = list()
    ENTROPY = list()
    KT_COR = list()
    CUMULATIVE_TIME = list()
    INFERENCE_REGRET = list()

    # evaluate with each seed
    for seed in range(NUM_SEEDS):
        logger.info(f"Loading initialization with random seed {seed}...")
        initialization = eval_task["initialization"][f"{seed}"]

        # record metrics averaged over all batches, [(1, T)]
        batch_simple_regret = list()
        batch_immediate_regret = list()
        batch_entropy = list()
        batch_kt_cor = list()
        batch_cumulative_time = list()
        batch_inference_regret = list()

        # evaluate on each dataset
        for b in tqdm(range(B), f"Evaluating on {B} GP samples..."):
            # evaluate on each dataset
            (
                dataset_simple_regret,
                dataset_immediate_regret,
                dataset_entropy,
                dataset_kt_cor,
                dataset_cumulative_time,
                dataset_inference_regret,
                fig,
            ) = evalate_on_a_dataset(
                model=model,
                seed=seed,
                Xopt=XOPT[b],
                yopt=YOPT[b],
                X_pending=test_data.X[b],
                y_pending=test_data.y[b],
                test_x_range=TEST_X_RANGE,
                train_x_range=TRAIN_X_RANGE,
                initial_pairs=initialization["initial_pairs"][b],
                initial_pairs_y=initialization["initial_pairs_y"][b],
                initial_c=initialization["initial_c"][b],
                eval_num_query_points=config.eval.eval_num_query_points,
                p_noise=config.eval.p_noise,
                eval_max_T=config.eval.eval_max_T,
                argmax=config.eval.argmax,
                logger=logger,
                plot_freq=config.eval.plot_freq,
            )
            # record metrics averaged over datasets
            batch_simple_regret.append(dataset_simple_regret)
            batch_immediate_regret.append(dataset_immediate_regret)
            batch_cumulative_time.append(dataset_cumulative_time)
            batch_entropy.append(dataset_entropy)
            batch_kt_cor.append(dataset_kt_cor)
            batch_inference_regret.append(dataset_inference_regret)

        SIMPLE_REGRET.append(torch.cat(batch_simple_regret, dim=0).mean(dim=0))  # (T)
        IMMEDIATE_REGRET.append(torch.cat(batch_immediate_regret, dim=0).mean(dim=0))
        ENTROPY.append(torch.cat(batch_entropy, dim=0).mean(dim=0))
        KT_COR.append(torch.cat(batch_kt_cor, dim=0).mean(dim=0))
        CUMULATIVE_TIME.append(torch.cat(batch_cumulative_time, dim=0).mean(dim=0))
        INFERENCE_REGRET.append(torch.cat(batch_inference_regret, dim=0).mean(dim=0))

    SIMPLE_REGRET = torch.stack(SIMPLE_REGRET, dim=0).cpu().numpy()
    IMMEDIATE_REGRET = torch.stack(IMMEDIATE_REGRET, dim=0).cpu().numpy()
    CUMULATIVE_REGRET = np.cumsum(IMMEDIATE_REGRET, axis=-1)
    ENTROPY = torch.stack(ENTROPY, dim=0).cpu().numpy()
    KT_COR = torch.stack(KT_COR, dim=0).cpu().numpy()
    CUMULATIVE_TIME = torch.stack(CUMULATIVE_TIME, dim=0).cpu().numpy()
    INFERENCE_REGRET = torch.stack(INFERENCE_REGRET, dim=0).cpu().numpy()

    # save results
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

        torch.save(SIMPLE_REGRET, f"{save_dir}/SIMPLE_REGRET.pt")
        torch.save(IMMEDIATE_REGRET, f"{save_dir}/IMMEDIATE_REGRET.pt")
        torch.save(CUMULATIVE_REGRET, f"{save_dir}/CUMULATIVE_REGRET.pt")
        torch.save(ENTROPY, f"{save_dir}/ENTROPY.pt")
        torch.save(KT_COR, f"{save_dir}/KT_COR.pt")
        torch.save(CUMULATIVE_TIME, f"{save_dir}/CUMULATIVE_TIME.pt")
        torch.save(INFERENCE_REGRET, f"{save_dir}/INFERENCE_REGRET.pt")

    return (
        SIMPLE_REGRET,
        IMMEDIATE_REGRET,
        CUMULATIVE_REGRET,
        ENTROPY,
        KT_COR,
        CUMULATIVE_TIME,
        INFERENCE_REGRET,
        fig,
    )


if __name__ == "__main__":
    main()
