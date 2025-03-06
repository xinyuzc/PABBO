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
from typing import Dict
from policies.transformer import TransformerModel
from data.utils import set_all_seeds
from data.sampler import SimpleGPSampler, OptimizationFunction, UtilitySampler
from data.kernel import *
from data.utils import scale_from_domain_1_to_domain_2
from data.environment import generate_query_pair_set
from data.evaluation import *
from data.candy_data_handler import *
from data.sushi_data_handler import *
from policy_learning import *
from utils.losses import kendalltau_correlation
from utils.plot import *
import matplotlib.pyplot as plt


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
from typing import Dict
from policies.transformer import TransformerModel
from data.utils import set_all_seeds
from data.sampler import SimpleGPSampler, OptimizationFunction, UtilitySampler
from data.kernel import *
from data.utils import scale_from_domain_1_to_domain_2
from data.environment import generate_query_pair_set
from data.evaluation import *
from data.candy_data_handler import *
from data.sushi_data_handler import *
from policy_learning import *
from utils.losses import kendalltau_correlation
from utils.plot import *
import matplotlib.pyplot as plt


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

    if config.data.name.startswith("GP"):
        res = evaluate_on_gp(config, model)
    else:
        res = evaluate_on_deterministic_function(config, model)

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

        print(f"Saving results in f{str(save_dir)}...")
        torch.save(
            res["simple_regret"],
            f"{save_dir}/SIMPLE_REGRET_{config.eval.eval_num_query_points}_{config.eval.num_parallel}.pt",
        )
        torch.save(
            res["immediate_regret"],
            f"{save_dir}/IMMEDIATE_REGRET_{config.eval.eval_num_query_points}_{config.eval.num_parallel}.pt",
        )
        torch.save(
            res["cumulative_regret"],
            f"{save_dir}/CUMULATIVE_REGRET_{config.eval.eval_num_query_points}_{config.eval.num_parallel}.pt",
        )
        torch.save(
            res["entropy"],
            f"{save_dir}/ENTROPY_{config.eval.eval_num_query_points}_{config.eval.num_parallel}.pt",
        )
        torch.save(
            res["kt_cor"],
            f"{save_dir}/KT_COR_{config.eval.eval_num_query_points}_{config.eval.num_parallel}.pt",
        )
        torch.save(
            res["cumulative_time"],
            f"{save_dir}/CUMULATIVE_TIME_{config.eval.eval_num_query_points}_{config.eval.num_parallel}.pt",
        )
        torch.save(
            res["inference_regret"],
            f"{save_dir}/INFERENCE_REGRET_{config.eval.eval_num_query_points}_{config.eval.num_parallel}.pt",
        )

    # log to W&B
    if config.experiment.wandb:
        num_seed = res["simple_regret"].shape[0]
        wandb.log(
            {
                f"simple_regret_vs_step_plot": wandb.plot.line_series(
                    xs=[*range(res["simple_regret"].shape[-1])],
                    ys=[res["simple_regret"].mean(dim=1)[i] for i in range(num_seed)],
                    keys=[f"seed {i}" for i in range(num_seed)],
                    title=f"Simple Regret vs Step Plot",
                    xname="Step",
                ),
                f"immediate_regret_vs_step_plot": wandb.plot.line_series(
                    xs=[*range(res["immediate_regret"].shape[-1])],
                    ys=[
                        res["immediate_regret"].mean(dim=1)[i] for i in range(num_seed)
                    ],
                    keys=[f"seed {i}" for i in range(num_seed)],
                    title=f"Immediate Regret vs Step Plot",
                    xname="Step",
                ),
                f"entropy_vs_step_plot": wandb.plot.line_series(
                    xs=[*range(res["entropy"].shape[-1])],
                    ys=[res["entropy"].mean(dim=1)[i] for i in range(num_seed)],
                    keys=[f"seed {i}" for i in range(num_seed)],
                    title=f"Entropy vs Step Plot",
                    xname="Step",
                ),
                f"inference_regret_vs_step_plot": wandb.plot.line_series(
                    xs=[
                        *range(res["inference_regret"].shape[-1])
                    ],  # list of x values: steps
                    ys=[
                        res["inference_regret"].mean(dim=1)[i] for i in range(num_seed)
                    ],
                    keys=[f"seed {i}" for i in range(num_seed)],
                    title=f"Inference Regret vs Step Plot",
                    xname="Step",
                ),
                f"kt_cor_vs_step_plot": wandb.plot.line_series(
                    xs=[*range(res["kt_cor"].shape[-1])],  # list of x values: steps
                    ys=[res["kt_cor"].mean(dim=1)[i] for i in range(num_seed)],
                    keys=[f"seed {i}" for i in range(num_seed)],
                    title=f"Kendall-tau Correlation vs Step Plot",
                    xname="Step",
                ),
            }
        )


def evaluate_on_a_utility(
    model: TransformerModel,
    seed: int,
    Xopt: torch.Tensor,
    yopt: torch.Tensor,
    utility: Callable,
    test_x_range: torch.Tensor,
    train_x_range: torch.Tensor,
    initial_pairs: torch.Tensor,
    initial_pairs_y: torch.Tensor,
    initial_c: torch.Tensor,
    num_query_points: int,
    p_noise: float,
    T: int,
    argmax: bool,
    num_parallel: int = 1,
    plot_freq: float = -1,
    sobol_grid: bool = True,
) -> Dict:
    """Evaluate PABBO on a utility function with parallel query pair sets.

    Args:
        model: PABBO.
        seed, scalar: random seed.
        Xopt, (num_global_optima, d_x): global optimum.
        yopt, (num_global_optima, 1): the optimal utility value.
        utility, callable: the utility function.
        test_x_range, (d_x, 2): the input range of the test function.
        train_x_range, (d_x, 2): the input range where PABBO was trained. We should scale all test inputs to the train_x_range.
        initial_pairs, (num_initial_pairs, 2 * d_x): the initial pairs.
        initial_pairs_y, (num_initial_pairs, 2): associated utility values.
        initial_c, (num_initial_pairs, 2): associated preference.
        num_query_points, scalar: search space is discretized into a query set, Q, and any two points in Q will be considered as a candidate query pair.
        p_noise, float: the observed noise when generating preference.
        T, scalar: time budget.
        argmax, bool: whether to choose the pair with maximal predicted acquisition function value as the next query or randomly sample from the policy.
        num_parallel, scalar: the number of parallel query sets.
        plot_freq, scalar: plot optimization history, inferred utility function shape with ground truth when plot_freq > 0 and (t % plot_freq == 0), t is the optimization step.

    Returns: a dict of metrics collected on the dataset with following elements:
        simple_regret, Tensor(1, T+1): the simple regret at each time step.
        immediate_regret, Tensor(1, T+1): the immediate regret at each time step.
        inference_regret, Tensor(1, T): the inference regret at each time step.
        cumulative_time, Tensor(1, T): the cumulative model inference time at each time step.
        entropy, Tensor(1, T): the policy entropy at each time step.
        kt_cor, Tensor(1, T): the kendall-tau corretion on the query set Q at each time step.
        plot, Figure: plot on the dataset.
    """
    set_all_seeds(seed)
    d_x = Xopt.shape[-1]
    # plot setup
    if plot_freq > 0:
        if d_x > 2:
            raise NotImplementedError
        num_axes = T // plot_freq
        nrows = max(int(math.ceil(num_axes / 3)), 1)
        ncols = min(3, num_axes)
        fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    else:
        fig = None

    # load initial data and add batch dim
    context_pairs_y = initial_pairs_y[None, :, :]  # (1, num_initial_pairs, 2)
    context_c = initial_c[None, :, :]

    # NOTE scale the test input to the range where PABBO was trained
    context_pairs = scale_from_domain_1_to_domain_2(
        x=initial_pairs.view(-1, d_x),
        bound1=test_x_range,
        bound2=train_x_range,
    ).view(-1, 2 * d_x)[None, :, :]

    # a sampler for the utility function
    sampler = UtilitySampler(
        d_x=d_x,
        utility_function=utility,
        Xopt=Xopt,
        yopt=yopt,
    )
    # NOTE create multiple parallel query sets, X of shape (num_parallel, num_query_points, d_X)
    X, y, _, _ = sampler.sample(
        batch_size=num_parallel,
        num_total_points=num_query_points,
        x_range=test_x_range,
        sobol_grid=sobol_grid,
    )
    # NOTE scale the test input to the range where PABBO was trained
    X = scale_from_domain_1_to_domain_2(
        x=X,
        bound1=test_x_range,
        bound2=train_x_range,
    )
    Xopt = scale_from_domain_1_to_domain_2(
        Xopt,
        bound1=test_x_range,
        bound2=train_x_range,
    )
    # create `num_parallel` candidate query pair set
    query_pair_idx, query_pair, query_pair_y, query_c = generate_query_pair_set(
        X=X,
        y=y,
        num_total_points=num_query_points,
        p_noise=p_noise,
    )
    # mask (num_parallel, num_query_pairs, 1), register sampled pairs for each candidate query pair set
    mask = torch.ones((query_pair.shape[0], query_pair.shape[1], 1)).bool()

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
        for t in range(1, T + 1):
            # evaluate kendall-tau correlation on the query set Q
            pred_f = model(
                query_src=context_pairs.tile(
                    num_parallel, 1, 1
                ),  # NOTE match size with X
                c_src=context_c.tile(num_parallel, 1, 1),
                eval_pos=X,
                acquire=False,
            )[1]
            kt_cor = kendalltau_correlation(
                pred=pred_f,
                y=y,
                reduce=True,
            )  # (1)
            dataset_kt_cor.append(kt_cor)

            t0 = time.time()
            # predict acquisition function values for pairs in each dataset
            acq_values, next_pair_idx, log_prob, entropy = action(
                model=model,
                context_pairs=context_pairs.tile(num_parallel, 1, 1),
                context_preference=context_c.tile(num_parallel, 1, 1),
                t=t,
                T=T,
                X_pending=X,
                pair_idx=query_pair_idx,
                mask=mask,
                argmax=argmax,
            )
            t1 = time.time()
            dataset_cumulative_time.append(t1 - t0)

            # acq_values for the sampled query in each query set
            next_pair_idx = next_pair_idx[:, None, None]  # (B, 1, 1)
            next_acq_values = gather_data_at_index(acq_values, next_pair_idx)

            # choose the one with maximal acq_value as the next query
            next_set_idx = torch.argmax(next_acq_values).int().item()  # (1)

            # mask out the selected pair from corresponding query set
            next_set_pair_idx = next_pair_idx[next_set_idx].int().item()
            mask[next_set_idx, next_set_pair_idx] = False

            # update observations and candidate set with the next query
            next_pair = query_pair[next_set_idx, next_set_pair_idx][None, None, :]
            next_pair_y = query_pair_y[next_set_idx, next_set_pair_idx][None, None, :]
            next_c = query_c[next_set_idx, next_set_pair_idx][None, None, :]
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
            infer_opt_pair = gather_data_at_index(
                data=query_pair, idx=inference_opt_idx
            )
            infer_opt_pair_y = gather_data_at_index(
                data=query_pair_y, idx=inference_opt_idx
            )
            dataset_inference_regret.append(
                yopt[0].view(1) - torch.max(infer_opt_pair_y)
            )  # (1)

            # plot history, inferred function values with ground truth
            if plot_freq > 0 and (t % plot_freq == 0):
                if d_x == 1:
                    ax = fig.add_subplot(nrows, ncols, t // plot_freq)
                    # NOTE scale the input back to test input range
                    plot_prediction_on_1d_function(
                        X=scale_from_domain_1_to_domain_2(
                            x=X[0].clone(), bound1=train_x_range, bound2=test_x_range
                        ),
                        y_pred=pred_f[0].clone(),
                        y=y[0].clone(),
                        Xopt=scale_from_domain_1_to_domain_2(
                            x=Xopt.clone(), bound1=train_x_range, bound2=test_x_range
                        ),
                        yopt=yopt.clone(),
                        x_range=test_x_range,
                        query_pair=scale_from_domain_1_to_domain_2(
                            x=context_pairs[0].clone().reshape(-1, d_x),
                            bound1=train_x_range,
                            bound2=test_x_range,
                        ).reshape(-1, 2 * d_x),
                        query_pair_y=context_pairs_y[0].clone(),
                        infer_opt_pair=scale_from_domain_1_to_domain_2(
                            x=infer_opt_pair[0].clone().reshape(-1, d_x),
                            bound1=train_x_range,
                            bound2=test_x_range,
                        ).reshape(-1, 2 * d_x),
                        infer_opt_pair_y=infer_opt_pair_y[0].clone(),
                        utility_function=utility,
                        ax=ax,
                    )
                elif d_x == 2:
                    ax = fig.add_subplot(nrows, ncols, t // plot_freq, projection="3d")
                    plot_prediction_on_2d_function(
                        X=scale_from_domain_1_to_domain_2(
                            x=X[0].clone(), bound1=train_x_range, bound2=test_x_range
                        ),
                        y_pred=pred_f[0].clone(),
                        y=y[0].clone(),
                        Xopt=scale_from_domain_1_to_domain_2(
                            x=Xopt.clone(), bound1=train_x_range, bound2=test_x_range
                        ),
                        yopt=yopt.clone(),
                        x_range=test_x_range,
                        query_pair=scale_from_domain_1_to_domain_2(
                            x=context_pairs[0].clone().reshape(-1, d_x),
                            bound1=train_x_range,
                            bound2=test_x_range,
                        ).reshape(-1, 2 * d_x),
                        query_pair_y=context_pairs_y[0].clone(),
                        infer_opt_pair=scale_from_domain_1_to_domain_2(
                            x=infer_opt_pair[0].clone().reshape(-1, d_x),
                            bound1=train_x_range,
                            bound2=test_x_range,
                        ).reshape(-1, 2 * d_x),
                        infer_opt_pair_y=infer_opt_pair_y[0].clone(),
                        utility_function=utility,
                        ax=ax,
                    )
                else:
                    raise NotImplementedError(
                        "Plotting data with more than two input dimensions is not supported."
                    )

    # cumulative time
    dataset_cumulative_time = (
        torch.from_numpy(np.cumsum(dataset_cumulative_time)).reshape(1, -1).to(Xopt)
    )
    print(
        f"final simple regret: {dataset_simple_regret[-1].item(): .3f}, final inference regret: {dataset_immediate_regret[-1].item(): .3f}, final kt-cor: {dataset_kt_cor[-1].item(): .3f}"
    )

    return {
        "simple_regret": torch.stack(dataset_simple_regret, dim=-1),
        "immediate_regret": torch.stack(dataset_immediate_regret, dim=-1),
        "inference_regret": torch.stack(dataset_inference_regret, dim=-1),
        "cumulative_time": dataset_cumulative_time,
        "entropy": torch.stack(dataset_entropy, dim=-1),
        "kt_cor": torch.from_numpy(np.stack(dataset_kt_cor)).reshape(1, -1).to(Xopt),
        "plot": fig,
    }


def evaluate_on_deterministic_function(
    config: DictConfig, model: TransformerModel
) -> Dict:
    """Evaluate PABBO on deterministic function."""
    datapath = get_evaluation_datapath(
        root=osp.join(hydra.utils.get_original_cwd(), DATASETS_PATH),
        dataname=config.data.name,
    )
    if not osp.exists(datapath):
        print(f"Generating PBBO evaluation task for {config.data.name}...")
        if config.data.name in ["candy", "sushi"]:
            # get dataset handlers for real-world datasets
            if config.data.name == "candy":
                handler = CandyDataHandler(interpolator_type="linear")
            else:
                handler = SushiDataHandler(interpolator_type="linear")

            _, _, Xopt, yopt = handler.get_data(add_batch_dim=True)
            eval_task = generate_real_world_evaluation_task(
                num_seeds=config.eval.num_seeds,
                handler=handler,
                test_x_range=[handler.x_range for _ in range(Xopt.shape[-1])],
                train_x_range=[
                    config.train.x_i_range for _ in range(len(config.data.x_range))
                ],
                num_initial_pairs=config.eval.num_initial_pairs,
                p_noise=config.eval.p_noise,
                device=config.experiment.device,
            )
            eval_task["utility"] = handler.get_utility()
            eval_task["Xopt"], eval_task["yopt"] = (
                torch.from_numpy(Xopt),
                torch.from_numpy(yopt),
            )
        else:
            # synthetic test functions
            eval_task = generate_test_function_evaluation_task(
                dataname=config.data.name,
                num_seeds=config.eval.num_seeds,
                test_x_range=config.data.x_range,
                train_x_range=[
                    config.train.x_i_range for _ in range(len(config.data.x_range))
                ],
                Xopt=torch.tensor(config.data.Xopt),
                yopt=torch.tensor(config.data.yopt),
                num_initial_pairs=config.eval.num_initial_pairs,
                p_noise=config.eval.p_noise,
            )

        torch.save(eval_task, datapath)
        print(f"Saved under {str(datapath)}.")
    else:
        print(
            f"Loading PBBO evaluation task for {config.data.name} from {str(datapath)}..."
        )
        eval_task = torch.load(
            datapath,
            map_location=config.experiment.device,
        )

    NUM_SEEDS = eval_task["num_seeds"]
    B = eval_task["num_datasets"]
    TEST_X_RANGE = torch.tensor(eval_task["test_x_range"])
    # .to(
    #     dtype=torch.float32, device=config.experiment.device
    # )  # (d_x, 2)
    TRAIN_X_RANGE = torch.tensor(eval_task["train_x_range"])
    # .to(
    #     dtype=torch.float32, device=config.experiment.device
    # )  # (d_x, 2)
    XOPT = eval_task["Xopt"]
    # .to(
    #     dtype=torch.float32, device=config.experiment.device
    # )  # (B, num_global_optima, d_x)
    YOPT = eval_task["yopt"]
    # .to(
    #     dtype=torch.float32, device=config.experiment.device
    # )  # (B, num_global_optima, 1)

    # record metrics, (num_seeds, B, T)
    SIMPLE_REGRET = list()
    IMMEDIATE_REGRET = list()
    ENTROPY = list()
    KT_COR = list()
    CUMULATIVE_TIME = list()
    INFERENCE_REGRET = list()

    # evaluate with each seed
    for seed in range(NUM_SEEDS):
        print(f"[Seed={seed}]\n")
        initial_pairs, initial_pairs_y, initial_c = (
            eval_task["initialization"][f"{seed}"]["initial_pairs"].to(
                dtype=torch.float32, device=config.experiment.device
            ),
            eval_task["initialization"][f"{seed}"]["initial_pairs_y"].to(
                dtype=torch.float32, device=config.experiment.device
            ),
            eval_task["initialization"][f"{seed}"]["initial_c"].to(
                dtype=torch.float32, device=config.experiment.device
            ),
        )

        # record metrics averaged over all datasets with seed, [(1, T)]
        batch_simple_regret = list()
        batch_immediate_regret = list()
        batch_entropy = list()
        batch_kt_cor = list()
        batch_cumulative_time = list()
        batch_inference_regret = list()

        # evaluate on each dataset
        for b in tqdm(range(B), f"Evaluating on {B} {config.data.name} samples..."):
            plot_freq = config.eval.plot_freq
            if config.eval.plot_dataset_id >= 0 and b != config.eval.plot_dataset_id:
                plot_freq = -1
            if config.eval.plot_seed_id >= 0 and seed != config.eval.plot_seed_id:
                plot_freq = -1
            dataset_metrics = evaluate_on_a_utility(
                model=model,
                seed=seed,
                Xopt=XOPT[b],
                yopt=YOPT[b],
                utility=eval_task["utility"],
                test_x_range=TEST_X_RANGE,
                train_x_range=TRAIN_X_RANGE,
                initial_pairs=initial_pairs[b],
                initial_pairs_y=initial_pairs_y[b],
                initial_c=initial_c[b],
                num_query_points=config.eval.eval_num_query_points,
                p_noise=config.eval.p_noise,
                T=config.eval.eval_max_T,
                argmax=config.eval.argmax,
                plot_freq=plot_freq,
                num_parallel=config.eval.num_parallel,
                sobol_grid=config.eval.sobol_grid,
            )

            # record metrics averaged over datasets
            batch_simple_regret.append(dataset_metrics["simple_regret"])
            batch_immediate_regret.append(dataset_metrics["immediate_regret"])
            batch_inference_regret.append(dataset_metrics["inference_regret"])
            batch_cumulative_time.append(dataset_metrics["cumulative_time"])
            batch_entropy.append(dataset_metrics["entropy"])
            batch_kt_cor.append(dataset_metrics["kt_cor"])

            # log to W&B
            if config.experiment.wandb and plot_freq > 0:
                wandb.log(
                    {
                        f"plot_dataset_{b}_seed_{seed}": wandb.Image(
                            dataset_metrics["plot"]
                        )
                    }
                )

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


def evaluate_on_gp(config: DictConfig, model: TransformerModel) -> Dict:
    """Evaluate PABBO on GP samples with global optimum structure.

    Returns: a dictionary of collected metrics of shape (NUM_SEED, B, H)"""
    datapath = get_evaluation_datapath(
        root=osp.join(hydra.utils.get_original_cwd(), DATASETS_PATH),
        dataname=config.data.name,
    )
    if not osp.exists(datapath):
        print(f"Generating PBBO evaluation task for {config.data.name}...")
        eval_task = generate_gp_evaluation_task(
            num_seeds=config.eval.num_seeds,
            num_datasets=config.eval.num_datasets,
            test_x_range=config.data.x_range,
            train_x_range=[
                config.train.x_i_range for _ in range(len(config.data.x_range))
            ],
            max_num_ctx_points=config.data.max_num_ctx,
            num_total_points=config.eval.num_total_points,
            num_initial_pairs=config.eval.num_initial_pairs,
            **config.eval.sampler,
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
            datapath,
            map_location=config.experiment.device,
        )

    NUM_SEEDS = eval_task["num_seeds"]
    B = eval_task["num_datasets"]
    TEST_X_RANGE = torch.tensor(eval_task["test_x_range"])
    TRAIN_X_RANGE = torch.tensor(eval_task["train_x_range"])
    XOPT = eval_task["Xopt"]
    YOPT = eval_task["yopt"]
    SAMPLER_KWARGS = eval_task["sampler_kwargs"]  # [sampler_kwargs for dataset b]
    FUNCTION_KWARGS = eval_task["function_kwargs"]  # [function_kwargs for dataset b]

    # record metrics, (num_seeds, B, T)
    SIMPLE_REGRET = list()
    IMMEDIATE_REGRET = list()
    ENTROPY = list()
    KT_COR = list()
    CUMULATIVE_TIME = list()
    INFERENCE_REGRET = list()

    # evaluate with each seed
    for seed in range(NUM_SEEDS):
        print(f"[Seed={seed}]\n")
        initial_pairs, initial_pairs_y, initial_c = (
            eval_task["initialization"][f"{seed}"]["initial_pairs"].to(
                dtype=torch.float32, device=config.experiment.device
            ),
            eval_task["initialization"][f"{seed}"]["initial_pairs_y"].to(
                dtype=torch.float32, device=config.experiment.device
            ),
            eval_task["initialization"][f"{seed}"]["initial_c"].to(
                dtype=torch.float32, device=config.experiment.device
            ),
        )

        # record metrics averaged over all datasets with seed, [(1, T)]
        batch_simple_regret = list()
        batch_immediate_regret = list()
        batch_entropy = list()
        batch_kt_cor = list()
        batch_cumulative_time = list()
        batch_inference_regret = list()

        # evaluate on each dataset
        for b in tqdm(range(B), f"Evaluating on {B} {config.data.name} samples..."):
            plot_freq = config.eval.plot_freq
            if config.eval.plot_dataset_id >= 0 and b != config.eval.plot_dataset_id:
                plot_freq = -1
            if config.eval.plot_seed_id >= 0 and seed != config.eval.plot_seed_id:
                plot_freq = -1
            # Build utility function for GP samples: we first create GP sampler with saved kernel function and mean
            sampler = SimpleGPSampler(
                kernel_function=globals()[SAMPLER_KWARGS[b]["kernel_function"]],
                mean=SAMPLER_KWARGS[b]["mean"],
                jitter=SAMPLER_KWARGS[b]["jitter"],
            )
            # then wrap it with `OptimizationFunction` class into the utility function
            utility = OptimizationFunction(sampler=sampler, **FUNCTION_KWARGS[b])

            dataset_metrics = evaluate_on_a_utility(
                model=model,
                seed=seed,
                Xopt=XOPT[b],
                yopt=YOPT[b],
                utility=utility,
                test_x_range=TEST_X_RANGE,
                train_x_range=TRAIN_X_RANGE,
                initial_pairs=initial_pairs[b],
                initial_pairs_y=initial_pairs_y[b],
                initial_c=initial_c[b],
                num_query_points=config.eval.eval_num_query_points,
                p_noise=config.eval.p_noise,
                T=config.eval.eval_max_T,
                argmax=config.eval.argmax,
                plot_freq=plot_freq,
                num_parallel=config.eval.num_parallel,
                sobol_grid=config.eval.sobol_grid,
            )

            # record metrics averaged over datasets
            batch_simple_regret.append(dataset_metrics["simple_regret"])
            batch_immediate_regret.append(dataset_metrics["immediate_regret"])
            batch_inference_regret.append(dataset_metrics["inference_regret"])
            batch_cumulative_time.append(dataset_metrics["cumulative_time"])
            batch_entropy.append(dataset_metrics["entropy"])
            batch_kt_cor.append(dataset_metrics["kt_cor"])

            # log to W&B
            if config.experiment.wandb and plot_freq > 0:
                wandb.log(
                    {
                        f"plot_dataset_{b}_seed_{seed}": wandb.Image(
                            dataset_metrics["plot"]
                        )
                    }
                )

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
