import time
import os.path as osp
import os
import hydra
from wandb_wrapper import init as wandb_init
import wandb
from omegaconf import DictConfig
import torch
from utils.log import get_logger
from utils.paths import DATASETS_PATH, RESULT_PATH
from data.utils import set_all_seeds, scale_from_domain_1_to_domain_2
from data.sampler import SimpleGPSampler, OptimizationFunction
from data.evaluation import *
from policies.pbo import PBOHandler
import botorch
from typing import Union, Callable
import logging


@hydra.main(version_base=None, config_path="configs")
def main(config: DictConfig):
    if config.experiment.wandb:
        wandb_init(config=config, **config.wandb)

    # botorch datatype and device
    torch.set_default_dtype(torch.float64)
    device = "cpu"
    torch.set_default_device(device)

    pbo_handler = PBOHandler(device=device, p_noise=config.eval.p_noise)
    dataname = config.data.name
    datapath = get_evaluation_datapath(
        root=osp.join(hydra.utils.get_original_cwd(), DATASETS_PATH), dataname=dataname
    )
    if dataname.startswith("GP"):
        (
            simple_regret,
            immediate_regret,
            cumulative_regret,
            cumulative_inference_time,
        ) = evaluate_on_gp(
            config,
            pbo_handler=pbo_handler,
            datapath=datapath,
            device=device,
        )
    elif dataname.startswith("HPOB"):
        (
            simple_regret,
            immediate_regret,
            cumulative_regret,
            cumulative_inference_time,
        ) = evaluate_on_hpob(
            config,
            pbo_handler=pbo_handler,
            datapath=datapath,
            device=device,
        )
    elif dataname == "candy":
        pass
    elif dataname == "sushi":
        pass
    elif dataname in [
        "forrester1D",
        "branin2D",
        "beale2D",
        "bukin2D",
        "Hartmann",
        "Ackley",
    ]:
        evaluate_on_test_function(
            config, pbo_handler=pbo_handler, datapath=datapath, device=device
        )
    else:
        raise ValueError(f"{dataname} is not supported.")

    print(
        f"[Step={config.eval.eval_max_T}]\n simple regret: {simple_regret[-1].item(): .2f}\n cumulative regret: {cumulative_regret[-1].item(): .2f}"
    )
    # log to W&B
    if config.experiment.wandb:
        srml = wandb.plot.line(
            table=wandb.Table(
                data=[[i, simple_regret[i]] for i in range(len(simple_regret))],
                columns=["Step", "Simple Regret"],
            ),  # (H)
            x="Step",
            y="Simple Regret",
            title="Simple Regret vs Step Plot",
        )

        wandb.log(
            {
                "simple_regret_plot": srml,
                #     "immediate_regret_plot": imrml,
                #     "cumulative_regret_plot": crml,
                #     "cumulative_inference_time_plot": ctml,
                #
            }
        )


def evaluate_on_test_function(config: DictConfig, pbo_handler, datapath, device="cpu"):
    if not osp.exists(datapath):
        print(f"Generating PBBO evaluation task for {config.data.name}...")
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
        print(
            f"Generated PBBO evaluation task for {config.data.name} is saved under {str(datapath)}."
        )
    else:
        # load evaluation data
        eval_task = torch.load(
            datapath,
            map_location=config.experiment.device,
        )

    NUM_SEEDS = eval_task["num_seeds"]
    B = eval_task["num_datasets"]
    # for each run, we only evaluate on one dataset with a specified seed to enable parallelism
    if config.eval.dataset_id >= B:
        raise ValueError("Invalid dataset_id.")
    else:
        dataset_id = config.eval.dataset_id
    if config.eval.seed_id >= NUM_SEEDS:
        raise ValueError("Invalid seed_id.")
    else:
        seed_id = config.eval.seed_id
    print(f"Random seed: {seed_id}, dataset: {dataset_id}")
    TEST_X_RANGE = torch.tensor(eval_task["test_x_range"]).to(device)
    YOPT = eval_task["yopt"][dataset_id].to(device)  # (B, num_global_optima, 1)

    utility = eval_task["utility"]

    (
        simple_regret,
        immediate_regret,
        cumulative_regret,
        cumulative_inference_time,
    ) = evaluate_on_a_continuous_dataset(
        pbo_handler=pbo_handler,
        seed=seed_id,
        acq_function_type=config.experiment.model,
        yopt=YOPT,
        test_x_range=TEST_X_RANGE,
        initial_pairs=eval_task["initialization"][f"{seed_id}"]["initial_pairs"][
            dataset_id
        ],
        initial_pairs_y=eval_task["initialization"][f"{seed_id}"]["initial_pairs_y"][
            dataset_id
        ],
        initial_c=eval_task["initialization"][f"{seed_id}"]["initial_c"][dataset_id],
        eval_max_T=config.eval.eval_max_T,
        utility=utility,
    )
    # save results for the dataset and random seed
    if config.experiment.override:
        save_dir = osp.join(
            hydra.utils.get_original_cwd(),
            RESULT_PATH,
            "evaluation",
            config.data.name,
            config.experiment.model,
            str(dataset_id),
            str(seed_id),
        )

        if not osp.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        torch.save(simple_regret, f"{save_dir}/simple_regret.pt")
        torch.save(immediate_regret, f"{save_dir}/immediate_regret.pt")
        torch.save(cumulative_regret, f"{save_dir}/cumulative_regret.pt")
        torch.save(
            cumulative_inference_time, f"{save_dir}/cumulative_inference_time.pt"
        )

    return (
        simple_regret,
        immediate_regret,
        cumulative_regret,
        cumulative_inference_time,
    )


def evaluate_on_hpob(config: DictConfig, pbo_handler, datapath, device="cpu"):
    if not osp.exists(datapath):
        print("Generating evaluation task...")
        eval_task = generate_hpob_evaluation_task(
            num_seeds=config.eval.num_seeds,
            num_datasets=config.eval.num_datasets,
            d_x=config.data.d_x,
            train_x_range=config.train.x_range,
            test_x_range=config.eval.x_range,
            max_num_ctx_points=config.eval.max_num_ctx,
            num_total_points=config.eval.num_total_points,
            num_initial_pairs=config.eval.num_initial_pairs,
            **config.eval.sampler,
            p_noise=config.eval.p_noise,
            maximize=True,
            device=device,
        )
        torch.save(eval_task, datapath)
        raise NotImplementedError
    else:
        eval_task = torch.load(
            datapath,
            map_location=config.experiment.device,
        )
    NUM_SEEDS = eval_task["num_seeds"]
    B = eval_task["num_datasets"]
    # for each run, we only evaluate on one dataset with a specified seed to enable parallelism
    if config.eval.dataset_id >= B:
        raise ValueError("Invalid dataset_id.")
    else:
        dataset_id = config.eval.dataset_id
    if config.eval.seed_id >= NUM_SEEDS:
        raise ValueError("Invalid seed_id.")
    else:
        seed_id = config.eval.seed_id
    print(f"Random seed: {seed_id}, dataset: {dataset_id}")

    (
        simple_regret,
        immediate_regret,
        cumulative_regret,
        cumulative_inference_time,
    ) = evaluate_on_a_discrete_dataset(
        pbo_handler=pbo_handler,
        seed=seed_id,
        acq_function_type=config.experiment.model,
        interpolator_type=config.eval.interpolator_type,
        X_pending=eval_task["X"][dataset_id],
        y_pending=eval_task["y"][dataset_id],
        initial_pairs=eval_task["initialization"][f"{seed_id}"]["initial_pairs"][
            dataset_id
        ],
        initial_pairs_y=eval_task["initialization"][f"{seed_id}"]["initial_pairs_y"][
            dataset_id
        ],
        initial_c=eval_task["initialization"][f"{seed_id}"]["initial_c"][dataset_id],
        eval_max_T=config.eval.eval_max_T,
    )
    return (
        simple_regret,
        immediate_regret,
        cumulative_regret,
        cumulative_inference_time,
    )


def evaluate_on_gp(config: DictConfig, pbo_handler, datapath, device="cpu"):
    """Evaluate on GP with PBO."""
    if not osp.exists(datapath):
        # generate evaluation data if file not found
        print("Generating evaluation task...")
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
            device=device,
        )
        torch.save(eval_task, datapath)
    else:
        # load evaluation data
        eval_task = torch.load(
            datapath,
            map_location=config.experiment.device,
        )

    NUM_SEEDS = eval_task["num_seeds"]
    B = eval_task["num_datasets"]
    # for each run, we only evaluate on one dataset with a specified seed to enable parallelism
    if config.eval.dataset_id >= B:
        raise ValueError("Invalid dataset_id.")
    else:
        dataset_id = config.eval.dataset_id
    if config.eval.seed_id >= NUM_SEEDS:
        raise ValueError("Invalid seed_id.")
    else:
        seed_id = config.eval.seed_id
    print(f"Random seed: {seed_id}, dataset: {dataset_id}")
    TEST_X_RANGE = torch.tensor(eval_task["test_x_range"]).to(device)
    YOPT = eval_task["yopt"][dataset_id].to(device)  # (B, num_global_optima, 1)

    sampler_kwargs = eval_task["sampler_kwargs"][dataset_id]
    function_kwargs = eval_task["function_kwargs"][dataset_id]
    # Create test utility function. We first create GP sampler with defined kernel function and mean
    sampler = SimpleGPSampler(
        kernel_function=globals()[sampler_kwargs["kernel_function"]],
        mean=sampler_kwargs["mean"],
        jitter=sampler_kwargs["jitter"],
    )
    # then wrap it with `OptimizationFunction` class
    utility = OptimizationFunction(sampler=sampler, **function_kwargs)

    (
        simple_regret,
        immediate_regret,
        cumulative_regret,
        cumulative_inference_time,
    ) = evaluate_on_a_continuous_dataset(
        pbo_handler=pbo_handler,
        seed=seed_id,
        acq_function_type=config.experiment.model,
        yopt=YOPT,
        test_x_range=TEST_X_RANGE,
        initial_pairs=eval_task["initialization"][f"{seed_id}"]["initial_pairs"][
            dataset_id
        ],
        initial_pairs_y=eval_task["initialization"][f"{seed_id}"]["initial_pairs_y"][
            dataset_id
        ],
        initial_c=eval_task["initialization"][f"{seed_id}"]["initial_c"][dataset_id],
        eval_max_T=config.eval.eval_max_T,
        utility=utility,
    )
    # save results for the dataset and random seed
    if config.experiment.override:
        save_dir = osp.join(
            hydra.utils.get_original_cwd(),
            RESULT_PATH,
            "evaluation",
            config.data.name,
            config.experiment.model,
            str(dataset_id),
            str(seed_id),
        )

        if not osp.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        torch.save(simple_regret, f"{save_dir}/simple_regret.pt")
        torch.save(immediate_regret, f"{save_dir}/immediate_regret.pt")
        torch.save(cumulative_regret, f"{save_dir}/cumulative_regret.pt")
        torch.save(
            cumulative_inference_time, f"{save_dir}/cumulative_inference_time.pt"
        )

    return (
        simple_regret,
        immediate_regret,
        cumulative_regret,
        cumulative_inference_time,
    )


def evaluate_on_a_discrete_dataset(
    pbo_handler: PBOHandler,
    seed: int,
    acq_function_type: str,
    interpolator_type: str,
    X_pending: torch.Tensor,
    y_pending: torch.Tensor,
    initial_pairs: torch.Tensor,
    initial_pairs_y: torch.Tensor,
    initial_c: torch.Tensor,
    eval_max_T: int,
):
    """PBO on one dataset with discrete input space, under specific random seed."""
    set_all_seeds(seed)
    print(f"Original initial pairs: \n {initial_pairs}")

    with botorch.settings.debug(True):
        (
            _,
            _,
            _,
            simple_regret,
            immediate_regret,
            cumulative_regret,
            cumulative_inference_time,
        ) = pbo_handler.optimize_on_discrete_space(
            acq_function_type=acq_function_type,
            initial_pairs=initial_pairs,
            initial_pairs_y=initial_pairs_y,
            initial_c=initial_c,
            X_pending=X_pending,
            y_pending=y_pending,
            T=eval_max_T,
            interpolator_type=interpolator_type,
        )

    return (
        simple_regret,
        immediate_regret,
        cumulative_regret,
        cumulative_inference_time,
    )


def evaluate_on_a_continuous_dataset(
    pbo_handler: PBOHandler,
    seed: int,
    acq_function_type: str,
    yopt: torch.Tensor,
    test_x_range: torch.Tensor,
    initial_pairs: torch.Tensor,
    initial_pairs_y: torch.Tensor,
    initial_c: torch.Tensor,
    eval_max_T: int,
    utility: Union[Callable, None] = None,
):
    """PBO on one dataset with continuous input space, under specific random seed.

    Args:
        initial_pairs, (num_initial_pairs, 2 * d_x)
        initial_pairs_y, (num_initial_pairs, 2)
        initial_c, (num_initial_pairs, 2)
    """
    set_all_seeds(seed)
    d_x = len(test_x_range)

    # normalize input to (0, 1)
    unit_range = torch.stack([torch.zeros(d_x), torch.ones(d_x)]).transpose(0, 1)
    print(f"Original initial pairs: \n {initial_pairs}")

    initial_pairs = scale_from_domain_1_to_domain_2(
        x=initial_pairs.view(-1, d_x),
        bound1=test_x_range,
        bound2=unit_range,
    ).view(-1, 2 * d_x)

    with botorch.settings.debug(True):
        (
            _,
            _,
            _,
            simple_regret,
            immediate_regret,
            cumulative_regret,
            cumulative_inference_time,
        ) = pbo_handler.optimize_on_continuous_space(
            acq_function_type=acq_function_type,
            initial_pairs=initial_pairs,
            initial_pairs_y=initial_pairs_y,
            initial_c=initial_c,
            yopt=yopt,
            T=eval_max_T,
            test_x_range=test_x_range,
            utility=utility,
        )

    return (
        simple_regret,
        immediate_regret,
        cumulative_regret,
        cumulative_inference_time,
    )


if __name__ == "__main__":
    main()
