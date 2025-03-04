import os.path as osp
import os
import hydra
from wandb_wrapper import init as wandb_init
import wandb
from omegaconf import DictConfig
import torch
from utils.paths import DATASETS_PATH, RESULT_PATH
from data.utils import set_all_seeds, scale_from_domain_1_to_domain_2
from data.sampler import SimpleGPSampler, OptimizationFunction
from data.evaluation import *
from policies.pbo import PBOHandler
import botorch
from typing import Union, Callable


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
    print(f"PBO on {dataname} with acquisition function {config.experiment.model}")
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
        )
    else: 
        (
            simple_regret,
            immediate_regret,
            cumulative_regret,
            cumulative_inference_time,
        ) = evaluate_on_test_function(
            config, pbo_handler=pbo_handler, datapath=datapath
        )

    print(
        f"[Step={config.eval.eval_max_T}]\n simple regret: {simple_regret[-1].item(): .2f}\n cumulative regret: {cumulative_regret[-1].item(): .2f}"
    )
    # save results for the dataset
    if config.experiment.override:
        save_dir = osp.join(
            hydra.utils.get_original_cwd(),
            RESULT_PATH,
            "evaluation",
            config.data.name,
            config.experiment.model,
            str(config.eval.dataset_id),
        )

        if not osp.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        print(f"Saving results under f{str(save_dir)}...")
        torch.save(
            simple_regret,
            f"{save_dir}/simple_regret_{str(config.eval.seed_id)}.pt",
        )
        torch.save(
            immediate_regret,
            f"{save_dir}/immediate_regret_{str(config.eval.seed_id)}.pt",
        )
        torch.save(
            cumulative_regret,
            f"{save_dir}/cumulative_regret_{str(config.eval.seed_id)}.pt",
        )
        torch.save(
            cumulative_inference_time,
            f"{save_dir}/cumulative_inference_time_{str(config.eval.seed_id)}.pt",
        )

    # log to W&B
    if config.experiment.wandb:
        wandb.log(
            {
                f"simple_regret_vs_step_plot": wandb.plot.line_series(
                    xs=[*range(simple_regret.shape[-1])],  # list of x values: steps
                    ys=[
                        simple_regret.mean(dim=1)[i]
                        for i in range(simple_regret.shape[0])
                    ],  # list of y values: simple regret for each seed
                    keys=[
                        f"seed {i}" for i in range(simple_regret.shape[0])
                    ],  # line labels: seed
                    title=f"Simple Regret vs Step Plot",
                    xname="Step",
                ),
                f"immediate_regret_vs_step_plot": wandb.plot.line_series(
                    xs=[*range(immediate_regret.shape[-1])],  # list of x values: steps
                    ys=[
                        immediate_regret.mean(dim=1)[i]
                        for i in range(immediate_regret.shape[0])
                    ],  # list of y values: immediate regret for each seed
                    keys=[
                        f"seed {i}" for i in range(immediate_regret.shape[0])
                    ],  # line labels: seed
                    title=f"Immediate Regret vs Step Plot",
                    xname="Step",
                ),
            }
        )


def evaluate_on_test_function(config: DictConfig, pbo_handler, datapath):
    """evaluate on test function."""
    if not osp.exists(datapath):
        print(f"Generating PBBO evaluation task for {config.data.name}...")
        if config.data.name in ["candy", "sushi"]:
            # we do linear extrapolation on real-world discrete datasets
            if config.data.name == "candy":
                handler = CandyDataHandler(interpolator_type="linear")
            else:
                handler = SushiDataHandler(interpolator_type="linear")

            # global optimum
            _, _, Xopt, yopt = handler.get_data(add_batch_dim=True)

            eval_task = generate_real_world_evaluation_task(
                num_seeds=config.eval.num_seeds,
                handler=handler,
                test_x_range=[handler.x_range for _ in range(Xopt.shape[-1])],
                train_x_range=[config.train.x_i_range for _ in range(Xopt.shape[-1])],
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
    TEST_X_RANGE = torch.tensor(eval_task["test_x_range"]).to(config.experiment.device)
    YOPT = eval_task["yopt"][dataset_id].to(
        config.experiment.device
    )  # (B, num_global_optima, 1)
    utility = eval_task["utility"]

    (
        simple_regret,
        immediate_regret,
        cumulative_regret,
        cumulative_inference_time,
    ) = continuous(
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
    return (
        simple_regret,
        immediate_regret,
        cumulative_regret,
        cumulative_inference_time,
    )


def evaluate_on_hpob(config: DictConfig, pbo_handler, datapath):
    """evaluate on HPO-B tasks."""
    if not osp.exists(datapath):
        print(f"Generating PBBO evaluation task for {config.data.name}...")
        eval_task = generate_hpob_evaluation_task(
            num_seeds=config.eval.num_seeds,
            train_x_i_range=config.train.x_i_range,
            test_x_range=config.data.x_range,
            search_space_id=str(config.data.search_space_id),  # NOTE
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
            datapath,
            map_location=config.experiment.device,
        )

    TEST_X_RANGE = torch.tensor(eval_task["test_x_range"]).to(config.experiment.device)
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
    ) = discrete(
        pbo_handler=pbo_handler,
        seed=seed_id,
        acq_function_type=config.experiment.model,
        X_pending=eval_task["X"][dataset_id],
        y_pending=eval_task["y"][dataset_id],
        initial_pairs=eval_task["initialization"][f"{seed_id}"]["initial_pairs"][
            dataset_id
        ],
        initial_pairs_y=eval_task["initialization"][f"{seed_id}"]["initial_pairs_y"][
            dataset_id
        ],
        initial_c=eval_task["initialization"][f"{seed_id}"]["initial_c"][dataset_id],
        test_x_range=TEST_X_RANGE,
        eval_max_T=config.eval.eval_max_T,
    )
    return (
        simple_regret,
        immediate_regret,
        cumulative_regret,
        cumulative_inference_time,
    )


def evaluate_on_gp(config: DictConfig, pbo_handler, datapath):
    """PBO on GP samples.

    Args:
        config: experiment configuration.
        pbo_handler, PBOHandler: PBO class.
        datapath, str: where evaluation data is stored.
        device, str: tensor device.
    """
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
            device=config.experiment.device,
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

    TEST_X_RANGE = torch.tensor(eval_task["test_x_range"]).to(config.experiment.device)
    YOPT = eval_task["yopt"][dataset_id].to(
        config.experiment.device
    )  # (B, num_global_optima, 1)
    sampler_kwargs = eval_task["sampler_kwargs"][dataset_id]
    function_kwargs = eval_task["function_kwargs"][dataset_id]

    # sampler for GP curves
    sampler = SimpleGPSampler(
        kernel_function=globals()[sampler_kwargs["kernel_function"]],
        mean=sampler_kwargs["mean"],
        jitter=sampler_kwargs["jitter"],
    )

    # transform the GP curves into utility functions with global optimum structure
    utility = OptimizationFunction(sampler=sampler, **function_kwargs)

    # optimize on continuous space
    (
        simple_regret,
        immediate_regret,
        cumulative_regret,
        cumulative_inference_time,
    ) = continuous(
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
    return (
        simple_regret,
        immediate_regret,
        cumulative_regret,
        cumulative_inference_time,
    )


def discrete(
    pbo_handler: PBOHandler,
    seed: int,
    acq_function_type: str,
    X_pending: torch.Tensor,
    y_pending: torch.Tensor,
    test_x_range: torch.Tensor,
    initial_pairs: torch.Tensor,
    initial_pairs_y: torch.Tensor,
    initial_c: torch.Tensor,
    eval_max_T: int,
):
    """PBO on a discrete dataset.

    Args:
        acq_function_type, str in ["rs", "qEI", "qEUBO", "qNEI", "qTS"]: acquisition function type.
        X_pending, (num_points, d_x): candidate points.
        y_pending, (num_points, 1): associated utility values.
        initial_pairs, (num_initial_pairs, 2 * d_x): starting pairs.
        initial_pairs_y, (num_initial_pairs, 2): associated utility values.
        initial_c, (num_initial_pairs, 1): preference.
        eval_max_T, scalar: time budget.

    Returns:
        simple_regret, (eval_max_T+1): simple regret at each step.
        immediate_regret, (eval_max_T+1): immediate regret at each step.
        cumulative_regret, (eval_max_T+1): cumulative regret at each step.
        cumulative_inference_time, (eval_max_T): cumulative inference time at each step.

    """
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
            test_x_range=test_x_range,
            T=eval_max_T,
        )

    return (
        simple_regret,
        immediate_regret,
        cumulative_regret,
        cumulative_inference_time,
    )


def continuous(
    pbo_handler: PBOHandler,
    seed: int,
    acq_function_type: str,
    yopt: torch.Tensor,
    test_x_range: torch.Tensor,
    initial_pairs: torch.Tensor,
    initial_pairs_y: torch.Tensor,
    initial_c: torch.Tensor,
    eval_max_T: int,
    utility: Callable,
):
    """PBO on one continuous data.

    Args:
        acq_function_type, str in ["rs", "qEI", "qEUBO", "qNEI", "qTS", "mpes"]: acquisition function type.
        yopt, (num_global_optimum, 1): global optimum.
        test_x_range, (d_x , 2): data range of input.
        initial_pairs, (num_initial_pairs, 2 * d_x): starting pairs.
        initial_pairs_y, (num_initial_pairs, 2): associated utility values.
        initial_c, (num_initial_pairs, 1): preference.
        eval_max_T, scalar: time budget.
        utility, Callable: utility function.

    Returns:
        simple_regret, (eval_max_T+1): simple regret at each step.
        immediate_regret, (eval_max_T+1): immediate regret at each step.
        cumulative_regret, (eval_max_T+1): cumulative regret at each step.
        cumulative_inference_time, (eval_max_T): cumulative inference time at each step.
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
