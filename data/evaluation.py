from typing import List, Dict, Union
import torch
from data.sampler import OptimizationSampler
from data.kernel import *
from data.utils import set_all_seeds, sample_from_full_combination
from data.environment import get_user_preference
from data.function import *
from data.candy_data_handler import *
from data.sushi_data_handler import *
from data.hpob import *
import os.path as osp


def get_evaluation_datapath(root, dataname):
    return osp.join(root, f"{dataname}_evaluation_task.pt")


def generate_test_function_evaluation_task(
    dataname: str,
    num_seeds: int,
    test_x_range: list,
    train_x_range: list,
    Xopt: torch.Tensor,
    yopt: torch.Tensor,
    num_initial_pairs: int = 1,
    p_noise: float = 0.0,
) -> Dict:
    """Generate initialization for preferential black-box optimization task from test functions.

    Args:
        num_seeds: the number of random seeds.
        test_x_range, List[d_x, 2]: the input range of test data.
        train_x_range, List[d_x, 2]: the input range of pre-training data. Saved for future scaling.
        num_initial_pairs, int: number of initial pairs.
        Xopt, (num_global_optima, d_x): the global optimum locations.
        yopt, (num_global_optima, 1): global optimum values.
        num_initial_pairs, int: number of initial pairs for each dataset.
        p_noise, float: the observed noise when providing preference.

    Returns:
        tasks, Dict: PBBO evaluation intialization generated from the test function.
    """
    assert len(test_x_range) == len(train_x_range)
    assert len(test_x_range[0]) == len(train_x_range[0])

    # call function in `data/function`
    if not dataname in globals():
        raise NotImplementedError(f"{dataname} is not supported.")
    utility = globals()[dataname]

    tasks = {
        "num_seeds": num_seeds,
        "num_datasets": 1,
        "test_x_range": test_x_range,
        "train_x_range": train_x_range,
        "Xopt": Xopt[None, :, :],  # only one dataset
        "yopt": -yopt[None, :, :],  # NOTE negate
        "utility": utility,
    }
    # generate intial pairs with different random seeds
    tasks["initialization"] = {}
    bound = torch.tensor(test_x_range)
    d_x = Xopt.shape[-1]
    for seed in range(num_seeds):
        set_all_seeds(seed)
        x = bound[:, 0] + torch.rand(2 * num_initial_pairs, d_x) * (
            bound[:, 1] - bound[:, 0]
        )
        y = utility(
            x=x, negate=True, add_dim=True
        )  # (num_datasets, 2 * num_initial_pairs, 1)
        initial_pairs = x.reshape(1, num_initial_pairs, -1)
        initial_pairs_y = y.reshape(1, num_initial_pairs, -1)
        initial_c = get_user_preference(
            pair_y=initial_pairs_y, maximize=True, p_noise=p_noise
        )
        tasks["initialization"][str(seed)] = {
            "initial_pairs": initial_pairs,
            "initial_pairs_y": initial_pairs_y,
            "initial_c": initial_c,
        }
    return tasks


def generate_real_world_evaluation_task(
    num_seeds: int,
    handler: Union[CandyDataHandler, SushiDataHandler],
    test_x_range: list,
    train_x_range: list,
    device: str,
    num_initial_pairs: int = 1,
    p_noise: float = 0.0,
) -> Dict:
    """Generate initialization for preferential black-box optimization task from real-world datasets.

    Args:
        num_seeds: the number of random seeds.
        handler, Callable: datahandler for real-world dataset `candy` or `sushi`.
        test_x_range, List[d_x, 2]: the input range of test data.
        train_x_range, List[d_x, 2]: the input range of pre-training data. Saved for future scaling.
        device, str: computational device, `cpu` or `cuda`.
        num_initial_pairs, int: number of initial pairs.
        p_noise, float: the observed noise when providing preference.

    Returns:
        tasks, Dict: PBBO evaluation intialization generated from real-world datasets.
    """
    assert len(test_x_range) == len(train_x_range)
    assert len(test_x_range[0]) == len(train_x_range[0])
    tasks = {
        "num_seeds": num_seeds,
        "num_datasets": 1,
        "test_x_range": test_x_range,
        "train_x_range": train_x_range,
    }

    tasks["initialization"] = {}
    X, y, _, _ = handler.get_data(add_batch_dim=True)

    for seed in range(num_seeds):
        set_all_seeds(seed)
        # sample from true data points
        indices = sample_from_full_combination(
            num_total_points=X.shape[1],
            device=device,
            num_random_pairs=num_initial_pairs,
        )
        initial_pairs = torch.from_numpy(X[:, indices]).flatten(start_dim=-2)
        initial_pairs_y = torch.from_numpy(y[:, indices]).flatten(start_dim=-2)
        initial_c = get_user_preference(
            pair_y=initial_pairs_y, maximize=True, p_noise=p_noise
        )
        tasks["initialization"][str(seed)] = {
            "initial_pairs": initial_pairs,
            "initial_pairs_y": initial_pairs_y,
            "initial_c": initial_c,
        }
    return tasks


def generate_gp_evaluation_task(
    num_seeds: int,
    num_datasets: int,
    test_x_range: list,
    train_x_range=list,
    max_num_ctx_points: int = 50,
    num_total_points: int = 100,
    num_initial_pairs: int = 1,
    kernel_list: list = [rbf, matern52, matern32, matern12],
    sample_kernel_weights: list = [0.25, 0.25, 0.25, 0.25],
    lengthscale_range: list = [0.05, 2],
    std_range: list = [0.1, 2],
    p_iso: float = 0.5,
    p_noise: float = 0.0,
    device: str = "cuda",
) -> Dict:
    """Generate initialization for preferential black-box optimization task from GP-based synthetic functions.

    Args:
        num_seeds: the number of random seeds.
        num_datasets: batch size.
        test_x_range, List[d_x, 2]: the input range of test data.
        train_x_range, List[d_x, 2]: the input range of PABBO pre-training.
        max_num_ctx_points, scalar: when sampling from a GP, we first sample some context points.
        num_total_points, scalar: then we sample the rest of total points as uncorrelated samples to speed up sampling. Refers to `OptimizationSampler` for details.
        num_initial_pairs, int: number of initial pairs for each PBBO dataset.
        kernel_list, List: we randomly sample a kernel.
        sample_kernel_weights, List: probabilities of each kernel being sampled.
        lengthscale_range, List: the range of kernel lengthscales.
        std_range: List: the range of kernel standard deviations.
        p_iso, float: the probability of isotropic lengthscale.
        p_noise, float: the observed noise when provding preference.

    Returns:
        tasks, Dict: PBBO evaluation intialization generated from GP-based synthetic functions.
    """
    assert len(test_x_range) == len(train_x_range)
    assert len(test_x_range[0]) == len(train_x_range[0])
    tasks = {
        "num_seeds": num_seeds,
        "num_datasets": num_datasets,
        "test_x_range": test_x_range,
        "train_x_range": train_x_range,
    }
    sampler = OptimizationSampler(
        kernel_list=kernel_list,
        sample_kernel_weights=sample_kernel_weights,
        lengthscale_range=lengthscale_range,
        std_range=std_range,
        p_iso=p_iso,
        maximize=True,
        device=device,
    )
    X, y, Xopt, yopt = sampler.sample(
        batch_size=num_datasets,
        max_num_ctx_points=max_num_ctx_points,
        num_total_points=num_total_points,
        x_range=test_x_range,
        evaluate=True,
    )
    tasks["Xopt"] = Xopt
    tasks["yopt"] = yopt

    # save attributes in each `OptimizationFunction` utility
    SAMPLER_KWARGS = list()
    FUNCTION_KWARGS = list()
    for u in sampler.utility:
        s = u.sampler
        function_kwargs = {
            "maximize": u.maximize,
            "x_range": u.x_range,
            "train_X": u.train_X,
            "train_y": u.train_y,
            "Xopt": u.Xopt,
            "length_scale": u.length_scale,
            "sigma_f": u.sigma_f,
            "f_offset": u.f_offset,
        }
        sampler_kwargs = {
            "kernel_function": s.kernel_function.__name__,
            "mean": s.mean,
            "jitter": s.jitter,
        }
        FUNCTION_KWARGS.append(function_kwargs)
        SAMPLER_KWARGS.append(sampler_kwargs)

    tasks["sampler_kwargs"] = SAMPLER_KWARGS
    tasks["function_kwargs"] = FUNCTION_KWARGS

    # generate intial pairs with different random seeds
    tasks["initialization"] = {}
    for seed in range(num_seeds):
        set_all_seeds(seed)
        # sample pairs on previously sampled points
        indices = sample_from_full_combination(
            num_total_points=X.shape[1],
            device=device,
            num_random_pairs=num_initial_pairs,
        )
        initial_pairs = X[:, indices].flatten(start_dim=-2)
        initial_pairs_y = y[:, indices].flatten(start_dim=-2)
        initial_c = get_user_preference(
            pair_y=initial_pairs_y, maximize=True, p_noise=p_noise
        )
        tasks["initialization"][str(seed)] = {
            "initial_pairs": initial_pairs,
            "initial_pairs_y": initial_pairs_y,
            "initial_c": initial_c,
        }
    return tasks


def generate_hpob_evaluation_task(
    num_seeds: int,
    search_space_id: str,
    standardize: bool,
    test_x_range: list,
    train_x_i_range: list,
    num_initial_pairs: int = 1,
    p_noise: float = 0.0,
    device: str = "cuda",
) -> Dict:
    """Generate initialization for preferential black-box optimization task from GP-based synthetic functions."""
    handler = HPOBHandler(root_dir="datasets/hpob-data", mode="v3-test")
    X, y, Xopt, yopt = handler.sample(
        search_space_id=search_space_id,
        num_total_points=1024,  # evaluate on pairs generated from 1024 samples
        standardize=standardize,
        split="test",
        device=device,
    )
    num_datasets = len(X)
    d_x = X.shape[-1]
    tasks = {
        "X": X,
        "y": y,
        "Xopt": Xopt,
        "yopt": yopt,
        "test_x_range": test_x_range,
        "train_x_range": [train_x_i_range for _ in range(d_x)],
        "num_seeds": num_seeds,
        "num_datasets": num_datasets,
    }
    tasks["initialization"] = {}
    for seed in range(num_seeds):
        set_all_seeds(seed)
        # sample pairs on previously sampled points
        indices = sample_from_full_combination(
            num_total_points=X.shape[1],
            device=X.device,
            num_random_pairs=num_initial_pairs,
        )
        initial_pairs = X[:, indices].flatten(start_dim=-2)
        initial_pairs_y = y[:, indices].flatten(start_dim=-2)
        initial_c = get_user_preference(
            pair_y=initial_pairs_y, maximize=True, p_noise=p_noise
        )
        tasks["initialization"][str(seed)] = {
            "initial_pairs": initial_pairs,
            "initial_pairs_y": initial_pairs_y,
            "initial_c": initial_c,
        }
    return tasks
