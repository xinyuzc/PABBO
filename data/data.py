import torch
import numpy as np
import math
from typing import Tuple, Union

import random
import sobol_seq


def rbf(
    x1: torch.Tensor,
    x2: torch.Tensor,
    length_scale: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """rbf kernel for N samples of d_x dimension, return the covariance matrix of shape [N, M]
    Args:
        x1, [N, D]: input 1.
        x2, [M, D]: input 2.
        length_scale, [d_x]: length scales over each input dimension.
        std, [1]: function scale.
    Returns:
        cov, [N, M]: covariance matrix.
    """
    dist = (x1.unsqueeze(-2) - x2.unsqueeze(-3)) / length_scale
    cov = std.pow(2) * torch.exp(-0.5 * dist.pow(2).sum(-1))
    return cov


def matern52(
    x1: torch.Tensor,
    x2: torch.Tensor,
    length_scale: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """matern52 kernel for N samples of d_x dimension, return the covariance matrix of shape [N, M]
    Args:
        x1, [N, D]: input 1.
        x2, [M, D]: input 2.
        length_scale, [d_x]: length scales over each input dimension.
        std, [1]: function scale.
    Returns:
        cov, [N, M]: covariance matrix.
    """
    dist = torch.norm((x1.unsqueeze(-2) - x2.unsqueeze(-3)) / length_scale, dim=-1)
    cov = (
        std.pow(2)
        * (1 + math.sqrt(5.0) * dist + 5.0 * dist.pow(2) / 3.0)
        * torch.exp(-math.sqrt(5.0) * dist)
    )
    return cov


def matern32(
    x1: torch.Tensor,
    x2: torch.Tensor,
    length_scale: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """matern32 kernel for N samples of d_x dimension, return the covariance matrix of shape [N, M]
    Args:
        x1, [N, D]: input 1.
        x2, [M, D]: input 2.
        length_scale, [d_x]: length scales over each input dimension.
        std, [1]: function scale.
    Returns:
        cov, [N, M]: covariance matrix.
    """
    dist = torch.norm((x1.unsqueeze(-2) - x2.unsqueeze(-3)) / length_scale, dim=-1)
    cov = std.pow(2) * (1 + math.sqrt(3.0) * dist) * torch.exp(-math.sqrt(3.0) * dist)
    return cov


def matern12(
    x1: torch.Tensor,
    x2: torch.Tensor,
    length_scale: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """matern32 kernel for N samples of d_x dimension, return the covariance matrix of shape [N, M]
    Args:
        x1, [N, D]: input 1.
        x2, [M, D]: input 2.
        length_scale, [d_x]: length scales over each input dimension.
        std, [1]: function scale.
    Returns:
        cov, [N, M]: covariance matrix.
    """
    dist = torch.norm((x1.unsqueeze(-2) - x2.unsqueeze(-3)) / length_scale, dim=-1)
    cov = std.pow(2) * torch.exp(-1.0 * dist)
    return cov


def scale_from_domain_1_to_domain_2(
    x: torch.Tensor,
    bound1: torch.Tensor,
    bound2: torch.Tensor,
) -> torch.Tensor:
    tmp = (x - bound1[..., 0]) / (bound1[..., 1] - bound1[..., 0])
    return bound2[..., 0] + tmp * (bound2[..., 1] - bound2[..., 0])


def scale_from_unit_to_domain(X: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
    return bound[..., 0] + X * (bound[..., 1] - bound[..., 0])


def create_sobol_grid(d_x: int, num: int, x_range=[[0, 1]]):
    """sample sobol grid locations and scale to target range.

    Args:
        d_x, scalar: input dimension.
        x_range, list: bound for each dimension
        num, scalar: number of samples.

    Returns:
        loc, [num, d_x]
    """
    loc = sobol_seq.i4_sobol_generate(dim_num=d_x, n=num)  # \in (0, 1)
    if not isinstance(x_range, np.ndarray):
        x_range = np.array(x_range)
    loc = scale_from_unit_to_domain(loc, x_range)
    return loc


class SimpleGPSampler(object):
    def __init__(self, mean: torch.Tensor, kernel_function: rbf, jitter=1e-4):
        """A GP sampler.

        Args:
            mean, [1]: function mean.
            kernel_function, K(x1, x2, lengthscale, std): kernel function.
        """
        self.kernel_function = kernel_function
        self.mean = mean  # [1]
        self.jitter = jitter

    def prior(self, test_X, length_scale, std, n_samples=1, correlated=True):
        if correlated:
            cov = self.kernel_function(test_X, test_X, length_scale, std)
            L = torch.linalg.cholesky(
                cov + torch.eye(len(test_X), device=cov.device) * self.jitter
            )
            return L @ torch.randn(len(test_X), n_samples, device=L.device) + self.mean
        else:
            var = std.pow(2) * torch.ones(len(test_X))
            mask = (
                var < self.jitter
            )  # replace var with jitter if the var is smaller than jitter
            var_jittered = torch.where(mask, torch.full_like(var, self.jitter), var)
            return self.mean + torch.sqrt(var_jittered).reshape(-1, 1) * torch.randn(
                len(test_X), n_samples
            )

    def posterior(
        self,
        test_X,
        train_X,
        train_y,
        length_scale,
        std,
        n_samples=1,
        correlated=True,
    ):
        K = self.kernel_function(train_X, train_X, length_scale, std)
        K_s = self.kernel_function(train_X, test_X, length_scale, std)

        L = torch.linalg.cholesky(K + torch.eye(len(train_X)).to(train_X) * self.jitter)
        Lk = torch.linalg.solve(L, K_s)
        Ly = torch.linalg.solve(L, train_y - self.mean)

        mu = self.mean + torch.mm(Lk.T, Ly).reshape((len(test_X),))

        if correlated:
            K_ss = self.kernel_function(test_X, test_X, length_scale, std)
            # , periodicity)
            L = torch.linalg.cholesky(
                K_ss
                + torch.eye(len(test_X)).to(train_X) * self.jitter
                - torch.mm(Lk.T, Lk)
            )
            f_post = mu.reshape(-1, 1) + torch.mm(
                L, torch.randn(len(test_X), n_samples).to(train_X)
            )
        else:
            K_ss_diag = std.pow(2) * torch.ones(len(test_X))
            var = K_ss_diag - torch.sum(Lk**2, dim=0)
            mask = (
                var < self.jitter
            )  # replace var with jitter if the var is smaller than jitter
            var_jittered = torch.where(mask, torch.full_like(var, self.jitter), var)
            f_post = mu.reshape(-1, 1) + torch.sqrt(var_jittered).reshape(
                -1, 1
            ) * torch.randn(len(test_X), n_samples).to(train_X)

        return f_post

    def sample(
        self,
        test_X,
        train_X=None,
        train_y=None,
        length_scale=None,
        std=None,
        n_samples: int = 1,
        correlated: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample from the GP.

        Parameters:
        test_X, [num_test, D]: input locations to predict.
        train_X, ([num_train, D], optional): training points.
        train_y,  ([num_train, 1], optional): function values at training points.
        length_scale: length scale for kernel.
        std: sigma for kernel.
        mean_f: function mean.
        n_samples: number of samples to draw.
        correlated: whether to draw correlated samples.

        Returns:
            f_post, [num_test, n_samples]: samples from the GP.
        """
        self.length_scale = length_scale
        self.std = std
        if train_X is None or train_y is None or len(train_X) == 0 or len(train_y) == 0:
            return self.prior(test_X, length_scale, std, n_samples, correlated)
        else:
            return self.posterior(
                test_X,
                train_X,
                train_y,
                length_scale,
                std,
                n_samples,
                correlated,
            )


class OptimizationFunction:
    def __init__(
        self,
        sampler: SimpleGPSampler,
        maximize: bool,
        x_range: list,
        train_X: torch.Tensor,
        train_y: torch.Tensor,
        Xopt: torch.Tensor,
        length_scale: torch.Tensor,
        sigma_f: torch.Tensor,
        f_offset: torch.Tensor,
    ):
        """A function wrapper for a GP sampler.

        Args:
            sampler: a GP sampler with assigned training data and kernel hyperparameters.
            maximize, bool: whether to maximize on the GP curve.
            x_range: same input range for all dimensions.
            train_X, [num_train, d_x]: training points
            train_y, [num_train, 1]: function values at training points.
            Xopt, [1, d_x]: optimum.
            length_scale, [d_x]: length scales over dimensions.
            sigma_f, [1]: function scale.
            f_offset, [1]: function offset from zero mean.
        """
        self.sampler = sampler
        self.maximize = maximize
        self.x_range = x_range
        self.d_x = train_X.shape[-1]
        self.train_X = train_X
        self.train_y = train_y
        self.Xopt = Xopt
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.f_offset = f_offset

    def __call__(self, test_X):
        # sample from the GP
        test_X = test_X.to(self.train_X)
        test_y = self.sampler.sample(
            test_X=test_X,
            train_X=self.train_X,
            train_y=self.train_y,
            length_scale=self.length_scale,
            std=self.sigma_f,
            correlated=False,
        )

        # add a broad quadratic bowl to the optima
        quad_lengthscale_squared = (
            2 * (float(self.x_range[1]) - float(self.x_range[0])) ** 2
        ) * self.d_x
        quadratic_factor = 1.0 / quad_lengthscale_squared
        f = (
            torch.abs(test_y)
            + quadratic_factor
            * torch.sum((test_X - self.Xopt) ** 2, dim=-1, keepdim=True)
            + self.f_offset
        )
        test_y = f

        # flip the function values if maximize
        if self.maximize:
            test_y = -test_y
        return test_y


class UtilitySampler(object):
    def __init__(
        self,
        d_x: int,
        utility_function,
        Xopt: torch.Tensor,
        yopt: torch.Tensor,
    ):
        """Sample data from a given utility function.

        Args:
            d_x, scalar: input dimension.
            utility_function: y=f(x).
            xopt, [num_opt, d_x]: optimum. Default as maximum.
            yopt, [num_opt, 1]: optimal function values.
        """
        self.d_x = d_x
        self.utility_function = utility_function
        self.Xopt = Xopt
        self.yopt = yopt

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """compute function values at given location x.
        Args:
            x, [B, N, d_x]
        """
        y = self.utility_function(x)
        return y

    def sample(
        self,
        batch_size: int = 16,
        num_total_points: int = 100,
        x_range=[[0, 1]],
        sobol_grid: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """sample a batch of points from the utility function.

        Args:
            batch_size, scalar.
            num_points, scalar: the number of points in a dataset.
            x_range, list: bounds over input dimensions.
            sobol_grid, bool: whether to sample from sobol sequences.

        Returns:
            x, [B, N, d_x]: sampled locations.
            y, [B, N, 1]: associated function values.
            Xopt, [B, num_opt, d_x]: optimum.
            yopt, [B, num_opt, 1]: optimal function values.
        """
        bound = torch.tensor(x_range)
        if sobol_grid:

            x = (
                torch.tensor(
                    create_sobol_grid(
                        d_x=self.d_x,
                        x_range=x_range,
                        num=num_total_points,
                    )
                )
                .to(dtype=torch.float32)[None, :, :]
                .tile(batch_size, 1, 1)
            )
            x = x[:, torch.randperm(num_total_points)]  # NOTE
        else:
            x = bound[:, 0] + (bound[:, 1] - bound[:, 0]) * torch.rand(
                [batch_size, num_total_points, self.d_x]
            )
        y = self.utility_function(x.reshape(-1, self.d_x)).reshape(
            batch_size, num_total_points, -1
        )
        return (
            x,
            y,
            self.Xopt[None, :, :].tile(batch_size, 1, 1).to(x.device),
            self.yopt[None, :, :].tile(batch_size, 1, 1).to(x.device),
        )


def sample_from_full_combination(
    num_total_points: int,
    device: torch.device,
    num_random_pairs: Union[None, int] = None,
) -> torch.Tensor:
    """sample pairs from the full combinations of points.

    Args:
        num_total_points, scalar: number of points.
        num_random_pairs, scalar: number of sampled pairs. If None, all pairs from the full combinations are sampled.

    Returns:
        pair_idx, [M, 2]: indices for sampled pairs.
    """
    num_total_pairs = num_total_points * (num_total_points - 1) // 2
    if num_random_pairs is not None and (
        num_random_pairs < num_total_pairs
    ):  # sample a subset
        pair_idx, _ = get_combinations_subset(
            num_query_points=num_total_points,
            num_random_pairs=num_random_pairs,
            device=device,
        )
    else:
        pair_idx = torch.combinations(
            torch.arange(0, num_total_points)
        )  # return the full combinations
    return pair_idx


def get_combinations_subset(
    device: torch.device,
    num_query_points: int,
    num_random_pairs: int,
) -> Tuple[torch.Tensor, int]:
    """Sample a subset of the full combinations.

    Args:
        num_total_points, scalar: number of points.
        num_random_pairs, scalar: number of sampled pairs. If None, all pairs from the full combinations are sampled.

    Returns:
        subset_pair_idx, [M, 2]: indices for sampled pairs.
        scalar, number of sampled pairs.
    """
    subset_pair_idx = []
    while len(subset_pair_idx) < num_random_pairs:
        idx = list(
            sorted(random.sample(range(num_query_points), 2))
        )  # NOTE sort the combination
        if idx not in subset_pair_idx:
            subset_pair_idx.append(idx)
    subset_pair_idx = torch.tensor(subset_pair_idx, dtype=torch.int64, device=device)
    return subset_pair_idx, len(subset_pair_idx)


def get_user_preference(
    pair_y: torch.Tensor, maximize: bool = True, p_noise: Union[float, None] = None
) -> torch.Tensor:
    """generates preference label given a pair of function values.

    Args:
        pairs_y, [..., 2]: function values for pairs.
        maximize, bool: whether to maximize the utility function. default as True.
        p_noise, scalar: observed Gaussian noise. Default as 0.1

    Returns:
        c, [..., 1]: the preference label.
    """
    p_noise = p_noise if p_noise is not None else 0.1
    pair_y_obs = pair_y + p_noise * torch.randn_like(pair_y, device=pair_y.device)
    c = (
        (pair_y_obs[..., 0] - pair_y_obs[..., 1] <= 0).float()
        if maximize
        else (pair_y_obs[..., 0] - pair_y_obs[..., 1] > 0).float()
    )
    return c.unsqueeze(-1)


def gather_data_at_index(data: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Retrieves specific elements from data based on indices.
    data, [B, N, d]: a batch of sequence consisting of N d-dimensional elements.
    idx, [B, M, 1]: M refers to the number of indices to gather.
    """
    B, N, d = data.shape
    tile_idx = idx.tile(1, 1, d)
    ele = torch.gather(input=data, dim=1, index=tile_idx)
    return ele
