import torch
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import math
from typing import Tuple, Union, List
import scipy.stats as sps
import random
from data.kernel import *
from data.utils import *


GRID_SIZE = 1000  # sobol grid size


def transform_with_global_optimum(
    x_range: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    xopt: torch.Tensor,
    f_offset: torch.Tensor,
    maximize: bool = True,
):
    """transform utility values to ensure a global optima at xopt.
    the global maximum = - f_offset if maximize=True,
    otherwise the global minimum = f_offset.

    Args:
        x_range, (d_x, 2): data range of each input dimension.
        x, (num_points, d_x): datapoints with d_x features.
        y, (num_points, 1): associated utility values.
        xopt, (1, d_x): global minimum location.
        f_offset, (1): function offset.
        maximize, bool: global maximum or minimum.

    Returns:
        f: transformed utility values.
        yopt: the global optimum.
    """
    # broader lengthscale for wider input space
    quad_lengthscale_squared = 2 * torch.mean(
        (x_range[:, 1].float() - x_range[:, 0].float()) ** 2
    )
    quadratic_factor = 1.0 / quad_lengthscale_squared

    # transform the GP samples by taking the absolute value, adding a quadratic bowl and an offset
    f = (
        torch.abs(y)
        + quadratic_factor
        * torch.sum((x - xopt) ** 2, dim=-1, keepdim=True)  # (B, num_test_points, 1)
        + f_offset
    )
    y = f
    yopt = f_offset

    # flip the function
    if maximize:
        f = -f
        yopt = -f_offset

    return f, yopt


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
        mean = self.mean.to(test_X)  # NOTE send `mean` to test device
        if correlated:
            cov = self.kernel_function(test_X, test_X, length_scale, std)
            L = torch.linalg.cholesky(
                cov + torch.eye(len(test_X), device=cov.device) * self.jitter
            )
            return L @ torch.randn(len(test_X), n_samples, device=L.device) + mean
        else:
            var = std.pow(2) * torch.ones(len(test_X))
            mask = (
                var < self.jitter
            )  # replace var with jitter if the var is smaller than jitter
            var_jittered = torch.where(mask, torch.full_like(var, self.jitter), var)
            return mean + torch.sqrt(var_jittered).reshape(-1, 1) * torch.randn(
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
        mean = self.mean.to(train_X)  # NOTE send mean to test device

        K = self.kernel_function(train_X, train_X, length_scale, std)
        K_s = self.kernel_function(train_X, test_X, length_scale, std)

        L = torch.linalg.cholesky(K + torch.eye(len(train_X)).to(train_X) * self.jitter)
        Lk = torch.linalg.solve(L, K_s)
        Ly = torch.linalg.solve(L, train_y - mean)

        mu = mean + torch.mm(Lk.T, Ly).reshape((len(test_X),))

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
            K_ss_diag = std.pow(2) * torch.ones(len(test_X)).to(train_X)
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
        x_range: torch.Tensor,
        train_X: torch.Tensor,
        train_y: torch.Tensor,
        Xopt: torch.Tensor,
        length_scale: torch.Tensor,
        sigma_f: torch.Tensor,
        f_offset: torch.Tensor,
    ):
        """function with global optimum structue by transforming GP datapoints.

        Args:
            sampler, SimpleGPSampler: sampler for a defined GP over the input space.
            maximize, bool: whether to create function with global maximum or minimum.
            x_range, (d_x, 2): data range for each input dimension.
            train_X, [num_train_points, d_x]: training datapoints
            train_y, [num_train_points, 1]: associated utility values.
            Xopt, [1, d_x]: optimum location.
            length_scale, [d_x]: lengthscales for the GP.
            sigma_f, [1]: std for the GP.
            f_offset, [1]: function offset.
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

    def __call__(self, test_X: torch.Tensor) -> torch.Tensor:
        """Sample utility values at `test_X`.

        Args:
            test_X, (B, num_test_points, d_x): test locations.

        Returns:
            test_y, (B, num_test_points, 1): associated utility values.
        """
        # first sample from the GP
        test_y = self.sampler.sample(
            test_X=test_X,
            train_X=self.train_X.to(test_X),
            train_y=self.train_y.to(test_X),
            length_scale=self.length_scale.to(test_X),
            std=self.sigma_f.to(test_X),
            correlated=False,
        )

        # then transform the GP utility values
        test_y, self.yopt = transform_with_global_optimum(
            x_range=self.x_range.to(test_X),
            x=test_X,
            y=test_y,
            xopt=self.Xopt.to(test_X),
            f_offset=self.f_offset.to(test_X),
            maximize=self.maximize,
        )

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
            xopt, [num_opt, d_x]: location of optimum.
            yopt, [num_opt, 1]: optimal function value.
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
        x_range: List[List[float]],
        batch_size: int = 16,
        num_total_points: int = 100,
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
            Xopt, [B, num_opt, d_x]: location of optimum.
            yopt, [B, num_opt, 1]: optimal function values.
        """
        x_range = torch.tensor(x_range)
        assert x_range.shape[0] == self.d_x
        # quasi-random samples over input space
        if sobol_grid:
            # [num_total_points, d_x]
            x = torch.tensor(
                create_sobol_grid(
                    d_x=self.d_x,
                    x_range=x_range,
                    num=num_total_points,
                )
            ).float()

            # add batch dim: [batch_size, num_total_points, d_x]
            x = x.unsqueeze(0).expand(batch_size, -1, -1)

            # randomly permute the grid points
            x = x[:, torch.randperm(num_total_points)]  
        else:  # randomly sample from uniform
            x = x_range[:, 0] + (x_range[:, 1] - x_range[:, 0]) * torch.rand(
                [batch_size, num_total_points, self.d_x]
            )
        y = self.utility_function(x.reshape(-1, self.d_x)).reshape(
            batch_size, num_total_points, -1
        )

        # add batch dim for optimum
        Xopt = self.Xopt.unsqueeze(0).expand(batch_size, -1, -1).to(x)
        yopt = self.yopt.unsqueeze(0).expand(batch_size, -1, -1).to(x)
        return (
            x,
            y,
            Xopt, 
            yopt
        )


class OptimizationSampler(object):
    def __init__(
        self,
        kernel_list: list = [rbf, matern52, matern32, matern12],
        sample_kernel_weights: list = [0.25, 0.25, 0.25, 0.25],
        lengthscale_range: list = [0.05, 2],
        std_range: list = [0.1, 2],
        p_iso: float = 0.5,
        maximize: bool = True,
        seed: Union[None, int] = None,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        """Sampler for GP-based functions with global optimum structure.

        Args:
            kernel_list: list of kernel functions.
            sample_kernel_weights: weights for sampling kernel functions.
            lengthscale_range: parameterize lengthscale.
            std_range: parameterize standard deviation.
            p_iso, float: probability of isotropic lengthscale.
            maximize, bool: whether to create function with global maximum or minimum.
            seed, int: random seed.
            device: device to use. Default as "cuda".
        """
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        assert len(kernel_list) == len(sample_kernel_weights)
        assert len(lengthscale_range) == 2
        assert len(std_range) == 2

        # get kernel functions by names
        if isinstance(kernel_list[0], str):
            self.kernel_list = []
            for k in kernel_list:
                if k == "rbf":
                    self.kernel_list.append(rbf)
                elif k == "matern52":
                    self.kernel_list.append(matern52)
                elif k == "matern32":
                    self.kernel_list.append(matern32)
                elif k == "matern12":
                    self.kernel_list.append(matern12)
                else:
                    raise ValueError(f"kernel function `{k}` is not supported.")
        else:
            self.kernel_list = kernel_list

        self.sample_kernel_weights = sample_kernel_weights
        self.lengthscale_range = lengthscale_range
        self.std_range = std_range
        self.p_iso = p_iso

        self.maximize = maximize
        self.device = device

        self.utility = []

    def sample(
        self,
        batch_size: int = 64,
        max_num_ctx_points: int = 128,
        num_total_points: int = 256,
        x_range: List[List[float]] = [[-1.0, 1.0]],
        sobol_grid: bool = True,
        evaluate: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """sample a batch of datapoints from GP-based functions with global optimum structure.

        Args:
            batch_size, int: the size of data batch.
            max_num_ctx_points, int: the maximal number of context points.
            num_total_points, int: the total number of context and target points.
            x_range, list: input range.
            sobol_grid, bool: whether to sample points from a sobol sequence or a uniform distribution.
            evaluate, bool: if true, we sample `num_total_points` sobol samples over the input space.

        Returns:
            x, (B, num_total_points, d_x): input location.
            y, (B, num_total_points, 1): corresponding function values.
            xopt, (B, 1, d_x): : location of optimum.
            yopt, (B, 1, 1): optimal function value.
            utility, [B]: the functions.
        """
        (_X, _Y, _XOPT, _YOPT, _UTILITY) = ([], [], [], [], [])
        x_range = torch.tensor(x_range)  # (d_x, 2)
        for _ in range(batch_size):
            x, y, xopt, yopt, utility = self.sample_a_function(
                max_num_ctx_points=max_num_ctx_points,
                num_total_points=num_total_points,
                x_range=x_range,
                sobol_grid=sobol_grid,
                evaluate=evaluate,
            )

            _X.append(x)
            _Y.append(y)
            _XOPT.append(xopt)
            _YOPT.append(yopt)

            # save the GP curve
            if utility is not None:
                _UTILITY.append(utility)

        self.utility = _UTILITY
        return (
            torch.stack(_X, dim=0),
            torch.stack(_Y, dim=0),
            torch.stack(_XOPT, dim=0),
            torch.stack(_YOPT, dim=0),
        )

    def sample_length_scale(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """sample lengthscale using truncated log-normal distribution and standard deviation using uniform distribution.

        Returns:
            length, (d_x): lengthscale for each input dimension.
            std (1): standard deviation.
        """
        mu, sigma = np.log(1 / 3), 0.75
        a = (np.log(self.lengthscale_range[0]) - mu) / sigma
        b = (np.log(self.lengthscale_range[1]) - mu) / sigma
        rv = sps.truncnorm(a, b, loc=mu, scale=sigma)
        length = torch.tensor(
            np.exp(rv.rvs(size=(self.d_x))), device=self.device, dtype=torch.float32
        )
        # scale along with input dimension
        length *= math.sqrt(self.d_x)

        # whether the process is isotropic
        is_iso = Bernoulli(self.p_iso).sample()  # (1)
        if self.d_x > 1 and is_iso:
            # adjust lengthscale if isotropic
            length[:] = length[0]  # (d_x)

        std = (
            torch.rand(1, device=self.device) * (self.std_range[1] - self.std_range[0])
            + self.std_range[0]
        )
        return length, std

    def sample_a_function(
        self,
        x_range: torch.Tensor,
        max_num_ctx_points: int = 50,
        num_total_points: int = 100,
        sobol_grid: bool = False,
        evaluate: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        OptimizationFunction,
    ]:
        """create and sample from a function.

        Args:
            x_range, (d_x, 2): input range.
            max_num_ctx_points, int: the maximal number of context points.
            num_total_points, int: the total number of context and target points.
            sobol_grid, bool: whether to sample points from a sobol sequence or a uniform distribution.
            evaluate, bool: if true, we sample `num_total_points` sobol samples over the input space.

        Returns:
            x, (num_total_points, d_x): input location.
            y, (num_total_points, 1): corresponding function values.
            xopt, (1, d_x): :the global optimum.
            yopt, (1, 1): the global optimal value.
            utility, OptimizationFunction: the function.
        """
        # sample context size
        self.d_x = x_range.shape[0]
        num_ctx_points = random.randint(3, max_num_ctx_points)

        # sample length_scale and sigma_f for kernel
        length_scale, sigma_f = self.sample_length_scale()

        # function mean sampling
        # the number of mean samples is determined based on input range and legnthscale (prod_i range_i / lengthscale_i)
        n_temp = int(
            torch.ceil(
                torch.prod(x_range[:, 1] - x_range[:, 0]) / torch.prod(length_scale)
            )
            .int()
            .item()
        )
        temp_mean = torch.zeros(size=(n_temp,))
        temp_stds = torch.full(size=(n_temp,), fill_value=sigma_f.item())
        # sample `n_temp` means from a zero-mean Normal distribution with function standard deviation.
        # then the one with maximal absolute value is chosen as the function mean
        temp_samples = torch.abs(torch.normal(temp_mean, temp_stds))  # [n_temp]
        mean_f = temp_samples.max()  # [1]

        #  a sharp optimum with a minor probability
        p_rare, rare_tau = 0.1, 1.0
        if torch.rand(1) < p_rare:
            mean_f = mean_f + torch.exp(torch.tensor(rare_tau))

        # sample input locations
        if evaluate:
            # for evaluation, sample quasi-random samples over input space
            x_tmp = torch.tensor(
                create_sobol_grid(
                    d_x=self.d_x,
                    x_range=x_range,
                    num=num_total_points - 1,
                ),
                device=self.device,
            ).to(dtype=torch.float32)
            x_tmp = x_tmp[torch.randperm(num_total_points - 1)]
            x_fix = x_tmp[: num_ctx_points - 1]
            x_ind = x_tmp[num_ctx_points - 1 :]
        else:
            # sampled quasi-random samples from a sufficiently dense grid
            if sobol_grid:
                x_tmp = torch.tensor(
                    create_sobol_grid(
                        d_x=self.d_x,
                        x_range=x_range,
                        num=GRID_SIZE,
                    ),
                    device=self.device,
                ).to(dtype=torch.float32)

                # randomly permute the grid points
                perm_idx = torch.randperm(GRID_SIZE)

                x_fix = x_tmp[perm_idx[: num_ctx_points - 1]]
                x_ind = x_tmp[perm_idx[num_ctx_points:num_total_points]]
            else:
                # uniform samples
                x_fix = x_range[:, 0] + (x_range[:, 1] - x_range[:, 0]) * torch.rand(
                    [num_ctx_points - 1, self.d_x],
                    device=self.device,
                )
                x_ind = x_range[:, 0] + (x_range[:, 1] - x_range[:, 0]) * torch.rand(
                    [num_total_points - num_ctx_points, self.d_x],
                    device=self.device,
                )

        # sample the optimum location and set utility value as zero
        xopt = (
            torch.rand(self.d_x, device=self.device) * (x_range[:, 1] - x_range[:, 0])
            + x_range[:, 0]
        ).view(1, -1)
        yopt = torch.zeros(size=(1, 1))

        # sample a kernel
        kernel = random.choices(
            population=self.kernel_list, weights=self.sample_kernel_weights, k=1
        )[0]

        # define a GP sampler
        gp_sampler = SimpleGPSampler(mean=mean_f, kernel_function=kernel)

        # correlated samples at fixed locations when conditioned on the optimum points
        y_fix = gp_sampler.sample(
            test_X=x_fix,
            train_X=xopt,
            train_y=yopt,
            length_scale=length_scale,
            std=sigma_f,
        )
        x_fix = torch.cat([x_fix, xopt], dim=0)
        y_fix = torch.cat([y_fix, yopt], dim=0)

        # uncorrelated samples at the rest of locations
        y_ind = gp_sampler.sample(
            test_X=x_ind,
            train_X=x_fix,
            train_y=y_fix,
            length_scale=length_scale,
            std=sigma_f,
            correlated=False,  # NOTE speed up the sampling
        )

        # combine data
        x = torch.cat([x_ind, x_fix], dim=0)
        y = torch.cat([y_ind, y_fix], dim=0)

        # sample a random function offset
        f_offset_range = [-5, 5]
        f_offset = (
            torch.rand(1) * (f_offset_range[1] - f_offset_range[0]) + f_offset_range[0]
        ).view(1, -1)

        # create function with global optimum structure by transforming the GP samples
        if evaluate:
            utility = OptimizationFunction(
                sampler=gp_sampler,
                maximize=self.maximize,
                x_range=x_range,
                train_X=x,
                train_y=y,
                Xopt=xopt,
                length_scale=length_scale,
                sigma_f=sigma_f,
                f_offset=f_offset,
            )
            y = utility(test_X=x)
            yopt = utility.yopt
        else:
            utility = None
            y, yopt = transform_with_global_optimum(
                x_range=x_range,
                x=x,
                y=y,
                xopt=xopt,
                f_offset=f_offset,
                maximize=self.maximize,
            )
        return x, y, xopt, yopt, utility
