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
        mean = self.mean.to(test_X)  # NOTE send mean to test device

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
        x_range: torch.Tensor,
        train_X: torch.Tensor,
        train_y: torch.Tensor,
        Xopt: torch.Tensor,
        length_scale: torch.Tensor,
        sigma_f: torch.Tensor,
        f_offset: torch.Tensor,
    ):
        """A wrapper for a GP curve with injected global optimal structure.

        Args:
            sampler, SimpleGPSampler: a GP sampler with assigned training data and kernel hyperparameters.
            maximize, bool: function with maximum or minimimum.
            x_range, (d_x, 2): input range.
            train_X, [num_train, d_x]: training points
            train_y, [num_train, 1]: function values at training points.
            Xopt, [1, d_x]: optimum.
            length_scale, [d_x]: length scales over dimensions.
            sigma_f, [1]: function scale.
            f_offset, [1]: function offset from zero mean.
        """
        # TODO fix bugs regarding device: can only be evaluated on device on which training data were.
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
        """Sample function values at `test_X` from the GP curve.
        Args:
            test_X, (B, num_test_points, d_x): test locations.

        Returns:
            test_y, (B, num_test_points, 1): sampled function values.
        """
        # sample from the GP
        device = test_X.device
        test_X = test_X.to(self.train_X)
        test_y = self.sampler.sample(
            test_X=test_X,
            train_X=self.train_X,
            train_y=self.train_y,
            length_scale=self.length_scale,
            std=self.sigma_f,
            correlated=False,
        )

        # add the quadratic bowl
        quad_lengthscale_squared = torch.sum(
            (2 * (self.x_range[:, 1].float() - self.x_range[:, 0].float())) ** 2
        )  # (1)

        quadratic_factor = 1.0 / quad_lengthscale_squared
        f = (
            torch.abs(test_y)
            + quadratic_factor
            * torch.sum(
                (test_X - self.Xopt) ** 2, dim=-1, keepdim=True
            )  # (B, num_test_points, 1)
            + self.f_offset
        )
        test_y = f

        # flip the function values if doing maximization
        if self.maximize:
            test_y = -test_y

        return test_y.to(device)


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
        batch_size: int = 16,
        num_total_points: int = 100,
        x_range: List[List[float]] = [[0.0, 1.0]],
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

        # quasi-random samples over input space
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
        else:  # randomly sample from uniform
            x = x_range[:, 0] + (x_range[:, 1] - x_range[:, 0]) * torch.rand(
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
        """sampler for GP samples with a global optimum.

        Args:
            kernel_list, list: the list of kernels.
            sample_kernel_weights, list: the probabilities of each kernel type being used.
            lengthscale_range: parameterize lengthscale distribution.
            std_range: parameterize std distribution.
            p_iso, float: probability of isotropic lengthscale.
            maximize, bool: whether to create maximum or minimum. Default true.
            seed, int: random seed.
            device: default cuda.

        Attrs:
            batch_utility, Sequence[OptimizationFunction]: save previous sampled batch of GP curves.
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
            # get a dict of all kernel functions in data.kernel module
            # kernel_func_dict = {
            #     key: value for key, value in getmembers(kernel_module, isfunction)
            # }
            # {
            #     "rbf": rbf,
            #     "matern52": matern52,
            #     "matern32": matern32,
            #     "matern12": matern12,
            # }
            self.kernel_list = []
            for k in kernel_list:
                if k in globals():  # we have import functions in data.kernel
                    # if k in kernel_func_dict:
                    self.kernel_list.append(globals()[k])
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

        self.batch_utility = []

    def sample(
        self,
        batch_size: int = 64,
        max_num_ctx_points: int = 128,
        num_total_points: int = 256,
        x_range: List[List[float]] = [[-1, 1]],
        sobol_grid: bool = True,
        evaluate: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """sample points from a batch of GP curves which are injected with a gobal optimum.

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
        """
        (_X, _Y, _XOPT, _YOPT, _UTILITY) = ([], [], [], [], [])
        x_range = torch.tensor(x_range)  # (d_x, 2)
        for _ in range(batch_size):
            # create a GP curve and sample points
            x, y, xopt, yopt, utiltiy = self.sample_a_function(
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
            if utiltiy is not None:
                _UTILITY.append(utiltiy)

        self.batch_utility = _UTILITY  # [batch_size]
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
        sobol_grid: bool = True,
        evaluate: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        OptimizationFunction,
    ]:
        """sample points from a single GP curve which are injected with a gobal optimum.

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
            utility, OptimizationFunction: the GP curve.
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
        )  # scalar
        temp_mean = torch.zeros(size=(n_temp,))
        temp_stds = torch.full(size=(n_temp,), fill_value=sigma_f.item())
        # sample `n_temp` means from a zero-mean Normal distribution with function standard deviation.
        # then the one with maximal absolute value is chosen as the function mean
        temp_samples = torch.abs(torch.normal(temp_mean, temp_stds))  # [n_temp]
        mean_f = temp_samples.max()  # [1]

        #  introduce a sharp optimum with a minor probability
        p_rare, rare_tau = 0.1, 1.0
        if torch.rand(1) < p_rare:
            mean_f = mean_f + torch.exp(torch.tensor(rare_tau))

        # sample input location
        if evaluate:
            # all locations are quasi-random samples over input space
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
            # all locations are sampled from a sufficiently large set of quasi-random samples
            if sobol_grid:
                x_tmp = torch.tensor(
                    create_sobol_grid(
                        d_x=self.d_x,
                        x_range=x_range,
                        num=GRID_SIZE,
                    ),
                    device=self.device,
                ).to(dtype=torch.float32)
                perm_idx = torch.randperm(GRID_SIZE)
                x_fix = x_tmp[perm_idx[: num_ctx_points - 1]]
                x_ind = x_tmp[perm_idx[num_ctx_points:num_total_points]]
            else:
                # sample from a uniform distribution
                x_fix = x_range[:, 0] + (x_range[:, 1] - x_range[:, 0]) * torch.rand(
                    [num_ctx_points - 1, self.d_x],
                    device=self.device,
                )
                x_ind = x_range[:, 0] + (x_range[:, 1] - x_range[:, 0]) * torch.rand(
                    [num_total_points - num_ctx_points, self.d_x],
                    device=self.device,
                )

        # create the global minimum, (x^*, 0)
        # randomly sample a location
        xopt = (
            torch.rand(self.d_x, device=self.device) * (x_range[:, 1] - x_range[:, 0])
            + x_range[:, 0]
        ).view(
            1, -1
        )  # [1, d_x]
        yopt = torch.zeros(size=(1, 1))  # [1, 1] NOTE zero-value minimum

        # sample a kernel
        kernel = random.choices(
            population=self.kernel_list, weights=self.sample_kernel_weights, k=1
        )[0]

        # define a GP sampler
        gp_sampler = SimpleGPSampler(mean=mean_f, kernel_function=kernel)

        # sample y values for the fixed points conditioned on the optimum
        y_fix = gp_sampler.sample(
            test_X=x_fix,
            train_X=xopt,
            train_y=yopt,
            length_scale=length_scale,
            std=sigma_f,
        )

        x_fix = torch.cat([x_fix, xopt], dim=0)
        y_fix = torch.cat([y_fix, yopt], dim=0)

        # sample y values for the additional points
        y_ind = gp_sampler.sample(
            test_X=x_ind,
            train_X=x_fix,
            train_y=y_fix,
            length_scale=length_scale,
            std=sigma_f,
            correlated=False,  # NOTE use uncorrelated samples to speed up the sampling.
        )

        # combine data
        x = torch.cat([x_ind, x_fix], dim=0)
        y = torch.cat([y_ind, y_fix], dim=0)

        # add a random offset to the function
        f_offset_range = [-5, 5]
        f_offset = (
            torch.rand(1) * (f_offset_range[1] - f_offset_range[0]) + f_offset_range[0]
        ).view(
            1, -1
        )  # [1, 1]

        # save the curve for evaluation
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
        else:
            utility = None

        # adjust the minimum from zero to f_offset
        yopt = f_offset

        # add a quadratic bowl with broad lengthscale around the optimum
        quad_lengthscale_squared = torch.sum(
            (2 * (x_range[:, 1].float() - x_range[:, 0].float())) ** 2
        )  # (1)

        quadratic_factor = 1.0 / quad_lengthscale_squared
        f = (
            torch.abs(y)
            + quadratic_factor
            * torch.sum(
                (x - xopt) ** 2, dim=-1, keepdim=True
            )  # (B, num_test_points, 1)
            + f_offset
        )
        y = f

        # we were creating minimum; flip the function if maximum is required
        if self.maximize:
            y = -y
            yopt = -yopt

        return x, y, xopt, yopt, utility
