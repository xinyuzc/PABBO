import torch
import numpy as np
from typing import Tuple, Callable, Union
from itertools import combinations
import botorch
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.model import Model
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling import SobolQMCNormalSampler
from botorch.generation import MaxPosteriorSampling
from botorch.acquisition.preference import (
    qExpectedUtilityOfBestOption,
)
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
)
from botorch.optim.optimize import optimize_acqf, optimize_acqf_discrete
from data.utils import My1DInterpolator, MyNDInterpolator
from policies.mpes import MultinomialPredictiveEntropySearch
import time

RAW_SAMPLES = 1024  # the number of raw samples before acqf optimization
NUM_RESTARTS = 20  # the number of best candidates from the raw samples for further local optimization
SAMPLE_SIZE = 64  # the number of sample when estimating the posterior with MC sampling
q = 2  # the number of candidates


class PBOHandler:
    def __init__(
        self, max_retries: int = 20, device="cpu", p_noise: float = 0.0
    ) -> None:
        """Preferential Bayesian optimization (minimization by default) based on `botorch` pairwiseGP."""
        self.device = device  # NOTE `fsolve` in PairwiseGP only works on CPU
        self.max_retries = max_retries
        self.p_noise = p_noise

    def load_pabbo_data(
        self, pair: torch.Tensor, pair_y: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map PABBO style data to botorch PBO style data.

        Args:
            pair, (num_pairs, 2 * d_x): N pair of points with d_x dimensions for comparison.
            pair_y, (num_pairs, 2): corresponding gound utility values.
            c, (num_pairs, 1): the location of preferred point in a pair. e.g. c[i] = 0 means point at pair[i, 0:d_x] is preferred than pair[i, d_x:2_dx].

        Returns: num_points := 2 * num_pairs
            X, (num_points, d_x): 2N datapoints.
            y, (num_points, 1): corresponding ground utility values.
            comp, LongTensor(num_pairs, 2): comparison[i] is an indicator that suggest  the utility value of comparisons[i, 0]-th is greater than comparisons[i, 1]-th
        """
        # data type and devices
        pair, pair_y, c = (
            pair.to(dtype=torch.float64, device=self.device),
            pair_y.to(dtype=torch.float64, device=self.device),
            c.to(dtype=torch.float64, device=self.device),
        )

        # flatten pair data into datapoints. e.g. X = [pair[0, 0:d_x], pair[0, d_x:2d_x]...]
        n_pairs = len(pair)
        X = pair.view(2 * n_pairs, -1)
        y = pair_y.view(2 * n_pairs, -1)

        # record comparison between datapoints in each pair. [[0, 1], [2, 3], ..., [2*n_pairs-2, 2*n_pairs-1]]
        comp = np.arange(2 * n_pairs).reshape(-1, 2)
        # flip when utility value at c[i, 1] is preferred
        flip_indices = (c == 1).cpu().reshape(n_pairs).numpy()  # (n_pairs)
        comp[flip_indices, :] = np.flip(comp[flip_indices, :], 1)
        comp = torch.tensor(comp).long().to(y.device)

        return X, y, comp

    def generate_comparisons(
        self, y: torch.Tensor, n_comp: int, replace: bool = False
    ) -> torch.Tensor:
        """Generate `n_comp` comparisons on the set of datapoints.

        Args:
            y, (num_points, 1): utility values associated with `num_points` datapoints.
            n_comp, scalar: the number of comparisons to generate.
            replace, bool: whether to sample comparisons with or without replacement.

        Returns:
            comp, (n_comp, 2)
        """
        # randomly sample `n_comp` pairs of datapoints from all the possible combinations
        all_combinations = np.array(list(combinations(range(y.shape[0]), 2)))
        comp = all_combinations[
            np.random.choice(range(len(all_combinations)), n_comp, replace=replace)
        ]  # (num_comp, 2)
        comp_y = y[comp].squeeze(-1)  # (num_comp, 2, 1) -> (num_comp, 2)
        # (noisy) observation
        comp_y += self.p_noise * torch.randn_like(comp_y, device=self.device)

        # flip when utility value at c[i, 1] is preferred
        flip_indices = (
            (comp_y[..., 0] < comp_y[..., 1]).reshape(n_comp).cpu().numpy()
        )  # (num_comp)
        comp[flip_indices, :] = np.flip(comp[flip_indices, :], 1)
        comp = torch.tensor(comp).long().to(y.device)
        return comp

    def update_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        comp: torch.Tensor,
        next_X: torch.Tensor,
        next_y: torch.Tensor,
        n_comp: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """update observations.
            X, (num_points, d_x)
            y, (num_points, 1)
            comp, (num_comparisons, 2)
            next_X, (num_new_points, d_x): new datapoints.
            next_y, (num_new_points, 1): corresponding gound truth utility values.
            n_comp, scalar: the number of observed comparison on the new datapoints.
        Returns:
            X, (num_points + num_new_points, d_x)
            y, (num_points + num_new_points, d_x)
            comp, (num_comparisons + n_comp, 2)
        """
        if isinstance(next_y, np.ndarray):
            next_y = torch.from_numpy(next_y)
        next_X = next_X.to(X)
        next_y = next_y.to(y)
        # generate new comparison
        next_comp = self.generate_comparisons(y=next_y, n_comp=n_comp)
        # NOTE shift indices of next_comp
        comp = torch.cat([comp, next_comp + X.shape[-2]])
        X = torch.cat([X, next_X])
        y = torch.cat([y, next_y])
        return X, y, comp

    def fit_model(
        self,
        X: torch.Tensor,
        comp: torch.Tensor,
    ):
        """Fit PairwiseGP on preferential data.
        Args:
            X, (num_points, , d_x): datapoints.
            y, (num_points, 1): corresponding ground utility values; for generating comparisons.
            comp, LongTensor(num_pairs, 2): comparison[i] is an indicator that suggest  the utility value of comparisons[i, 0]-th is greater than comparisons[i, 1]-th

        Returns:
            mll, model
        """
        model = PairwiseGP(
            X,
            comp,
        ).to(
            device=X.device,
            dtype=X.dtype,
        )
        mll = PairwiseLaplaceMarginalLogLikelihood(
            model.likelihood,
            model,
        )
        try:
            fit_gpytorch_mll(
                mll,
                max_retries=self.max_retries,
            )
        except (RuntimeError, botorch.exceptions.errors.ModelFittingError) as e:
            print(e)
        return mll, model

    def optimize_step_on_continuous_space(
        self,
        model: Model,
        acq_function_type: str,
        X: torch.Tensor,
        X_bound: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PBO step on bounded continuous input space.

        Args:
            model: fitted GP with pairwise preferential likelihood.
            acq_function_type, str in ["rs", "qTS", "qEUBO", "qEI", "qNEI", "mpes"]
            X, (num_points, d_x): the training datapoints.
            X_bound, (2, d_x): the bound on input space.

        Returns:
            next_X, (2, d_x): the next query
            acq_vals: the associated acqf values.
        """
        d_x = X.shape[-1]
        acq_vals = None
        if acq_function_type == "rs":
            # randomly sample two points from the input space
            next_X = torch.rand((q, d_x)).to(X) * (X_bound[1] - X_bound[0]) + X_bound[0]
        elif acq_function_type == "qTS":
            next_X = []

            # sample 2 draws from the posterior and choose the maximum respectively
            for _ in range(q):
                # (RAW_SAMPLES, d_x) sobol samples from input space
                draws = draw_sobol_samples(bounds=X_bound, n=RAW_SAMPLES, q=1).squeeze(
                    -2
                )
                thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
                # (1, d_x), the maximum of each draw
                X_next = thompson_sampling(draws, num_samples=1)
                next_X.append(X_next)
            next_X = torch.concat(next_X, dim=0)  # (q, d_x)
        else:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([SAMPLE_SIZE]))
            if acq_function_type == "qEUBO":
                acq_func = qExpectedUtilityOfBestOption(
                    pref_model=model, sampler=sampler
                )
            elif acq_function_type == "qEI":
                posterior = model.posterior(X)
                mean = posterior.mean  # type:ignore
                acq_func = qExpectedImprovement(
                    model=model,
                    best_f=mean.max().item(),
                    sampler=sampler,
                )
            elif acq_function_type == "mpes":
                acq_func = MultinomialPredictiveEntropySearch(
                    model=model,
                    bounds=X_bound,
                )
            elif acq_function_type == "qNEI":
                acq_func = qNoisyExpectedImprovement(
                    model=model,
                    X_baseline=X,
                    sampler=sampler,
                    prune_baseline=True,
                )
            else:
                raise ValueError(f"{acq_function_type} is not supported.")

            # optimize the acqf to find the next query
            next_X, acq_vals = optimize_acqf(
                acq_function=acq_func,
                bounds=X_bound,
                q=q,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )
        return next_X.detach(), acq_vals

    def optimize_on_continuous_space(
        self,
        acq_function_type: str,
        initial_pairs: torch.Tensor,
        initial_pairs_y: torch.Tensor,
        initial_c: torch.Tensor,
        yopt: torch.Tensor,
        T: int,
        test_x_range: torch.Tensor,
        utility: Callable,
        **kwargs,
    ) -> Tuple:
        """PBO on either 1) bounded countinuous input space, or 2) discrete intput space.
        NOTE continuous input are assumed to be normalised, i.e., x \in (0, 1)

        Args:
            acq_function_type, str in ["rs", "qEI", "qEUBO", "qTS", "qNEI", "mpes"]
            initial_pairs, (num_initial_pairs, 2 * d_x): initial pair of points with d_x dimensions for comparison.
            initial_pairs_y, (num_initial_pairs, 2): associated gound utility values.
            initial_c, (num_initial_pairs, 1): the location of preferred point in a pair. e.g. c[i] = 0 means point at pair[i, 0:d_x] is preferred than pair[i, d_x:2_dx].
            yopt, (num_gobal_optima, 1): the optimal utility values.
            T, scalar: time budget.
            test_x_range, (d_x, 2): the original input range.
            utility: y = utility(x) gives the ground truth utility value at x.

        Returns:
            X, (2 * num_initial_pairs + 2 * T, d_x): the acquired datapoints.
            y, (2 * num_initial_pairs + 2 * T, 1): associated ground truth utility values.
            comp, (num_initial_pairs + T, 2): the preference comparisons.
            simple_regret, (T+1): simple regret at each step t as yopt - max_{i=1}^t max {y_{i,1}, y_{i,2}}
            immediate_regret, (T+1): immediate regret at each step t as yopt - max {y_{t,1}, y_{t,2}}
            cumulative_regret, (T+1): cumulative regret at each step t cumsum(immediate_regret_t)
            cumulative_inference_time, (T)
        """
        # load PABBO data
        X, y, comp = self.load_pabbo_data(
            pair=initial_pairs, pair_y=initial_pairs_y, c=initial_c
        )

        yopt = yopt[0].to(X)
        d_x = X.shape[-1]

        # unit bound. We assume input has been normalised.
        X_bound = torch.stack([torch.zeros(d_x), torch.ones(d_x)]).to(X)  # (2, d_x)

        # fit model
        t0 = time.time()
        if acq_function_type != "rs":
            _, model = self.fit_model(X=X, comp=comp)
        else:
            model = None
        t1 = time.time()
        model_fitting_time = t1 - t0

        # record metrics
        t = []
        simple_regret = [yopt - torch.max(y)]  # [(1)]
        immediate_regret = [yopt - torch.max(y)]

        for _ in range(T):
            # find next query
            t0 = time.time()
            next_X, _ = self.optimize_step_on_continuous_space(
                model=model,
                acq_function_type=acq_function_type,
                X=X,
                X_bound=X_bound,
            )  # (2, d_x)
            t1 = time.time()
            acq_time = t1 - t0

            # NOTE scale unit X to test range before computing the utility value
            next_X_scale = test_x_range[..., 0] + next_X.clone() * (
                test_x_range[..., 1] - test_x_range[..., 0]
            )
            next_y = utility(next_X_scale)  # (2, 1)

            # update observations
            X, y, comp = self.update_data(
                X=X, y=y, comp=comp, next_X=next_X, next_y=next_y, n_comp=1
            )

            # record metrics
            t.append(model_fitting_time + acq_time)
            simple_regret.append(yopt - torch.max(y))
            immediate_regret.append(yopt - torch.max(next_y))

            # fit model
            t0 = time.time()
            if acq_function_type != "rs":
                _, model = self.fit_model(X=X, comp=comp)
            t1 = time.time()
            model_fitting_time = t1 - t0

        # cumulative inference time
        cumulative_inference_time = torch.from_numpy(np.cumsum(t)).to(X)
        simple_regret = torch.cat(simple_regret)  # (T+1)
        immediate_regret = torch.cat(immediate_regret)  # (T+1)
        cumulative_regret = np.cumsum(immediate_regret, axis=-1)  # (T+1)
        return (
            X,
            y,
            comp,
            simple_regret,
            immediate_regret,
            cumulative_regret,
            cumulative_inference_time,
        )

    def get_utility_on_discrete_space(
        self,
        X_pending: torch.Tensor,
        y_pending: torch.Tensor,
        interpolator_type: str = "nearest",
    ) -> Callable:
        """fit a utility function on discrete candidate set through interpolation.

        Args:
            X_pending, (num_pending_points, d_x)
            y_pending, (num_pending_points, 1)
            interpolator_type, str in ["nearest", "linear"]

        Returns:
            utility
        """
        if X_pending.shape[-1] == 1:
            utility = My1DInterpolator(
                X=X_pending, y=y_pending, interpolator_type=interpolator_type
            )
        else:
            utility = MyNDInterpolator(
                X=X_pending, y=y_pending, interpolator_type=interpolator_type
            )
        return utility

    def optimize_step_on_discrete_space(
        self,
        model: Model,
        acq_function_type: str,
        X: torch.Tensor,
        X_pending: torch.Tensor,
        X_bound: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PBO step on discrete input space.

        Args:
            model: fitted GP with pairwise preferential likelihood.
            acq_function_type, str in ["rs", "qTS", "qEUBO", "qEI", "qNEI"]
            X, (num_training_points, d_x): the training datapoints.
            X_pending, (num_candidates, d_x): the candidates.

        Returns:
            next_X, (2, d_x): the next query
            acq_vals: the associated acqf values.
        """
        acq_vals = None
        if acq_function_type == "rs":
            # randomly choose two points without replacement.
            next_idx = np.random.choice(len(X_pending), size=(q), replace=False)
            next_X = X_pending[next_idx]
        elif acq_function_type == "qTS":
            next_X = []
            # we sample 2 draws from the GP posterior on candidates and choose the maximum respectively to form the next query
            for _ in range(q):
                draws = X_pending  # (num_candidates, d_x)
                thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
                X_next = thompson_sampling(draws, num_samples=1)
                next_X.append(X_next)
            next_X = torch.concat(next_X, dim=0)
        else:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([SAMPLE_SIZE]))
            if acq_function_type == "qEUBO":
                acq_func = qExpectedUtilityOfBestOption(
                    pref_model=model, sampler=sampler
                )
            elif acq_function_type == "qEI":
                posterior = model.posterior(X)
                mean = posterior.mean  # type:ignore
                acq_func = qExpectedImprovement(
                    model=model,
                    best_f=mean.max().item(),
                    sampler=sampler,
                )
            elif acq_function_type == "qNEI":
                acq_func = qNoisyExpectedImprovement(
                    model=model,
                    X_baseline=X,
                    sampler=sampler,
                    prune_baseline=True,
                )
            elif acq_function_type == "mpes":
                acq_func = MultinomialPredictiveEntropySearch(
                    model=model,
                    bounds=X_bound,
                )
            else:
                raise ValueError(f"{acq_function_type} is not supported.")

            next_X, acq_vals = optimize_acqf_discrete(
                acq_function=acq_func, q=q, choices=X_pending
            )  # (2, d_x)
        return next_X.detach(), acq_vals

    def optimize_on_discrete_space(
        self,
        acq_function_type: str,
        initial_pairs: torch.Tensor,
        initial_pairs_y: torch.Tensor,
        initial_c: torch.Tensor,
        X_pending: torch.Tensor,
        y_pending: torch.Tensor,
        test_x_range: torch.Tensor,
        T: int,
        interpolator_type: str = "nearest",
        **kwargs,
    ):
        """PBO on discrete input space.

        Args:
            acq_function_type, str in ["rs", "qEI", "qEUBO", "qTS", "qNEI"]
            initial_pairs, (num_initial_pairs, 2 * d_x): initial pair of points with d_x dimensions for comparison.
            initial_pairs_y, (num_initial_pairs, 2): associated gound utility values.
            initial_c, (num_initial_pairs, 1): the location of preferred point in a pair. e.g. c[i] = 0 means point at pair[i, 0:d_x] is preferred than pair[i, d_x:2_dx].
            yopt, (num_gobal_optima, 1): the optimal utility values.
            T, scalar: time budget.
            X_pending, (num_candidates, d_x): the discrete candidate set.
            y_pending, (num_candidates, 1): associated ground truth utility values.
        Returns:
            X, (2 * num_initial_pairs + 2 * T, d_x): the acquired datapoints.
            y, (2 * num_initial_pairs + 2 * T, 1): associated ground truth utility values.
            comp, (num_initial_pairs + T, 2): the preference comparisons.
            simple_regret, (T+1): simple regret at each step t as yopt - max_{i=1}^t max {y_{i,1}, y_{i,2}}
            immediate_regret, (T+1): immediate regret at each step t as yopt - max {y_{t,1}, y_{t,2}}
            cumulative_regret, (T+1): cumulative regret at each step t cumsum(immediate_regret_t)
            cumulative_inference_time, (T)
        """
        X_pending, y_pending = (
            X_pending.to(dtype=torch.float64),
            y_pending.to(dtype=torch.float64),
        )
        # build a utility on the discrete candidate set
        utility = self.get_utility_on_discrete_space(
            X_pending=X_pending,
            y_pending=y_pending,
            interpolator_type=interpolator_type,
        )
        # load PABBO data
        X, y, comp = self.load_pabbo_data(
            pair=initial_pairs, pair_y=initial_pairs_y, c=initial_c
        )

        yopt = torch.max(y_pending)  # (1)
        # fit model
        t0 = time.time()
        if acq_function_type != "rs":
            _, model = self.fit_model(X=X, comp=comp)
        else:
            model = None
        t1 = time.time()
        model_fitting_time = t1 - t0

        # record metrics
        t = []
        simple_regret = [yopt - torch.max(y)]  # [(1)]
        immediate_regret = [yopt - torch.max(y)]

        for _ in range(T):
            # find next query
            t0 = time.time()
            next_X, _ = self.optimize_step_on_discrete_space(
                model=model,
                acq_function_type=acq_function_type,
                X=X,
                X_pending=X_pending,
                X_bound=test_x_range.transpose(0, 1),
            )
            t1 = time.time()
            acq_time = t1 - t0

            # compute ground truth utility value
            next_y = utility(next_X)  # (q, 1)

            # update observations
            X, y, comp = self.update_data(
                X=X, y=y, comp=comp, next_X=next_X, next_y=next_y, n_comp=1
            )

            # record metrics
            t.append(model_fitting_time + acq_time)
            simple_regret.append(yopt - torch.max(y))
            immediate_regret.append(yopt - torch.max(next_y))

            # fit model
            t0 = time.time()
            if acq_function_type != "rs":
                _, model = self.fit_model(X=X, comp=comp)
            t1 = time.time()
            model_fitting_time = t1 - t0
        # cumulative inference time
        cumulative_inference_time = torch.from_numpy(np.cumsum(t)).to(X)
        simple_regret = torch.stack(simple_regret)  # (T+1)
        immediate_regret = torch.stack(immediate_regret)  # (T+1)
        cumulative_regret = np.cumsum(immediate_regret, axis=-1)  # (T+1)
        return (
            X,
            y,
            comp,
            simple_regret,
            immediate_regret,
            cumulative_regret,
            cumulative_inference_time,
        )
