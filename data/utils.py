import torch
import numpy as np
import random
import sobol_seq
from typing import List, Union, Tuple, Dict
from scipy.interpolate import LinearNDInterpolator as LND
from scipy.interpolate import NearestNDInterpolator as NND
from scipy import interpolate


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def scale_from_domain_1_to_domain_2(
    x: torch.Tensor,
    bound1: torch.Tensor,
    bound2: torch.Tensor,
) -> torch.Tensor:
    tmp = (x - bound1[..., 0]) / (bound1[..., 1] - bound1[..., 0])
    return bound2[..., 0] + tmp * (bound2[..., 1] - bound2[..., 0])


def scale_from_unit_to_domain(X: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
    return bound[..., 0] + X * (bound[..., 1] - bound[..., 0])


def create_sobol_grid(d_x: int, num: int, x_range: List[List[float]] = [[0.0, 1.0]]):
    """sample sobol grid locations and scale to target range.

    Args:
        d_x, scalar: input dimension.
        x_range, [d_x, 2]: bound for each dimension
        num, scalar: number of samples.

    Returns:
        loc, [num, d_x]
    """
    loc = sobol_seq.i4_sobol_generate(dim_num=d_x, n=num)  # \in (0, 1)

    if not isinstance(x_range, np.ndarray):
        x_range = np.array(x_range.detach().cpu().numpy())

    loc = scale_from_unit_to_domain(loc, x_range)
    return loc


def sample_from_full_combination(
    num_total_points: int,
    device: torch.device,
    num_random_pairs: Union[None, int] = None,
) -> torch.Tensor:
    """sample pairs from the full combinations of points.

    Args:
        num_total_points, scalar: number of points.
        num_random_pairs, scalar: number of sampled pairs. If None, full combination is returned.

    Returns:
        pair_idx, [M, 2]: indices for sampled pairs.
    """
    num_total_pairs = num_total_points * (num_total_points - 1) // 2

    # sample a subset
    if num_random_pairs is not None and (num_random_pairs < num_total_pairs):
        pair_idx, _ = get_combinations_subset(
            num_query_points=num_total_points,
            num_random_pairs=num_random_pairs,
            device=device,
        )
    else:
        # the full combination
        pair_idx = torch.combinations(torch.arange(0, num_total_points))
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


def gather_data_at_index(data: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Retrieves specific elements from data based on indices.
    data, [B, N, d]: a batch of sequence consisting of N d-dimensional elements.
    idx, [B, M, 1]: M refers to the number of indices to gather.
    """
    B, N, d = data.shape
    tile_idx = idx.tile(1, 1, d)
    ele = torch.gather(input=data, dim=1, index=tile_idx)
    return ele


def get_random_split_data(
    total_num: int,
    min_num_ctx: int = 1,
    max_num_ctx: int = 30,
):
    """randomly split a sequence of `total_num` elements into context and target set."""
    if max_num_ctx >= total_num:
        raise ValueError("`max_num_ctx` should be smaller than `total_num`.")
    num_ctx_pairs = random.randint(min_num_ctx, max_num_ctx)
    num_tar_pairs = total_num - num_ctx_pairs
    rand_idx = [*range(total_num)]
    random.shuffle(rand_idx)
    ctx_idx, tar_idx = rand_idx[:num_ctx_pairs], rand_idx[num_ctx_pairs:]
    return num_ctx_pairs, ctx_idx, num_tar_pairs, tar_idx


class My1DInterpolator:
    def __init__(self, X, y: np.ndarray, interpolator_type: str = "nearest"):
        """1-dimensional interpolator.

        Attrs:
            X, (N, 1): datapoints.
            y, (N, 1): utility values.
            interpolator_type, str: interpolation type in ["nearest", "linear"]
        """
        assert X.shape[-1] == 1
        assert y.shape[-1] == 1
        X = X.squeeze(-1) if X.ndim == 2 else X
        y = y.squeeze(-1) if y.ndim == 2 else y

        self.interpolator_type = interpolator_type
        # sort datapoints by X ascending order
        idx = np.argsort(self.X)
        X, y = X[idx], y[idx]

        if interpolator_type == "nearest":
            # X is expected as ascending
            self.ext = interpolate.interp1d(
                X, y, bounds_error=False, fill_value=(y[0], y[-1])
            )
        elif interpolator_type == "linear":
            self.ext = lambda x: np.interp(x, X, y, left=y[0], right=y[-1])
        else:
            raise ValueError(f"{interpolator_type} is not supported.")

    def __call__(self, xx: torch.Tensor) -> torch.Tensor:
        """Interpolate at points xx.

        Args:
            xx, (..., 1)

        Returns:
            yy, (..., 1)
        """
        assert xx.shape[-1] == 1
        xx = xx.detach().cpu().numpy()
        if xx.ndim == 2:
            xx = xx[None, :, :]  # (1, N, 1)
        B = len(xx)
        yy = list()
        for b in range(B):
            y = self.ext(xx[b])
            yy.append(y)
        yy = np.stack(yy, axis=0)
        return yy


def LND_fill_oob_with_nn(
    xx: np.ndarray, ext: LND, X: torch.Tensor, y: torch.Tensor
) -> np.ndarray:
    """linear interpolation at points xx;
    out-of-bound value is filled with its nearest neighbors.

    Args:
        xx, (M, d_x): point to evalaute.
        ext, LND: linearNDInterpolator, where out-of-bound values are nan by default.
        X, (N, d_x): training datapoints.
        y, (N, 1): associated utility values.
    """
    zz = ext(*[xx[:, i] for i in range(xx.shape[-1])])  # interpolation at xx
    nans = np.isnan(zz).flatten()
    # fill nan point with the value of its nearest neighbor
    if nans.any():
        inds = np.argmin(
            sum(
                [
                    (np.expand_dims(X[..., i], axis=-1) - xx[nans, i]) ** 2
                    for i in range(xx.shape[-1])
                ]
            ),
            axis=0,
        )
        zz[nans] = y[inds]
    return zz


class MyNDInterpolator:
    def __init__(self, X, y, interpolator_type: str = "nearest"):
        """n-dimensional interpolator"""
        if X.shape[-1] < 2:
            raise ValueError
        self.X, self.y = X, y
        self.interpolator_type = interpolator_type
        if interpolator_type == "nearest":
            self.ext = NND(X, y)
        elif interpolator_type == "linear":
            self.ext = LND(X, y)
        else:
            raise ValueError(f"{interpolator_type} is not supported.")

    def __call__(self, xx: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Interpolote at points xx.

        Args:
            xx, (..., d_x): the locations to predict.

        Returns:
            yy, (..., 1): predicted values.
        """
        to_tensor = False
        if isinstance(xx, torch.Tensor):
            to_tensor = True
            device = xx.device
            xx = xx.detach().cpu().numpy()
        if xx.ndim == 2:  # no batch dim
            if self.interpolator_type == "nearest":
                yy = self.ext(*[xx[:, i] for i in range(xx.shape[-1])])
            else:
                yy = LND_fill_oob_with_nn(xx=xx, ext=self.ext, X=self.X, y=self.y)
        else:
            B = len(xx)
            yy = list()
            for b in range(B):
                if self.interpolator_type == "nearest":
                    y = self.ext(*[xx[b, :, i] for i in range(xx.shape[-1])])
                else:
                    y = LND_fill_oob_with_nn(xx=xx[b], ext=self.ext, X=self.X, y=self.y)
                yy.append(y)
            yy = np.stack(yy, axis=0)
        return torch.from_numpy(yy).to(device) if to_tensor else yy
