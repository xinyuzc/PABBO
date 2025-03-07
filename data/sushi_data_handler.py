import os
import pandas as pd
import numpy as np
from data.utils import (
    MyNDInterpolator,
)
import os.path as osp
from typing import List, Tuple


class SushiDataHandler:
    def __init__(
        self, root_dir: str = "datasets/sushi-data", interpolator_type: str = "linear"
    ):
        """Data handler for sushi dataset.
        Adapted from qEUBO (MIT license)
        Source: https://github.com/facebookresearch/qEUBO/blob/21cd661efc25b242c9fdf5230f5828f01ff0872b/experiments/sushi_runner.py

        Args:
            root_dir, str: the datapath.
            interpolator_type, str: we do linear / nearest interpolation on the dataset and fill the out-of-bound point with its closest neighbor's value.

        Attrs:
            num_total_points, scalar: total number of datapoints.
            d_x, scalar: input dimension.
            raw_X, np.ndarray [num_total_points, 2]: two continuous raw features `sugarpercent` and `pricepercent`.
            X, np.ndarray [num_total_points, 2]: min-max scaled `raw_X`.
            raw_y, np.ndarray [num_total_points, 1]: raw utility values, `winpercent`.
            y, np.ndarray [num_total_points, 1] min-max scaled `raw_y`.
            x_range, list: data range for feature.
        """
        self.root_dir = root_dir
        df_features = pd.read_csv(
            osp.join(root_dir, f"sushi3.idata"), sep="\t", header=None
        )
        df_score = pd.read_csv(
            osp.join(root_dir, "sushi3b.5000.10.score"), sep=" ", header=None
        )

        # generate utility values f = win_count / total
        scores = []
        for item_a in range(df_score.values.shape[1]):
            score = 0
            for item_b in range(df_score.values.shape[1]):
                if self._prefer_a(item_a, item_b, df_score):
                    score += 1
            scores.append(score / float(df_score.values.shape[1]))

        self.x_range = [0, 1]
        self.raw_X = df_features.values[:, [6, 5, 7, 8][:4]].astype(np.float64)
        self.X = (self.raw_X - np.min(self.raw_X, axis=0)[None, :]) / (
            np.max(self.raw_X, axis=0) - np.min(self.raw_X, axis=0)
        )[None, :]

        self.raw_y = np.array(scores).reshape((-1, 1))
        self.y = (self.raw_y - np.min(self.raw_y)) / (
            np.max(self.raw_y) - np.min(self.raw_y)
        )
        self.interpolator_type = interpolator_type
        self.fit(interpolator_type=interpolator_type)

    def _prefer_a(self, item_a: int, item_b: int, df_scores: List):
        """Check from data if item_a has higher score that item_b"""
        # find records where both item_a and item_b are rated
        # generate preference by comparing the averaged scores from all records
        ix = (df_scores[item_a].values > -1) * (df_scores[item_b].values > -1)
        prefer_a = np.mean(df_scores[item_a].values[ix] > df_scores[item_b].values[ix])
        prefer_b = np.mean(df_scores[item_b].values[ix] > df_scores[item_a].values[ix])
        return prefer_a > prefer_b

    def fit(
        self,
        interpolator_type: str = "linear",
    ) -> Tuple[MyNDInterpolator, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit an extrapolator on the discrete sushi dataset.

        Args:
            interpolator_type, str in ["linear", "nearest"]: the out-of-bound values will be filled with its nearest neighbor's value when using linear interpolator.
        """
        self.yopt = np.max(self.y, axis=0, keepdims=True)  # (1, 1)
        self.Xopt = np.take_along_axis(
            self.X, np.argmax(self.y, axis=0, keepdims=True), axis=0
        )
        self.utility = MyNDInterpolator(
            X=self.X, y=self.y, interpolator_type=interpolator_type
        )

    def get_utility(self):
        """Get fitted extrapolator."""
        return self.utility

    def get_data(
        self, add_batch_dim: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve the sushi dataset.

        Args:
            add_batch_dim, bool: whether to add a batch dimension.

        Returns:
            X, (1, num_total_points, 4) if add_batch_dim else (num_total_points, 4): datapoints.
            y, (1, num_total_points, 1) if add_batch_dim else (num_total_points, 1): associated utility values.
            Xopt, (1, 1, 4) if add_batch_dim else (1, 4): global optimum locations.
            yopt, (1, 1, 1) if add_batch_dim else (1, 1): global optimum value.

        """
        if add_batch_dim:
            return (
                self.X[None, :, :],
                self.y[None, :, :],
                self.Xopt[None, :, :],
                self.yopt[None, :, :],
            )
        else:
            return self.X, self.y, self.Xopt, self.yopt

    def sample(
        self,
        batch_size: int = 1,
        num_points: int = 200,
    ):
        """Sample from the fitted extrapolator.

        Args:
            batch_size, scalar: batch size.
            num_points, scalar: number of datapoints in each sampled dataset.

        Returns:
            XX, (batch_size, num_points, 4): sampled datapoints.
            YY, (batch_size, num_points, 1): associated utility values.
            Xopt, (batch_size, 1, 4): global optimum location.
            yopt, (batch_size, 1, 1): global optimum value.
        """
        XX = (
            np.random.random((batch_size, num_points, 4))
            * (self.x_range[1] - self.x_range[0])
            + self.x_range[0]
        )
        YY = self.utility(XX)
        return (
            XX,
            YY,
            np.tile(self.Xopt, (batch_size, 1, 1)),
            np.tile(self.yopt, (batch_size, 1, 1)),
        )
