import pandas as pd
import numpy as np
import os.path as osp
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from data.utils import (
    MyNDInterpolator,
)
from typing import Tuple


class CandyDataHandler:
    def __init__(
        self,
        root_dir: str = "datasets/candy-data",
        filename: str = "candy-data.csv",
        interpolator_type: str = "linear",
    ):
        """Handler for the candy dataset.

        Args:
            root_dir, str: the datapath.
            filename, str: the csv file under the datapath.
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
        self.raw_data = pd.read_csv(osp.join(root_dir, filename)).drop(
            labels=[
                "chocolate",
                "fruity",
                "caramel",
                "peanutyalmondy",
                "nougat",
                "crispedricewafer",
                "hard",
                "bar",
                "pluribus",
            ],
            axis=1,
        )

        self.num_total_points = len(self.raw_data)
        self.x_range = [0, 1]
        self.raw_X = self.raw_data[["sugarpercent", "pricepercent"]].to_numpy()
        self.X = (self.raw_X - np.min(self.raw_X, axis=0)[None, :]) / (
            np.max(self.raw_X, axis=0) - np.min(self.raw_X, axis=0)
        )[None, :]
        self.raw_y = (
            self.raw_data["winpercent"].to_numpy().reshape(self.num_total_points, -1)
        )
        self.y = MinMaxScaler().fit_transform(self.raw_y)

        # fit a linear interpolator on the datapoints
        self.interpolator_type = interpolator_type
        self.fit(interpolator_type=interpolator_type)

    def fit(
        self,
        interpolator_type: str = "linear",
    ) -> Tuple[MyNDInterpolator, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit an extrapolator on the discrete candy dataset.

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
        return self.utility

    def get_data(
        self, add_batch_dim: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve the candy dataset.

        Args:
            add_batch_dim, bool: whether to add a batch dimension.

        Returns:
            X, (1, num_total_points, 2) if add_batch_dim else (num_total_points, 2): datapoints.
            y, (1, num_total_points, 1) if add_batch_dim else (num_total_points, 1): associated utility values.
            Xopt, (1, 1, 2) if add_batch_dim else (1, 2): global optimum locations.
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
            XX, (batch_size, num_points, 2): sampled datapoints.
            YY, (batch_size, num_points, 1): associated utility values.
            Xopt, (batch_size, 1, 2): global optimum location.
            yopt, (batch_size, 1, 1): global optimum value.
        """
        XX = (
            np.random.random((batch_size, num_points, 2))
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
