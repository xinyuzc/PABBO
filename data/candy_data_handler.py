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
    ):
        """Data handler for candy dataset.

        Attrs:
            num_total_points, scalar: the number of objects.
            d_x, scalar: number of features.
            raw_X, np.ndarray[num_total_points, 2]: candy object containing two features.
            scaled_X, np.ndarray[num_total_points, 2]: standardized candy object.
            raw_y, np.ndarray[num_total_points, 1]: associated utility values.
            std_y, np.ndarray[num_total_points, 1] standardized utility values.
            x_range, list of two elements: input range.
        """
        filename = "candy-data.csv"
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
        )  # drop useless columns

        self.num_total_points = len(self.raw_data)
        self.raw_X = self.raw_data[["sugarpercent", "pricepercent"]].to_numpy()
        self.scaled_X = (self.raw_X - np.min(self.raw_X, axis=0)[None, :]) / (
            np.max(self.raw_X, axis=0) - np.min(self.raw_X, axis=0)
        )[None, :]
        self.raw_y = (
            self.raw_data["winpercent"].to_numpy().reshape(self.num_total_points, -1)
        )
        self.std_y = MinMaxScaler().fit_transform(self.raw_y)
        self.x_range = [0, 1]
        self.d_x = self.raw_X.shape[-1]
        self.utility = None

    def get_data(
        self, add_batch_dim: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load candy dataset.

        Args:
            add_batch_dim, bool: whether to add batch dimension.

        Returns:
            X, (1, num_total_points, 2) if add_batch_dim else (num_total_points, 2)
            y, (1, num_total_points, 1) if add_batch_dim else (num_total_points, 1)
            Xopt, (1, 1, 2) if add_batch_dim else (1, 2)
            yopt, (1, 1, 1) if add_batch_dim else (1, 1)

        """
        X = self.scaled_X
        y = self.std_y
        yopt = np.max(y, axis=0, keepdims=True)  # (1, 1)
        Xopt = np.take_along_axis(X, np.argmax(y, axis=0, keepdims=True), axis=0)
        if add_batch_dim:
            X, y, Xopt, yopt = (
                X[None, :, :],
                y[None, :, :],
                Xopt[None, :, :],
                yopt[None, :, :],
            )
        return X, y, Xopt, yopt

    def linear_extrapolation(
        self,
        interpolator_type: str = "linear",
    ) -> Tuple[MyNDInterpolator, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generalize the candy dataset to continous space with linear extrapolation.

        Args:
            interpolator_type, str in ["linear", "nearest"]

        Returns:
            utility: utility function through linear extrapolation on the candy dataset
            Xopt, (1, 1, 2)
            yopt, (1, 1, 1)
            X, (1, num_total_points, 2)
            y, (1, num_total_points, 1)
        """
        X, y, Xopt, yopt = self.get_data(add_batch_dim=False)
        self.utility = MyNDInterpolator(X=X, y=y, interpolator_type=interpolator_type)
        X, y, Xopt, yopt = (
            X[None, :, :],
            y[None, :, :],
            Xopt[None, :, :],
            yopt[None, :, :],
        )
        return self.utility, Xopt, yopt, X, y

    def sample(
        self,
        batch_size: int = 1,
        num_points: int = 200,
    ):
        """Sample from the generalized candy dataset.

        Args:
            batch_size, scalar
            num_points, scalar: number of points in each set.

        Returns:
            XX, (batch_size, num_points, 2)
            YY, (batch_size, num_points, 1)
            Xopt, (batch_size, 1, 2)
            yopt, (batch_size, 1, 1)
        """
        if self.utility is None:
            self.utility, Xopt, yopt, _, _ = self.linear_extrapolation()
        XX = (
            np.random.random((batch_size, num_points, 2))
            * (self.x_range[1] - self.x_range[0])
            + self.x_range[0]
        )
        YY = self.utility(XX=XX)
        return (
            XX,
            YY,
            np.tile(Xopt, (batch_size, 1, 1)),
            np.tile(yopt, (batch_size, 1, 1)),
        )
