import os
import pandas as pd
import numpy as np
from data.utils import (
    MyNDInterpolator,
)
import os.path as osp
from typing import List, Tuple


class SushiDataHandler:
    def __init__(self, root_dir: str = "datasets/sushi-data"):
        """Data handler for sushi dataset.

        Attrs:
            num_total_points, scalar: the number of objects.
            d_x, scalar: number of features.
            raw_X, np.ndarray[num_total_points, 2]: candy object containing two features.
            scaled_X, np.ndarray[num_total_points, 2]: standardized candy object.
            raw_y, np.ndarray[num_total_points, 1]: associated utility values.
            std_y, np.ndarray[num_total_points, 1] standardized utility values.
            x_range, list of two elements: input range.
        """
        self.root_dir = root_dir
        df_features = pd.read_csv(
            osp.join(root_dir, f"sushi3.idata"), sep="\t", header=None
        )
        df_score = pd.read_csv(
            osp.join(root_dir, "sushi3b.5000.10.score"), sep=" ", header=None
        )
        # Generate scrores from preferences
        scores = []
        for item_a in range(df_score.values.shape[1]):
            score = 0
            for item_b in range(df_score.values.shape[1]):
                if self._prefer_a(item_a, item_b, df_score):
                    score += 1
            scores.append(score / float(df_score.values.shape[1]))

        self.raw_X = df_features.values[:, [6, 5, 7, 8][:4]].astype(np.float64)
        self.num_total_points = len(self.raw_X)
        self.d_x = self.raw_X.shape[-1]
        self.scaled_X = (self.raw_X - np.min(self.raw_X, axis=0)[None, :]) / (
            np.max(self.raw_X, axis=0) - np.min(self.raw_X, axis=0)
        )[None, :]
        self.x_range = [0, 1]

        self.raw_y = np.array(scores).reshape((-1, 1))
        self.std_y = (self.raw_y - np.min(self.raw_y)) / (
            np.max(self.raw_y) - np.min(self.raw_y)
        )

    def _prefer_a(self, item_a: int, item_b: int, df_scores: List):
        """
        Check from data if item_a has higher score that item_b

        :param item_a: index of the first item to be compared
        :param item_b: index of the second item to be compared
        :param df_scores: Scores of all dat points
        :return: True if item_a is preferred over item_b
        """
        # find records where both item_a and item_b are rated
        # generate preference by comparing the averaged scores from all records
        ix = (df_scores[item_a].values > -1) * (df_scores[item_b].values > -1)
        prefer_a = np.mean(df_scores[item_a].values[ix] > df_scores[item_b].values[ix])
        prefer_b = np.mean(df_scores[item_b].values[ix] > df_scores[item_a].values[ix])
        return prefer_a > prefer_b

    def get_data(self, add_batch_dim: bool = False):
        """Load sushi dataset.

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
        yopt = np.max(y, axis=0, keepdims=True)
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
        """Generalize the sushi dataset to continous space with linear extrapolation.

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
