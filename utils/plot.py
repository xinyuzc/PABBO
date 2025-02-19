import torch
import numpy as np
from typing import Union, Callable
import matplotlib.pyplot as plt


def tnp(data: Union[torch.Tensor, np.ndarray]):
    return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data


def center_scale(data: np.ndarray, t_range: np.ndarray):
    """Center and scale data to target range.

    Args:
        x, (N, d_x)
        t_range, (d_x, 2)
    """
    data_std = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    data_scaled = data_std * (t_range[..., 1] - t_range[..., 0]) + t_range[..., 0]
    return data_scaled


def plot_predictions_on_1d_function(
    X: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    Xopt: Union[torch.Tensor, np.ndarray],
    yopt: Union[torch.Tensor, np.ndarray],
    x_range: torch.Tensor,
    query_pair: Union[torch.Tensor, np.ndarray],
    query_pair_y: Union[torch.Tensor, np.ndarray],
    infer_opt_pair: Union[torch.Tensor, np.ndarray],
    infer_opt_pair_y: Union[torch.Tensor, np.ndarray],
    utility_function: Union[Callable, None] = None,
    ax=None,
):
    """visualize predictions on 1-dimensional function.

    Args:
        X, (N, 1): points in prediction target set.
        y_pred, (N, 1): predicted funtion values on X.
        y, (N): ground truth function values on X.
        Xopt, (num_global_opt, 1): the locations of global optima.
        yopt, (num_global_opt, 1): the global optima.
        x_range, (1, 2): the range of X.
        query_pair, (num_queries, 2): queries.
        query_pair_y, (num_queries, 2): corresponding function values.
        infer_opt_pair, (1, 2): inference optimal next query pair.
        infer_opt_pair_y, (1, 2): corresponding ground true function values.
        utility_function: __call__(x: torch.Tensor) -> y: torch.Tensor, the ground truth function.
    """
    (
        X,
        y_pred,
        y,
        Xopt,
        yopt,
        x_range,
        query_pair,
        query_pair_y,
        infer_opt_pair,
        infer_opt_pair_y,
    ) = (
        tnp(X),
        tnp(y_pred),
        tnp(y),
        tnp(Xopt),
        tnp(yopt),
        tnp(x_range),
        tnp(query_pair),
        tnp(query_pair_y),
        tnp(infer_opt_pair),
        tnp(infer_opt_pair_y),
    )
    if ax is None:
        ax = plt.figure(figsize=(6, 4)).add_subplot()
    ax.spines[["right", "top"]].set_visible(False)
    ax.set(xlabel="X", ylabel="Y")

    # visualize the true function on grid
    if utility_function is not None:
        a = np.linspace(x_range[:, 0].item(), x_range[:, 1].item(), 100)[:, None]
        b = utility_function(torch.from_numpy(a))
        a, b = tnp(a), tnp(b)
        ax.plot(a, b, color="lightgrey", zorder=0)

        # set lims for each axis
        x_offset = 0.1 * (x_range[:, 1].item() - x_range[:, 0].item())
        y_offset = 0.1 * (b.max() - b.min())
        ax.set(
            xlim=(
                x_range[:, 0] - x_offset,
                x_range[:, 1] + x_offset,
            ),
            ylim=(b.min() - y_offset, b.max() + y_offset),
        )

        # we can't recover the true function values from preferential data
        # center and scale the predictions for better comparison
        y_pred = center_scale(data=y_pred, t_range=(b.min(), b.max()))

    # flatten pair into points
    query_pair = query_pair.reshape(-1, 1)  # (2 * num_queries, 1)
    query_pair_y = query_pair_y.reshape(-1, 1)
    infer_opt_pair = infer_opt_pair.reshape(-1, 1)
    infer_opt_pair_y = infer_opt_pair_y.reshape(-1, 1)

    # NOTE `zorder` works for ax.plot but fails for ax.scatter in 3d plot
    # plot queries
    ax.plot(
        query_pair,
        query_pair_y,
        label="Queries",
        marker="o",
        color="royalblue",
        markersize=5,
        zorder=2,
        linestyle="",
    )
    # plot next query
    ax.plot(
        query_pair[-2:],
        query_pair_y[-2:],
        label="Next Query",
        marker="D",
        color="red",
        markersize=4,
        zorder=3,
        linestyle="",
    )

    # plot inference optima
    ax.plot(
        infer_opt_pair,
        infer_opt_pair_y,
        c="fuchsia",
        label="Inference Optima",
        marker="*",
        zorder=4,
        linestyle="",
    )

    # plot ground truth optima
    ax.plot(
        Xopt,
        yopt,
        color="lime",
        linestyle="",
        marker="o",
        zorder=4,
        label="True Optima",
    )

    # plot ground truth function values
    ax.plot(
        X,
        y,
        color="gray",
        zorder=0,
        marker=".",
        linestyle="",
        label="Ground Truth",
    )
    # plot predictions
    ax.plot(
        X,
        y_pred,
        color="orange",
        zorder=1,
        marker=".",
        linestyle="",
        label="Prediction",
    )

    return ax
