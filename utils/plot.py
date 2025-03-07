import torch
import numpy as np
from typing import Union, Callable, List
import matplotlib.pyplot as plt


def confidence_interval(value: np.ndarray):
    return 1.96 * value.std(axis=0) / np.sqrt(value.shape[0])


def tnp(data: Union[torch.Tensor, np.ndarray]):
    return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data


def center_scale(data: np.ndarray, t_range: np.ndarray):
    """Center and scale data to target range.

    Args:
        x, (N, d_x): original data.
        t_range, (d_x, 2): target data range.

    Returns:
        data_scaled, (N, d_x): scaled data.
    """
    data_std = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    data_scaled = data_std * (t_range[..., 1] - t_range[..., 0]) + t_range[..., 0]
    return data_scaled


def plot_prediction_on_1d_function(
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

    # visualize the utility on evenly spaced points
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
        y_pred = center_scale(
            data=y_pred, t_range=np.array([b.min(), b.max()]).reshape(1, -1)
        )

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


def plot_prediction_on_2d_function(
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
    """visualize predictions on 2-dimensional function.

    Args:
        X, (N, 2): points in prediction target set.
        y_pred, (N, 1): predicted funtion values on X.
        y, (N): ground truth function values on X.
        Xopt, (num_global_opt, 2): the locations of global optima.
        yopt, (num_global_opt, 1): the global optima.
        x_range, (2, 2): the range of X.
        query_pair, (num_queries, 4): queries.
        query_pair_y, (num_queries, 2): corresponding function values.
        infer_opt_pair, (1, 4): inference optimal next query pair.
        infer_opt_pair_y, (1, 2): corresponding ground true function values.
        utility_function: __call__(x: torch.Tensor) -> y: torch.Tensor, the ground truth function.
    """

    def plot_projected(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        xy_plane: np.ndarray,
        xz_plane: np.ndarray,
        yz_plane: np.ndarray,
        ax,
        **kwargs,
    ):
        """Plot point (x, y, z) on x-y, x-z, y-z planes respectively.

        Args:
            x, (num_points,): x-value.
            y, (num_points,): y-value.
            z, (num_points,): z-value.
            xy_plane, (1): z-value where xy plane locates.
            xz_plane, (1): z-value where xz plane locates.
            yz_plane, (1): z-value where yz plane locates.
        """
        num_points = len(x)
        ax.plot(np.tile(yz_plane, (num_points,)), y, z, **kwargs)
        ax.plot(x, np.tile(xz_plane, (num_points,)), z, **kwargs)
        ax.plot(x, y, np.tile(xy_plane, (num_points,)), **kwargs)

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
        tnp(y_pred).flatten(),
        tnp(y),
        tnp(Xopt),
        tnp(yopt).flatten(),
        tnp(x_range),
        tnp(query_pair),
        tnp(query_pair_y).flatten(),
        tnp(infer_opt_pair),
        tnp(infer_opt_pair_y).flatten(),
    )
    if ax is None:
        ax = plt.figure(figsize=(10, 10)).add_subplot(projection="3d")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    x_offset, y_offset = [0.6 * (x_range[i, 1] - x_range[i, 0]) for i in range(2)]
    z_offset = 0.6 * (y.max() - y.min())
    # visualize the utility on evenly spaced points
    if utility_function is not None:
        # plot function shape: wireframe ground truth and projected contour.
        a = np.linspace(x_range[0, 0], x_range[0, 1], 100)
        b = np.linspace(x_range[1, 0], x_range[1, 1], 100)
        A, B = np.meshgrid(a, b)
        inputs = torch.from_numpy(np.stack((A, B), axis=-1)).reshape(-1, 2)
        C = tnp(utility_function(inputs).reshape(100, 100))
        # the location of x-y, x-z, and y-z planes
        planes = {
            "yz_plane": A.min() - x_offset,
            "xz_plane": B.max() + y_offset,
            "xy_plane": C.min() - z_offset,
        }
        ax.plot_wireframe(
            A,
            B,
            C,
            edgecolor="lightgrey",
            lw=0.5,
            rstride=8,
            cstride=8,
            zorder=0,
        )
        ax.contour(A, B, C, zdir="x", offset=planes["yz_plane"], cmap="coolwarm")
        ax.contour(A, B, C, zdir="y", offset=planes["xz_plane"], cmap="coolwarm")
        ax.contour(A, B, C, zdir="z", offset=planes["xy_plane"], cmap="coolwarm")
        ax.set(
            xlim=(x_range[0, 0] - x_offset, x_range[0, 1] + x_offset),
            ylim=(x_range[1, 0] - y_offset, x_range[1, 1] + y_offset),
            zlim=(C.min() - z_offset, C.max() + z_offset),
        )
        # we can't recover the true function values from preferential data
        # center and scale the predictions for better comparison
        y_pred = center_scale(
            y_pred, t_range=np.array([C.min(), C.max()]).reshape(1, -1)
        )
    else:
        planes = {
            "yz_plane": x_range[0, 0] - x_offset,
            "xz_plane": x_range[1, 1] + y_offset,
            "xy_plane": y.min() - z_offset,
        }
        ax.set(
            xlim=(x_range[0, 0] - x_offset, x_range[0, 1] + x_offset),
            ylim=(x_range[1, 0] - y_offset, x_range[1, 1] + y_offset),
            zlim=(y.min() - z_offset, y.max() + z_offset),
        )
        y_pred = center_scale(
            y_pred, t_range=np.array([y.min(), y.max()]).reshape(1, -1)
        )
        ax.plot(
            X[..., 0],
            X[..., 1],
            y,
            color="gray",
            zorder=0,
            marker=".",
            linestyle="",
            label="Ground Truth",
        )

    # plot ground truth optima
    ax.plot(
        Xopt[..., 0],
        Xopt[..., 1],
        yopt,
        color="lime",
        marker="o",
        linestyle="",
        zorder=4,
        label="True Optima",
    )
    plot_projected(
        x=Xopt[..., 0],
        y=Xopt[..., 1],
        z=yopt,
        **planes,
        ax=ax,
        color="lime",
        marker="o",
        zorder=4,
        linestyle="",
    )

    # plot queries
    # flatten pair into points
    query_pair = query_pair.reshape(-1, 2)  # (2 * num_queries, 1)
    infer_opt_pair = infer_opt_pair.reshape(-1, 2)
    ax.plot(
        query_pair[..., 0],
        query_pair[..., 1],
        query_pair_y,
        label="Queries",
        marker="o",
        color="royalblue",
        markersize=5,
        zorder=2,
        linestyle="",
    )
    plot_projected(
        query_pair[..., 0],
        query_pair[..., 1],
        query_pair_y,
        **planes,
        ax=ax,
        marker="o",
        color="royalblue",
        markersize=5,
        zorder=2,
        linestyle="",
    )
    # plot next query
    ax.plot(
        query_pair[-2:, 0],
        query_pair[-2:, 1],
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
        infer_opt_pair[..., 0],
        infer_opt_pair[..., 1],
        infer_opt_pair_y,
        c="fuchsia",
        label="Inference Optima",
        marker="*",
        zorder=4,
        linestyle="",
    )
    plot_projected(
        infer_opt_pair[..., 0],
        infer_opt_pair[..., 1],
        infer_opt_pair_y,
        **planes,
        ax=ax,
        marker="o",
        color="fuchsia",
        markersize=5,
        zorder=2,
        linestyle="",
    )
    # plot predictions
    ax.plot(
        X[..., 0],
        X[..., 1],
        y_pred,
        color="orange",
        zorder=1,
        marker=".",
        linestyle="",
        label="Prediction",
    )

    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    return ax


def plot_metric_along_trajectory(
    metrics: List[torch.Tensor],
    model_names: List,
    ax=None,
):
    """Plot metric along trajectory for a list of models.

    Args:
        metrics, List[torch.Tensor]: metrics for different models, each of shape (num_seed, H)
        model_names, List: the associated model name.

    Return:
        ax, Axes.
    """
    colors = {
        "PABBO256": "violet",
        "PABBO512": "darkviolet",
        "PABBO1024": "deeppink",
        "PABBO256_5": "yellow",
        "PABBO": "violet",
        "PABBO_synthetic": "fuchsia",
        "qTS": "blue",
        "qEUBO": "orange",
        "qEI": "green",
        "qNEI": "grey",
        "mpes": "red",
        "rs": "black",
    }
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    for i, model_tra in enumerate(metrics):
        model_tra = model_tra.detach().cpu().numpy()
        x = np.arange(model_tra.shape[-1])
        mean = model_tra.mean(axis=0).flatten()
        ci = confidence_interval(model_tra)
        ax.plot(x, mean, "o-", color=colors[model_names[i]], label=f"{model_names[i]}")
        ax.fill_between(
            x, mean + ci, mean - ci, alpha=0.3, color=colors[model_names[i]]
        )

    plt.tight_layout()
    return ax
