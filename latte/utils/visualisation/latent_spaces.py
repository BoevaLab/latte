"""
Various visualisation helper functions for plotting heatmaps and factor trends in small-dimensional (1D or 2D) latent
spaces.
"""
from typing import Optional, List

import numpy as np
import torch
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

import matplotlib.pyplot as plt

from latte.dataset.utils import RandomGenerator


def factor_trend(
    Z: torch.Tensor,
    Y: torch.Tensor,
    target: str,
    discrete_factors: Optional[List[int]] = None,
    interpolate: bool = True,
    k_continuous: int = 5,
    k_discrete: int = 5,
    N: int = 4000,
    q: float = 0.05,
    attribute_names: Optional[List[str]] = None,
    file_name: Optional[str] = None,
    rng: RandomGenerator = 42,
) -> None:
    """
    Plots a graph of the values of a set of 1-dimensional variables against a 1-dimensional random variable.
    It plots the values of the provided set of points or samples `N` points uniformly from the interval containing the
    1-dimensional variable, and interpolates the values of the set of other variables using `k`-nearest neighbours.

    The function can be used to plot the relationship between a ground-truth factor of variation and a two-dimensional
    learned representation space or the other way around.
    Args:
        Z: The tensor of values of the learned representations of `M` samples.
        Y: The tensor of values of the true factors of variation of `M` samples.
        target: The meaning of the target.
                If "ground-truth", the hue corresponds to the values of the grond-truth factor.
                In that case, Z.shape[1] must be 1 and Y.shape[1] figures will be produces, one heatmap for each
                of the columns of `Y`.
                If "learned", it corresponds to the values of a learned dimension.
                In that case, Y.shape[1] must be 1 and Z.shape[1] figures will be produces, one heatmap for each
                of the columns of `Z`.
        discrete_factors: List of indices of discrete factors.
                          If None, it is assumed all factors are continuous.
        interpolate: Whether to plot the heatmaps as the values of *sampled* points (interpolate = True) based on a
                     kNN model learned on the true values or just plot the distribution and the heatmap of the original
                     `M` points provided.
        k_continuous: Number of nearest neighbours to use for interpolation of the values if the targets are continuous.
        k_discrete: Number of nearest neighbours to use for interpolation of the values if the targets are discrete.
        N: Number of data points to plot.
        q: The quantile at which to cut off the values of `Y`.
        attribute_names: A list of length Z.shape[1] (target = "ground-truth") or Y.shape[1] (target = "learned") to
                         set the titles of all the heatmaps produced.
        file_name: An optional file name.
                   If specified, the figure will be saved there, otherwise, it will be displayed.
        rng: A rando generator for the uniform sampling.

    """
    assert target in ["ground-truth", "learned"]

    rng = np.random.default_rng(rng)

    if target == "learned":
        assert Y.shape[1] == 1
        # Plot Z against Y
        X = Y
        T = Z

        xlabel, ylabel = "Factor", "Learned Dimension"
    else:
        # Plot Y against Z
        assert Z.shape[1] == 1
        X = Z
        T = Y

        xlabel, ylabel = "Learned Dimension", "Factor"

    fig, ax = plt.subplots(nrows=1, ncols=len(attribute_names), figsize=(2 * len(attribute_names), 2), dpi=120)
    ax = ax if len(attribute_names) > 1 else [ax]  # in case only a single axes object is returned

    rng = np.random.default_rng(rng)

    for jj in range(len(attribute_names)):

        X_jj, T_jj = (X, T[:, jj]) if target == "ground-truth" else (X[:, [jj]], T)

        if interpolate:

            # Generate a uniform sample from the range of `X`
            x = np.sort(
                rng.uniform(low=-1, high=1, size=(N,))
                * np.asarray([torch.quantile(X_jj, 1 - q) - torch.quantile(X_jj, q)])
            )

            model = (
                KNeighborsClassifier(n_neighbors=k_discrete).fit(X_jj, T_jj)
                if jj in discrete_factors
                else KNeighborsRegressor(n_neighbors=k_continuous).fit(X_jj, T_jj)
            )

            # Compute the values of the variable in `T[:, jj` for the sampled points based on kNN
            t = model.predict(x.reshape(-1, 1))

        else:
            ixs = rng.choice(len(X), size=min(N, len(X)), replace=False)
            x = X_jj[ixs]
            t = T_jj[ixs]

        ax[jj].plot(x, t, label="Latent dimension")
        ax[jj].set(
            xlabel=xlabel,
            ylabel=ylabel,
            title=attribute_names[jj] if attribute_names is not None else "",
        )
        ax[jj].get_xaxis().set_ticks([])
        ax[jj].get_yaxis().set_ticks([])
        ax[jj].tick_params(labelsize=6)

    fig.tight_layout()

    if file_name is not None:
        fig.savefig(file_name, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def factor_heatmap(
    Z: torch.Tensor,
    Y: torch.Tensor,
    target: str,
    discrete_factors: Optional[List[int]] = None,
    interpolate: bool = True,
    k_continuous: int = 5,
    k_discrete: int = 5,
    N: int = 4000,
    q: float = 0.05,
    attribute_names: Optional[List[str]] = None,
    file_name: Optional[str] = None,
    rng: RandomGenerator = 42,
) -> None:
    """
    Plots a heatmap of the values of a set of 1-dimensional variables against a 2-dimensional random variable.
    It plots the values of the provided set of points or samples `N` points uniformly from the square containing the
    2-dimensional variable, and interpolates the values of the set of other variables using `k`-nearest neighbours.

    The function can be used to plot the relationship between a ground-truth factor of variation and a two-dimensional
    learned representation space or the other way around.
    Args:
        Z: The tensor of values of the learned representations of `M` samples.
        Y: The tensor of values of the true factors of variation of `M` samples.
        target: The meaning of the target.
                If "ground-truth", the hue corresponds to the values of the grond-truth factor.
                In that case, Z.shape[1] must be 2 and Y.shape[1] figures will be produces, one heatmap for each
                of the columns of `Y`.
                If "learned", it corresponds to the values of a learned dimension.
                In that case, Y.shape[1] must be 2 and Z.shape[1] figures will be produces, one heatmap for each
                of the columns of `Z`.
        discrete_factors: List of indices of discrete factors.
                          If None, it is assumed all factors are continuous.
        interpolate: Whether to plot the heatmaps as the values of *sampled* points (interpolate = True) based on a
                     kNN model learned on the true values or just plot the distribution and the heatmap of the original
                     `M` points provided.
        k_continuous: Number of nearest neighbours to use for interpolation of the values if the targets are continuous.
        k_discrete: Number of nearest neighbours to use for interpolation of the values if the targets are discrete.
        N: Number of data points to plot.
        q: The quantile at which to cut off the values of `Y`.
        attribute_names: A list of length Z.shape[1] (target = "ground-truth") or Y.shape[1] (target = "learned") to
                         set the titles of all the heatmaps produced.
        file_name: An optional file name.
                   If specified, the figure will be saved there, otherwise, it will be displayed.
        rng: A rando generator for the uniform sampling.

    """
    assert target in ["ground-truth", "learned"]

    if target == "learned":
        # Plot Z against Y
        assert Y.shape[1] == 2
        X = Y
        T = Z

        axis_name = "Factor"
    else:
        # Plot Y against Z
        assert Z.shape[1] == 2
        X = Z
        T = Y

        axis_name = "Learned Dimension"

    fig, ax = plt.subplots(nrows=1, ncols=len(attribute_names), figsize=(2 * len(attribute_names), 2), dpi=120)
    ax = ax if len(attribute_names) > 1 else [ax]  # in case only a single axes object is returned

    rng = np.random.default_rng(rng)

    for jj in range(len(attribute_names)):
        X_jj, T_jj = (X, T[:, jj]) if target == "ground-truth" else (X[:, [jj]], T)

        if interpolate:

            # Generate a uniform sample from the range of `X`
            x = rng.uniform(low=-1, high=1, size=(N, 2)) * np.asarray(
                [
                    torch.quantile(X[:, 0], 1 - q) - torch.quantile(X[:, 0], q),
                    torch.quantile(X[:, 1], 1 - q) - torch.quantile(X[:, 1], q),
                ]
            )

            # Compute the values of the variable in `T[:, jj]` for the sampled points based on kNN
            model = (
                KNeighborsClassifier(n_neighbors=k_discrete).fit(X_jj, T_jj)
                if jj in discrete_factors
                else KNeighborsRegressor(n_neighbors=k_continuous).fit(X_jj, T_jj)
            )

            t = model.predict(x)

        else:
            ixs = rng.choice(len(X), size=min(N, len(X)), replace=False)
            x = X_jj[ixs]
            t = T_jj[ixs]

        ax[jj].scatter(x[:, 0], x[:, 1], c=t, cmap="plasma" if jj in discrete_factors else "jet", s=6, label=t)
        ax[jj].set(
            xlabel=f"{axis_name} 1",
            ylabel=f"{axis_name} 2",
            title=attribute_names[jj] if attribute_names is not None else "",
        )
        ax[jj].get_xaxis().set_ticks([])
        ax[jj].get_yaxis().set_ticks([])
        ax[jj].tick_params(labelsize=6)

    fig.tight_layout()

    if file_name is not None:
        fig.savefig(file_name, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
