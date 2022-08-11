import pathlib
from typing import Callable, Dict, Optional, List, Any, Union, Tuple
from itertools import product

import geoopt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.neighbors import KNeighborsRegressor

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import art3d
import seaborn as sns

from pythae.models.vae.vae_model import VAE

from latte.dataset.utils import RandomGenerator
from latte.models.probabilistic_pca import probabilistic_pca

left, width = -0.25, 0.5
bottom, height = 0.25, 0.5
right = left + width
top = bottom + height


def vae_reconstructions(
    dataset: Union[Dataset, torch.Tensor],
    vae_model: VAE,
    latent_transformation: Optional[Callable] = None,
    nrows: int = 1,
    ncols: int = 12,
    device: str = "cuda",
    cmap: Optional[str] = "Greys",
    file_name: Optional[str] = None,
    rng: RandomGenerator = 42,
) -> None:
    """
    Plots a sample of the original images and the corresponding reconstructions from the VAE.
    Optionally, also transform the original VAE representations and reconstruct from the transformed representations.
    Args:
        dataset: The data to be considered.
                 It can be a torch dataset or just a tensor of observations.
        vae_model: The trained VAE to use for embedding and reconstructing.
        latent_transformation: Optional transformation of the latent representation to induce wanted properties (e.g.,
                               erasure of a concept).
                               It should return a vector of the same dimensionality.
        nrows: Number of rows of images to plot.
        ncols: Number of columns of images to plot.
        device: The pytorch device to load the data onto.
        cmap: The color map for the matplotlib plotting function.
        file_name: Optional file name to save the produced figure.
        rng: The random generator to use to generate a sample.

    """

    rng = np.random.default_rng(rng)
    # Generate a random sample of images
    # nrows * ncols images will be drawn
    index = rng.choice(len(dataset), size=nrows * ncols, replace=len(dataset) <= nrows * ncols)
    # `replace` configured such that it handles cases when len(dataset) <= nrows * ncols

    xs = dataset[index]
    x = xs[0] if isinstance(xs, tuple) else xs  # Accommodate for datasets returning both observables and targets

    # Produce the reconstructions
    with torch.no_grad():
        y = vae_model.encoder(x.to(device))
        # We take the mean of the distribution as the latent representation for plotting
        z_original = y.embedding
        # Decode the original representation into the reconstruction
        x_hat_original = vae_model.decoder(z_original).reconstruction.detach().cpu()
        if latent_transformation is not None:
            # If a transformation of the latent space is provided, transform the original representation and decode it
            z_transformed = latent_transformation(y.embedding)
            x_hat_transformed = vae_model.decoder(z_transformed).reconstruction.detach().cpu()

    rows_per_sample = 2 if latent_transformation is None else 3
    fig, axes = plt.subplots(
        ncols=ncols, nrows=rows_per_sample * nrows, figsize=(0.75 * ncols, 0.75 * rows_per_sample * nrows), dpi=200
    )

    for i, j in product(range(nrows), range(ncols)):
        # Plot the original and the reconstruction in the same column
        # We assume the images are in the stored in pytorch (c, w, h) format, so we permute them to plot them
        axes[rows_per_sample * i, j].imshow(x[ncols * i + j].permute(1, 2, 0), cmap=cmap)
        axes[rows_per_sample * i, j].get_xaxis().set_ticks([])
        axes[rows_per_sample * i, j].get_yaxis().set_ticks([])
        axes[rows_per_sample * i + 1, j].imshow(x_hat_original[ncols * i + j].permute(1, 2, 0), cmap=cmap)
        axes[rows_per_sample * i + 1, j].get_xaxis().set_ticks([])
        axes[rows_per_sample * i + 1, j].get_yaxis().set_ticks([])
        if latent_transformation is not None:
            axes[rows_per_sample * i + 2, j].imshow(x_hat_transformed[ncols * i + j].permute(1, 2, 0), cmap=cmap)
            axes[rows_per_sample * i + 2, j].get_xaxis().set_ticks([])
            axes[rows_per_sample * i + 2, j].get_yaxis().set_ticks([])

        if j > 0:
            continue

        # Add some information on the left-most column
        axes[rows_per_sample * i, j].text(
            -1.5,
            0.5 * (bottom + top),
            "Original",
            horizontalalignment="center",
            verticalalignment="center",
            rotation="horizontal",
            size=8,
            transform=axes[rows_per_sample * i, j].transAxes,
        )

        axes[rows_per_sample * i + 1, j].text(
            -1.5,
            0.5 * (bottom + top),
            "Reconstruction" if latent_transformation is None else "Unmodified\nReconstruction",
            horizontalalignment="center",
            verticalalignment="center",
            rotation="horizontal",
            size=8,
            transform=axes[rows_per_sample * i + 1, j].transAxes,
        )

        if latent_transformation is not None:
            axes[rows_per_sample * i + 2, j].text(
                -1.5,
                0.5 * (bottom + top),
                "Modified\nReconstruction",
                horizontalalignment="center",
                verticalalignment="center",
                rotation="horizontal",
                size=8,
                transform=axes[rows_per_sample * i + 2, j].transAxes,
            )

    plt.suptitle("Reconstructions of the VAE", size=10)
    fig.tight_layout()

    if file_name is not None:
        fig.savefig(file_name)
        plt.close()
    else:
        plt.show()


def _get_axis_values(
    Z: Optional[torch.Tensor],
    axis: Optional[int],
    n_values: Optional[int] = None,
    axis_values: Optional[np.ndarray] = None,
    q: Tuple[float, float] = (0.01, 0.99),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    qs = None
    # Intervene on the currently chosen latent dimension
    if axis_values is None:
        # The values to which the dimensions will be set correspond to quantiles
        # of the distribution over the entire dataset
        qs = np.linspace(q[0], q[1], n_values)
        values = np.quantile(Z[:, axis], q=qs)
    elif axis_values.shape[0] == 1:
        values = axis_values
    else:
        values = axis_values[axis, :]

    return values, qs


# TODO (Anej): Documentation
def latent_traversals(
    vae_model: nn.Module,
    starting_x: torch.Tensor,
    intervention: bool = False,
    A_hat: Optional[geoopt.ManifoldTensor] = None,
    x_hat_transformation: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    n_axes: int = 1,
    Z: torch.Tensor = None,
    n_values: int = 12,
    axis_values: Optional[np.ndarray] = None,
    q: Tuple[float, float] = (0.01, 0.99),
    shift: bool = False,
    sort_by: str = "var",
    cmap: Optional[str] = "Greys",
    device: str = "cuda",
    file_name: Optional[str] = None,
) -> None:
    """
    Plot the latent traversals of the `n_axes` most-variable axes produced by the VAE `vae_model`.

    Args:
        vae_model: The VAE to use for embedding and reconstructing.
        starting_x: The starting image of the latent traversals.
                    Individual axes of the embedding of this image will be intervened on.
        Z: The latent representations of the entire dataset.
           This is used to calculate the variance statistics and quantiles of all the axes.
           Can be either the full VAE latent space or the projections onto the linear subspace as found by ManifoldMINE.
        n_values: Number of values to try in each dimension. `values` overrides this.
        axis_values: Optional. Manually chosen values for interventions.
                If it is a 1D array, the same values will be used for all axes.
                Otherwise, the first dimension of the array has to match the number of plotted axes;
                the values of the i-th row of `values` will be used to intervene on the i-th axis
        shift: If true, shift the axes by the quantile values (add them to the original value) instead of
               setting them to the raw quantile value.
        n_axes: Number of axes to traverse.
        sort_by: The criterion by which to sort the axes - either "KL" or "var" (KL version not working yet).
        cmap: The color map for the matplotlib plotting function.
        file_name: Optional file name to save the produced figure.
    """

    assert not intervention or A_hat is not None

    # TODO (Anej): This is not really correct...
    kl_divergences = np.asarray(
        [
            -0.5 * (1 + torch.log(torch.var(Z[:, j])) - torch.mean(Z[:, j]) ** 2 - torch.var(Z[:, j]))
            for j in range(Z.shape[-1])
        ]
    )
    variances = np.asarray([torch.var(Z[:, j]) for j in range(Z.shape[-1])])
    axis_order = np.argsort(-kl_divergences if sort_by == "KL" else -variances)

    fig, axes = plt.subplots(
        nrows=n_axes, ncols=n_values + 1, figsize=(0.75 * (n_values + 1), 0.65 * (n_axes + 1)), dpi=200
    )

    # To represent `x`, we just take the mean of the Gaussian distribution
    with torch.no_grad():
        starting_z = vae_model.encoder(starting_x.unsqueeze(0)).embedding  # We extract `mu`
        starting_x_hat = x_hat_transformation(
            vae_model.decoder(starting_z).reconstruction.detach().cpu()[0]  # First image from the batch
        )

    ax = axes[0, 0] if n_axes > 1 else axes[0]
    ax.text(
        0.5,
        1.1,
        "Original",
        horizontalalignment="center",
        verticalalignment="bottom",
        rotation="horizontal",
        size=8,
        transform=ax.transAxes,
    )

    with torch.no_grad():
        for i in range(n_axes):

            # The manipulated axes are plotted in the order of descending variance
            axis = axis_order[i]

            ax = axes[i, 0] if n_axes > 1 else axes[0]
            var_expr = r"$\sigma^2$"
            ax.text(
                -1.5,
                0.5 * (bottom + top),
                f"Axis {axis}\n KL = {kl_divergences[axis]: .2f}"
                if sort_by == "KL"
                else f"Axis {axis}\n {var_expr} = {variances[axis]: .2f}",
                horizontalalignment="center",
                verticalalignment="center",
                rotation="horizontal",
                size=8,
                transform=ax.transAxes,
            )

            ax.imshow(starting_x_hat.permute(1, 2, 0), cmap=cmap)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            values, qs = _get_axis_values(Z, axis, n_values, axis_values, q)

            for j in range(n_values):

                z = starting_z.clone()
                if intervention and A_hat is not None:
                    # The model has to be the original so that we can actually work with these representations

                    if not shift:
                        erase = z @ A_hat
                        z = z - erase @ A_hat.T

                    add = torch.zeros((1, A_hat.shape[1])).to(device)
                    add[0, axis] = values[j]
                    z = z + add @ A_hat.T

                else:
                    # Intervene on a dimension
                    z[0, axis] = z[0, axis] + values[j] if shift else values[j]

                # Decode
                with torch.no_grad():
                    x_hat = x_hat_transformation(
                        vae_model.decoder(z).reconstruction.detach().cpu()[0]  # Take the first image from the batch
                    )

                ax = axes[i, j + 1] if n_axes > 1 else axes[j + 1]
                ax.imshow(x_hat.permute(1, 2, 0), cmap=cmap)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

                if i == 0 and axis_values is None:  # Only print if values were not pre-specified

                    ax.text(
                        0.5,
                        1.1,
                        f"{qs[j]: .2f}",
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        rotation="horizontal",
                        size=8,
                        transform=ax.transAxes,
                    )

    plt.suptitle("Latent traversals", size=10)
    fig.tight_layout()

    if file_name is not None:
        fig.savefig(file_name)
        plt.close()
    else:
        plt.show()


def _get_axis_values_2d(
    Z: Optional[torch.Tensor],
    axis: Optional[Tuple[int, int]],
    n_values: Optional[int] = None,
    axis_values: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    q: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.01, 0.99), (0.01, 0.99)),
) -> Tuple[Tuple[np.ndarray, np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]]:
    qs = None
    # Intervene on the currently chosen latent dimension
    if axis_values is None:
        # The values to which the dimensions will be set correspond to quantiles
        # of the distribution over the entire dataset
        qs = np.linspace(q[0][0], q[0][1], n_values), np.linspace(q[1][0], q[1][1], n_values)
        values = np.quantile(Z[:, axis[0]], q=qs[0]), np.quantile(Z[:, axis[1]], q=qs[1])
    else:
        values = axis_values

    return values, qs


def latent_traversals_2d(
    vae_model: nn.Module,
    starting_x: torch.Tensor,
    axis: Tuple[int, int],
    intervention: bool = False,
    A_hat: Optional[geoopt.ManifoldTensor] = None,
    x_hat_transformation: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    Z: torch.Tensor = None,
    n_values: int = 12,
    axis_values: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    q: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.01, 0.99), (0.01, 0.99)),
    shift: bool = False,
    cmap: Optional[str] = "Greys",
    device: str = "cuda",
    file_name: Optional[str] = None,
) -> None:
    """
    Plot the 2-dimensional latent traversals of 2 specified axes produced by the VAE `vae_model`.

    Args:
        vae_model: The VAE to use for embedding and reconstructing.
        starting_x: The starting image of the latent traversals.
                    Individual axes of the embedding of this image will be intervened on.
        Z: The latent representations of the entire dataset.
           This is used to calculate the variance statistics and quantiles of all the axes.
           Can be either the full VAE latent space or the projections onto the linear subspace as found by ManifoldMINE.
        n_values: Number of values to try in each dimension. `values` overrides this.
        shift: If true, shift the axes by the quantile values (add them to the original value) instead of
               setting them to the raw quantile value.
        cmap: The color map for the matplotlib plotting function.
        file_name: Optional file name to save the produced figure.
    """

    assert not intervention or A_hat is not None

    fig, axes = plt.subplots(
        nrows=n_values + 1, ncols=n_values + 1, figsize=(0.75 * (n_values + 1), 0.75 * (n_values + 1)), dpi=200
    )

    # To represent `x`, we just take the mean of the Gaussian distribution
    with torch.no_grad():
        starting_z = vae_model.encoder(starting_x.unsqueeze(0)).embedding  # We extract `mu`
        starting_x_hat = x_hat_transformation(
            vae_model.decoder(starting_z).reconstruction.detach().cpu()[0]  # First image from the batch
        )

    values, qs = _get_axis_values_2d(Z, axis, n_values, axis_values, q)

    axes[0, 0].text(
        -0.5,
        0.5,
        "Original",
        horizontalalignment="center",
        verticalalignment="center",
        rotation="horizontal",
        size=6,
        transform=axes[0, 0].transAxes,
    )
    axes[0, 0].text(
        0.5,
        1.15,
        "Original",
        horizontalalignment="center",
        verticalalignment="center",
        rotation="horizontal",
        size=6,
        transform=axes[0, 0].transAxes,
    )

    axes[0, 0].imshow(starting_x_hat.permute(1, 2, 0), cmap=cmap)
    axes[0, 0].get_xaxis().set_ticks([])
    axes[0, 0].get_yaxis().set_ticks([])

    for a in range(n_values):

        axes[a + 1, 0].imshow(starting_x_hat.permute(1, 2, 0), cmap=cmap)
        axes[a + 1, 0].get_xaxis().set_ticks([])
        axes[a + 1, 0].get_yaxis().set_ticks([])

        axes[0, a + 1].imshow(starting_x_hat.permute(1, 2, 0), cmap=cmap)
        axes[0, a + 1].get_xaxis().set_ticks([])
        axes[0, a + 1].get_yaxis().set_ticks([])

        if axis_values is None:  # Only print if values were not pre-specified

            axes[0, a + 1].text(
                0.5,
                1.1,
                f"{qs[0][a]: .2f}",
                horizontalalignment="center",
                verticalalignment="bottom",
                rotation="horizontal",
                size=8,
                transform=axes[0, a + 1].transAxes,
            )

            axes[a + 1, 0].text(
                -0.5,
                0.5,
                f"{qs[1][a]: .2f}",
                horizontalalignment="center",
                verticalalignment="bottom",
                rotation="horizontal",
                size=8,
                transform=axes[a + 1, 0].transAxes,
            )

    with torch.no_grad():

        for i in range(n_values):

            for j in range(n_values):

                z = starting_z.clone()
                if intervention and A_hat is not None:
                    # The model has to be the original so that we can actually work with these representations

                    if not shift:
                        erase = z @ A_hat
                        z = z - erase @ A_hat.T

                    add = torch.zeros((1, A_hat.shape[1])).to(device)
                    add[0, axis[0]] = values[0][j]
                    add[0, axis[1]] = values[1][i]
                    z = z + add @ A_hat.T

                else:
                    # Intervene on a dimension
                    z[0, axis[0]] = z[0, axis[0]] + values[0][j] if shift else values[0][j]
                    z[0, axis[1]] = z[0, axis[1]] + values[1][i] if shift else values[1][i]

                # Decode
                with torch.no_grad():
                    x_hat = x_hat_transformation(
                        vae_model.decoder(z).reconstruction.detach().cpu()[0]  # Take the first image from the batch
                    )

                axes[i + 1, j + 1].imshow(x_hat.permute(1, 2, 0), cmap=cmap)
                axes[i + 1, j + 1].get_xaxis().set_ticks([])
                axes[i + 1, j + 1].get_yaxis().set_ticks([])

    plt.suptitle("Latent traversals", size=10)
    fig.tight_layout()

    if file_name is not None:
        fig.savefig(file_name)
        plt.close()
    else:
        plt.show()


def _get_lines(
    X: np.ndarray,
    A: Optional[np.ndarray] = None,
    mu: Optional[np.ndarray] = None,
    A_hat: Optional[np.ndarray] = None,
    mu_hat: Optional[np.ndarray] = None,
    dim: int = 3,
) -> List:
    """
    Utility function to get the lines denoting the linear basis of a subspace defined by some projection matrix A and
    its estimate A_hat.
    """

    # To plot lines of the appropriate length
    scale = max([np.max(X[:, j]) - np.min(X[:, j]) for j in range(X.shape[-1])]) / 2

    lines = []
    params = [(M, v, t) for M, v, t in zip([A, A_hat], [mu, mu_hat], ["truth", "hat"]) if M is not None]
    for ix, (M, v, t) in enumerate(params):
        for j in range(M.shape[-1]):
            v = v if v is not None else np.zeros(dim)
            if dim == 3:
                lines.append(
                    art3d.Line3D(
                        (-scale * M[0, j] + v[0], scale * M[0, j] + v[0]),
                        (-scale * M[1, j] + v[1], scale * M[1, j] + v[1]),
                        (-scale * M[2, j] + v[2], scale * M[2, j] + v[2]),
                        color="tab:green" if t == "truth" else "tab:orange",
                        lw=3,
                    )
                )
            elif dim == 2:
                lines.append(
                    Line2D(
                        (-scale * M[0, j] + v[0], scale * M[0, j] + v[0]),
                        (-scale * M[1, j] + v[1], scale * M[1, j] + v[1]),
                        lw=2,
                    )
                )

    return lines


def _plot_points(
    X: np.ndarray,
    ax: matplotlib.axes.Axes,
    dim: int,
    f: Optional[np.ndarray],
    color: str,
    alpha: float,
    legend_elements: List[Any],
    legend_names: List[str],
    legend_colors: List[str],
    element_name: str,
) -> None:
    """
    Utility function for plotting a set of points onto an axes.
    It also manages the UI elements which are kept track of.
    """

    if dim == 2:
        if f is not None:
            points = ax.scatter(X[:, 0], X[:, 1], s=10, c=f, alpha=alpha)
        else:
            points = ax.scatter(X[:, 0], X[:, 1], s=10, color=color, alpha=alpha)
    else:
        if f is not None:
            points = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10, c=f, alpha=alpha)
        else:
            points = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10, color=color, alpha=alpha)
    legend_elements.append(points)
    legend_names.append(element_name)
    legend_colors.append(color)


def synthetic_data(
    X: np.ndarray,
    A: Optional[np.ndarray] = None,
    A_hat: Optional[np.ndarray] = None,
    f: Optional[np.ndarray] = None,
    alphas: Dict[str, float] = {"observable": 0.1, "true_projections": 0.2, "estimated_projections": 0.3},
    file_name: Optional[Union[str, pathlib.Path]] = None,
) -> None:
    """
    Visualises 2D or 3D synthetic data with factors of variation from a nonlinear manifold.
    Mainly used for data generated with the `nonlinear_manifold` function from the `synthetic` module.
    Optionally, it also visualises the projections of the points onto 2D or 1D linear subspaces; either the ground-truth
    projections or the projections on some estimated subspace spanned by the matrices `A` and `A_hat`, respectively.
    Args:
        X: A `N x n` array holding the observable data
        A: A `n x d` matrix projecting X onto the true linear subspace.
        A_hat: An estimate of the `A` matrix.
        f: An `N` dimensional vector holding the values of a factor of variation to be color-visualised.
        alphas: The alpha values for plotting the different points in the figure (the observable ones, the true
                projections, and the estimated projections).
                It should contain the keys "observable", "true_projections", "estimated_projections", where the latter
                two are only needed if also plotting the corresponding points.
                Useful to configure depending on the number of points plotting.
        file_name: Optional file name to save the produced figure.

    """
    dim = X.shape[1]
    assert dim in [2, 3], "The dimensionality of the data should be 2 or 3 for visualisation."

    fig = plt.figure()
    if dim == 2:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection="3d")
    # Construct the lines showing the basis defined by the matrices
    for line in _get_lines(X=X, A=A, A_hat=A_hat, dim=dim):
        ax.add_line(line)

    # Useful for the legend on the image
    legend_elements, legend_names, legend_colors = [], [], []

    _plot_points(
        X,
        ax,
        dim,
        f,
        "tab:blue",
        alphas["observable"],
        legend_elements,
        legend_names,
        legend_colors,
        "Observable points",
    )

    if A is not None:
        true_plane = X @ A @ A.T
        _plot_points(
            true_plane,
            ax,
            dim,
            f,
            "tab:green",
            alphas["true_projections"],
            legend_elements,
            legend_names,
            legend_colors,
            "True linear subspace",
        )

    if A_hat is not None:
        predicted_plane = X @ A_hat @ A_hat.T
        _plot_points(
            predicted_plane,
            ax,
            dim,
            f,
            "tab:orange",
            alphas["estimated_projections"],
            legend_elements,
            legend_names,
            legend_colors,
            "Predicted linear subspace",
        )

    if dim == 2:
        ax.set(
            title=f"{dim}D Gaussian with nonlinear factors of variation",
            xlabel="x",
            ylabel="y",
        )
    else:
        ax.set(
            title=f"{dim}D Gaussian with nonlinear factors of variation",
            xlabel="x",
            ylabel="y",
            zlabel="z" if dim == 3 else None,
        )

    if f is None:
        ax.legend(legend_elements, legend_names, labelcolor=legend_colors)
    else:
        ax.legend(*legend_elements[0].legend_elements())

    if file_name is not None:
        fig.savefig(file_name)
        plt.close()
    else:
        plt.show()


def probabilistic_pca_data(
    X: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    A_hat: Optional[np.ndarray] = None,
    mu_hat: Optional[np.ndarray] = None,
    sigma_hat: Optional[float] = None,
    alphas: Dict[str, float] = {"observable": 0.1, "true_projections": 0.2, "estimated_projections": 0.3},
    title: Optional[str] = None,
    file_name: Optional[Union[str, pathlib.Path]] = None,
) -> None:
    """
    Plots 3D observable data coming from the probabilistic PCA generative model, and optionally the subspace spanned by
    the (estimate of) the loading matrix together with its projections.
    Mainly used for data generated with the `probabilistic_pca` function from the `synthetic` module.
    Args:
        X: A `N x n` matrix of the noisy observations
        Z: A `N x d` matrix of the generative factors
        A: The ground-truth `d x d` loading matrix
        mu: The ground-truth mean of the observable distribution.
        A_hat: Estimate of `A`
        mu_hat: Estimate of `mu`
        sigma_hat: Estimate of `sigma`
        alphas: The alpha values for plotting the different points in the figure (the observable ones, the true
                projections, and the estimated projections).
                It should contain the keys "observable", "true_projections", "estimated_projections", where the latter
                two are only needed if also plotting the corresponding points.
                Useful to configure depending on the number of points plotting.
        title: Optional plot title
        file_name: Optional file name to save the produced figure.

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    legend_elements, legend_names, legend_colors = [], [], []

    # Plot the actual noisy observations `X`
    noise_observations = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10, color="tab:blue", alpha=alphas["observable"])
    legend_elements.append(noise_observations)
    legend_names.append("Noisy observations")
    legend_colors.append("tab:blue")

    true_plane = probabilistic_pca.get_observations(Z, A, mu)

    # Construct the lines showing the basis defined by the matrices
    for line in _get_lines(X=true_plane, A=A, mu=mu, A_hat=A_hat, mu_hat=mu_hat):
        ax.add_line(line)

    # Plot the projections onto the true subspace
    true_point_projections = ax.scatter(
        true_plane[:, 0], true_plane[:, 1], true_plane[:, 2], s=10, color="tab:green", alpha=alphas["true_projections"]
    )
    legend_elements.append(true_point_projections)
    legend_names.append("Projections on the true linear subspace")
    legend_colors.append("tab:green")

    if A_hat is not None:
        # Plot the projections onto the estimated subspace
        Z_hat = probabilistic_pca.get_latent_representations(X, A_hat, mu_hat, sigma_hat)
        predicted_plane = probabilistic_pca.get_observations(Z_hat, A_hat, mu_hat)

        predicted_point_projections = ax.scatter(
            predicted_plane[:, 0],
            predicted_plane[:, 1],
            predicted_plane[:, 2],
            s=10,
            color="tab:orange",
            alpha=alphas["estimated_projections"],
        )
        legend_elements.append(predicted_point_projections)
        legend_names.append("Projections on the predicted linear subspace")
        legend_colors.append("tab:orange")

    ax.set(title="Probabilistic PCA" if title is None else title, xlabel="x", ylabel="y", zlabel="z")
    ax.legend(legend_elements, legend_names, labelcolor=legend_colors)

    if file_name is not None:
        fig.savefig(file_name)
        plt.close()
    else:
        plt.show()


def metrics_boxplot(
    data: pd.DataFrame,
    x: str = "Metric",
    quantity_name: Optional[str] = None,
    label_rotation: int = 30,
    file_name: Optional[Union[str, pathlib.Path]] = None,
) -> None:
    """
    Plots the boxplot of the metrics logged in the `data` dataframe.
    The dataframe should contain a column "Metric" with the name of the metric and a column "Value" which contains the
    values of the metrics recorded.

    Args:
        data (pd.DataFrame): The data frame containing the logged metrics.
        x: The x-axis value of the boxplot.
        quantity_name (str): The name of the quantity of interest (if one was varied and the values of metrics were
                             measured at these different points).
        label_rotation: The rotation of the metric name labels when plotting.
        file_name (Optional[Union[str, pathlib.Path]], optional): Optional name of the file to save the plot to.
                                                                  Defaults to None.
    """
    n_metrics = len(set(data["Metric"]))
    fig, ax = plt.subplots(figsize=(max(1.0, 1.0 * n_metrics), 2), dpi=200)

    b = sns.boxplot(data=data, x=x, y="Value", ax=ax)
    ax.set(
        title="Metrics" if n_metrics > 1 else list(data["Metric"])[0],
        xlabel="Metric" if quantity_name is None else quantity_name,
        ylabel="Value",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=label_rotation)
    b.tick_params(labelsize=6)

    if file_name is not None:
        fig.savefig(file_name, bbox_inches="tight")
        plt.close()
    else:
        plt.show(bbox_inches="tight")


def metric_trend(
    data: pd.DataFrame,
    quantity_name: str,
    file_name: Optional[Union[str, pathlib.Path]] = None,
) -> None:
    """
    Plots the trends in metrics stored in the data frame `data` over different values of a quantity of interest.
    It assumes that the data frame contains a column names "Metric" with the name of the metrics of interest,
    a column names "Value" with the measurement of that metric, and a column named "x" which contains the value of
    the quantity at which the value of the metric was measured.

     Args:
         data (pd.DataFrame): The data frame containing the data as described above.
         quantity_name (str): The name of the quantity of interest (which was varied and the values of metrics were
                              measured at these different points).
         file_name (Optional[Union[str, pathlib.Path]], optional): Optional name of the file to save the plot to.
                                                                   Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(4, 2), dpi=200)

    sns.lineplot(data=data, x="x", y="Value", hue="Metric", legend="full", ci="sd", ax=ax)
    ax.set(xlabel=quantity_name)
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    if file_name is not None:
        fig.savefig(file_name, bbox_extra_artists=(legend,), bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def factor_trend(
    Z: torch.Tensor,
    Y: torch.Tensor,
    target: str,
    k: int = 5,
    N: int = 4000,
    q: float = 0.05,
    file_name: Optional[str] = None,
    rng: RandomGenerator = 42,
) -> None:
    """
    Plots the relationship between values of the 1-dimensional variable `Z` (e.g., a learned latent dimension)
    and the variable `Y`.
    It samples `N` points uniformly from the support of `Y` (or a subinterval), and interpolates the values of `Z`
    using `k`-nearest neighbours.

    The function can be used to plot the relationship between a ground-truth factor of variation and a learned
    dimension or the other way around.
    Args:
        Z: A 1-dimensional array containing the values of the learned dimension for a sample of data points.
        Y: A 1-dimensional array containing the ground-truth factors of variation for the same data points.
        target: The meaning of the target.
                If "ground-truth", the y-axis corresponds to the values of the grond-truth factor, if "learned", it
                corresponds to the values of a learned dimension.
        k: Number of nearest neighbours to use for interpolation of the values.
        N: Number of data points to plot.
        q: The quantile at which to cut off the values of `Y`.
        file_name: An optional file name.
                   If specified, the figure will be saved there, otherwise, it will be displayed.
        rng: A rando generator for the uniform sampling.

    """
    assert target in ["ground-truth", "learned"]

    rng = np.random.default_rng(rng)

    if target == "learned":
        # Plot Z against Y
        X = Y
        T = Z
    else:
        # Plot Y against Z
        X = Z
        T = Y

    regressor = KNeighborsRegressor(n_neighbors=k).fit(X, T)

    # Generate a uniform sample from the range of `X`
    x = np.sort(rng.uniform(low=-1, high=1, size=(N,)) * np.asarray([torch.quantile(X, 1 - q) - torch.quantile(X, q)]))
    # Compute the values of the variable in `T` for the sampled points based on kNN
    t = regressor.predict(x.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(3.25, 3.25), dpi=200)
    ax.plot(x, t, label="Latent dimension")

    if target == "learned":
        xlabel, ylabel = "Learned Dimension", "Factor"
    else:
        xlabel, ylabel = "Factor", "Learned Dimension"
    ax.set(xlabel=xlabel, ylabel=ylabel)

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
    k: int = 5,
    N: int = 4000,
    q: float = 0.05,
    file_name: Optional[str] = None,
    rng: RandomGenerator = 42,
) -> None:
    """
    Plots a "jet" heatmap of the values of a 1-dimensional latent variable against a two-dimensional random vector.
    It samples `N` points uniformly from the square containing the two-dimensional vector, and interpolates the values
    of the other variable using `k`-nearest neighbours.

    The function can be used to plot the relationship between a ground-truth factor of variation and a two learned
    dimensions or the other way around.
    Args:
        Z: A 1-dimensional array containing the values of the learned dimension for a sample of data points.
        Y: A 2-dimensional array containing the ground-truth factors of variation for the same data points.
        target: The meaning of the target.
                If "ground-truth", the hue corresponds to the values of the grond-truth factor, if "learned", it
                corresponds to the values of a learned dimension.
        k: Number of nearest neighbours to use for interpolation of the values.
        N: Number of data points to plot.
        q: The quantile at which to cut off the values of `Y`.
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
    else:
        # Plot Y against Z
        assert Z.shape[1] == 2
        X = Z
        T = Y

    rng = np.random.default_rng(rng)

    regressor = KNeighborsRegressor(n_neighbors=k).fit(X, T)

    # Generate a uniform sample from the range of `X`
    x = rng.uniform(low=-1, high=1, size=(N, 2)) * np.asarray(
        [
            torch.quantile(X[:, 0], 1 - q) - torch.quantile(X[:, 0], q),
            torch.quantile(X[:, 1], 1 - q) - torch.quantile(X[:, 1], q),
        ]
    )

    # Compute the values of the variable in `T` for the sampled points based on kNN
    t = regressor.predict(x)

    fig, ax = plt.subplots(figsize=(3.25, 3.25), dpi=200)
    ax.scatter(x[:, 0], x[:, 1], c=t, cmap="jet", s=10, label=t)

    axis_name = "Learned Dimension" if target == "learned" else "Factor"
    ax.set(xlabel=f"{axis_name} 1", ylabel=f"{axis_name} 2")

    fig.tight_layout()

    if file_name is not None:
        fig.savefig(file_name, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def generated_images(
    images: torch.Tensor, nrows: int, ncols: int, file_name: Optional[str] = None, cmap: Optional[str] = None
):
    """
    Plots a set of (generated) images from a VAE in a grid.
    Args:
        images: The set of nrows * ncols images to plot.
        nrows: Number of rows to plot.
        ncols: Number of columsn to plot.
        file_name: Optional file name to save the image in.
        cmap: The colormap to use for plotting.
    """

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(0.75 * ncols, 0.75 * nrows), dpi=200)

    for i in range(nrows):
        for j in range(ncols):
            axes[i][j].imshow(images[i * ncols + j].cpu().permute(1, 2, 0), cmap=cmap)
            axes[i][j].get_xaxis().set_ticks([])
            axes[i][j].get_yaxis().set_ticks([])

    plt.suptitle("Generated images", size=10)
    fig.tight_layout()

    if file_name is not None:
        fig.savefig(file_name, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
