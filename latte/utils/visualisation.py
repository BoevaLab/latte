from typing import Dict, Optional, List, Any
from itertools import product

import numpy as np
import torch
import pytorch_lightning as pl

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import art3d

from latte.models import datamodules as dm
from latte.dataset.utils import RandomGenerator
from latte.models import vae

left, width = -0.25, 0.5
bottom, height = 0.25, 0.5
right = left + width
top = bottom + height


def plot_originals_and_reconstructions(
    data: dm.BaseDataModule,
    vae_model: vae.VAE,
    nrows: int = 1,
    ncols: int = 12,
    file_name: Optional[str] = None,
    rng: RandomGenerator = 42,
) -> None:
    """
    Plots a sample of the original images and the corresponding reconstructions from the VAE.
    Args:
        data: The Lightning datamodule containing the data to be considered.
        vae_model: The trained VAE to use for embedding and reconstructing.
        nrows: Number of rows of images to plot.
        ncols: Number of columns of images to plot.
        file_name: Optional file name to save the produced figure.
        rng: The random generator to use to generate a sample.

    """
    test_dataset = data.test_dataloader().dataset

    rng = np.random.default_rng(rng)
    # Generate a random sample of images
    # nrows * ncols images will be drawn
    index = rng.choice(len(test_dataset), size=nrows * ncols, replace=len(test_dataset) <= nrows * ncols)
    # `replace` configured such that it handles cases when len(test_dataset) <= nrows * ncols

    x, _ = test_dataset[index]

    # Produce the reconstructions
    with torch.no_grad():
        x_hat, _, _ = vae_model(x)
        x_hat = x_hat.detach().cpu()

    fig, axes = plt.subplots(ncols=ncols, nrows=2 * nrows, figsize=(ncols, 2 * nrows))

    for i, j in product(range(nrows), range(ncols)):
        # Plot the original and the reconstruction in the same column
        axes[2 * i, j].imshow(x[ncols * i + j][0], cmap="Greys")
        axes[2 * i + 1, j].imshow(torch.sigmoid(x_hat[ncols * i + j][0]), cmap="Greys")
        axes[2 * i, j].get_xaxis().set_ticks([])
        axes[2 * i, j].get_yaxis().set_ticks([])
        axes[2 * i + 1, j].get_xaxis().set_ticks([])
        axes[2 * i + 1, j].get_yaxis().set_ticks([])

        if j > 0:
            continue

        # Add some information on the left-most column
        axes[2 * i, j].text(
            left,
            0.5 * (bottom + top),
            "original",
            horizontalalignment="right",
            verticalalignment="center",
            rotation="horizontal",
            transform=axes[2 * i, j].transAxes,
        )

        axes[2 * i + 1, j].text(
            left,
            0.5 * (bottom + top),
            "reconstruction",
            horizontalalignment="right",
            verticalalignment="center",
            rotation="horizontal",
            transform=axes[2 * i + 1, j].transAxes,
        )

    plt.suptitle("Reconstructions of the VAE")
    fig.tight_layout()

    if file_name is not None:
        fig.savefig(file_name)
        plt.close()
    else:
        plt.show()


def plot_latent_traversals(
    vae_model: pl.LightningModule,
    starting_x: torch.Tensor,
    Z: Dict[str, torch.Tensor],  # Can be either the full VAE latent space or the projections
    n_values: int = 12,
    A: Optional[torch.Tensor] = None,
    shift: bool = False,
    n_axes: Optional[int] = None,
    file_name: Optional[str] = None,
) -> None:
    """
    Plot the latent traversals of the `n_axes` most-variable axes produced by the VAE `vae_model`, optionally projected
    to a subspace using the A matrix.

    Args:
        vae_model: The VAE to use for embedding and reconstructing.
        starting_x: The starting image of the latent traversals.
                    Individual axes of the embedding of this image will be intervened on.
        Z: The latent representations of the entire dataset.
           This is used to calculate the variance statistics and quantiles of all the axes.
        n_values: Number of values to try in each dimension
        A: The projection matrix.
        shift: If true, shift the axes by the quantile values (add them to the original value) instead of
               setting them to the raw quantile value.
        n_axes: Number of axes to traverse.
        file_name: Optional file name to save the produced figure.
    """

    variances = np.asarray([torch.var(Z["train"][:, j]) for j in range(Z["train"].shape[-1])])
    axis_order = np.argsort(-variances)

    if n_axes is None:
        n_axes = vae_model.latent_size
    fig, axes = plt.subplots(nrows=n_axes, ncols=n_values + 1, figsize=(n_values + 1, n_axes + 1))

    # To represent `x`, we just take the mean of the Gaussian distribution
    with torch.no_grad():
        starting_z = vae_model.encoder(starting_x)[:, : vae_model.latent_size][0]  # We extract `mu`
        starting_x_hat = (
            vae_model.decoder(starting_z)[0, 0] if A is None else vae_model.decoder(starting_z @ A @ A.T)[0, 0]
        )

    if A is not None:
        # If projection matrix is provided, project to the MINE-learned subspace
        starting_z = starting_z @ A

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
            ax.text(
                left,
                0.5 * (bottom + top),
                f"axis {axis}, var = {variances[axis]: .2f}",
                horizontalalignment="right",
                verticalalignment="center",
                rotation="horizontal",
                size=9,
                transform=ax.transAxes,
            )

            ax.imshow(torch.sigmoid(starting_x_hat), cmap="Greys")
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            # The values to which the dimensions will be set correspond to quantiles
            # of the distribution over the entire dataset
            qs = np.linspace(0.1, 0.9, n_values)
            values = np.quantile(Z["train"][:, axis], q=qs)

            for j in range(n_values):

                z = starting_z.clone()

                # Intervene on a dimension
                z[axis] = z[axis] + values[j] if shift else values[j]

                if A is not None:
                    # Project back onto the full VAE latent space
                    z = z @ A.T

                # Decode
                with torch.no_grad():
                    x_hat = vae_model.decoder(z).detach().cpu()

                ax = axes[i, j + 1] if n_axes > 1 else axes[j + 1]
                ax.imshow(torch.sigmoid(x_hat[0, 0]), cmap="Greys")
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

                if i == 0:

                    ax.text(
                        0.5,
                        1.1,
                        f"q = {qs[j]: .2f}",
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        rotation="horizontal",
                        size=8,
                        transform=ax.transAxes,
                    )

    plt.suptitle("Latent traversals")
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

    # To plot lines of the appropriate length
    scale = max([np.max(X[:, j]) - np.min(X[:, j]) for j in range(X.shape[-1])]) / 2

    lines = []
    params = [(M, v) for M, v in zip([A, A_hat], [mu, mu_hat]) if M is not None]
    for ix, (M, v) in enumerate(params):
        for j in range(M.shape[-1]):
            v = v if v is not None else np.zeros(dim)
            if dim == 3:
                lines.append(
                    art3d.Line3D(
                        (-scale * M[0, j] + v[0], scale * M[0, j] + v[0]),
                        (-scale * M[1, j] + v[1], scale * M[1, j] + v[1]),
                        (-scale * M[2, j] + v[2], scale * M[2, j] + v[2]),
                        color="tab:green" if ix == 0 else "tab:orange",
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


def visualise_synthetic_data(
    X: np.ndarray,
    A: Optional[np.ndarray] = None,
    A_hat: Optional[np.ndarray] = None,
    f: Optional[np.ndarray] = None,
    alphas: Dict[str, float] = {"observable": 0.1, "true_projections": 0.2, "estimated_projections": 0.3},
    file_name: Optional[str] = None,
) -> None:
    """
    Visualises 2D or 3D synthetic data.
    Optionally, it also visualises the projections of the points onto 2D or 1D linear subspaces; either the ground-truth
    projections or the projections on some estimated subspace spanned by the matrices `A` and `A_hat`, respectively.
    Args:
        X: A `n x d` array holding the observable data
        A: A `d x k` matrix projecting X onto the true linear subspace.
        A_hat: An estimate of the `A` matrix.
        f: A `n` dimensional vector holding the values of a factor of variation to be color-visualised.
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
        ax = fig.add_subplot(11)
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

    ax.set(
        title=f"{dim}D Gaussian with nonlinear factors of variation",
        xlabel="x",
        ylabel="y",
        zlabel="z" if dim == 3 else None,
    )
    ax.legend(legend_elements, legend_names, labelcolor=legend_colors)

    if file_name is not None:
        fig.savefig(file_name)
        plt.close()
    else:
        plt.show()


def visualise_generative_model_in_3d(
    Z: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    A_hat: Optional[np.ndarray] = None,
    mu_hat: Optional[np.ndarray] = None,
    alphas: Dict[str, float] = {"observable": 0.1, "true_projections": 0.2, "estimated_projections": 0.3},
    file_name: Optional[str] = None,
) -> None:
    """
    Plots 3D observable data coming from a generative model in the form of factor analysis, and optionally the
    estimate of the loading matrix together with its projections.
    This is, in a way, the opposite direction of what `visualise_subspace_in_3d` plots.
    Args:
        Z: A `n x k` matrix of the generative factors
        A: The ground-truth `d x k` loading matrix
        mu: The ground-truth mean of the observable distribution.
        A_hat: Estimate of `A`
        mu_hat: Estimate of `mu`
        alphas: The alpha values for plotting the different points in the figure (the observable ones, the true
                projections, and the estimated projections).
                It should contain the keys "observable", "true_projections", "estimated_projections", where the latter
                two are only needed if also plotting the corresponding points.
                Useful to configure depending on the number of points plotting.
        file_name: Optional file name to save the produced figure.

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    X = Z @ A.T + mu

    # Construct the lines showing the basis defined by the matrices
    for line in _get_lines(X=X, A=A, mu=mu, A_hat=A_hat, mu_hat=mu_hat):
        ax.add_line(line)

    legend_elements, legend_names, legend_colors = [], [], []

    true_plane = X
    true_point_projections = ax.scatter(
        true_plane[:, 0], true_plane[:, 1], true_plane[:, 2], s=10, color="tab:green", alpha=alphas["true_projections"]
    )
    legend_elements.append(true_point_projections)
    legend_names.append("Projections on the true linear subspace")
    legend_colors.append("tab:green")

    if A_hat is not None:
        predicted_plane = Z @ A_hat.T + mu_hat
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

    ax.set(title="3D generative Gaussian", xlabel="x", ylabel="y", zlabel="z")
    ax.legend(legend_elements, legend_names, labelcolor=legend_colors)

    if file_name is not None:
        fig.savefig(file_name)
        plt.close()
    else:
        plt.show()
