"""
Various visualisation helper functions for plotting image data, such as homotopies or latent traversals.
"""

from itertools import product
from typing import Callable, Optional, List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from pythae.models.vae.vae_model import VAE
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset

from latte.dataset.utils import RandomGenerator
from latte.models.vae import utils as vae_utils

left, width = -0.25, 0.5
bottom, height = 0.25, 0.5
right = left + width
top = bottom + height


def vae_reconstructions(
    dataset: Union[Dataset, torch.Tensor],
    vae_model: VAE,
    latent_transformation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
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


def latent_traversals(
    vae_model: nn.Module,
    starting_x: torch.Tensor,
    A_hat: Optional[torch.Tensor] = None,
    x_hat_transformation: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    n_axes: int = 1,
    Z: torch.Tensor = None,
    n_values: int = 12,
    axis_values: Optional[np.ndarray] = None,
    q: Tuple[float, float] = (0.01, 0.99),
    shift: bool = False,
    cmap: Optional[str] = "Greys",
    device: str = "cuda",
    file_name: Optional[str] = None,
) -> None:
    """
    Plot the latent traversals of the `n_axes` most-variable axes produced by the VAE `vae_model`.
    It can also optionally (1) transform the latent representation before decoding it or (2) transform the produced
    reconstructions.

    Args:
        vae_model: The VAE to use for embedding and reconstructing.
        starting_x: The starting image of the latent traversals.
                    Individual axes of the embedding of this image will be intervened on.
        Z: The latent representations of the entire dataset.
           This is used to calculate the variance statistics and quantiles of all the axes.
           Can be either the full VAE latent space or the projections onto the linear subspace as found by ManifoldMINE.
        A_hat: A `n x d` matrix projection from the `n`-dimensional latent space to a `d` dimensional linear subspace.
               It is used to transform the latent representations before decoding them (and thus enables projecting onto
               a specified linear subspace e.g. capturing some attribute).
        x_hat_transformation: Transformation performed on the reconstructed image before plotting.
                              Can be used to make permute the channels.
        n_values: Number of values to try in each dimension. `values` overrides this.
        axis_values: Optional. Manually chosen values for interventions.
                If it is a 1D array, the same values will be used for all axes.
                Otherwise, the first dimension of the array has to match the number of plotted axes;
                the values of the i-th row of `values` will be used to intervene on the i-th axis
        q: The quantiles of the individual axes distributions between which to traverse.
        shift: If true, shift the axes by the quantile values (add them to the original value) instead of
               setting them to the raw quantile value.
        n_axes: Number of axes to traverse.
        cmap: The color map for the matplotlib plotting function.
        device: The pytorch device to use for tensors.
        file_name: Optional file name to save the produced figure.
    """

    variances = np.asarray([torch.var(Z[:, j]) for j in range(Z.shape[-1])])
    axis_order = np.argsort(-variances)

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
                f"Axis {axis}\n {var_expr} = {variances[axis]: .2f}",
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
                if A_hat is not None:
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


def latent_traversals_2d(
    vae_model: nn.Module,
    starting_x: torch.Tensor,
    axis: Tuple[int, int],
    A_hat: Optional[torch.Tensor] = None,
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
        axis: The indices of the two axes to traverse.
        A_hat: A `n x d` matrix projection from the `n`-dimensional latent space to a `d` dimensional linear subspace.
               It is used to transform the latent representations before decoding them (and thus enables projecting onto
               a specified linear subspace e.g. capturing some attribute).
        x_hat_transformation: Transformation performed on the reconstructed image before plotting.
                              Can be used to make permute the channels.
        Z: The latent representations of the entire dataset.
           This is used to calculate the variance statistics and quantiles of all the axes.
           Can be either the full VAE latent space or the projections onto the linear subspace as found by ManifoldMINE.
        n_values: Number of values to try in each dimension. `values` overrides this.
        shift: If true, shift the axes by the quantile values (add them to the original value) instead of
               setting them to the raw quantile value.
        axis_values: Optional. Manually chosen values for interventions.
                If it is a 1D array, the same values will be used for all axes.
                Otherwise, the first dimension of the array has to match the number of plotted axes;
                the values of the i-th row of `values` will be used to intervene on the i-th axis
        q: The quantiles of the individual axes distributions between which to traverse.
        cmap: The color map for the matplotlib plotting function.
        device: The pytorch device to use for tensors.
        file_name: Optional file name to save the produced figure.
    """

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
                if A_hat is not None:
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


def homotopies(
    xs: List[torch.Tensor],
    vae_model: VAE,
    A: Optional[torch.Tensor] = None,
    n_cols: int = 10,
    x_hat_transformation: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    cmap: Optional[str] = "Greys",
    file_name: Optional[str] = None,
    device: str = "cuda",
) -> None:
    """
    Produces plots of homotopies (interpolations between pairs of images) as defined in Section 6.3 in
    https://arxiv.org/abs/1511.06349.

    Args:
        xs: The starting images.
            The function will interpolate between all the pairs of images in the list.
        vae_model: The VAE to use for embedding and reconstructing.
        A: A `n x d` matrix projection from the `n`-dimensional latent space to a `d` dimensional linear subspace.
           If provided, the function will only interpolate between the pairs *on these subspaces* and leave all other
           aspects of the latent representations the same as in the original image.
           Practically, this means that *only aspects captured in the subspace defined by `A`* will be applied from
           the second image to the first one.
        x_hat_transformation: Transformation performed on the reconstructed image before plotting.
                              Can be used to make permute the channels.
        n_cols: Number of values to produce.
        cmap: The color map for the matplotlib plotting function.
        device: The pytorch device to use for tensors.
        file_name: Optional file name to save the produced figure.
    """

    if A is None:
        A = torch.eye(vae_model.model_config.latent_dim).to(device)

    N = len(xs)
    fig, axes = plt.subplots(
        nrows=N * (N - 1), ncols=n_cols + 2, figsize=(0.75 * (n_cols + 2), 0.65 * N * (N - 1)), dpi=200
    )

    # To represent `x`, we just take the mean of the Gaussian distribution
    with torch.no_grad():
        starting_zs = [vae_model.encoder(x.unsqueeze(0)).embedding for x in xs]  # We extract `mu`

    ts = np.linspace(0, 1, n_cols)
    for jj in range(n_cols + 2):
        axes[0, jj].text(
            0.5,
            1.1,
            "Original" if jj in [0, n_cols + 1] else f"{ts[jj - 1]: .2f}",
            horizontalalignment="center",
            verticalalignment="bottom",
            rotation="horizontal",
            size=8,
            transform=axes[0, jj].transAxes,
        )

    row = 0
    for ii, jj in product(range(N), range(N)):
        if ii == jj:
            continue

        axes[row, 0].imshow(xs[ii].detach().cpu().permute(1, 2, 0), cmap=cmap)
        axes[row, 0].get_xaxis().set_ticks([])
        axes[row, 0].get_yaxis().set_ticks([])

        axes[row, n_cols + 1].imshow(xs[jj].detach().cpu().permute(1, 2, 0), cmap=cmap)
        axes[row, n_cols + 1].get_xaxis().set_ticks([])
        axes[row, n_cols + 1].get_yaxis().set_ticks([])

        with torch.no_grad():
            for t_ix, t in enumerate(ts):
                z = starting_zs[ii]
                d = ((1 - t) * z + t * starting_zs[jj]) @ A
                z = z - z @ A @ A.T  # Erase the subspace from z
                z = z + d @ A.T  # Add back just the convex combination
                x_hat = x_hat_transformation(
                    vae_model.decoder(z).reconstruction.detach().cpu()[0]
                )  # First image from the batch
                axes[row, t_ix + 1].imshow(x_hat.permute(1, 2, 0), cmap=cmap)
                axes[row, t_ix + 1].get_xaxis().set_ticks([])
                axes[row, t_ix + 1].get_yaxis().set_ticks([])
        row += 1

    plt.suptitle("Homotopies", size=10)

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


def graphically_evaluate_model(
    model: VAE,
    X: torch.Tensor,
    A: Optional[torch.Tensor] = None,
    starting_x: Optional[torch.Tensor] = None,
    starting_xs: Optional[List[torch.Tensor]] = None,
    cmap: Optional[str] = None,
    latent_transformation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    repeats: int = 1,
    homotopy_n: int = 3,
    n_rows: int = 4,
    n_cols: int = 6,
    rng: RandomGenerator = 1,
    device: str = "cuda",
    file_prefix: Optional[str] = None,
) -> None:
    """
    A convenience function to produce plots qualitatively evaluating a VAE model for images.
    It produces the VAE reconstructions of a subset of the passed dataset, the latent traversals of a subset of chosen
    starting images, homotopies of a subset of the dataset, and the 2D traversals in case the (projected) subspace is
    two-dimensional.
    Optionally, the VAE model representations can be projected with the passed `A` projection matrix and the
    produced plots can be repeated `repeats` times.

    Args:
        model: The VAE model to evaluate.
        X: A dataset of original images.
        A: An optional `d x K` projection matrix to project the VAE space to a linear subspace capturing the information
           to be kept in the reconstructions.
           `k` is the dimensionality of the subspace.
        starting_x: An optional starting image for latent traversals.
        starting_xs: An optional set of starting images for homotopies.
        cmap: The colormap to use for plotting.
        latent_transformation: An optional transformation of the latent traversals before
                               being reconstructed by `model`.
        repeats: Number of figures of each type to generate.
        homotopy_n: If starting_xs is not provided, the number of images to generate for homotopies.
        n_rows: Number of rows of samples to plot for VAE reconstructions and the latent traversals.
        n_cols: Number of columns of samples to plot for VAE reconstructions, the latent traversals, and homotopies.
        rng: The random number generator.
        device: The pytorch device to use.
        file_prefix: The prefix of the filenames used to save all the figures.
                     Different suffixes will be added to the individual files of the reconstructions, traversals,
                     and homotopies.

    """

    rng = np.random.default_rng(rng)

    # Get the (projected) latent representations
    Z = vae_utils.get_latent_representations(model, X)
    Z = torch.from_numpy(StandardScaler().fit_transform(Z.numpy())).float()
    if A is not None:
        Z = Z @ A.detach().cpu()
        Z = torch.from_numpy(StandardScaler().fit_transform(Z.numpy())).float()

    for ii in range(repeats):
        vae_reconstructions(
            dataset=X,
            vae_model=model,
            nrows=n_rows,
            ncols=n_cols,
            latent_transformation=latent_transformation,
            cmap=cmap,
            rng=rng,
            file_name=file_prefix + f"_reconstructions_{ii}.png" if file_prefix is not None else None,
        )

        latent_traversals(
            vae_model=model,
            starting_x=starting_x if starting_x is not None else X[rng.choice(len(X))].to(device),
            Z=Z,
            A_hat=A,
            n_values=n_cols,
            n_axes=n_rows,
            cmap=cmap,
            file_name=file_prefix + f"_traversals_{ii}.png" if file_prefix is not None else None,
        )

        homotopies(
            vae_model=model,
            A=A,
            xs=starting_xs
            if starting_xs is not None
            else [X[rng.choice(len(X))].to(device) for _ in range(homotopy_n)],
            n_cols=n_cols,
            cmap=cmap,
            file_name=file_prefix + f"_homotopies_{ii}.png" if file_prefix is not None else None,
        )

        if A is None and Z.shape[1] == 2 or A is not None and A.shape[1] == 2:
            latent_traversals_2d(
                vae_model=model,
                starting_x=X[rng.choice(len(X))].to(device),
                axis=(0, 1),
                A_hat=A,
                Z=Z,
                n_values=n_cols,
                device=device,
                cmap=cmap,
                file_name=file_prefix + f"_homotopies_{ii}.png",
            )
