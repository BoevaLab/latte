"""
Various visualisation helper functions for plotting low-dimensional (2D or 3D) synthetic dataset.
"""
import pathlib
from typing import Dict, Optional, Union, List, Any

import matplotlib
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import art3d
import matplotlib.pyplot as plt
import numpy as np

from latte.models.probabilistic_pca import probabilistic_pca


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


def ppca_data(
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
