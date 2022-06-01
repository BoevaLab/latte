import pathlib
import dataclasses
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import latte.hydra_utils as hy
from latte.direction.api import linear_regression as lr, unit_vector
from latte.dataset import dsprites


DEFAULT_TARGET = "figures/dsprites"


@hy.config
@dataclasses.dataclass
class DSpritesExplorationConfig:
    factors: List[dsprites.DSpritesFactor] = dataclasses.field(
        default_factory=lambda: [
            dsprites.DSpritesFactor.SCALE,
            dsprites.DSpritesFactor.X,
            dsprites.DSpritesFactor.Y,
        ]
    )
    ignored_factors: List[dsprites.DSpritesFactor] = dataclasses.field(
        default_factory=lambda: [dsprites.DSpritesFactor.SHAPE, dsprites.DSpritesFactor.ORIENTATION]
    )
    n_components: int = 4
    method: str = "PCA"


def plot_variance_explained(ratios: np.ndarray, filepath: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(np.arange(1, len(ratios) + 1), ratios)
    axes[0].set_title("Proportion of variance explained")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Proportion of variance")

    axes[1].plot(np.arange(1, len(ratios) + 1), np.cumsum(ratios))
    axes[1].set_title("Proportion of variance explained")
    axes[1].set_xlabel("Number of principal components")
    axes[1].set_ylabel("Proportion of variance")

    fig.savefig(filepath / "variance_explained.png")
    plt.close()


def get_optimal_directions(
    X: np.ndarray, y: np.ndarray, factors: List[dsprites.DSpritesFactor], factor2idx: Dict[dsprites.DSpritesFactor, int]
) -> List[lr.OptimizationResult]:
    results = []
    for factor in factors:
        result = lr.find_best_direction(X, y[:, factor2idx[factor]])
        result.direction = unit_vector(result.direction)
        results.append(result)
    return results


def plot_2d(
    X: np.ndarray,
    y: np.ndarray,
    factors: List[dsprites.DSpritesFactor],
    factor2idx: Dict[dsprites.DSpritesFactor, int],
    optimal_directions: List[lr.OptimizationResult],
    filepath: pathlib.Path,
) -> None:
    """Plot the 2D projections with the two most informative axes
    for each factor individually."""
    fig, axes = plt.subplots(1, len(factors), figsize=((len(factors) + 1) * 4, 4))
    for ii, (factor, direction) in enumerate(zip(factors, optimal_directions)):

        # get the two most informative components for that specific factor
        pc_x, pc_y = np.argsort(-np.abs(direction.direction))[:2]

        sns.scatterplot(x=X[:, pc_x], y=X[:, pc_y], hue=y[:, factor2idx[factor]], ax=axes[ii])

        # plot the direction as well
        # scale the direction so that it can be seen in the figure well
        scale = (np.max(X[:, (pc_x, pc_y)], axis=0) - np.min(X[:, (pc_x, pc_y)], axis=0)) / 2
        v = np.asarray([direction.direction[pc_x], direction.direction[pc_y]])
        v = v / np.linalg.norm(v) * scale
        axes[ii].plot((0, v[0]), (0, v[1]), linewidth=2, color="green")

        axes[ii].set_title(f"{dsprites.factor_names[factor]} Gradient")
        axes[ii].set(xlabel=f"PC{pc_x + 1}", ylabel=f"PC{pc_y + 1}")
        axes[ii].legend(title=dsprites.factor_names[factor], loc="upper right")

    fig.savefig(filepath / "2d_plots.png")
    plt.close()


@hy.main
def main(cfg: DSpritesExplorationConfig):

    n_components = cfg.n_components
    factors = cfg.factors
    ignored_factors = cfg.ignored_factors
    method = cfg.method

    assert method in ["PCA", "FA"], f"Dimensionality reduction method {method} not supported."
    assert len(set(factors).intersection(set(ignored_factors))) == 0, "Some factors are both ignored and of interest."

    filepath = pathlib.Path(DEFAULT_TARGET) / method
    filepath.mkdir(exist_ok=True, parents=True)

    data = dsprites.load_dsprites_preprocessed(
        factors=factors, ignored_factors=ignored_factors, n_components=n_components, method=method
    )

    best_directions = get_optimal_directions(data.X, data.y, factors, data.factor2idx)

    plot_2d(data.X, data.y, factors, data.factor2idx, best_directions, filepath)

    if method == "PCA":
        plot_variance_explained(data.pca_ratios, filepath)

    # TODO (Anej): Add MINE when it is finished


if __name__ == "__main__":
    main()
