import pathlib
import dataclasses
from typing import Tuple, List

import hydra
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import latte.hydra_utils as hy
from latte.direction.api import linear_regression as lr, unit_vector
from latte.dataset import dsprites
from latte.dataset.dsprites import DSpritesFactor


DEFAULT_TARGET = "figures/dsprites"


@hy.config
@dataclasses.dataclass
class DSpritesExplorationConfig:
    factors: List[int]
    ignored_factors: List[int]
    n_components: int = 4
    method: str = "PCA"


factor_names = {
    DSpritesFactor.SHAPE: "Shape",
    DSpritesFactor.SCALE: "Scale",
    DSpritesFactor.ORIENTATION: "Orientation",
    DSpritesFactor.X: "Position x",
    DSpritesFactor.Y: "Position y",
}


fixed_values = {
    DSpritesFactor.SHAPE: 1,
    DSpritesFactor.SCALE: 1,
    DSpritesFactor.ORIENTATION: 0,
    DSpritesFactor.X: 0.5,
    DSpritesFactor.Y: 0.5,
}


def prepare_data(data: dsprites.DSpritesDataset, ignored_factors: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Converts the raw dSprites data into Numpy arrays which will be
    processed by PCA."""
    X = data.imgs.reshape((len(data.imgs), -1))

    # select the images of interest by ignoring the variation of factors
    # we are not interested in
    sample_mask = np.ones(data.latents_values.shape[0], dtype=bool)
    for factor in ignored_factors:
        sample_mask *= data.latents_values[:, factor] == fixed_values[DSpritesFactor(factor)]

    X = X[sample_mask, :]
    y = data.latents_values[sample_mask, :]

    # filter out constant pixels
    X = X[:, X.max(axis=0) == 1]

    return X, y


def pca(X: np.ndarray, n_components: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    t = PCA(n_components=n_components, copy=False)
    X = StandardScaler().fit_transform(X)
    X_pca = t.fit_transform(X)
    return X_pca, t.explained_variance_ratio_


def fa(X: np.ndarray, n_components: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    t = FactorAnalysis(n_components=n_components, copy=False)
    X = StandardScaler().fit_transform(X)
    X_pca = t.fit_transform(X)
    return X_pca


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


def get_optimal_directions(X: np.ndarray, y: np.ndarray, factors: List[int]) -> List[lr.OptimizationResult]:
    results = []
    for j in factors:
        result = lr.find_best_direction(X, y[:, j])
        result.direction = unit_vector(result.direction)
        results.append(result)
    return results


def plot_2d(
    X: np.ndarray,
    y: np.ndarray,
    factors: List[int],
    optimal_directions: List[lr.OptimizationResult],
    filepath: pathlib.Path,
) -> None:
    """Plot the 2D projections with the two most informative axes
    for each factor individually."""
    fig, axes = plt.subplots(1, len(factors), figsize=((len(factors) + 1) * 4, 4))
    for ii, (factor, direction) in enumerate(zip(factors, optimal_directions)):

        # get the two most informative components for that specific factor
        pc_x, pc_y = np.argsort(-np.abs(direction.direction))[:2]

        sns.scatterplot(x=X[:, pc_x], y=X[:, pc_y], hue=y[:, factor], ax=axes[ii])

        # plot the direction as well
        # scale the direction so that it can be seen in the figure well
        scale = (np.max(X[:, (pc_x, pc_y)], axis=0) - np.min(X[:, (pc_x, pc_y)], axis=0)) / 2
        v = np.asarray([direction.direction[pc_x], direction.direction[pc_y]])
        v = v / np.linalg.norm(v) * scale
        axes[ii].plot((0, v[0]), (0, v[1]), linewidth=2, color="green")

        axes[ii].set_title(f"{factor_names[DSpritesFactor(factor)]} Gradient")
        axes[ii].set(xlabel=f"PC{pc_x + 1}", ylabel=f"PC{pc_y + 1}")
        axes[ii].legend(title=factor_names[DSpritesFactor(factor)], loc="upper right")

    fig.savefig(filepath / "2d_plots.png")
    plt.close()


@hydra.main(config_path="../config", config_name="dsprites_factors")
def main(cfg: DSpritesExplorationConfig):

    n_components = cfg.n_components
    factors = cfg.factors
    ignored_factors = cfg.ignored_factors
    method = cfg.method

    assert method in ["PCA", "FA"], f"Dimensionality reduction method {method} not supported."
    assert len(set(factors).intersection(set(ignored_factors))) == 0, "Some factors are both ignored and of interest."

    filepath = pathlib.Path(DEFAULT_TARGET) / method
    filepath.mkdir(exist_ok=True, parents=True)

    data = dsprites.load_dsprites()
    X, y = prepare_data(data, ignored_factors=ignored_factors)

    if method == "PCA":
        X, ratios = pca(X, n_components)
        plot_variance_explained(ratios, filepath)
    else:
        X = fa(X, n_components)

    best_directions = get_optimal_directions(X, y, factors)

    plot_2d(X, y, factors, best_directions, filepath)

    # TODO (Anej): Add MINE when it is finished


if __name__ == "__main__":
    main()
