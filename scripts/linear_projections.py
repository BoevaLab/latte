"""
Evaluates and demonstrates the random directions optimisation on 2D and 3D datasets with synthetic data.
"""

import pathlib
import dataclasses
from typing import Dict, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3D
import seaborn as sns

import latte.hydra_utils as hy
from latte.direction.api import (
    linear_regression as lin_reg,
    random as rnd,
    unit_vector,
    SpearmanCorrelationMetric,
    KendallTauMetric,
    MutualInformationMetric,
)

_DEFAULT_TARGET = "figures/linear"


def create_data(n: int, d: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.random.normal(size=(n, d))
    direction = unit_vector(np.random.normal(size=(d,)))
    scores = np.einsum("ij,j->i", points, direction)
    return points, scores, direction


def optimal_directions(
    points: np.ndarray, scores: np.ndarray, budget: int, seed: int
) -> Dict[str, Union[rnd.OptimizationResult, lin_reg.OptimizationResult]]:
    """Computes the optimal directions found by various implemented procedures."""
    results = {}
    results["spearman"] = rnd.find_best_direction(
        points, scores, SpearmanCorrelationMetric(), budget=budget, seed=seed + 1
    )
    results["kendall"] = rnd.find_best_direction(points, scores, KendallTauMetric(), budget=budget, seed=seed + 2)
    results["mi"] = rnd.find_best_direction(points, scores, MutualInformationMetric(), budget=budget, seed=seed + 3)
    results["lr"] = lin_reg.find_best_direction(points, scores)
    return results


def score_distributions(points: np.ndarray, scores: np.ndarray, m: int, seed: int) -> Dict[str, np.ndarray]:
    """Samples random directions in the representation space and calculates
    various metrics of agreement between those and the ground-truth direction."""
    distributions = {}
    distributions["spearman"] = rnd.find_metric_distribution(
        points=points,
        scores=scores,
        metric=SpearmanCorrelationMetric(),
        budget=m,
        seed=seed,
    ).metrics
    distributions["kendall"] = rnd.find_metric_distribution(
        points=points, scores=scores, metric=KendallTauMetric(), budget=m, seed=seed
    ).metrics
    distributions["mi"] = rnd.find_metric_distribution(
        points=points, scores=scores, metric=MutualInformationMetric(), budget=m, seed=seed
    ).metrics
    return distributions


def plot_2d(
    points: np.ndarray,
    direction: np.ndarray,
    results: Dict[str, Union[rnd.OptimizationResult, lin_reg.OptimizationResult]],
    filepath: pathlib.Path,
) -> None:
    fig, ax = plt.subplots()
    lines = [
        Line2D((0, 3 * result.direction[0]), (0, 3 * result.direction[1]), color=color, lw=2)
        for result, color in zip(results.values(), ["red", "orange", "blue", "yellow"])
    ]
    lines.append(Line2D((0, 3 * direction[0]), (0, 3 * direction[1]), color="green", lw=3))

    ax.scatter(points[:, 0], points[:, 1], s=40, alpha=0.4)

    for line in lines:
        ax.add_line(line)

    ax.set(title="2D Gaussian with linear projections", xlabel="x", ylabel="y")
    ax.legend(lines, ["Spearman", "Kendall", "Mutual Information", "Linear Regression", "True Direction"])

    fig.savefig(filepath / "lines_2d.png")
    plt.close()


def plot_3d(
    points: np.ndarray,
    direction: np.ndarray,
    results: Dict[str, Union[rnd.OptimizationResult, lin_reg.OptimizationResult]],
    filepath: pathlib.Path,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    lines = [
        Line3D(
            (0, 3 * result.direction[0]), (0, 3 * result.direction[1]), (0, 3 * result.direction[2]), color=color, lw=2
        )
        for result, color in zip(results.values(), ["red", "orange", "blue", "yellow"])
    ]
    lines.append(Line3D((0, 3 * direction[0]), (0, 3 * direction[1]), (0, 3 * direction[2]), color="green", lw=3))

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=40, alpha=0.4)

    for line in lines:
        ax.add_line(line)

    ax.set(title="3D Gaussian with linear projections", xlabel="x", ylabel="y", zlabel="z")
    ax.legend(lines, ["Spearman", "Kendall", "Mutual Information", "Linear Regression", "True Direction"])

    fig.savefig(filepath / "lines_3d.png")
    plt.close()


def plot_metric_distribution(distribution: np.ndarray, metric_name: str, d: int, filepath: pathlib.Path) -> None:
    plot = sns.histplot(distribution, stat="density", kde=True)
    plot.axes.set_title(f"{metric_name} coefficient distribution in {d}D")
    plot.set(xlabel=f"{metric_name}")
    plt.savefig(filepath / (metric_name + f"_{d}D_" + ".png"))
    plt.close()


@hy.config
@dataclasses.dataclass
class RandomLinesConfig:
    d: int = 2
    n: int = 100
    budget: int = 200
    m: int = 1000
    seed: int = 1


@hy.main
def main(cfg: RandomLinesConfig):

    # create the directory for storing the figures
    filepath = pathlib.Path(_DEFAULT_TARGET)
    filepath.mkdir(exist_ok=True, parents=True)

    n = cfg.n
    d = cfg.d
    m = cfg.m
    budget = cfg.budget
    seed = cfg.seed

    assert d in [2, 3], "Can not plot for dimensions other than 2 or 3"

    # create the artificial dataset
    points, scores, direction = create_data(n=n, d=d)

    # get the directions found by all the methods we consider
    results = optimal_directions(points, scores, budget, seed=seed)

    # get the distribution of the Spearman correlation coefficient
    distributions = score_distributions(points, scores, m=m, seed=seed)

    # plot
    if d == 2:
        plot_2d(points, direction, results, filepath)
    elif d == 3:
        plot_3d(points, direction, results, filepath)
    plot_metric_distribution(distributions["spearman"], "Spearman Coefficient", d, filepath)


if __name__ == "__main__":
    main()
