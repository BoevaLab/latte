"""
This is the script for generating the results for "Experiment 1" for the thesis.

It evaluates the solutions found by simple probabilistic PCA on data generated using the probabilistic PCA model.
It runs pPCA on the data and then rotates the found mixing matrix to align with the true one based on a labeled
subsample of the data.
"""
from typing import List
from dataclasses import dataclass, field
import pathlib
from math import ceil

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, ParameterGrid

from tqdm import tqdm, trange

import latte.hydra_utils as hy
from latte.dataset import utils as dsutils
from latte.utils.visualisation import datasets as dataset_visualisation, metrics as metrics_visualisation
from latte.models.probabilistic_pca import probabilistic_pca, orientation, evaluation as ppca_evaluation

import logging
import coloredlogs

coloredlogs.install()
logging.basicConfig(level=logging.INFO)


# Names of the quantities over which the script can iterate to inspect the impact of on the results
quantity_names = ["N", "n", "d", "d_hat", "d_measured", "noise_std", "M", "manifold_optimisation"]
quantity_x_labels = {
    "N": r"$N$",
    "n": r"$n$",
    "d": r"$d$",
    "d_hat": r"$\hat{d}$",
    "d_measured": r"$d_m$",
    "noise_std": r"$\sigma^2$",
    "M": r"$M$",
    "manifold_optimisation": "Manifold optimisation",
}

# What will be plotted in the end
metrics_to_plot = [
    (
        ppca_evaluation.OrientationMetrics.DISTANCE_TO_PERMUTATION_MATRIX,
        ppca_evaluation.OrientationMetrics.MEAN_DISTANCE_TO_PERMUTATION_MATRIX,
    ),
    (ppca_evaluation.OrientationMetrics.ORIENTED_DISTANCE_TO_A,),
    # (ppca_evaluation.OrientationMetrics.ORIENTED_DISTANCE_TO_A_NORMALISED,),
    (ppca_evaluation.OrientationMetrics.SIGMA_SQUARED_HAT,),
    (ppca_evaluation.OrientationMetrics.DISTANCE_MU,),
    (
        ppca_evaluation.OrientationMetrics.TRAIN_LOSS,
        ppca_evaluation.OrientationMetrics.TEST_LOSS,
    ),
    (
        "Difference between subspaces",
    ),  # Manually added entries about the PCA subspace error since they are not in the class OrientationMetrics
    # ("Normalised difference between subspaces",),
    ("Number of epochs required to orient the model",),
]

metric_titles = {
    (
        ppca_evaluation.OrientationMetrics.DISTANCE_TO_PERMUTATION_MATRIX,
        ppca_evaluation.OrientationMetrics.MEAN_DISTANCE_TO_PERMUTATION_MATRIX,
    ): "Distance to a Permutation Matrix",
    (ppca_evaluation.OrientationMetrics.ORIENTED_DISTANCE_TO_A,): "Matrix Error",
    # (ppca_evaluation.OrientationMetrics.ORIENTED_DISTANCE_TO_A_NORMALISED,),
    (ppca_evaluation.OrientationMetrics.SIGMA_SQUARED_HAT,): r"$\hat{\sigma}^2$",
    (ppca_evaluation.OrientationMetrics.DISTANCE_MU,): r"$\mu$ Error",
    ("Difference between subspaces",): "Subspace Error",
    # ("Normalised difference between subspaces",),
    ("Number of epochs required to orient the model",): "Epochs Required for Orientation",
    (
        ppca_evaluation.OrientationMetrics.TRAIN_LOSS,
        ppca_evaluation.OrientationMetrics.TEST_LOSS,
    ): "Train and Test Loss",
}


@hy.config
@dataclass
class Experiment1Config:
    """
    The configuration of the synthetic data with subspace estimation experiment.

    Members:
        N: The number of points to generate.
        n: the dimensionality of the observed space.
        d: The dimensionality of the subspace
        d_hat: The assumed number of components (the number of components fit by pPCA)
        d_measured: The number of relevant factors we are interested in / have measurements of
        noise_std: The scale of the gaussian noise to add to the observations
        M: The number of samples to use for orientation of the mixing matrix
        manifold_optimisation: Whether to orient the estimate constrained to the Stiefel manifold or not.
        N_eval: Number of data points used for independent evaluation in the end
        n_runs: How many times to fit each parameter setting to get a reliable estimate of the performance
        loss: The loss to use for the orientation fitting.
              Currently, only "mse" is supported.
        learning_rate, batch_size, max_epochs, min_delta, patience, num_workers, gpus: Parameters for the Orientator
        seed: The seed for the random generators
    """

    N: List[int] = field(default_factory=lambda: [1024])
    n: List[int] = field(default_factory=lambda: [64])
    d: List[int] = field(default_factory=lambda: [8])
    d_hat: List[int] = field(default_factory=lambda: [8])
    d_measured: List[int] = field(default_factory=lambda: [8])
    noise_std: List[float] = field(default_factory=lambda: [0.1 ** (1 / 2)])
    M: List[int] = field(default_factory=lambda: [1024])
    manifold_optimisation: List[bool] = field(default_factory=lambda: [True])
    N_eval: int = 1024
    n_runs: int = 20
    loss: str = "mse"
    learning_rate: float = 1e-2
    batch_size: int = 32
    max_epochs: int = 512
    min_delta: float = 1e-3
    patience: int = 16
    seed: int = 42
    num_workers: int = 12
    gpus: int = 1


def fit(
    N: int,
    n: int,
    d: int,
    d_hat: int,
    d_measured: int,
    loss: str,
    num_workers: int,
    noise_std: float,
    learning_rate: float,
    M: int,
    manifold_optimisation: bool,
    batch_size: int,
    max_epochs: int,
    min_delta: float,
    patience: int,
    N_eval: int,
    seed: int,
    gpus: int,
    filepath: pathlib.Path,
    rng: dsutils.RandomGenerator,
):
    """Fits and evaluates one set of parameters for the pPCA model."""

    # Generate the synthetic data based on the pPCA model
    # Adjust the number of generated data points to divide it into train and test data later on
    M_full = ceil(M / 0.5)
    dataset = probabilistic_pca.generate(
        N=N + M_full + N_eval, n=n, d=d, d_measured=d_measured, sigma=noise_std, rng=rng
    )

    X_fit, X_eval, Z_fit, Z_eval = train_test_split(dataset.X, dataset.Z, test_size=N_eval, random_state=seed)

    # We do not need `Z_train`, since we only use the "train" split for pPCA
    X_train, X_orient, _, Z_orient = train_test_split(
        X_fit,
        Z_fit,
        test_size=M_full,
        random_state=seed,
    )

    # Run probabilistic PCA to get the principal components of the observed data
    pca = PCA(n_components=d_hat, random_state=seed)
    pca.fit(X_train)

    # Evaluate the quality of the fit to the *subspace*
    subspace_evaluation_results = ppca_evaluation.subspace_fit(dataset, pca.components_.T)
    print("--- Quality of the fit. ---")
    print(subspace_evaluation_results)
    print()

    # Construct the original/raw pPCA estimate of the mixing matrix. See (Tipping, Bishop, 1999)
    A_hat = probabilistic_pca.construct_mixing_matrix_estimate(pca)

    if n == 3 and d_measured <= 2:

        # If you only need a figure, you should run the script with n_runs=1
        # and only a small number of possible settings of parameters
        dataset_visualisation.probabilistic_pca_data(
            X=X_eval,
            Z=Z_eval[:, dataset.measured_factors],
            A=dataset.A[:, dataset.measured_factors],
            mu=dataset.mu,
            alphas={"observable": 0.4, "true_projections": 0.7, "estimated_projections": 0.6},
            file_name=filepath / f"original_est_1_N{N}_n{n}_d{d}_dh{d_hat}_dm{d_measured}"
            f"_s{noise_std:.3f}_o{len(X_orient)}_mo_{manifold_optimisation}.png",
            title="Observable data and the true subspace",
        )
        dataset_visualisation.probabilistic_pca_data(
            X=X_eval,
            Z=Z_eval[:, dataset.measured_factors],
            A=dataset.A[:, dataset.measured_factors],
            mu=dataset.mu,
            A_hat=A_hat,
            mu_hat=pca.mean_,
            sigma_hat=np.sqrt(pca.noise_variance_),
            file_name=filepath / f"original_est_2_N{N}_n{n}_d{d}_dh{d_hat}_dm{d_measured}"
            f"_s{noise_std:.3f}_o{len(X_orient)}_mo_{manifold_optimisation}.png",
            title="Original pPCA mixing matrix estimate",
        )

    # Orient the computed mixing matrix to be able to compute the true factors of variation
    orientation_result = orientation.orient(
        X=X_orient,
        Z=Z_orient[:, dataset.measured_factors],
        A_hat=A_hat,
        mu_hat=pca.mean_,
        sigma_hat=np.sqrt(pca.noise_variance_),
        manifold_optimisation=manifold_optimisation,
        p_train=0.4,
        p_val=0.1,
        loss=loss,
        batch_size=batch_size,
        learning_rate=learning_rate,
        min_delta=min_delta,
        patience=patience,
        max_epochs=max_epochs,
        num_workers=num_workers,
        verbose=False,
        gpus=gpus,
    )

    # print(f"Test loss of the orientation: {orientation_result.loss}.")

    R_hat = orientation_result.R
    A_hat_oriented = A_hat @ R_hat
    stopped_epoch = orientation_result.stopped_epoch

    # Evaluate the orientation results
    orientation_evaluation_results = ppca_evaluation.evaluate_full_result(
        X=X_eval,
        Z=Z_eval[:, dataset.measured_factors],
        A=dataset.A[:, dataset.measured_factors],
        A_hat=A_hat,
        A_hat_oriented=A_hat_oriented,
        R=R_hat,
        mu=dataset.mu,
        mu_hat=pca.mean_,
        sigma=dataset.sigma,
        sigma_hat=np.sqrt(pca.noise_variance_),
        stopped_epoch=stopped_epoch,
        train_loss=orientation_result.train_loss,
        loss=orientation_result.loss,
    )

    # print("--- Results of the fit. ---")
    # print(orientation_evaluation_results)
    # print()

    # To confirm the rotational invariance of the log-likelihood, we calculate it for the rotated matrices
    # and some random ones

    # Log-likelihood based on only on the factors we can measure
    log_likelihood_evaluation_results_constrained = ppca_evaluation.evaluate_log_likelihood(
        X_eval, A_hat, A_hat_oriented, pca.mean_, np.sqrt(pca.noise_variance_), rng=rng
    )

    print("--- Log-likelihood evaluation results constrained on the measured factors. ---")
    print(log_likelihood_evaluation_results_constrained)
    print()

    # Save the evaluation results
    subspace_evaluation_results.to_csv(
        filepath / f"subspace_evaluation_N{N}_n{n}_d{d}_dh{d_hat}_dm{d_measured}"
        f"_s{noise_std:.3f}_o{len(X_orient)}_mo_{manifold_optimisation}.csv"
    )
    orientation_evaluation_results.to_csv(
        filepath / f"orientation_evaluation_N{N}_n{n}_d{d}_dh{d_hat}_dm{d_measured}"
        f"_s{noise_std:.3f}_o{len(X_orient)}_mo_{manifold_optimisation}.csv"
    )
    log_likelihood_evaluation_results_constrained.to_csv(
        filepath / f"log_likelihood_evaluation_constrained_N{N}_n{n}_d{d}_dh{d_hat}_dm{d_measured}"
        f"_s{noise_std:.3f}_o{len(X_orient)}_mo_{manifold_optimisation}.csv"
    )

    if n == 3 and d == 2:
        dataset_visualisation.probabilistic_pca_data(
            X=X_eval,
            Z=Z_eval,
            A=dataset.A[:, dataset.measured_factors],
            mu=dataset.mu,
            A_hat=A_hat_oriented,
            mu_hat=pca.mean_,
            sigma_hat=np.sqrt(pca.noise_variance_),
            alphas={"observable": 0.05, "true_projections": 0.4, "estimated_projections": 0.4},
            file_name=filepath / f"oriented_est_N{N}_n{n}_d{d}_dh{d_hat}_dm{d_measured}"
            f"_s{noise_std:.3f}_o{len(X_orient)}_mo_{manifold_optimisation}.png",
            title="Oriented pPCA mixing matrix estimate",
        )

    # Additionally include the evaluation of the subspace found by PCA
    evaluation_results = pd.concat(
        [orientation_evaluation_results, subspace_evaluation_results.reset_index().rename(columns={"index": "Metric"})]
    )

    return evaluation_results


@hy.main
def main(cfg: Experiment1Config):
    assert cfg.loss in ["mse"], "The selected loss function is not supported."

    rng = np.random.default_rng(cfg.seed)

    # We evaluate the effect of each hyperparameter individually - only one array can contain more than 1 element.
    arrays = [
        cfg.N,
        cfg.n,
        cfg.d,
        cfg.d_hat,
        cfg.d_measured,
        cfg.noise_std,
        cfg.M,
        cfg.manifold_optimisation,
    ]
    assert sum([len(a) > 1 for a in arrays]) == 1 or all([len(a) == 1 for a in arrays])
    # assert sum(c > 1 for c in [len(a) for a in arrays]) == 1

    # Extract the *only* array with more than one value specified
    if all([len(a) == 1 for a in arrays]):
        quantity_name = "run"
    else:
        quantity_name = [quantity_names[i] for i, a in enumerate(arrays) if len(a) > 1][0]

    filepath = pathlib.Path(f"experiment_1/{quantity_name}")
    filepath.mkdir(exist_ok=True, parents=True)

    # Construct the parameter grid of hyperparameters for the pPCA fit and orientation
    parameters = ParameterGrid(
        {
            "N": cfg.N,
            "n": cfg.n,
            "d": cfg.d,
            "d_hat": cfg.d_hat,
            "d_measured": cfg.d_measured,
            "noise_std": cfg.noise_std,
            "M": cfg.M,
            "manifold_optimisation": cfg.manifold_optimisation,
            "run": [1],
        }
    )
    results = list()
    for p in tqdm(parameters):
        print(f"Running pPCA with configuration {p}.")
        parameter_results = []
        for _ in trange(cfg.n_runs, leave=False):
            result = fit(
                N=p["N"],
                n=p["n"],
                d=p["d"],
                d_measured=p["d_measured"] if quantity_name in ["run", "d_hat", "d_measured"] else p["d"],
                d_hat=p["d_hat"] if quantity_name in ["run", "d_hat", "d_measured"] else p["d"],
                # `d_measured` and `d_hat` only differ from `d` if we are explicitly varying them
                # (measuring the effect of misspecification)
                loss=cfg.loss,
                num_workers=cfg.num_workers,
                noise_std=p["noise_std"],
                learning_rate=cfg.learning_rate,
                M=p["M"],
                manifold_optimisation=p["manifold_optimisation"],
                batch_size=cfg.batch_size,
                max_epochs=cfg.max_epochs,
                min_delta=cfg.min_delta,
                patience=cfg.patience,
                N_eval=cfg.N_eval,
                seed=cfg.seed,
                gpus=cfg.gpus,
                filepath=filepath,
                rng=rng,
            )

            # Save the value which was used in this particular run for plotting trends
            result["x"] = float(p[quantity_name]) if quantity_name != "noise_std" else p[quantity_name] ** 2

            parameter_results.append(result)

        print("--- Results of the evaluation ---")
        print(pd.concat(parameter_results, ignore_index=True).groupby("Metric").median().reset_index())

        results.extend(parameter_results)

    # Concat results from all runs
    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(filepath / "full_results.csv")

    summarised_df = results_df.groupby(["x", "Metric"]).mean()
    summarised_df["std"] = results_df.groupby(["x", "Metric"]).std()["Value"]
    summarised_df.to_csv(filepath / "mean_results.csv")

    if quantity_name == "manifold_optimisation" or all([len(a) == 1 for a in arrays]):

        # Plot standard deviations
        for metric in results_df["Metric"]:
            metrics_visualisation.metrics_boxplot(
                results_df[results_df["Metric"] == metric],
                x="x" if quantity_name == "manifold_optimisation" else "Metric",
                quantity_name="Manifold optimisation" if quantity_name == "manifold_optimisation" else None,
                label_rotation=0,
                file_name=filepath / f"{metric}_histogram.png",
            )
        plot_df = results_df[
            ~results_df["Metric"].isin(
                [
                    "Number of epochs required to orient the model",
                    "Observation error of the original estimate",
                    "Matrix distance of the original estimate",
                    "Matrix distance of the oriented estimate",
                ]
            )
        ]
        if all([len(a) == 1 for a in arrays]):
            metrics_visualisation.metrics_boxplot(
                plot_df,
                label_rotation=80,
                file_name=filepath / "metrics_histogram.png",
            )
        else:
            metrics_visualisation.metrics_boxplot(
                plot_df[plot_df["x"] == 0],
                label_rotation=80,
                file_name=filepath / "metrics_histogram_moFalse.png",
            )
            metrics_visualisation.metrics_boxplot(
                plot_df[plot_df["x"] == 1],
                label_rotation=80,
                file_name=filepath / "metrics_histogram_moTrue.png",
            )
    else:
        # Plot trends
        for metrics in metrics_to_plot:
            metrics_visualisation.metric_trend(
                results_df[results_df["Metric"].isin(metrics)],
                quantity_x_labels[quantity_name],
                title=metric_titles[metrics],
                file_name=filepath / f"{','.join(metrics)}_effects.png",
            )


if __name__ == "__main__":
    main()
