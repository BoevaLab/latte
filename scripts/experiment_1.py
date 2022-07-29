"""
This is the script for generating the results for "Experiment 1" for the thesis.

It evaluates the solutions found by simple probabilistic PCA on data generated using the probabilistic PCA model.
It runs pPCA on the data and then rotates the found mixing matrix to align with the true one based on a labeled
subsample of the data.
"""
from typing import List
from dataclasses import dataclass, field
import pathlib

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, ParameterGrid

from tqdm import tqdm, trange

import latte.hydra_utils as hy
from latte.dataset import utils as dsutils
from latte.utils import visualisation, evaluation
from latte.models.probabilistic_pca.orientator import orientation
from latte.models.probabilistic_pca import probabilistic_pca, evaluation as ppca_evaluation

import logging
import coloredlogs

coloredlogs.install()
logging.basicConfig(level=logging.INFO)


# Names of the quantities over which the script can iterate to inspect the impact of on the results
quantity_names = ["N", "n", "d", "d_measured", "noise_std", "orientation_ratio", "orientation_N"]
quantity_x_labels = {
    "N": "Number of training points",
    "n": "Dimensionality of the observable space",
    "d": "Dimensionality of the latent space",
    "d_measured": "Number of factors of interest",
    "noise_std": "Noise standard deviation",
    "orientation_ratio": "Proportion of the data used for orientation",
    "orientation_N": "Number of examples for orientation",
    "run": "Run",
}

# What will be plotted in the end
metrics_to_plot = [
    (
        ppca_evaluation.OrientationMetrics.DISTANCE_TO_PERMUTATION_MATRIX,
        ppca_evaluation.OrientationMetrics.MEAN_DISTANCE_TO_PERMUTATION_MATRIX,
    ),
    (
        ppca_evaluation.OrientationMetrics.ORIGINAL_DISTANCE_TO_A_NORMALISED,
        ppca_evaluation.OrientationMetrics.ORIENTED_DISTANCE_TO_A_NORMALISED,
    ),
    (ppca_evaluation.OrientationMetrics.ORIGINAL_DISTANCE_TO_A_NORMALISED,),
    (ppca_evaluation.OrientationMetrics.ORIENTED_DISTANCE_TO_A_NORMALISED,),
    (
        ppca_evaluation.OrientationMetrics.ORIGINAL_REPRESENTATION_ERROR,
        ppca_evaluation.OrientationMetrics.OPTIMAL_REPRESENTATION_ERROR,
        ppca_evaluation.OrientationMetrics.ORIENTED_REPRESENTATION_ERROR,
    ),
    (
        ppca_evaluation.OrientationMetrics.ORIGINAL_OBSERVATION_ERROR,
        ppca_evaluation.OrientationMetrics.OPTIMAL_OBSERVATION_ERROR,
        ppca_evaluation.OrientationMetrics.ORIENTED_OBSERVATION_ERROR,
    ),
    (ppca_evaluation.OrientationMetrics.RELATIVE_ORIENTED_REPRESENTATION_ERROR,),
    (ppca_evaluation.OrientationMetrics.RELATIVE_ORIENTED_OBSERVATION_ERROR,),
    (ppca_evaluation.OrientationMetrics.SIGMA_HAT,),
    (ppca_evaluation.OrientationMetrics.SIGMA_SQUARED_HAT,),
    (
        "Difference between subspaces",
    ),  # Manually added entries about the PCA subspace error since they are not in the class OrientationMetrics
    ("Normalised difference between subspaces",),
    ("Number of epochs required to orient the model",),
]


@hy.config
@dataclass
class Experiment1Config:
    """
    The configuration of the synthetic data with subspace estimation experiment.

    Members:
        N: The number of points to generate.
        n: the dimensionality of the observed space.
        d: The dimensionality of the subspace
        d_measured: The number of relevant factors we are interested in / have measurements of
        orientation_ratio: The proportion of the data to use for orientation of the mixing matrix
        orientation_N: The number of samples to use for orientation of the mixing matrix (overrides `orientation_ratio`)
        n_runs: How many times to fit each parameter setting to get a reliable estimate of the performance
        learning_rate: The learning rate for the Orientator network
        noise: The scale of the gaussian noise to add to the observations
        max_epochs: The number of epochs to train the MINE model for
        seed: The seed for the random generators
    """

    N: List[int] = field(default_factory=lambda: [65536])
    n: List[int] = field(default_factory=lambda: [256])
    d: List[int] = field(default_factory=lambda: [32])
    d_measured: List[int] = field(default_factory=lambda: [32])
    noise_std: List[float] = field(default_factory=lambda: [0.1 ** (1 / 2)])
    orientation_ratio: List[float] = field(default_factory=lambda: [-1.0])
    orientation_N: List[int] = field(default_factory=lambda: [32768])
    n_runs: int = 20
    loss: str = "mse"
    learning_rate: float = 1e-2
    batch_size: int = 32
    max_epochs: int = 512
    min_delta: float = 1e-3
    patience: int = 16
    p_eval: float = 0.2
    seed: int = 42
    num_workers: int = 12
    gpus: int = 1


def fit(
    N: int,
    n: int,
    d: int,
    d_measured: int,
    loss: str,
    num_workers: int,
    noise_std: float,
    learning_rate: float,
    orientation_ratio: float,
    orientation_N: int,
    batch_size: int,
    max_epochs: int,
    min_delta: float,
    patience: int,
    p_eval: float,
    seed: int,
    gpus: int,
    filepath: pathlib.Path,
    rng: dsutils.RandomGenerator,
):
    """Fits and evaluates one set of parameters for the pPCA model."""
    assert (
        orientation_ratio != -1 or orientation_N != -1
    ), "At least one of the orientation_ratio or orientation_n has to be specified."

    # Generate the synthetic data based on the pPCA model
    dataset = probabilistic_pca.generate(N=N, n=n, d=d, d_measured=d_measured, sigma=noise_std, rng=rng)

    X_fit, X_eval, Z_fit, Z_eval = train_test_split(dataset.X, dataset.Z, test_size=p_eval, random_state=seed)

    # We do not need `Z_train`, since we only use the "train" split for pPCA
    X_train, X_orient, _, Z_orient = train_test_split(
        X_fit,
        Z_fit,
        test_size=orientation_N if orientation_N != -1 else int(orientation_ratio * len(X_fit)),
        random_state=seed,
    )

    # Run probabilistic PCA to get the principal components of the observed data
    pca = PCA(n_components=d, random_state=seed)
    pca.fit(X_train)

    # Evaluate the quality of the fit to the *subspace*
    subspace_evaluation_results = evaluation.subspace_fit(pca.components_.T, dataset.Q)
    logging.info("--- Quality of the fit. ---")
    print(subspace_evaluation_results)
    print()

    # Construct the original/raw pPCA estimate of the mixing matrix. See (Tipping, Bishop, 1999)
    A_hat = probabilistic_pca.construct_mixing_matrix_estimate(pca)

    if n == 3 and d_measured <= 2:

        # If you only need a figure, you should run the script with n_runs=1
        # and only a small number of possible settings of parameters
        visualisation.probabilistic_pca_data(
            X=X_eval,
            Z=Z_eval,
            A=dataset.A[:, dataset.factors_of_interest],
            mu=dataset.mu,
            alphas={"observable": 0.4, "true_projections": 0.7, "estimated_projections": 0.6},
            file_name=filepath / f"original_est_1_N{N}_n{n}_d{d}_dm{d_measured}_s{noise_std:.3f}_o{len(X_orient)}.png",
            title="Observable data and the true subspace",
        )
        visualisation.probabilistic_pca_data(
            X=X_eval,
            Z=Z_eval,
            A=dataset.A[:, dataset.factors_of_interest],
            mu=dataset.mu,
            A_hat=A_hat,
            mu_hat=pca.mean_,
            sigma_hat=np.sqrt(pca.noise_variance_),
            file_name=filepath / f"original_est_2_N{N}_n{n}_d{d}_dm{d_measured}_s{noise_std:.3f}_o{len(X_orient)}.png",
            title="Original pPCA mixing matrix estimate",
        )

    # Orient the computed mixing matrix to be able to compute the true factors of variation
    orientation_result = orientation.orient(
        X=X_orient,
        Z=Z_orient,
        A_hat=A_hat,
        mu_hat=pca.mean_,
        sigma_hat=np.sqrt(pca.noise_variance_),
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

    print(f"Test loss of the orientation: {orientation_result.loss}.")

    R_hat = orientation_result.R
    A_hat_oriented = A_hat @ R_hat
    stopped_epoch = orientation_result.stopped_epoch

    # Evaluate the orientation results
    orientation_evaluation_results = ppca_evaluation.evaluate_orientation(
        X=X_eval,
        Z=Z_eval,
        A=dataset.A[:, dataset.factors_of_interest],
        A_hat=A_hat[:, dataset.factors_of_interest],
        A_hat_oriented=A_hat_oriented,
        R=R_hat,
        mu=dataset.mu,
        mu_hat=pca.mean_,
        sigma=dataset.sigma,
        sigma_hat=np.sqrt(pca.noise_variance_),
        stopped_epoch=stopped_epoch,
    )

    logging.info("--- Results of the fit. ---")
    print(orientation_evaluation_results)
    print()

    # To confirm the rotational invariance of the log-likelihood, we calculate it for the rotated matrices
    # and some random ones

    # Log-likelihood based on only on the factors we can measure
    log_likelihood_evaluation_results_constrained = ppca_evaluation.evaluate_log_likelihood(
        X_eval, A_hat[:, dataset.factors_of_interest], A_hat_oriented, pca.mean_, np.sqrt(pca.noise_variance_), rng=rng
    )

    logging.info("--- Log-likelihood evaluation results constrained on the measured factors. ---")
    print(log_likelihood_evaluation_results_constrained)
    print()

    # Log-likelihood based on all factors
    log_likelihood_evaluation_results_unconstrained = ppca_evaluation.evaluate_log_likelihood(
        X_eval, A_hat, A_hat_oriented, pca.mean_, np.sqrt(pca.noise_variance_), rng=rng
    )

    logging.info("--- Log-likelihood evaluation results on all factors. ---")
    print(log_likelihood_evaluation_results_unconstrained)
    print()

    # Save the evaluation results
    subspace_evaluation_results.to_csv(
        filepath / f"subspace_evaluation_N{N}_n{n}_d{d}_dm{d_measured}_s{noise_std:.3f}_o{len(X_orient)}.csv"
    )
    orientation_evaluation_results.to_csv(
        filepath / f"orientation_evaluation_N{N}_n{n}_d{d}_dm{d_measured}_s{noise_std:.3f}_o{len(X_orient)}.csv"
    )
    log_likelihood_evaluation_results_constrained.to_csv(
        filepath
        / f"log_likelihood_evaluation_constrained_N{N}_n{n}_d{d}_dm{d_measured}_s{noise_std:.3f}_o{len(X_orient)}.csv"
    )
    log_likelihood_evaluation_results_unconstrained.to_csv(
        filepath
        / f"log_likelihood_evaluation_unconstrained_N{N}_n{n}_d{d}_dm{d_measured}_s{noise_std:.3f}_o{len(X_orient)}.csv"
    )

    if n == 3 and d == 2:
        visualisation.probabilistic_pca_data(
            X=X_eval,
            Z=Z_eval,
            A=dataset.A[:, dataset.factors_of_interest],
            mu=dataset.mu,
            A_hat=A_hat_oriented,
            mu_hat=pca.mean_,
            sigma_hat=np.sqrt(pca.noise_variance_),
            alphas={"observable": 0.05, "true_projections": 0.4, "estimated_projections": 0.4},
            file_name=filepath / f"oriented_est_N{N}_n{n}_d{d}_dm{d_measured}_s{noise_std:.3f}_o{len(X_orient)}.png",
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
        cfg.d_measured,
        cfg.noise_std,
        cfg.orientation_ratio,
        cfg.orientation_N,
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
            "d_measured": cfg.d_measured,
            "noise_std": cfg.noise_std,
            "orientation_ratio": cfg.orientation_ratio,
            "orientation_N": cfg.orientation_N,
            "run": [1],
        }
    )
    results = list()
    for p in tqdm(parameters):
        logging.info(f"Running pPCA with configuration {p}.")
        for _ in trange(cfg.n_runs, leave=False):
            result = fit(
                N=p["N"],
                n=p["n"],
                d=p["d"],
                d_measured=p["d_measured"] if quantity_name == "d_measured" else p["d"],
                # `d_measured` only differs from `d` if we are explicitly varying `d_measured`
                # (measuring the effect of misspecification)
                loss=cfg.loss,
                num_workers=cfg.num_workers,
                noise_std=p["noise_std"],
                learning_rate=cfg.learning_rate,
                orientation_ratio=p["orientation_ratio"],
                orientation_N=min(p["N"] // 2, p["orientation_N"]),
                # That's because not all `N` samples will be used for orientation
                batch_size=cfg.batch_size,
                max_epochs=cfg.max_epochs,
                min_delta=cfg.min_delta,
                patience=cfg.patience,
                p_eval=cfg.p_eval,
                seed=cfg.seed,
                gpus=cfg.gpus,
                filepath=filepath,
                rng=rng,
            )

            # Save the value which was used in this particular run for plotting trends
            result["x"] = p[quantity_name]

            results.append(result)

    # Concat results from all runs
    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(filepath / "full_results.csv")

    if all([len(a) == 1 for a in arrays]):
        # Plot standard deviations
        visualisation.metrics_boxplot(
            results_df[results_df["Metric"].isin(["Number of epochs required to orient the model"])],
            label_rotation=0,
            file_name=filepath / "num_epochs_histogram.png",
        )
        visualisation.metrics_boxplot(
            results_df[
                ~results_df["Metric"].isin(
                    [
                        "Number of epochs required to orient the model",
                        "Observation error of the original estimate",
                        "Matrix distance of the original estimate",
                        "Matrix distance of the oriented estimate",
                    ]
                )
            ],
            label_rotation=80,
            file_name=filepath / "metrics_histogram.png",
        )
    else:
        # Plot trends
        for metrics in metrics_to_plot:
            visualisation.metric_trend(
                results_df[results_df["Metric"].isin(metrics)],
                quantity_x_labels[quantity_name],
                filepath / f"{','.join(metrics)}_effects.png",
            )


if __name__ == "__main__":
    main()
