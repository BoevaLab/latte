import numpy as np
import pandas as pd
import geoopt
import torch

from latte.dataset import utils as dsutils
from latte.models.probabilistic_pca import probabilistic_pca
from latte.utils import evaluation


class OrientationMetrics:
    """
    A collection of metrics measured when fitting a pPCA model to generated data.
    These are for example used to index data frames containing the evaluation results.
    """

    DISTANCE_TO_PERMUTATION_MATRIX = "Distance to a permutation matrix"
    MEAN_DISTANCE_TO_PERMUTATION_MATRIX = "Mean distance to a permutation matrix"
    ORIGINAL_DISTANCE_TO_A = "Matrix distance of the original estimate"
    DISTANCE_MU = "Distance of the estimated mean"
    ORIGINAL_DISTANCE_TO_A_NORMALISED = "Relative matrix distance of the original estimate"
    ORIENTED_DISTANCE_TO_A = "Matrix distance of the oriented estimate"
    ORIENTED_DISTANCE_TO_A_NORMALISED = "Relative matrix distance of the oriented estimate"
    OPTIMAL_REPRESENTATION_ERROR = "Optimal latent representation error"
    ORIGINAL_REPRESENTATION_ERROR = "Latent representation error of the original estimate"
    ORIENTED_REPRESENTATION_ERROR = "Latent representation error of the oriented estimate"
    RELATIVE_ORIENTED_REPRESENTATION_ERROR = "Relative representation error of the oriented estimate"
    OPTIMAL_OBSERVATION_ERROR = "Optimal observation error"
    ORIGINAL_OBSERVATION_ERROR = "Observation error of the original estimate"
    ORIENTED_OBSERVATION_ERROR = "Observation error of the oriented estimate"
    RELATIVE_ORIENTED_OBSERVATION_ERROR = "Relative observation error of the oriented estimate"
    NUMBER_OF_EPOCHS_REQUIRED = "Number of epochs required to orient the model"
    SIGMA_HAT = "Observational noise standard deviation"
    SIGMA_SQUARED_HAT = "Observational noise variance"


def subspace_fit(dataset: probabilistic_pca.ProbabilisticPCADataset, U_hat: np.ndarray) -> pd.DataFrame:
    """
    Evaluates the goodness of fit of the found subspace containing the *measured factors* spanned by the estimate
    of the mixing matrix `A_hat` compared to the ground-truth one.
    Args:
        dataset: The generated pPCA dataset.
        U_hat: The orthogonal basis of the estimated subspace

    Returns:
        A pandas dataframe containing two evaluation metrics:
            - the norm of the difference between the projection matrices *in the original space*
            - the same norm normalised by the norm of the ground-truth mixing matrix
    """
    assert dataset.A.shape[1] >= len(dataset.measured_factors)

    # We take out the columns which correspond to the factors we assume to have measured
    A = dataset.A[:, dataset.measured_factors]

    # We decompose the induced true covariance matrix to get the orthogonal basis of the principal subspace
    U, _, _ = np.linalg.svd(A @ A.T + dataset.sigma**2 * np.eye(dataset.n))
    # Take the first `d_measured` components to identify the true subspace of the measured factors
    # (it only captures the measured ones since we copied them out above, and it captures them entirely, since we
    #  take all `d_measured` components)
    U = U[:, : dataset.d_measured]

    # We look at the distance between the subspace of the `d_measured` factors and *its* projection onto the found
    # subspace defined by `U_hat`
    return evaluation.subspace_fit(U, U_hat)


def evaluate_log_likelihood(
    X: np.ndarray,
    A_hat: np.ndarray,
    A_hat_oriented: np.ndarray,
    mu_hat: np.ndarray,
    sigma_hat: float,
    rng: dsutils.RandomGenerator,
    n_runs: int = 50,
) -> pd.DataFrame:
    """Evaluates the log-likelihood of the observable data `X` given the pPCA-estimated matrix `A_hat`, the oriented
       version of `A_hat`, `A_hat_oriented`, other random rotations of `A_hat`, and random matrices of the same
       dimensionality.
       The point is to show that all possible rotations have the same log-likelihood under this model.

    Args:
        X (np.ndarray): The data matrix.
        A_hat (np.ndarray): The original estimate of the mixing matrix (with columns as scaled principal directions).
        A_hat_oriented (np.ndarray): The oriented version of `A_hat` such that it corresponds as well as possible to the
                                     true mixing matrix.
        mu_hat (np.ndarray): The estimate of the observable data mean.
        sigma_hat (float): The estimate of the observable data noise standard deviation.
        rng (dsutils.RandomGenerator): Random generator.
        n_runs (int, optional): How many random matrices to use to calculate the mean log-likelihood of a random matrix.
                                Defaults to 50.

    Returns:
        pd.DataFrame: A data frame with the measured quantities;
            - LL(A_hat): log-likelihood of `A_hat`, `mu_hat`, and `sigma_hat`
            - LL(A_oriented): log-likelihood of `A_hat_oriented`, `mu_hat`, and `sigma_hat`
            - LL(A_hat R)": log-likelihood of a random rotation of `A_hat`, `mu_hat`, and `sigma_hat`
            - LL(M): mean log-likelihood of a random mixing matrix, `mu_hat`, and `sigma_hat`
    """
    df = pd.DataFrame(
        [
            ("LL(A_hat)", probabilistic_pca.gaussian_log_likelihood(X, A_hat, mu_hat, sigma_hat)),
            # Log likelihood of the original (non-oriented) estimate of the mixing matrix
            (
                "LL(A_oriented)",
                probabilistic_pca.gaussian_log_likelihood(X, A_hat_oriented, mu_hat, sigma_hat),
            ),  # Log likelihood of the fit matrix rotated towards the true one
            (
                "LL(A_hat R)",
                np.mean(
                    [
                        probabilistic_pca.gaussian_log_likelihood(
                            X,
                            A_hat @ geoopt.Stiefel().random(A_hat.shape[1], A_hat.shape[1]).detach().numpy(),
                            mu_hat,
                            sigma_hat,
                        )
                        for _ in range(n_runs)
                    ]
                ),
            ),  # Log likelihood of a random rotation of the estimate
            (
                "LL(M)",
                np.mean(
                    [
                        probabilistic_pca.gaussian_log_likelihood(
                            X, rng.normal(size=A_hat_oriented.shape), mu_hat, sigma_hat
                        )
                        for _ in range(n_runs)
                    ]
                ),
            ),  # Log likelihood of a random matrix of the same shape
        ],
        columns=["Metric", "Value"],
    )

    df = df.set_index("Metric")

    return df


def evaluate_full_result(
    X: np.ndarray,
    Z: np.ndarray,
    A: np.ndarray,
    A_hat: np.ndarray,
    A_hat_oriented: np.ndarray,
    R: np.ndarray,
    mu: np.ndarray,
    mu_hat: np.ndarray,
    sigma: float,
    sigma_hat: float,
    stopped_epoch: int,
    nruns: int = 50,
) -> pd.DataFrame:
    """Evaluates the orientation of the original mixing matrix estimate `A_hat` - `A_hat_oriented` - fit to correspond
       to the true mixing matrix `A`.
       On top of that, it also evaluates other possible solutions obtained by rotating the original estimate.

    Args:
        X (np.ndarray): The observable data matrix.
        Z (np.ndarray): The matrix of true observations of the latent factors
        A (np.ndarray): The true mixing matrix.
        A_hat (np.ndarray): The original estimate of the mixing matrix as returned by pPCA.
        A_hat_oriented (np.ndarray): The oriented estimate of the mixing matrix.
        R (np.ndarray): The rotation matrix used to construct A_hat_oriented.
        mu (np.ndarray): The true observation data mean.
        mu_hat (np.ndarray): The estimate of the observed mean.
        sigma (float): The true observable data noise standard deviation.
        sigma_hat (float): The estimate of the observable data noise standard deviation.
        stopped_epoch (int): Number of epochs required to orient the model.
        n_runs (int, optional): How many random matrices to use to calculate the mean distance to a permutation matrix
                                of a random matrix orthogonal matrix.
                                Defaults to 50.

    Returns:
        pd.DataFrame: A data frame containing the values of the metrics grouped in the OrientationMetrics class.
                      The data frame contains a column naming the metric, "Metric", and a column for the value
                      of the metric, "Value".
    """
    # We keep a list of (metric, value) pairs to keep the order in the data frame
    evaluation_results = []

    # Calculate the distance of the estimated rotation matrix to a permutation matrix to assess how well aligned the
    # original mixing matrix captured individual factors
    evaluation_results.append(
        (
            OrientationMetrics.DISTANCE_TO_PERMUTATION_MATRIX,
            round(evaluation.distance_to_permutation_matrix(torch.from_numpy(R)), 3),
        )
    )
    evaluation_results.append(
        (
            OrientationMetrics.MEAN_DISTANCE_TO_PERMUTATION_MATRIX,
            round(
                float(
                    np.mean(
                        [
                            evaluation.distance_to_permutation_matrix(
                                geoopt.Stiefel().random(max(R.shape), min(R.shape))
                            )
                            for _ in range(nruns)
                        ]
                    )
                ),
                3,
            ),
        )
    )

    if A_hat.shape == A.shape:
        # If the estimate is of the correct shape, also see how close it is to the true mixing matrix
        evaluation_results.append((OrientationMetrics.ORIGINAL_DISTANCE_TO_A, round(np.linalg.norm(A_hat - A), 4)))
        evaluation_results.append(
            (
                OrientationMetrics.ORIGINAL_DISTANCE_TO_A_NORMALISED,
                round(np.linalg.norm(A_hat - A) / np.linalg.norm(A), 4),
            )
        )

    # Evaluate how close the oriented matrix is to the true one
    evaluation_results.append((OrientationMetrics.ORIENTED_DISTANCE_TO_A, round(np.linalg.norm(A_hat_oriented - A), 4)))
    evaluation_results.append(
        (
            OrientationMetrics.ORIENTED_DISTANCE_TO_A_NORMALISED,
            round(np.linalg.norm(A_hat_oriented - A) / np.linalg.norm(A), 4),
        )
    )

    # Calculate the posterior means given the mixing matrices
    Z_pca_true = probabilistic_pca.get_latent_representations(X, A, mu, sigma)
    Z_pca_original = probabilistic_pca.get_latent_representations(X, A_hat, mu_hat, sigma_hat)
    Z_pca_oriented = probabilistic_pca.get_latent_representations(X, A_hat_oriented, mu_hat, sigma_hat)

    # Calculate the mean latent error from the true observations
    r_error_oriented = np.linalg.norm(Z_pca_oriented - Z, axis=1).mean()
    r_error_true = np.linalg.norm(Z_pca_true - Z, axis=1).mean()

    if A_hat.shape == A.shape:
        r_error_original = np.linalg.norm(Z_pca_original - Z, axis=1).mean()
        evaluation_results.append((OrientationMetrics.ORIGINAL_REPRESENTATION_ERROR, round(r_error_original, 4)))

    evaluation_results.append((OrientationMetrics.OPTIMAL_REPRESENTATION_ERROR, round(r_error_true, 4)))
    evaluation_results.append((OrientationMetrics.ORIENTED_REPRESENTATION_ERROR, round(r_error_oriented, 4)))
    evaluation_results.append(
        (
            OrientationMetrics.RELATIVE_ORIENTED_REPRESENTATION_ERROR,
            round((r_error_oriented - r_error_true) / r_error_true, 4),
        )
    )

    # Calculate the mean observation given the mixing matrices
    X_hat_true = probabilistic_pca.get_observations(Z, A, mu)
    X_hat_oriented = probabilistic_pca.get_observations(Z, A_hat_oriented, mu_hat)

    # Calculate the mean observation error based on the true latent factors
    if A_hat.shape == A.shape:
        X_hat_original = probabilistic_pca.get_observations(Z, A_hat, mu_hat)
        o_error_original = np.linalg.norm(X - X_hat_original, axis=1).mean()
        evaluation_results.append((OrientationMetrics.ORIGINAL_OBSERVATION_ERROR, round(o_error_original, 4)))

    o_error_oriented = np.linalg.norm(X - X_hat_oriented, axis=1).mean()
    o_error_true = np.linalg.norm(X - X_hat_true, axis=1).mean()

    evaluation_results.append((OrientationMetrics.OPTIMAL_OBSERVATION_ERROR, round(o_error_true, 4)))
    evaluation_results.append((OrientationMetrics.ORIENTED_OBSERVATION_ERROR, round(o_error_oriented, 4)))
    evaluation_results.append(
        (
            OrientationMetrics.RELATIVE_ORIENTED_OBSERVATION_ERROR,
            round((o_error_oriented - o_error_true) / o_error_true, 4),
        )
    )

    # Record the error of the mean estimate
    evaluation_results.append((OrientationMetrics.DISTANCE_MU, np.linalg.norm(mu - mu_hat)))

    # Record the estimate of the standard deviation
    evaluation_results.append((OrientationMetrics.SIGMA_HAT, sigma_hat))
    evaluation_results.append((OrientationMetrics.SIGMA_SQUARED_HAT, sigma_hat**2))

    # Keep track of the number of epochs needed to orient the model
    evaluation_results.append((OrientationMetrics.NUMBER_OF_EPOCHS_REQUIRED, stopped_epoch))

    df = pd.DataFrame(evaluation_results, columns=["Metric", "Value"])
    return df
