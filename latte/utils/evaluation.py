# TODO (Anej): documentation
import numpy as np
import pandas as pd


def evaluate_found_subspace(A: np.ndarray, A_hat: np.ndarray) -> pd.DataFrame:
    """
    Evaluates the goodness of fit of the found subspace spanned by the k-frame `A_hat` compared to the ground-truth
    subspace spanned by the k-frame `A`.
    Args:
        A: The true k-frame
        A_hat: The estimated k-frame

    Returns:
        A pandas dataframe containing two evaluation metrics:
            - the norm of the difference between the projection matrices *in the original space* corresponding to the
              k-frames
            - the same norm normalised by the norm of the projection matrix of the true k-frame
    """
    subspace_distance = np.linalg.norm(A @ A.T - A_hat @ A_hat.T)
    normalised_subspace_distance = np.linalg.norm(A @ A.T - A_hat @ A_hat.T) / np.linalg.norm(A @ A.T)
    return pd.DataFrame(
        {
            "Value": {
                "Difference between subspaces": subspace_distance,
                "Normalised difference between subspaces": normalised_subspace_distance,
            }
        }
    )


def evaluate_found_mixing_matrix(A: np.ndarray, A_hat: np.ndarray) -> pd.DataFrame:
    """
    Evaluates the goodness of fit of the found mixing matrix `A_hat` compared to the ground-truth matrix `A`,
    for example in probabilistic PCA.
    Args:
        A: The true mixing matrix
        A_hat: The estimated mixing matrix

    Returns:
        A pandas dataframe containing two evaluation metrics:
            - the norm of the difference between the matrices
            - the same norm normalised by the norm of the true mixing matrix
    """
    subspace_distance = np.linalg.norm(A - A_hat)
    normalised_subspace_distance = np.linalg.norm(A - A_hat) / np.linalg.norm(A)
    return pd.DataFrame(
        {
            "Value": {
                "Difference between matrices": subspace_distance,
                "Normalised difference between matrices": normalised_subspace_distance,
            }
        }
    )


# TODO (Anej): This is a work-in-progress implementation
def evaluate_alignment_to_axes(A_hat: np.ndarray) -> pd.DataFrame:
    """
    Evaluates the degree to which a given projection matrix (its columns) is
    aligned to the canonical basis of the latent space.
    This can be used to evaluate how well the individual axes of the original latent space
    capture factors on their own (and not as a component of a linear combination of the others);
    in effect, this measures how well the factors are aligned with the original axes.

    Args:
        A_hat: The estimated projection matrix onto the linear subspace capturing the most information
        about the factors of variation.

    Returns:
        A dataframe containing the results of the evaluation.
    """
    alignments = {"Alignment": dict()}
    for new_axis in range(A_hat.shape[1]):
        # If some value in a column of the transformation matrix is close to 1 (in absolute value),
        # it means that the new basis is well aligned to the original axis whose coefficient is close to 1
        alignments["Alignment"][f"Axis {new_axis}"] = 1 - np.abs(A_hat[:, new_axis])

    return pd.DataFrame(alignments)
