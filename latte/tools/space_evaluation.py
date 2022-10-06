"""
This module contains main wrapper functions around the various tools for estimating the quality and relevant of found
linear subspaces in terms of the mutual information, predictability, and entropy, captured in the subspace.
"""

from typing import List, Dict, Protocol, Any, Optional, Union, Sequence
from enum import Enum
from collections import Counter

import torch
import numpy as np
from sklearn import metrics, preprocessing, feature_selection, linear_model, svm

from tqdm import tqdm

from pythae.models import VAE

from latte.utils.visualisation import images as image_visualisation, latent_spaces as latent_space_visualisation
from latte.tools.ksg import naive_ksg, ksg
from latte.dataset import utils as ds_utils
from latte.tools import model_comparison


class MIEstimationMethod(Enum):
    """
    Specifies the implemented methods of estimating the mutual information.
    `MIST` refers to the extension of `MINE` with optimisation over the Stiefel manifold.
    `NAIVE_KSG` refers to a naive, more biased, version of the KSG estimator, which is, however, faster.
    `KSG` refers to a more polished estimator of the mutual information with less bias.
    `SKLEARN` refers to the `sklearn` estimator of mutual information which only works for two 1-dimensional variables.

    Note that all methods only work when the sum of the two space dimensionalities is at most 20.
    """

    MIST = "mist"
    KSG = "latte-ksg"
    NAIVE_KSG = "ksg"
    SKLEARN = "sklearn"


class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None) -> Any:
        ...

    def predict(self, X) -> Any:
        ...


def _get_Z(
    Z: Union[torch.Tensor, np.ndarray],
    A: Optional[torch.Tensor] = None,
    standardise: bool = True,
    to_numpy: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Utility function to optionally project and standardise the data.

    Args:
        Z: The dataset of representations.
        A: THe optional projection matrix
        standardise: Whether to standardise the data.
        to_numpy: Whether to return the data in numpy format.

    Returns:
        Processed dataset.
    """

    if isinstance(Z, torch.Tensor):
        Z = Z.numpy()

    if standardise:
        Z = preprocessing.StandardScaler().fit_transform(Z)

    if A is not None:
        Z = Z @ A.detach().cpu().numpy()
        Z = preprocessing.StandardScaler().fit_transform(Z)

    return Z if to_numpy else torch.Tensor(Z).float()


def baseline_predictability(
    Y: torch.Tensor,
    factors: List[str],
    dataset: str,
) -> Dict[str, Dict[str, float]]:
    """
    Computes the baseline predictability of class-valued factors specified in `factors`.

    Args:
        Y: The array of factor values of all the factors in the dataset.
        factors: The factors whose baseline accuracies to compute.
        dataset: The string name of the dataset correctly map the factor names to the indices in `Y`.

    Returns:
        A dictionary of baseline accuracies and F1 scores of the specified factors.
    """

    results = dict()

    for attr in tqdm(factors):

        Y_c = preprocessing.LabelEncoder().fit_transform(Y[:, ds_utils.get_dataset_module(dataset).attribute2idx[attr]])

        counts = Counter(Y_c)
        majority = list(counts.keys())[np.argmax(counts.values())] * np.ones(Y_c.shape[0])
        acc, f1 = metrics.accuracy_score(Y_c, majority), metrics.f1_score(Y_c, majority, average="macro")
        results[attr] = {"accuracy": round(acc, 3), "f1": round(f1, 3)}

    return results


def evaluate_space_with_a_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    model: ScikitModel,
    mode: str = "class",
) -> Dict[str, float]:
    """
    Evaluates the subspace with representations `X_train` and `y_train` in terms of the predictability of the factor
    captured in `y_train` and `y_val`.
    The predictability can be determined with a linear or nonlinear classification or regression model.
    Args:
        X_train: The training split of the representations to use for predicting.
        y_train: The training split of the factor/target to predict.
        X_val: The evaluation split of the representations to use for predicting.
        y_val: The validation split of the factor/target to predict.
        model: The model to use for predicting.
               Works with a general `sklearn` model.
        mode: Either "class" or "regression" depending on whether `y` contains class labels or real valued targets.

    Returns:
        A set of predictability measures of the provided factor.
        If mode = "class", it returns a dictionary of the accuracy and F1 scores on the train and the evaluation splits,
        and if mode = "regression", the function returns a dictionary with the train and evaluation mean squared error
        and R^2 measures.
    """

    # Scale the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # Fit the model
    model = model.fit(X_train, y_train)

    # Predict
    y_val_hat = model.predict(X_val)
    y_train_hat = model.predict(X_train)

    return (
        {
            "accuracy": round(metrics.accuracy_score(y_val, y_val_hat), 3),
            "f1": round(metrics.f1_score(y_val, y_val_hat, average="macro"), 3),
            "train_accuracy": round(metrics.accuracy_score(y_train, y_train_hat), 3),
            "train_f1": round(metrics.f1_score(y_train, y_train_hat, average="macro"), 3),
        }
        if mode == "class"
        else {
            "r2": round(metrics.r2_score(y_val, y_val_hat), 3),
            "mse": round(metrics.mean_squared_error(y_val, y_val_hat), 3),
            "train_r2": round(metrics.r2_score(y_train, y_train_hat), 3),
            "train_mse": round(metrics.mean_squared_error(y_train, y_train_hat), 3),
        }
    )


def mutual_information(
    Z: Union[torch.Tensor, np.ndarray],
    Y: Union[torch.Tensor, np.ndarray],
    factors: Optional[Union[Sequence[str], List[Sequence[str]]]] = None,
    dataset: Optional[str] = None,
    fitting_indices: Optional[List[int]] = None,
    A: Optional[torch.Tensor] = None,
    method: MIEstimationMethod = MIEstimationMethod.KSG,
    standardise: bool = True,
) -> Dict[Union[Sequence[str], str], float]:
    """
    Evaluates the representations `Z` (optionally projected with the projection matrix `A`) w.r.t. how much information
    they contain about the factors captured in `Y`.

    The mutual information can be estimated using any of the methods specified in `MIEstimationMethod`.

    Args:
        Z: The original representations.
        Y: The array of ground-truth factors of variations.
        fitting_indices: Optional list of indices of the datapoints to use for estimation.
        dataset: The string name of the dataset correctly map the factor names to the indices in `Y`.
        factors: A list of list.
                 Each individual list is the set of factors to evaluate the representations against, i.e., the function
                 will compute the amount of mutual information the representations contain about each of the set of
                 factors specified by the lists.
        A: An optional projection matrix to a linear subspace to investigate.
        method: Which method to use to compute the mutual information.
        standardise: Whether to standardise the data.

    Returns:
        A dictionary with the estimate of the mutual information in the representation for each set of the factors.
    """
    results = dict()

    if fitting_indices is None:
        # If no subset is provided, train on the entire dataset.
        fitting_indices = list(range(Z.shape[0]))

    Z = _get_Z(Z, A, standardise, to_numpy=True)

    if dataset is not None:
        targets = (
            [ds_utils.get_dataset_module(dataset).attribute2idx[f] for f in factor_group] for factor_group in factors
        )
    else:
        targets = [[ii] for ii in range(Y.shape[1])] if factors is not None else [list(range(Y.shape[1]))]

    # Loop through all the specified sets of factors and compute the mutual information
    for jj, target in enumerate(tqdm(targets)):

        if method == MIEstimationMethod.MIST:
            mi = model_comparison.find_common_subspaces_with_mutual_information(
                Z_1=torch.from_numpy(Z).float(),
                Z_2=torch.from_numpy(Y[:, target]).float(),
                fitting_indices=fitting_indices,
                subspace_sizes=(Z.shape[1], len(target)),
                standardise=standardise,
            )[-1]
        elif method == MIEstimationMethod.KSG:
            mi = ksg.estimate_mi_ksg(
                Z[fitting_indices],
                Y[fitting_indices][:, target].numpy() if isinstance(Y, torch.Tensor) else Y[fitting_indices][:, target],
                neighborhoods=(5,),
            )[5]
        elif method == MIEstimationMethod.NAIVE_KSG:
            mi = naive_ksg.mutual_information(
                (
                    Z[fitting_indices],
                    Y[fitting_indices][:, target].numpy()
                    if isinstance(Y, torch.Tensor)
                    else Y[fitting_indices][:, target],
                ),
                k=5,
            )
        elif method == MIEstimationMethod.SKLEARN:
            assert Z.shape[1] == 1
            assert len(target) == 1
            mi = feature_selection.mutual_info_regression(
                Z[fitting_indices],
                Y[fitting_indices][:, target].numpy() if isinstance(Y, torch.Tensor) else Y[fitting_indices][:, target],
            )[0]
        else:
            raise NotImplementedError

        # Save the computed mutual information
        key = factors[jj] if factors is not None else "Y"
        results[key] = round(mi, 3)

    return results


def evaluate_space_with_predictability(
    Z: torch.Tensor,
    Y: torch.Tensor,
    dataset: str,
    factors: List[str],
    fitting_indices: Optional[List[int]] = None,
    A: Optional[torch.Tensor] = None,
    nonlinear: bool = False,
    standardise: bool = True,
) -> Dict[str, float]:
    """
    Wrapper function around the `evaluate_space_with_a_model` function for the discrete target case.
    Args:
        Z: The latent representations of the entire dataset.
        Y: The ground-truth values of all the factors of the entire dataset.
        fitting_indices: Optional list of indices of the datapoints to use for estimation.
        A: Optional projection matrix to a linear subspace to evaluate.
        nonlinear: Whether to evaluate with a nonlinear model (an SVM classifier).
        dataset, factors, standardise: Parameters for the `evaluate_space_with_a_model` function.

    Returns:
        Dictionary of `evaluate_space_with_a_model` results for all factors specified in `factors`.
    """
    results = dict()

    Z = _get_Z(Z, A, standardise, to_numpy=True)

    if fitting_indices is None:
        # If no subset is provided, train on the entire dataset.
        fitting_indices = list(range(Z.shape[0]))

    for attr in tqdm(factors):

        Y_c = preprocessing.LabelEncoder().fit_transform(Y[:, ds_utils.get_dataset_module(dataset).attribute2idx[attr]])

        if nonlinear:
            r = evaluate_space_with_a_model(
                Z[fitting_indices],
                Y_c[fitting_indices],
                Z[list(set(range(len(Z))) - set(fitting_indices))],
                Y_c[list(set(range(len(Z))) - set(fitting_indices))],
                model=svm.SVC(C=0.1, random_state=1),
                mode="class",
            )

        else:
            r = evaluate_space_with_a_model(
                Z[fitting_indices],
                Y_c[fitting_indices],
                Z[list(set(range(len(Z))) - set(fitting_indices))],
                Y_c[list(set(range(len(Z))) - set(fitting_indices))],
                model=linear_model.SGDClassifier(loss="log_loss", random_state=1),
                mode="class",
            )

        results[attr] = round(r["accuracy"], 3)

    return results


def subspace_entropy(
    Z: torch.Tensor,
    fitting_indices: Optional[List[int]] = None,
    A: Optional[torch.Tensor] = None,
    standardise: bool = True,
) -> float:
    """
    Estimates the entropy of the (projected) representation space `Z` with `MIST`.

    Args:
        Z: The latent representations of the entire dataset.
        fitting_indices: Optional list of indices of the datapoints to use for estimation.
        A: Optional projection matrix to a linear subspace to evaluate.
        standardise: Whether to standardise the data.

    Returns:
        Estimate of the entropy.
    """

    Z = _get_Z(Z, A, standardise, to_numpy=False)

    return model_comparison.find_common_subspaces_with_mutual_information(
        Z_1=Z, Z_2=Z, fitting_indices=fitting_indices, subspace_sizes=(Z.shape[1], Z.shape[1])
    )[-1]


def subspace_mutual_information(
    Z: torch.Tensor,
    fitting_indices: List[int],
    A: torch.Tensor,
    standardise: bool = True,
) -> float:
    """
    Estimates the mutual information between a linear subspace defined by the projection matrix `A` and the original
    representation space `Z` with `MIST`.

    Args:
        Z: The latent representations of the entire dataset.
        fitting_indices: Optional list of indices of the datapoints to use for estimation.
        A: Optional projection matrix to a linear subspace to evaluate.
        standardise: Whether to standardise the data.

    Returns:
        Estimate of the mutual information.
    """

    Z_p = _get_Z(Z, A, standardise, to_numpy=False)

    return model_comparison.find_common_subspaces_with_mutual_information(
        Z_1=Z, Z_2=Z_p, fitting_indices=fitting_indices, subspace_sizes=(Z.shape[1], Z_p.shape[1])
    )[-1]


def subspace_latent_traversals(
    Z: torch.Tensor,
    A: torch.Tensor,
    model: VAE,
    X: Optional[torch.Tensor] = None,
    starting_x: Optional[torch.Tensor] = None,
    standardise: bool = True,
    rng: ds_utils.RandomGenerator = 1,
    device: str = "cuda",
    file_name: Optional[str] = None,
) -> None:
    """
    A wrapper function to traverse the latent subspace defined by the projection matrix `A`.
    Args:
        Z: The latent representations of the entire dataset.
        A: Optional projection matrix to a linear subspace whose traversals to produce.
        model: The VAE to use for embedding and reconstructing.
        X: The dataset of the original images to select a starting image from.
        standardise: Whether to standardise the data.
        rng: The random generator to generate the initial sample randomly if one is not provided by `starting_x`.
        starting_x, device, file_name: Arguments for the `latent_traversals` function in `images.py`.

    """

    if starting_x is None:
        assert X is not None
        rng = np.random.default_rng(rng)
        starting_x = X[rng.choice(len(X))].to(device)

    Z = _get_Z(Z, A, standardise, to_numpy=False)

    image_visualisation.latent_traversals(
        vae_model=model,
        A_hat=A.cuda(),
        Z=Z,
        starting_x=starting_x.cuda(),
        n_values=8,
        n_axes=Z.shape[1],
        file_name=file_name,
    )


def subspace_heatmaps(
    Z: torch.Tensor,
    Y: torch.Tensor,
    factor_names: List[str],
    discrete_factors: Optional[List[int]] = None,
    A: Optional[torch.Tensor] = None,
    standardise: bool = True,
    target: str = "ground-truth",
    interpolate: bool = True,
    N: int = 5000,
    k_continuous: int = 250,
    k_discrete: int = 250,
    q: float = 0.25,
    file_name: Optional[str] = None,
) -> None:
    """
    Splot the (sub)space heatmaps; either using `factor_trend` or `factor_heatmap` in the `visualisation.py` module.
    Args:
        Z: The original representations.
        Y: The ground-truth values of the factors of variation.
        factor_names: The names of the factors whose heatmaps to plot.
        discrete_factors: List of indices of discrete factors.
                          If None, it is assumed all factors are continuous.
        A: Optional projection matrix to a linear subspace whose heatmaps to produce.
        standardise: Whether to standardise the data.
        target, interpolate, N, k_continuous, k_discrete, q, file_name: parameters for the `factor_heatmap` or
                                                                        `factor_trend` functions.

    """

    Z = _get_Z(Z, A, standardise, to_numpy=False)

    assert Z.shape[1] <= 2

    plotting_method = (
        latent_space_visualisation.factor_heatmap if Z.shape[1] == 2 else latent_space_visualisation.factor_trend
    )

    plotting_method(
        Z,
        Y,
        target=target,
        discrete_factors=discrete_factors,
        interpolate=interpolate,
        N=N,
        k_continuous=k_continuous,
        k_discrete=k_discrete,
        q=q,
        attribute_names=factor_names,
        file_name=file_name,
    )
