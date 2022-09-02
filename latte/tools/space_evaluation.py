# TODO (Anej): Documentation

from typing import List, Dict, Protocol, Any, Optional, Union, Tuple
from enum import Enum
from collections import Counter

import torch
import numpy as np
from sklearn import metrics, preprocessing, feature_selection, linear_model, svm

from tqdm import tqdm

from pythae.models import VAE

from latte.utils import ksg_estimator, visualisation
from latte.dataset import utils as ds_utils
from latte.tools import model_comparison
from latte import ksg


class MIEstimationMethod(Enum):
    MIST = "mist"
    LATTE_KSG = "latte-ksg"
    KSG = "ksg"
    SKLEARN = "sklearn"


class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None) -> Any:
        ...

    def predict(self, X) -> Any:
        ...


def _get_Z(
    Z: torch.Tensor,
    A: Optional[torch.Tensor] = None,
    standardise: bool = True,
    to_numpy: bool = False,
) -> Union[torch.Tensor, np.ndarray]:

    if standardise:
        Z = preprocessing.StandardScaler().fit_transform(Z.numpy())
    else:
        Z = Z.numpy()

    if A is not None:
        Z = Z @ A.detach().cpu().numpy()
        Z = preprocessing.StandardScaler().fit_transform(Z)

    return Z if to_numpy else torch.Tensor(Z).float()


def evaluate_space_with_a_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    model: ScikitModel,
    mode: str = "class",
) -> Dict[str, float]:

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    model = model.fit(X_train, y_train)

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


def evaluate_space_with_mutual_information(
    Z: torch.Tensor,
    Y: torch.Tensor,
    ixs: List[int],
    dataset: str,
    factors: List[Tuple[str]],
    A: Optional[torch.Tensor] = None,
    method: MIEstimationMethod = MIEstimationMethod.LATTE_KSG,
    standardise: bool = True,
) -> Dict[Tuple[str], float]:
    results = dict()

    Z = _get_Z(Z, A, standardise, to_numpy=True)

    for factor_group in tqdm(factors):

        if method == MIEstimationMethod.MIST:
            mi = model_comparison.find_common_subspaces_with_mutual_information(
                torch.from_numpy(Z).float(),
                Y[:, [ds_utils.get_dataset_module(dataset).attribute2idx[f] for f in factor_group]],
                ixs,
                (Z.shape[1], 1),
                standardise=False,
            )[-1]
        elif method == MIEstimationMethod.LATTE_KSG:
            mi = ksg.estimate_mi_ksg(
                Z[ixs],
                Y[ixs][:, [ds_utils.get_dataset_module(dataset).attribute2idx[f] for f in factor_group]].numpy(),
                neighborhoods=(5,),
            )[5]
        elif method == MIEstimationMethod.KSG:
            mi = ksg_estimator.mutual_information(
                (
                    Z[ixs],
                    Y[ixs][:, [ds_utils.get_dataset_module(dataset).attribute2idx[f] for f in factor_group]].numpy(),
                ),
                k=5,
            )
        elif method == MIEstimationMethod.SKLEARN:
            assert Z.shape[1] == 1
            assert len(factor_group) == 1
            mi = feature_selection.mutual_info_regression(
                Z[ixs], Y[ixs][:, ds_utils.get_dataset_module(dataset).attribute2idx[factor_group[0]]].numpy()
            )[0]
        else:
            raise NotImplementedError

        results[factor_group] = round(mi, 3)

    return results


def baseline_predictability(
    Y: torch.Tensor,
    dataset: str,
    factors_of_interest: List[str],
) -> Dict[str, Dict[str, float]]:

    results = dict()

    for attr in tqdm(factors_of_interest):

        Y_c = preprocessing.LabelEncoder().fit_transform(Y[:, ds_utils.get_dataset_module(dataset).attribute2idx[attr]])

        counts = Counter(Y_c)
        majority = list(counts.keys())[np.argmax(counts.values())] * np.ones(Y_c.shape[0])
        acc, f1 = metrics.accuracy_score(Y_c, majority), metrics.f1_score(Y_c, majority, average="macro")
        results[attr] = {"accuracy": round(acc, 3), "f1": round(f1, 3)}

    return results


def evaluate_space_with_predictability(
    Z: torch.Tensor,
    Y: torch.Tensor,
    ixs: List[int],
    dataset: str,
    factors_of_interest: List[str],
    A: Optional[torch.Tensor] = None,
    nonlinear: bool = False,
    standardise: bool = True,
) -> Dict[str, float]:
    results = dict()

    Z = _get_Z(Z, A, standardise, to_numpy=True)

    for attr in tqdm(factors_of_interest):

        Y_c = preprocessing.LabelEncoder().fit_transform(Y[:, ds_utils.get_dataset_module(dataset).attribute2idx[attr]])

        if nonlinear:
            r = evaluate_space_with_a_model(
                Z[ixs],
                Y_c[ixs],
                Z[list(set(range(len(Z))) - set(ixs))],
                Y_c[list(set(range(len(Z))) - set(ixs))],
                model=svm.SVC(C=0.1, random_state=1),
                mode="class",
            )

        else:
            r = evaluate_space_with_a_model(
                Z[ixs],
                Y_c[ixs],
                Z[list(set(range(len(Z))) - set(ixs))],
                Y_c[list(set(range(len(Z))) - set(ixs))],
                model=linear_model.SGDClassifier(loss="log_loss", random_state=1),
                mode="class",
            )

        results[attr] = round(r["accuracy"], 3)

    return results


def subspace_entropy(
    Z: torch.Tensor,
    ixs: List[int],
    A: Optional[torch.Tensor] = None,
    standardise: bool = True,
) -> float:

    Z = _get_Z(Z, A, standardise, to_numpy=False)

    return model_comparison.find_common_subspaces_with_mutual_information(Z, Z, ixs, (Z.shape[1], Z.shape[1]))[-1]


def subspace_mutual_information(
    Z: torch.Tensor,
    ixs: List[int],
    A: torch.Tensor,
    standardise: bool = True,
) -> float:

    Z_p = _get_Z(Z, A, standardise, to_numpy=False)

    return model_comparison.find_common_subspaces_with_mutual_information(Z, Z_p, ixs, (Z.shape[1], Z_p.shape[1]))[-1]


def subspace_latent_traversals(
    Z: torch.Tensor,
    A: torch.Tensor,
    trained_model: VAE,
    X: Optional[torch.Tensor] = None,
    starting_x: Optional[torch.Tensor] = None,
    standardise: bool = True,
    rng: ds_utils.RandomGenerator = 1,
    device: str = "cuda",
    file_name: Optional[str] = None,
) -> None:

    if starting_x is None:
        assert X is not None
        rng = np.random.default_rng(rng)
        starting_x = X[rng.choice(len(X))].to(device)

    Z = _get_Z(Z, A, standardise, to_numpy=False)

    visualisation.latent_traversals(
        vae_model=trained_model,
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
    attribute_names: List[str],
    A: Optional[torch.Tensor] = None,
    standardise: bool = True,
    target: str = "ground-truth",
    interpolate: bool = True,
    N: int = 5000,
    k: int = 250,
    q: float = 0.25,
    file_name: Optional[str] = None,
) -> None:

    Z = _get_Z(Z, A, standardise, to_numpy=False)

    assert Z.shape[1] <= 2

    plotting_method = visualisation.factor_heatmap if Z.shape[1] == 2 else visualisation.factor_trend

    plotting_method(
        Z,
        Y,
        target=target,
        interpolate=interpolate,
        N=N,
        k=k,
        q=q,
        attribute_names=attribute_names,
        file_name=file_name,
    )
