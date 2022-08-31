# TODO (Anej): Documentation

from typing import List, Dict, Protocol, Any, Tuple, Optional

import torch
from sklearn import metrics, preprocessing, feature_selection, linear_model, svm

from tqdm import tqdm

from pythae.models import VAE

from latte.models.cca import estimation as cca_estimation
from latte.utils import ksg_estimator, visualisation
from latte.dataset import utils as ds_utils
from latte.tools import model_comparison
from latte import ksg


class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None) -> Any:
        ...

    def predict(self, X) -> Any:
        ...


def evaluate_with_a_model(
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


def _train_cca(
    Z_1: torch.Tensor, Z_2: torch.Tensor, ixs: List[int], subspace_size: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    Z_1 = preprocessing.StandardScaler().fit_transform(Z_1.numpy())
    Z_2 = preprocessing.StandardScaler().fit_transform(Z_2.numpy())

    cca_result = cca_estimation.find_subspace(Z_1[ixs], Z_2[ixs], subspace_size)

    return cca_result.A_1, cca_result.E_1, cca_result.A_2, cca_result.E_2


def evaluate_space_with_mutual_information(
    Z: torch.Tensor,
    Y: torch.Tensor,
    ixs: List[int],
    dataset: str,
    A: Optional[torch.Tensor] = None,
    method: str = "latte-ksg",
) -> Dict[str, float]:
    results = dict()

    Z = (
        preprocessing.StandardScaler().fit_transform(Z.numpy()) @ A.detach().cpu().numpy()
        if A is not None
        else Z.numpy()
    )
    Z = preprocessing.StandardScaler().fit_transform(Z)

    for attribute in tqdm(ds_utils.get_dataset_module(dataset).attribute_names):

        if method == "mist":
            mi = model_comparison.find_subspaces_with_mutual_information(
                torch.from_numpy(Z).float(),
                Y[:, [ds_utils.get_dataset_module(dataset).attribute2idx[attribute]]],
                ixs,
                (Z.shape[1], 1),
                standardise=False,
            )[-1]
        elif method == "latte-ksg":
            mi = ksg.estimate_mi_ksg(
                Z[ixs],
                Y[ixs][:, [ds_utils.get_dataset_module(dataset).attribute2idx[attribute]]].numpy(),
                neighborhoods=(5,),
            )[5]
        elif method == "ksg":
            mi = ksg_estimator.mutual_information(
                (
                    Z[ixs],
                    Y[ixs][:, [ds_utils.get_dataset_module(dataset).attribute2idx[attribute]]].numpy(),
                ),
                k=5,
            )
        elif method == "sklearn":
            assert Z.shape[1] == 1
            mi = feature_selection.mutual_info_regression(
                Z[ixs], Y[ixs][:, ds_utils.get_dataset_module(dataset).attribute2idx[attribute]].numpy()
            )[0]
        else:
            raise NotImplementedError

        results[attribute] = round(mi, 3)

    return results


def evaluate_space_with_predictability(
    Z: torch.Tensor,
    Y: torch.Tensor,
    ixs: List[int],
    dataset: str,
    A: Optional[torch.Tensor] = None,
    nonlinear: bool = False,
) -> Dict[str, float]:
    results = dict()

    Z_projected = (
        torch.from_numpy(preprocessing.StandardScaler().fit_transform(Z.numpy())).float() @ A.detach().cpu()
        if A is not None
        else Z
    )
    Z_projected = torch.from_numpy(preprocessing.StandardScaler().fit_transform(Z_projected)).float()

    for attr in tqdm(ds_utils.get_dataset_module(dataset).attribute_names):

        Y_c = preprocessing.LabelEncoder().fit_transform(Y[:, ds_utils.get_dataset_module(dataset).attribute2idx[attr]])

        if nonlinear:
            r = evaluate_with_a_model(
                Z_projected[ixs],
                Y_c[ixs],
                Z_projected[list(set(range(len(Z))) - set(ixs))],
                Y_c[list(set(range(len(Z))) - set(ixs))],
                model=svm.SVC(C=0.1, random_state=1),
                mode="class",
            )

        else:
            r = evaluate_with_a_model(
                Z_projected[ixs],
                Y_c[ixs],
                Z_projected[list(set(range(len(Z))) - set(ixs))],
                Y_c[list(set(range(len(Z))) - set(ixs))],
                model=linear_model.SGDClassifier(loss="log_loss", random_state=1),
                mode="class",
            )

        results[attr] = round(r["accuracy"], 3)

    return results


def subspace_entropy(
    Z: torch.Tensor,
    A_hat: torch.Tensor,
    ixs: List[int],
) -> float:

    Z = torch.from_numpy(preprocessing.StandardScaler().fit_transform(Z.numpy())).float() @ A_hat.detach().cpu()

    return model_comparison.find_subspaces_with_mutual_information(Z, Z, ixs, (Z.shape[1], Z.shape[1]))[-1]


def subspace_mutual_information(
    Z: torch.Tensor,
    A_hat: torch.Tensor,
    ixs: List[int],
) -> float:

    Z_p = torch.from_numpy(preprocessing.StandardScaler().fit_transform(Z.numpy())).float() @ A_hat.detach().cpu()

    return model_comparison.find_subspaces_with_mutual_information(Z, Z_p, ixs, (Z.shape[1], Z_p.shape[1]))[-1]


def subspace_latent_traversals(
    starting_x: torch.Tensor, Z: torch.Tensor, A_hat: torch.Tensor, trained_model: VAE, file_name: Optional[str] = None
) -> None:

    Z = torch.from_numpy(preprocessing.StandardScaler().fit_transform(Z.numpy())).float() @ A_hat.detach().cpu()
    Z = torch.from_numpy(preprocessing.StandardScaler().fit_transform(Z)).float()

    visualisation.latent_traversals(
        vae_model=trained_model,
        A_hat=A_hat.cuda(),
        Z=Z,
        starting_x=starting_x.cuda(),
        n_values=8,
        n_axes=Z.shape[1],
        file_name=file_name,
    )


def subspace_heatmaps(
    Z: torch.Tensor,
    Y: torch.Tensor,
    A_hat: torch.Tensor,
    dataset: str,
    interpolate: bool = True,
    N: int = 5000,
) -> None:

    Z = torch.from_numpy(preprocessing.StandardScaler().fit_transform(Z.numpy())).float() @ A_hat.detach().cpu()
    Z = torch.from_numpy(preprocessing.StandardScaler().fit_transform(Z)).float()

    visualisation.factor_heatmap(
        Z,
        Y,
        target="ground-truth",
        interpolate=interpolate,
        N=N,
        k=50,
        q=0.25,
        attribute_names=ds_utils.get_dataset_module(dataset).attribute_names,
    )
