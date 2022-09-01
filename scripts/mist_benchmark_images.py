# TODO (Anej): If 2D or 3D, add heatmaps and latent space plots with factor values (heatmaps and from the notebook)
"""
This script trains a supervised MIST model on a supported dataset of images.
Currently, it works for the `CelebA` and `Shapes3D` datasets.
It takes the representations of a pre-trained VAE and discovers in them the subspaces of specified dimensionality
capturing the most information about a set of given attributes/latent factors.
The results are quantified, plotted, and saved.
"""
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, replace, field
from collections import Counter

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import geoopt

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics, linear_model, svm

from pythae.models import (
    BetaVAE,
    BetaTCVAE,
    AE,
    FactorVAE,
    IWAE,
    INFOVAE_MMD,
    VAMP,
    DisentangledBetaVAE,
    RHVAE,
    VAE,
    AutoModel,
)
from pythae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig

from latte.dataset import celeba, shapes3d, utils as dsutils
from latte.models.vae import utils as vaeutils
from latte.utils import evaluation, visualisation, ksg_estimator
from latte.models.mist import estimation as mist_estimation
from latte.models.rlace import estimation as rlace_estimation
from latte.models.linear_regression import estimation as linear_regression_estimation
from latte.models.vae.projection_vae import ProjectionBaseEncoder, ProjectionBaseDecoder
from latte.utils import space_evaluation
import latte.hydra_utils as hy


@hy.config
@dataclass
class MISTConfig:
    """
    The configuration of the script.

    Members:
        dataset: The dataset to analyse
        file_paths_x: Paths to the train, validation, and test image datasets, in this order.
        file_paths_y: Paths to the train, validation, and test attribute datasets, in this order.
        vae_model_file_path: Path to the file of the trained VAE model.
        factors_of_interest: A list of lists.
                             Each element of the outer list is a list of elements for which we want to find the linear
                             subspace capturing the most information about them *together*.
                             Multiple such lists can be provided to explore subspaces capturing information about
                             different sets of factors.
        alternative_factors: A list of lists.
                             After the subspace containing the most information about a certain set of factors has been
                             found, the script will check how much information there is in that subspace about all the
                             factors specified in `alternative_factors`.
                             It should be of the same length as `factors_of_interest`.
                             It for each set of factors in `factors_of_interest`, it will check how much information
                             there is the found space about each of the factors specified in the corresponding list in
                             `alternative_factors`.
        subspace_sizes: A list of dimensionalities of the subspaces to consider.
                        For each set of factors of interest, a subspace of each size specified in `subspace_sizes` will
                        be computed.
        training_fractions: The list of the different fractions of all the entire dataset to use to fit the MIST model.
                        This is useful to inspect the sensitivity of the model to the number of training points.
        training_Ns: The list of the different numbers of data points to use to fit the MIST model.
                             This is useful to inspect the sensitivity of the model to the number of training points.
                             It overrides `training_fractions`
        model_class: The VAE flavour to use.
                     Currently supported: "BetaVAE", "BetaTCVAE", "FactorVAE".
        early_stopping_min_delta: Min delta parameter for the early stopping callback of the MIST model.
        early_stopping_patience: Patience parameter for the early stopping callback of the MIST model.
        mine_network_width: The script uses the default implemented MINE statistics network.
                            This specified the width to be used.
        mist_batch_size: Batch size for training MIST.
        mine_learning_rate: Learning rate for training `MINE`.
                                       Should be around `1e-4`.
        manifold_learning_rate: Learning rate for optimising over the Stiefel manifold of projection matrices.
                                Should be larger, around `1e-2`.
        mist_max_epochs: Maximum number of epochs for training MIST.
        lr_scheduler_patience: Patience parameter for the learning rate scheduler of the MIST model.
        lr_scheduler_min_delta: Min delta parameter for the learning rate scheduler of the MIST model.
        n_gmm_components: Number of GMM components to use when fitting a GMM model to the marginal distributions
                          over the latent factors given the encoder.
                          This is used for sampling.
        plot_nrows: Number of rows to plot when plotting generated images and reconstructions by the VAE.
        plot_ncols: Number of columns to plot when plotting generated images and reconstructions by the VAE.
        plot_n_images: Number of images to use for homotopies.
        seed: The random seed.
        gpus: The `gpus` parameter for Pytorch Lightning.
        num_workers:  The `num_workers` parameter for Pytorch Lightning data modules.
    """

    dataset: str
    file_paths_x: Tuple[str, str, str]
    file_paths_y: Tuple[str, str, str]
    vae_model_file_path: str
    factors_of_interest: List[List[str]] = field(
        default_factory=lambda: [
            ["Male"],
            ["Young"],
            ["Smiling"],
            ["Heavy_Makeup"],
            ["Blond_Hair"],
        ]
    )
    alternative_factors: List[List[str]] = field(default_factory=lambda: [[], [], [], [], []])
    subspace_sizes: List[int] = field(default_factory=lambda: [1, 2])
    training_fractions: Optional[List[float]] = None
    training_Ns: Optional[List[int]] = field(default_factory=lambda: [10000])
    model_class: str = "BetaVAE"
    early_stopping_min_delta: float = 5e-3
    early_stopping_patience: int = 12
    mine_network_width: int = 256
    mist_batch_size: int = 128
    mine_learning_rate: float = 1e-4
    manifold_learning_rate: float = 1e-2
    mist_max_epochs: int = 256
    lr_scheduler_patience: int = 6
    lr_scheduler_min_delta: float = 5e-3
    n_gmm_components: int = 8
    plot_nrows: int = 8
    plot_ncols: int = 12
    plot_n_images: int = 4
    seed: int = 1
    gpus: int = 1
    num_workers: int = 6


def _load_data(
    cfg: MISTConfig,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Load the data of the specified dataset.
    Args:
        cfg: The configuration object of the experiment.

    Returns:
        Two tuples; the first one is the observation data split into the train, validation, and test splits, and the
        second one is the latent-factor data split into the same splits.
    """
    print("Loading the data.")

    X_train = torch.from_numpy(np.load(cfg.file_paths_x[0])).float()
    Y_train = torch.from_numpy(np.load(cfg.file_paths_y[0])).float()
    X_val = torch.from_numpy(np.load(cfg.file_paths_x[1])).float()
    Y_val = torch.from_numpy(np.load(cfg.file_paths_y[1])).float()
    X_test = torch.from_numpy(np.load(cfg.file_paths_x[2])).float()
    Y_test = torch.from_numpy(np.load(cfg.file_paths_y[2])).float()

    if cfg.dataset in ["celeba", "shapes3d"]:
        X_train /= 255
        X_val /= 255
        X_test /= 255

    print("Data loaded.")
    return (X_train, X_val, X_test), (Y_train, Y_val, Y_test)


def _get_dataset_module(cfg: MISTConfig) -> Any:
    """Returns the module to use for various utilities according to the dataset."""
    if cfg.dataset == "celeba":
        return celeba
    elif cfg.dataset == "shapes3d":
        return shapes3d


def _train_mist(
    Z: torch.Tensor, Y: torch.Tensor, subspace_size: int, fit_subspace: bool, cfg: MISTConfig
) -> mist_estimation.MISTResult:
    """
    Trains a MIST model to find the optimal subspace of dimensionality `subspace_size` about the factors captured in `Y`
    Args:
        Z: Full VAE representations
        Y: Ground-truth factor values
        subspace_size: The current size of the subspace considered.
        fit_subspace: Whether subspace is to be fit (and the train/test fractions should be different)
        cfg: The configuration object of the experiment.

    Returns:
        The MIST estimation result

    """

    print("Training MIST.")

    p_train, p_val = (0.4, 0.2) if fit_subspace else (0.7, 0.1)

    Z = torch.from_numpy(StandardScaler().fit_transform(Z)).float()

    mi_estimation_result = mist_estimation.find_subspace(
        X=Z,
        Z_max=Y,
        datamodule_kwargs=dict(
            num_workers=cfg.num_workers, batch_size=cfg.mist_batch_size, p_train=p_train, p_val=p_val
        ),
        model_kwargs=dict(
            x_size=Z.shape[1],
            z_max_size=Y.shape[1],
            subspace_size=subspace_size,
            mine_network_width=cfg.mine_network_width,
            mine_learning_rate=cfg.mine_learning_rate,
            manifold_learning_rate=cfg.manifold_learning_rate,
            lr_scheduler_patience=cfg.lr_scheduler_patience,
            lr_scheduler_min_delta=cfg.lr_scheduler_min_delta,
            verbose=False,
        ),
        trainer_kwargs=dict(
            max_epochs=cfg.mist_max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            gpus=cfg.gpus,
        ),
    )

    print("Training MIST done.")

    return mi_estimation_result


# TODO (Anej)
def _get_model_class(vae_flavour: str) -> Any:  # noqa: C901
    """Returns the appropriate model class according to the specification."""
    if vae_flavour == "RHVAE":
        return RHVAE
    elif vae_flavour == "BetaVAE":
        return BetaVAE
    elif vae_flavour == "BetaTCVAE":
        return BetaTCVAE
    elif vae_flavour == "AE":
        return AE
    elif vae_flavour == "FactorVAE":
        return FactorVAE
    elif vae_flavour == "IWAE":
        return IWAE
    elif vae_flavour == "INFOVAE_MMD":
        return INFOVAE_MMD
    elif vae_flavour == "VAMP":
        return VAMP
    elif vae_flavour == "DisentangledBetaVAE":
        return DisentangledBetaVAE
    else:
        raise NotImplementedError


def _process_alternative_factors(
    Z: Dict[str, torch.Tensor],
    Z_train_projected: torch.Tensor,
    Y: Dict[str, torch.Tensor],
    model: VAE,
    method: str,
    factors: List[str],
    alternative_factors: List[str],
    subspace_size: int,
    estimation_N: int,
    experiment_results: Dict[str, float],
    cfg: MISTConfig,
) -> None:

    for alternative_factor in alternative_factors:
        print(f"Checking how much information about {alternative_factor} was also captured in the subspace.")
        print("Training MIST on the entire VAE space.")
        full_alternative_factors_result = _train_mist(
            Z=Z["train"],
            Y=Y["train"][:, [_get_dataset_module(cfg).attribute2idx[alternative_factor]]],
            subspace_size=model.model_config.latent_dim,
            fit_subspace=False,
            cfg=cfg,
        )
        full_alternative_factors_mi = full_alternative_factors_result.mutual_information
        print(f"The information about {alternative_factor} in the full VAE space is {full_alternative_factors_mi}")
        experiment_results[f"I({alternative_factor}, VAE | {method} N={estimation_N})"] = full_alternative_factors_mi

        print(f"Training MIST for {alternative_factor} on the {subspace_size}-dimensional subspace of {factors}.")
        alternative_factors_subspace_result = _train_mist(
            Z=Z_train_projected,
            Y=Y["train"][:, [_get_dataset_module(cfg).attribute2idx[alternative_factor]]],
            subspace_size=subspace_size,
            fit_subspace=False,
            cfg=cfg,
        )
        alternative_factors_mi = alternative_factors_subspace_result.mutual_information
        print(f"The information about {alternative_factor} in the found space is {alternative_factors_mi}")
        experiment_results[
            f"I({alternative_factor}, {subspace_size}-{factors}-subspace | {method} N={estimation_N})"
        ] = alternative_factors_mi


def _compute_predictability_of_attributes(
    Z_val_projected: torch.Tensor,
    Y_val: torch.Tensor,
    predictor_train_ixs: List[int],
    method: str,
    factor: str,
    subspace_size: int,
    erased: bool,
    estimation_N: int,
    experiment_results: Dict[str, float],
    cfg: MISTConfig,
) -> None:

    # If we are only looking at a single factor, investigate its predictability
    le = LabelEncoder().fit(Y_val[:, _get_dataset_module(cfg).attribute2idx[factor]])

    Y_val_c = le.transform(Y_val[:, _get_dataset_module(cfg).attribute2idx[factor]])

    counts = Counter(Y_val_c)
    majority = list(counts.keys())[np.argmax(counts.values())] * np.ones(Y_val_c.shape[0])
    acc, f1 = metrics.accuracy_score(Y_val_c, majority), metrics.f1_score(Y_val_c, majority, average="macro")
    print(f"Naive baseline accuracy = {acc}.")
    print(f"Naive baseline F1 = {f1}.")
    experiment_results[f"Naive baseline accuracy ({factor})"] = acc
    experiment_results[f"Naive baseline F1 score ({factor})"] = f1

    for f in _get_dataset_module(cfg).attribute_names:
        ksg_mi_estimate = ksg_estimator.mutual_information(
            (Z_val_projected.numpy(), Y_val[:, [_get_dataset_module(cfg).attribute2idx[f]]].numpy()), k=3
        )
        print(f"KSG MI estimate ({f}) = {ksg_mi_estimate}.")
        experiment_results[
            f"KSG MI estimate ({f} | {method}, {subspace_size}, {estimation_N}, {factor}, erased = {erased})"
        ] = ksg_mi_estimate

    for model_type, model in zip(
        ["linear", "nonlinear"],
        [linear_model.SGDClassifier(loss="log_loss", random_state=1), svm.SVC(C=0.1, random_state=1)],
    ):
        prediction_result = space_evaluation.evaluate_space_with_a_model(
            Z_val_projected[predictor_train_ixs],
            Y_val_c[predictor_train_ixs],
            Z_val_projected[list(set(range(len(Z_val_projected))) - set(predictor_train_ixs))],
            Y_val_c[list(set(range(len(Z_val_projected))) - set(predictor_train_ixs))],
            model=model,
            mode="class",
        )
        print(f"Prediction result with {model_type} = {prediction_result}")
        for key, value in prediction_result.items():
            experiment_results[
                f"Prediction {key} ({method}, {subspace_size}, {estimation_N}, {factor}, erased = {erased})"
            ] = value


def _process_subspace(
    X: Dict[str, torch.Tensor],
    Z: Dict[str, torch.Tensor],
    Y: Dict[str, torch.Tensor],
    predictor_train_ixs: List[int],
    A_hat: geoopt.ManifoldTensor,
    model: VAE,
    method: str,
    factors: List[str],
    alternative_factors: List[str],
    subspace_size: int,
    erased: bool,
    estimation_N: int,
    experiment_results: Dict[str, float],
    cfg: MISTConfig,
    device: str,
    rng: dsutils.RandomGenerator,
) -> None:
    """
    Takes in a projection matrix onto a linear subspace of the full VAE representation space, construct a VAE which
    projects onto that space, and analyses the results by doing latent traversals, calculating the mutual information
    with regard to each of the axes, and sampling from the subspace.
    Args:
        X: Dictionary of the observations split into "train", "val", and "test" splits
        Z: Dictionary of the full VAE representations split into "train", "val", and "test" splits
        Y: Dictionary of the ground-truth factor values split into "train", "val", and "test" splits
        A_hat: The projection matrix.
        model: The original trained model
        factors: The factors of interest
        alternative_factors: The set of factors of interest to consider.
        subspace_size: The current size of the subspace considered.
                       This is needed since in the case when looking at the "erasing" projection matrix, the matrix
                       will already be of dimensionality `n x n`, where `n` is the size of the full representation space
                       This means we can not get the dimensionality of the subspace it is erasing just from the size
                       of the matrix.
        erased: Whether the A_hat projects onto the complement of the maximum-information subspace or not.
        estimation_N: Number of data points used for training.
        experiment_results: The dictionary in which to store the results of the experiment.
        cfg: The configuration object of the experiment.
        device: The device to load the data to.
        rng: The random generator.
    """
    rng = np.random.default_rng(rng)

    # `subspace_size` only refers to the dimensionality of the subspace the "maximisation matrix" projects onto
    # In case this function is called with the "erasing" matrix, we keep that dimensionality for logging,
    # but we need to actually use the entire space, wince the erasing matrix is square
    projection_subspace_size = A_hat.shape[1]

    # Construct the VAE model which projects onto the subspace defined by `A_hat`
    # It has the same config as the original model, so we copy it, we just change the `latent_dim`
    ModelClass = _get_model_class(cfg.model_class)
    projection_model = ModelClass(
        model_config=replace(model.model_config),
        encoder=ProjectionBaseEncoder(model.encoder, A_hat),
        decoder=ProjectionBaseDecoder(model.decoder, A_hat),
    )
    projection_model.model_config.latent_dim = projection_subspace_size

    # Construct the representations by the projection VAE
    Z_train_projected = Z["train"] @ A_hat.detach().cpu()
    Z_val_projected = Z["val"] @ A_hat.detach().cpu()

    projected_scaler = StandardScaler().fit(Z_train_projected)
    Z_train_projected = torch.from_numpy(projected_scaler.transform(Z_train_projected)).float()
    Z_val_projected = torch.from_numpy(projected_scaler.transform(Z_val_projected)).float()

    # Plot the latent traversals in the subspace
    visualisation.latent_traversals(
        vae_model=projection_model,
        Z=Z_train_projected,
        starting_x=X["val"][rng.choice(len(X["val"]))].to(device),
        n_values=cfg.plot_ncols,
        n_axes=min(projection_subspace_size, cfg.plot_nrows),
        file_name=f"lat_trav_erased{erased}_{method}_{subspace_size}_{factors}_{estimation_N}.png",
    )

    # Plot homotopies
    starting_xs = [X[rng.choice(len(X))] for _ in range(cfg.plot_n_images)]
    visualisation.homotopies(
        vae_model=projection_model,
        xs=[x.to(device) for x in starting_xs],
        n_cols=cfg.plot_ncols,
        device=device,
        file_name="homotopies_erased{erased}_{method}_{subspace_size}_{factors}_{estimation_N}.png",
    )

    if not erased:
        if subspace_size == 2:
            for ii in range(cfg.plot_nrows):
                visualisation.latent_traversals_2d(
                    vae_model=model,
                    starting_x=X["val"][rng.choice(len(X["val"]))].to(device),
                    axis=(0, 1),
                    A_hat=A_hat,
                    Z=Z_train_projected,
                    n_values=cfg.plot_ncols,
                    q=((0.000001, 0.999999), (0.000001, 0.999999)),
                    device=device,
                    cmap=None,
                    file_name=f"lat_trav2d_intervened_{method}_{subspace_size}_{factors}_{estimation_N}_{ii}.png",
                )

        # Intervene on the values of the subspace on a sample of the images to change their features
        for ii in range(cfg.plot_nrows):
            visualisation.latent_traversals(
                vae_model=model,
                Z=Z_train_projected,
                A_hat=A_hat,
                starting_x=X["val"][rng.choice(len(X["val"]))].to(device),
                n_values=cfg.plot_ncols,
                q=(0.000001, 0.999999),
                device=device,
                cmap=None,
                n_axes=min(projection_subspace_size, cfg.plot_nrows),
                file_name=f"lat_trav_intervened_{method}_{subspace_size}_{factors}_{estimation_N}_{ii}.png",
            )

        # Only calculate this for the projected subspace, since the number of dimensions it too large otherwise, so
        # it takes too long
        print("Calculating the mutual information about the factors captured in the axes of the projected subspace.")
        axis_mi = evaluation.axis_factor_mutual_information(
            Z_val_projected, Y["val"], factor_names=_get_dataset_module(cfg).attribute_names
        )
        print(f"Average amount of information captured about:\n {axis_mi.mean(axis=1).sort_values(ascending=False)}.")
        print(f"Maximum amount of information captured about:\n {axis_mi.max(axis=1).sort_values(ascending=False)}.")
        axis_mi.to_csv(f"axis_factor_mi_{method}_{subspace_size}_{factors}_{estimation_N}.csv")

        _process_alternative_factors(
            Z,
            Z_train_projected,
            Y,
            model,
            method,
            factors,
            alternative_factors,
            subspace_size,
            estimation_N,
            experiment_results,
            cfg,
        )

    # Reconstructions together with their original non-projected reconstructions
    visualisation.vae_reconstructions(
        dataset=X["val"],
        vae_model=model,
        latent_transformation=lambda z: z @ A_hat @ A_hat.T,
        nrows=cfg.plot_nrows,
        ncols=cfg.plot_ncols,
        file_name=f"reconstructions_transformation_{method}"
        f"_{'erased' if erased else 'projected'}_{subspace_size}_{factors}_{estimation_N}.png",
    )

    # Construct a sampler from the subspace defined by the projection matrix
    print("Fitting a GMM on the projected subspace.")
    projection_gmm_sampler = GaussianMixtureSampler(
        sampler_config=GaussianMixtureSamplerConfig(n_components=cfg.n_gmm_components), model=projection_model
    )
    projection_gmm_sampler.fit(X["val"])
    print("Fitting a GMM on the projected subspace done.")

    visualisation.generated_images(
        projection_gmm_sampler.sample(num_samples=cfg.plot_nrows * cfg.plot_ncols),
        cfg.plot_nrows,
        cfg.plot_ncols,
        file_name=f"gen_{method}_{'erased' if erased else 'projected'}_{subspace_size}_{factors}_{estimation_N}.png",
    )

    print(f"Training MIST on the erased = {erased} {method} {subspace_size}-dimensional subspace for {factors}.")
    subspace_result = _train_mist(
        Z=Z_train_projected,
        Y=Y["train"][:, [_get_dataset_module(cfg).attribute2idx[factor_of_interest] for factor_of_interest in factors]],
        subspace_size=projection_subspace_size
        if erased
        else subspace_size,  # The matrix `A_hat` is square, so this is over the entire space
        fit_subspace=False,
        cfg=cfg,
    )
    mi = subspace_result.mutual_information
    print(f"The information about {factors} in the projected space is {mi}")
    experiment_results[f"I({factors}, erased={erased}-{subspace_size}-subspace | {method} N={estimation_N})"] = mi

    print(
        "Calculating entropy with MIST on the erased "
        f"= {erased} {method} {subspace_size}-dimensional subspace for {factors}."
    )
    entropy_result = _train_mist(
        Z=Z_train_projected,
        Y=Z_train_projected,
        subspace_size=projection_subspace_size
        if erased
        else subspace_size,  # The matrix `A_hat` is square, so this is over the entire space
        fit_subspace=False,
        cfg=cfg,
    )
    mi = entropy_result.mutual_information
    print(f"The entropy of the projected space is {mi}")
    experiment_results[f"H({factors} erased={erased}-{subspace_size}-subspace | {method} N={estimation_N})"] = mi

    if len(factors) == 1:
        _compute_predictability_of_attributes(
            Z_val_projected,
            Y["val"],
            predictor_train_ixs,
            method,
            factors[0],
            subspace_size,
            erased,
            estimation_N,
            experiment_results,
            cfg,
        )


def _get_fitting_Ns(cfg: MISTConfig, N: int) -> List[int]:
    """
    Returns the number of data points to use to train the MIST model.
    Note that the entire number of data points to use is larger because a large portion is used to *evaluate* the
    full mutual information in the end.
    Only 40 % of the provided data is used to train the MIST model when fitting the subspace.
    Args:
        cfg: The script configuration.
        N: The entire size of the training dataset.

    Returns:
        List of the number of data points to use for MIST subspace fitting.
    """
    if cfg.training_Ns is not None:
        return [int(2.5 * N) for N in cfg.training_Ns]
    else:
        return [int(2.5 * fraction * N) for fraction in cfg.training_fractions]


def _find_subspace(
    X: Dict[str, torch.Tensor],
    Z: Dict[str, torch.Tensor],
    Y: Dict[str, torch.Tensor],
    model: VAE,
    factors_of_interest: List[str],
    factors_of_interest_idx: List[int],
    alternative_factors: List[str],
    experiment_results: Dict[str, float],
    subspace_size: int,
    fitting_N: int,
    method: str,
    train_ixs: np.ndarray,
    cfg: MISTConfig,
    device: str,
    rng: dsutils.RandomGenerator,
) -> None:

    print(f"Determining the most informative {subspace_size}-dimensional subspace about {factors_of_interest}.")
    if method == "MIST":
        subspace_result = _train_mist(
            Z=Z["train"][train_ixs],
            Y=Y["train"][train_ixs][:, factors_of_interest_idx],  # Train on the chosen subset of the data
            subspace_size=subspace_size,
            fit_subspace=True,
            cfg=cfg,
        )
        # mi = subspace_result.mutual_information
    elif method == "rLACE":
        assert len(factors_of_interest_idx) == 1

        # default hyperparameters
        rlace_params = dict(
            out_iters=4000,
            rank=subspace_size,
            optimizer_class=torch.optim.SGD,
            optimizer_params_P={"lr": 0.005, "weight_decay": 1e-4, "momentum": 0.0},
            optimizer_params_predictor={"lr": 0.005, "weight_decay": 1e-5, "momentum": 0.9},
            epsilon=0.01,  # 1 %
            batch_size=256,
            device="cuda",
        )

        le = LabelEncoder().fit(Y["train"][:, factors_of_interest_idx[0]])

        Y_c = le.transform(Y["train"][:, factors_of_interest_idx[0]])
        subspace_result = rlace_estimation.find_subspace(Z["train"][train_ixs], Y_c[train_ixs], rlace_params)

    elif method == "Linear Regression":
        assert len(factors_of_interest_idx) == 1
        subspace_result = linear_regression_estimation.find_subspace(
            Z["train"][train_ixs].numpy(), Y["train"][train_ixs, factors_of_interest_idx[0]].numpy()
        )
    else:
        raise NotImplementedError

    print("\n --------------------------------------- \n\n")

    predictor_train_ixs = rng.choice(int(Z["val"].shape[0]), size=16000, replace=False)

    # Process and analyze the subspace defined by the projection matrix onto the optimal subspace
    _process_subspace(
        X=X,
        Z=Z,  # just standardize here... (you can even do it with a new StandardScaler)
        Y=Y,
        predictor_train_ixs=predictor_train_ixs,
        A_hat=subspace_result.A_1.to(device),
        model=model,
        method=method,
        factors=factors_of_interest,
        alternative_factors=alternative_factors,
        subspace_size=subspace_size,
        erased=False,
        estimation_N=fitting_N,
        experiment_results=experiment_results,
        cfg=cfg,
        device=device,
        rng=rng,
    )

    print("\n --------------------------------------- \n\n")

    # Process and analyze the subspace defined by the projection matrix onto the *complement* of the optimal
    # subspace, i.e., the subspace with as much as possible of the information removed
    # Note that here, we pass the *entire* training dataset, since we will evaluate the amount of retained
    # information on the entire dataset
    _process_subspace(
        X=X,
        Z=Z,
        Y=Y,
        predictor_train_ixs=predictor_train_ixs,
        A_hat=subspace_result.E_1.to(device),
        model=model,
        method=method,
        factors=factors_of_interest,
        alternative_factors=[],
        subspace_size=subspace_size,
        erased=True,
        estimation_N=fitting_N,
        experiment_results=experiment_results,
        cfg=cfg,
        device=device,
        rng=rng,
    )

    print("\n ------------------------------------------------------------------------------ \n\n\n")


def _fit_gmm(
    X: torch.Tensor,
    Z: torch.Tensor,
    trained_model: VAE,
    cfg: MISTConfig,
    experiment_results: Dict[str, float],
):

    # Fit a GMM model on the produced latent space
    print("Fitting the GMM on the original latent space.")
    gmm_sampler = GaussianMixtureSampler(
        sampler_config=GaussianMixtureSamplerConfig(n_components=cfg.n_gmm_components), model=trained_model
    )
    gmm_sampler.fit(X)
    print("Fitting the GMM on the original latent space done.")

    # Generate a sample of images based on the fit sampler, plot, and save them.
    visualisation.generated_images(
        gmm_sampler.sample(num_samples=cfg.plot_nrows * cfg.plot_ncols),
        cfg.plot_nrows,
        cfg.plot_ncols,
        file_name="generated_images_full_model.png",
    )

    # Compute the log likelihood of the validation data according to the sampler
    org_ll = gmm_sampler.gmm.score(Z)
    print(f"The log likelihood of the validation data is {org_ll}.")
    experiment_results["Original model validation log likelihood"] = org_ll


def _process_factors_of_interest(
    X: Dict[str, torch.Tensor],
    Z: Dict[str, torch.Tensor],
    Y: Dict[str, torch.Tensor],
    model: VAE,
    factors_of_interest: List[str],
    alternative_factors: List[str],
    experiment_results: Dict[str, float],
    cfg: MISTConfig,
    device: str,
    rng: dsutils.RandomGenerator,
) -> None:
    """
    Finds the optimal subspace capturing the most information about the set of factors of interest specified.
    It trains a MIST model on the *entire* VAE representation space to figure out how much information is captured
    in the entire space, and then find the optimal linear subspaces capturing as much information as possible about the
    factors, for each dimensionality specified in the config.
    Args:
        X: Dictionary of the observations split into "train", "val", and "test" splits
        Z: Dictionary of the full VAE representations split into "train", "val", and "test" splits
        Y: Dictionary of the ground-truth factor values split into "train", "val", and "test" splits
        model: The loaded trained VAE model to be applied to the data in `X`
        factors_of_interest: The set of factors of interest to consider.
        alternative_factors: The set of factors for which to check how much information was captured about them in the
                             found subspace.
        experiment_results: The dictionary in which to store the results of the experiment.
        cfg: The configuration object of the experiment.
        device: The device to load the data to.
        rng: The random generator.
    """

    # Compute the indices of the factors of interest in the ground-truth data matrix `Y`
    factors_of_interest_idx = [
        _get_dataset_module(cfg).attribute2idx[factor_of_interest] for factor_of_interest in factors_of_interest
    ]

    # Estimate the entropy of each of the factors on the validation split
    for ii, factor_of_interest_idx in enumerate(factors_of_interest_idx):
        H = mutual_info_regression(
            Y["val"][:, factor_of_interest_idx].reshape((-1, 1)), Y["val"][:, factor_of_interest_idx].flatten()
        )[0]
        print(f"H({factors_of_interest[ii]}) = {H}.")
        experiment_results[f"H({factors_of_interest[ii]})"] = H

    # To find out how much information about the factors of interest there is in the entire learned representation
    # space, we train a MIST model on the entire space
    # This is done on the *entire* dataset
    print(f"Training MIST to determine the information about {factors_of_interest} in the full VAE space.")
    full_space_result = _train_mist(
        Z=Z["train"],
        Y=Y["train"][:, factors_of_interest_idx],
        subspace_size=model.model_config.latent_dim,
        fit_subspace=False,
        cfg=cfg,
    )
    full_mi = full_space_result.mutual_information
    print(f"The information about {factors_of_interest} in the full VAE space is {full_mi}")
    experiment_results[f"I({factors_of_interest}, VAE)"] = full_mi

    # Get the number of data points the MIST model will be trained with
    fitting_Ns = _get_fitting_Ns(cfg, len(Z["train"]))
    print(
        f"Will fit the optimal subspaces with {fitting_Ns} data points "
        f"(40 %, {[int(0.4 * N) for N in fitting_Ns]}, of that for the training)."
    )

    # Next, we find, for each dimensionality specified in the config, the linear subspace capturing the most
    # information about these factors
    for subspace_size in cfg.subspace_sizes:

        # The MIST model is trained in different number of data points to test the sensitivity to the dataset size
        for fitting_N in fitting_Ns:

            # Get a random subset of the training dataset
            train_ixs = rng.choice(int(Z["train"].shape[0]), size=fitting_N, replace=False)

            # Train MIST on the subspace of size `subspace_size` with `fitting_N` data points
            print(
                f"Looking into the subspace of size {subspace_size} for factors {factors_of_interest} with "
                f"{fitting_N} data points ({0.4 * 100 * fitting_N / len(Z['train']):2.2f} % of the dataset)."
            )

            print(">>> INVESTIGATING MIST")
            _find_subspace(
                X=X,
                Z=Z,
                Y=Y,
                model=model,
                factors_of_interest=factors_of_interest,
                factors_of_interest_idx=factors_of_interest_idx,
                alternative_factors=alternative_factors,
                experiment_results=experiment_results,
                subspace_size=subspace_size,
                fitting_N=fitting_N,
                method="MIST",
                train_ixs=train_ixs,
                cfg=cfg,
                device=device,
                rng=rng,
            )
            print()
            print()
            print()

            print(">>> INVESTIGATING RLACE")
            _find_subspace(
                X=X,
                Z=Z,
                Y=Y,
                model=model,
                factors_of_interest=factors_of_interest,
                factors_of_interest_idx=factors_of_interest_idx,
                alternative_factors=alternative_factors,
                experiment_results=experiment_results,
                subspace_size=subspace_size,
                fitting_N=fitting_N,
                method="rLACE",
                train_ixs=train_ixs,
                cfg=cfg,
                device=device,
                rng=rng,
            )
            print()
            print()
            print()

            if subspace_size == 1:
                print(">>> INVESTIGATING LINEAR REGRESSION")
                _find_subspace(
                    X=X,
                    Z=Z,
                    Y=Y,
                    model=model,
                    factors_of_interest=factors_of_interest,
                    factors_of_interest_idx=factors_of_interest_idx,
                    alternative_factors=alternative_factors,
                    experiment_results=experiment_results,
                    subspace_size=subspace_size,
                    fitting_N=fitting_N,
                    method="Linear Regression",
                    train_ixs=train_ixs,
                    cfg=cfg,
                    device=device,
                    rng=rng,
                )


@hy.main
def main(cfg: MISTConfig):

    assert cfg.dataset in ["celeba", "shapes3d"], "The dataset not supported."
    assert (
        cfg.training_fractions is not None or cfg.training_Ns is not None
    ), "At least one of `training_fractions` and `mist_Ns` has to be specified."
    assert cfg.training_fractions is None or all(fraction < 0.4 for fraction in cfg.training_fractions), (
        "To get a good estimate of the mutual information, "
        "at most 40 % of the data can be used to fit the subspace with the MIST model."
    )

    # This will keep various metrics from the experiment
    experiment_results = dict()

    pl.seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    (X_train, X_val, X_test), (Y_train, Y_val, Y_test) = _load_data(cfg)

    device = "cuda" if cfg.gpus > 0 else "cpu"

    # Load the trained VAE
    trained_model = AutoModel.load_from_folder(cfg.vae_model_file_path).to(device)

    # Plot the original VAE reconstructions
    visualisation.vae_reconstructions(
        dataset=X_val,
        vae_model=trained_model,
        nrows=cfg.plot_nrows,
        ncols=cfg.plot_ncols,
        cmap=None,
        file_name="reconstructions_full_model.png",
    )

    # Get the latent representations given by the original VAE
    Z_train = vaeutils.get_latent_representations(trained_model, X_train)
    Z_val = vaeutils.get_latent_representations(trained_model, X_val)
    Z_test = vaeutils.get_latent_representations(trained_model, X_test)

    full_scaler = StandardScaler().fit(Z_train)
    Z_train = torch.from_numpy(full_scaler.transform(Z_train.numpy())).float()
    Z_val = torch.from_numpy(full_scaler.transform(Z_val.numpy())).float()
    Z_test = torch.from_numpy(full_scaler.transform(Z_test.numpy())).float()

    _fit_gmm(X=X_val, Z=Z_val, trained_model=trained_model, cfg=cfg, experiment_results=experiment_results)

    # Plot the latent traversals
    visualisation.latent_traversals(
        vae_model=trained_model,
        starting_x=X_val[rng.choice(len(X_val))].to(device),
        Z=Z_val,
        n_values=cfg.plot_ncols,
        n_axes=cfg.plot_nrows,
        cmap=None,
        file_name="latent_traversals_full_model.png",
    )

    # Plot homotopies
    starting_xs = [X_val[rng.choice(len(X_val))] for _ in range(cfg.plot_n_images)]
    visualisation.homotopies(
        vae_model=trained_model,
        xs=[x.to(device) for x in starting_xs],
        n_cols=cfg.plot_ncols,
        device=device,
        file_name="homotopies_full_model.png",
    )

    # Go through the specified factors of interest, and for each provided set, find the subspace capturing the most
    # information about then, and analyse it
    for ii in range(len(cfg.factors_of_interest)):
        print(f"Processing the set of factors {cfg.factors_of_interest[ii]}.")
        _process_factors_of_interest(
            {"train": X_train, "val": X_val, "test": X_test},
            {"train": Z_train, "val": Z_val, "test": Z_test},
            {"train": Y_train, "val": Y_val, "test": Y_test},
            trained_model,
            cfg.factors_of_interest[ii],
            cfg.alternative_factors[ii],
            experiment_results,
            cfg,
            device,
            rng,
        )

    print("Finished.")

    # Save the experiment results in a data frame
    results_df = pd.DataFrame({"Value": experiment_results})
    print("The final experiment results:")
    print(results_df)
    results_df.to_csv("experiment_results.csv")


if __name__ == "__main__":
    main()
