"""
This script trains a supervised MIST model on a supported dataset of images.
Currently, it works for the `CelebA` and `Shapes3D` datasets.
It takes the representations of a pre-trained VAE and discovers in them the subspaces of specified dimensionality
capturing the most information about a set of given attributes/latent factors.
The results are quantified, plotted, and saved.
"""
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, replace, field

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import geoopt

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler, LabelEncoder

from pythae.models import VAE, AutoModel
from pythae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig

from latte.dataset import utils as dsutils
from latte.models.vae import utils as vaeutils
from latte.utils import evaluation, visualisation
from latte.models.mist import estimation as mist_estimation
from latte.models.rlace import estimation as rlace_estimation
from latte.models.linear_regression import estimation as linear_regression_estimation
from latte.models.vae.projection_vae import ProjectionBaseEncoder, ProjectionBaseDecoder
from latte.tools import space_evaluation
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
    factors_of_interest: List[List[str]]
    alternative_factors: List[List[str]]
    model_class: str
    subspace_sizes: List[int] = field(default_factory=lambda: [1, 2])
    training_fractions: Optional[List[float]] = None
    training_Ns: Optional[List[int]] = field(default_factory=lambda: [8000])
    estimation_N: int = 16000
    early_stopping_min_delta: float = 5e-3
    early_stopping_patience: int = 12
    mine_network_width: int = 200
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


def _process_alternative_factors(
    Z: torch.Tensor,
    Z_p: torch.Tensor,
    Y: torch.Tensor,
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
        full_alternative_factors_result = _train_mist(
            Z=Z,
            Y=Y[:, [dsutils.get_dataset_module(cfg.dataset).attribute2idx[alternative_factor]]],
            subspace_size=model.model_config.latent_dim,
            fit_subspace=False,
            cfg=cfg,
        )
        full_alternative_factors_mi = full_alternative_factors_result.mutual_information
        print(f"The information about {alternative_factor} in the full VAE space is {full_alternative_factors_mi}")
        experiment_results[f"I({alternative_factor}, VAE | {method} N={estimation_N})"] = full_alternative_factors_mi

        print(f"Training MIST for {alternative_factor} on the {subspace_size}-dimensional subspace of {factors}.")
        alternative_factors_subspace_result = _train_mist(
            Z=Z_p,
            Y=Y[:, [dsutils.get_dataset_module(cfg.dataset).attribute2idx[alternative_factor]]],
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
    Z: torch.Tensor,
    Y: torch.Tensor,
    method: str,
    factors: List[str],
    subspace_size: int,
    erased: bool,
    estimation_N: int,
    experiment_results: Dict[str, float],
    cfg: MISTConfig,
    rng: dsutils.RandomGenerator,
) -> None:

    baseline_scores = space_evaluation.baseline_predictability(Y, cfg.dataset, factors)

    predictor_train_ixs = rng.choice(int(Z.shape[0]), size=cfg.estimation_N, replace=False)
    factor_predictability = space_evaluation.evaluate_space_with_predictability(
        Z, Y, predictor_train_ixs, cfg.dataset, factors, nonlinear=False, standardise=False
    )

    for attr in baseline_scores:
        experiment_results[
            f"Accuracy({attr} | {subspace_size}, {erased}, {estimation_N}, {method})"
        ] = factor_predictability[attr]
        experiment_results[
            f"Baseline Accuracy({attr} | {subspace_size}, {erased}, {estimation_N}, {method})"
        ] = baseline_scores[attr]["accuracy"]


def _project_onto_subspace(Z: torch.Tensor, A: torch.Tensor, standardise: bool = True) -> torch.Tensor:

    # Construct the representations by the projection VAE
    Z_p = Z @ A.detach().cpu()

    if standardise:
        Z_p = torch.from_numpy(StandardScaler().fit_transform(Z_p)).float()

    return Z_p


def _process_subspace(
    X: torch.Tensor,
    Z: torch.Tensor,
    Y: torch.Tensor,
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

    evaluation_ixs = rng.choice(len(Z), size=cfg.estimation_N, replace=False)

    # `subspace_size` only refers to the dimensionality of the subspace the "maximisation matrix" projects onto
    # In case this function is called with the "erasing" matrix, we keep that dimensionality for logging,
    # but we need to actually use the entire space, wince the erasing matrix is square
    projection_subspace_size = A_hat.shape[1]

    # Construct the VAE model which projects onto the subspace defined by `A_hat`
    # It has the same config as the original model, so we copy it, we just change the `latent_dim`
    projection_model = vaeutils.get_vae_class(cfg.model_class)(
        model_config=replace(model.model_config),
        encoder=ProjectionBaseEncoder(model.encoder, A_hat),
        decoder=ProjectionBaseDecoder(model.decoder, A_hat),
    )
    projection_model.model_config.latent_dim = projection_subspace_size

    Z_p = _project_onto_subspace(Z, A_hat)

    _fit_gmm(
        X=X,
        Z=Z_p,
        model=projection_model,
        cfg=cfg,
        experiment_results=experiment_results,
        model_type="projected",
        file_name=f"generated_images_projected_{method}_"
        f"{'erased' if erased else 'projected'}_{subspace_size}_{factors}_{estimation_N}.png",
    )

    for lt, model_type in zip([None, lambda t: t @ A_hat @ A_hat.T], ["full", "projected"]):
        visualisation.graphically_evaluate_model(
            model,
            X,
            Z,
            homotopy_n=cfg.plot_n_images,
            latent_transformation=lt,
            n_rows=min(projection_subspace_size, cfg.plot_nrows),
            n_cols=cfg.plot_ncols,
            repeats=cfg.plot_nrows,
            rng=rng,
            device=device,
            file_prefix=f"m{model_type}_e{erased}_{method}_{subspace_size}_{factors}_{estimation_N}",
        )

    if not erased:

        # Only calculate this for the projected subspace, since the number of dimensions it too large otherwise, so
        # it takes too long
        print("Calculating the mutual information about the factors captured in the axes of the projected subspace.")
        axis_mi = evaluation.axis_factor_mutual_information(
            Z_p, Y, factor_names=dsutils.get_dataset_module(cfg.dataset).attribute_names
        )
        print(f"Average amount of information captured about:\n {axis_mi.mean(axis=1).sort_values(ascending=False)}.")
        print(f"Maximum amount of information captured about:\n {axis_mi.max(axis=1).sort_values(ascending=False)}.")
        axis_mi.to_csv(f"axis_factor_mi_{method}_{subspace_size}_{factors}_{estimation_N}.csv")

        if projection_subspace_size <= 2:
            space_evaluation.subspace_heatmaps(
                Z_p,
                Y,
                attribute_names=factors + alternative_factors,
                standardise=False,
                file_name=f"heatmaps_{method}_{subspace_size}_{factors}_{estimation_N}.png",
            )

        _process_alternative_factors(
            Z[evaluation_ixs],
            Z_p[evaluation_ixs],
            Y[evaluation_ixs],
            model,
            method,
            factors,
            alternative_factors,
            subspace_size,
            estimation_N,
            experiment_results,
            cfg,
        )

    if erased:
        print(f"Training MIST on the erased {method} {subspace_size}-dimensional subspace for {factors}.")
        subspace_result = _train_mist(
            Z=Z_p[evaluation_ixs],
            Y=Y[evaluation_ixs][
                :, [dsutils.get_dataset_module(cfg.dataset).attribute2idx[factor] for factor in factors]
            ],
            subspace_size=projection_subspace_size,  # The matrix `A_hat` is square, so this is over the entire space
            fit_subspace=False,
            cfg=cfg,
        )
        mi = subspace_result.mutual_information
        print(f"The information about {factors} in the projected space is {mi}")
        experiment_results[f"I({factors}, erased={erased}-{subspace_size}-subspace | {method} N={estimation_N})"] = mi

    _compute_predictability_of_attributes(
        Z_p,
        Y,
        method,
        factors,
        subspace_size,
        erased,
        estimation_N,
        experiment_results,
        cfg,
        rng,
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
    factors: List[str],
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

    factors_of_interest_idx = [dsutils.get_dataset_module(cfg.dataset).attribute2idx[factor] for factor in factors]
    print(f"Determining the most informative {subspace_size}-dimensional subspace about {factors}.")
    if method == "MIST":
        result = _train_mist(
            Z=Z["train"][train_ixs],
            Y=Y["train"][train_ixs][:, factors_of_interest_idx],  # Train on the chosen subset of the data
            subspace_size=subspace_size,
            fit_subspace=True,
            cfg=cfg,
        )
        experiment_results[
            f"I({factors}, {subspace_size}-subspace | {method} N={fitting_N})"
        ] = result.mutual_information

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

        Y_c = LabelEncoder().fit_transform(Y["train"][:, factors_of_interest_idx[0]])
        result = rlace_estimation.find_subspace(Z["train"][train_ixs], Y_c[train_ixs], rlace_params)

    elif method == "Linear Regression":
        assert len(factors_of_interest_idx) == 1
        result = linear_regression_estimation.find_subspace(
            Z["train"][train_ixs].numpy(), Y["train"][train_ixs, factors_of_interest_idx[0]].numpy()
        )
    else:
        raise NotImplementedError

    print("\n --------------------------------------- \n\n")

    # Process and analyze the subspace defined by the projection matrix onto the optimal subspace and the out defined
    # by the projection matrix onto the *complement* of the optimal subspace, i.e., the subspace with as much as
    # possible of the information removed
    # Note that here, we pass the *entire* training dataset, since we will evaluate the amount of retained
    # information on the entire dataset
    for A, a_factors, erased in zip([result.A_1, result.E_1], [alternative_factors, []], [False, True]):
        _process_subspace(
            X=X["val"],
            Z=Z["val"],
            Y=Y["val"],
            A_hat=A.to(device),
            model=model,
            method=method,
            factors=factors,
            alternative_factors=a_factors,
            subspace_size=subspace_size,
            erased=erased,
            estimation_N=fitting_N,
            experiment_results=experiment_results,
            cfg=cfg,
            device=device,
            rng=rng,
        )
        print("\n --------------------------------------- \n\n")

    print("\n ------------------------------------------------------------------------------ \n\n\n")


def _fit_gmm(
    X: torch.Tensor,
    Z: torch.Tensor,
    model: VAE,
    cfg: MISTConfig,
    experiment_results: Dict[str, float],
    model_type: str,
    file_name: str,
) -> None:

    # Fit a GMM model on the produced latent space
    print("Fitting the GMM.")
    gmm_sampler = GaussianMixtureSampler(
        sampler_config=GaussianMixtureSamplerConfig(n_components=cfg.n_gmm_components), model=model
    )
    gmm_sampler.fit(X)

    # Generate a sample of images based on the fit sampler, plot, and save them.
    visualisation.generated_images(
        gmm_sampler.sample(num_samples=cfg.plot_nrows * cfg.plot_ncols),
        cfg.plot_nrows,
        cfg.plot_ncols,
        file_name=file_name,
    )

    # Compute the log likelihood of the validation data according to the sampler
    org_ll = gmm_sampler.gmm.score(Z)
    print(f"The log likelihood of the validation data is {org_ll}.")
    experiment_results[f"{model_type} model validation log likelihood"] = org_ll


def _estimate_factor_entropies(
    Y: torch.Tensor,
    factors: List[str],
    factors_idx: List[int],
    experiment_results: Dict[str, float],
) -> None:

    # Estimate the entropy of each of the factors on the validation split
    for ii, factor_idx in enumerate(factors_idx):
        H = mutual_info_regression(Y[:, factor_idx].reshape((-1, 1)), Y[:, factor_idx].flatten())[0]
        print(f"H({factors[ii]}) = {H}.")
        experiment_results[f"H({factors[ii]})"] = H


def _determine_factor_mutual_information(
    Z: torch.Tensor,
    Y: torch.Tensor,
    model: VAE,
    factors: List[str],
    experiment_results: Dict[str, float],
    cfg: MISTConfig,
) -> None:

    print(f"Training MIST to determine the information about {factors} in the space.")
    result = _train_mist(
        Z=Z,
        Y=Y[:, [dsutils.get_dataset_module(cfg.dataset).attribute2idx[factor] for factor in factors]],
        subspace_size=model.model_config.latent_dim,
        fit_subspace=False,
        cfg=cfg,
    )
    mi = result.mutual_information
    print(f"The information about {factors} in the space is {mi}")
    experiment_results[f"I({factors}, VAE)"] = mi


def _process_factors_of_interest(
    X: Dict[str, torch.Tensor],
    Z: Dict[str, torch.Tensor],
    Y: Dict[str, torch.Tensor],
    model: VAE,
    factors: List[str],
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
        factors: The set of factors of interest to consider.
        alternative_factors: The set of factors for which to check how much information was captured about them in the
                             found subspace.
        experiment_results: The dictionary in which to store the results of the experiment.
        cfg: The configuration object of the experiment.
        device: The device to load the data to.
        rng: The random generator.
    """

    _estimate_factor_entropies(
        Y=Y["val"],
        factors=factors,
        factors_idx=[dsutils.get_dataset_module(cfg.dataset).attribute2idx[factor] for factor in factors],
        experiment_results=experiment_results,
    )

    # To find out how much information about the factors of interest there is in the entire learned representation
    # space, we train a MIST model on the entire space
    evaluation_ixs = rng.choice(len(Z["val"]), size=cfg.estimation_N, replace=False)
    _determine_factor_mutual_information(
        Z=Z["val"][evaluation_ixs],
        Y=Y["val"][evaluation_ixs],
        model=model,
        factors=factors,
        experiment_results=experiment_results,
        cfg=cfg,
    )

    # Get the number of data points the MIST model will be trained with
    fitting_Ns = _get_fitting_Ns(cfg, len(Z["train"]))
    print(f"Will fit the optimal subspaces with {[int(0.4 * N) for N in fitting_Ns]} data points.")

    # Get a random subset of the training dataset
    train_ixs = {N: rng.choice(int(Z["train"].shape[0]), size=N, replace=False) for N in fitting_Ns}

    # Next, we find, for each dimensionality specified in the config, the linear subspace capturing the most
    # information about these factors
    for subspace_size in cfg.subspace_sizes:

        # The MIST model is trained in different number of data points to test the sensitivity to the dataset size
        for fitting_N in fitting_Ns:

            # Train MIST on the subspace of size `subspace_size` with `fitting_N` data points
            print(
                f"Looking into the subspace of size {subspace_size} for factors {factors} with "
                f"{fitting_N} data points ({0.4 * 100 * fitting_N / len(Z['train']):2.2f} % of the dataset)."
            )

            for method in ["MIST", "rLACE", "Linear Regression"]:
                print(f">>> INVESTIGATING {method}")
                _find_subspace(
                    X=X,
                    Z=Z,
                    Y=Y,
                    model=model,
                    factors=factors,
                    alternative_factors=alternative_factors,
                    experiment_results=experiment_results,
                    subspace_size=subspace_size,
                    fitting_N=fitting_N,
                    method=method,
                    train_ixs=train_ixs[fitting_N],
                    cfg=cfg,
                    device=device,
                    rng=rng,
                )
                print("\n\n\n")


def _get_representations(
    trained_model: VAE, X_train: torch.Tensor, X_val: torch.Tensor, X_test: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # Get the latent representations given by the original VAE
    Z_train = vaeutils.get_latent_representations(trained_model, X_train)
    Z_val = vaeutils.get_latent_representations(trained_model, X_val)
    Z_test = vaeutils.get_latent_representations(trained_model, X_test)

    full_scaler = StandardScaler().fit(Z_train)
    Z_train = torch.from_numpy(full_scaler.transform(Z_train.numpy())).float()
    Z_val = torch.from_numpy(full_scaler.transform(Z_val.numpy())).float()
    Z_test = torch.from_numpy(full_scaler.transform(Z_test.numpy())).float()

    return Z_train, Z_val, Z_test


def _save_results(experiment_results: Dict[str, float]) -> None:
    # Save the experiment results in a data frame
    results_df = pd.DataFrame({"Value": experiment_results})
    print("The final experiment results:")
    print(results_df)
    results_df.to_csv("experiment_results.csv")


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

    pl.seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # This will keep various metrics from the experiment
    experiment_results = dict()
    device = "cuda" if cfg.gpus > 0 else "cpu"

    # Load the trained VAE
    trained_model = AutoModel.load_from_folder(cfg.vae_model_file_path).to(device)

    (X_train, X_val, X_test), (Y_train, Y_val, Y_test) = dsutils.load_split_data(
        cfg.file_paths_x, cfg.file_paths_y, cfg.dataset
    )
    Z_train, Z_val, Z_test = _get_representations(trained_model, X_train, X_val, X_test)

    # Generate samples from the model
    _fit_gmm(
        X=X_val,
        Z=Z_val,
        model=trained_model,
        cfg=cfg,
        experiment_results=experiment_results,
        model_type="full",
        file_name="full_model_generated_images.png",
    )

    # Visually inspect the reconstructions and traversals
    visualisation.graphically_evaluate_model(
        trained_model,
        X_val,
        Z_val,
        homotopy_n=cfg.plot_n_images,
        n_rows=cfg.plot_nrows,
        n_cols=cfg.plot_ncols,
        rng=rng,
        device=device,
        file_prefix="full_model",
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

    _save_results(experiment_results)

    print("Finished.")


if __name__ == "__main__":
    main()
