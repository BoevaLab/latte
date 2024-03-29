"""
This script trains a supervised MIST model on a supported dataset of images.
Currently, it works for the `CelebA` and `Shapes3D` datasets.
It takes the representations of a pre-trained VAE and discovers in them the subspaces of specified dimensionality
capturing the most information about a set of given attributes/latent factors.
The results are quantified, plotted, and saved.
"""
from typing import Tuple, Dict, Optional, List, Any
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

from latte.dataset import utils as ds_utils
from latte.models.vae import utils as vae_utils
from latte.evaluation import subspace_evaluation, metrics
from latte.utils.visualisation import images as image_visualisation
from latte.models.rlace import estimation as rlace_estimation
from latte.models.linear_regression import estimation as linear_regression_estimation
from latte.models.vae.projection_vae import ProjectionBaseEncoder, ProjectionBaseDecoder
from latte.tools import space_evaluation, subspace_search
from latte.tools.space_evaluation import MIEstimationMethod
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
        estimation_N: Number of data points used for evaluation.
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


def _process_alternative_factors(
    Z: torch.Tensor,
    Y: torch.Tensor,
    A: torch.Tensor,
    ixs: Dict[MIEstimationMethod, List[int]],
    method: str,
    factors: List[str],
    alternative_factors: List[str],
    subspace_size: int,
    estimation_N: int,
    factor_mutual_informations: List[List[Any]],
    cfg: MISTConfig,
) -> None:
    """
    Computes the information about a specified set of alternative factors in the entire VAE representation space and
    the optimal projection space w.r.t. some other factors to determine the amount of disentanglement.
    """

    for factor in alternative_factors:
        for M, result_key in zip(
            [None, A],
            [
                [[factor], method, "full", "N/A", "N/A", estimation_N],
                [[factor], method, subspace_size, factors, False, estimation_N],
            ],
        ):
            # Determine the amount of mutual information in the (projected) representation space about the
            # specified factor of variation.
            _determine_factor_mutual_information(
                Z=Z,
                Y=Y,
                A=M,
                ixs=ixs,
                factors=[factor],
                prefix=result_key,
                cfg=cfg,
                factor_mutual_informations=factor_mutual_informations,
            )


def _compute_predictability_of_attributes(
    Z: torch.Tensor,
    Y: torch.Tensor,
    method: str,
    factors: List[str],
    subspace_size: int,
    erased: bool,
    estimation_N: int,
    attribute_predictability: List[List[Any]],
    baseline_attribute_predictability: List[List[Any]],
    cfg: MISTConfig,
    rng: ds_utils.RandomGenerator,
) -> None:
    """
    Computes the linear predictability of factors in `Y` with `evaluate_space_with_predictability`.
    """

    baseline_scores = space_evaluation.baseline_predictability(Y=Y, dataset=cfg.dataset, factors=factors)

    predictor_train_ixs = rng.choice(int(Z.shape[0]), size=cfg.estimation_N, replace=False)
    factor_predictability = space_evaluation.evaluate_space_with_predictability(
        Z=Z,
        Y=Y,
        fitting_indices=predictor_train_ixs,
        dataset=cfg.dataset,
        factors=factors,
        nonlinear=False,
        standardise=True,
    )

    for attribute in baseline_scores:
        attribute_predictability.append(
            [subspace_size, erased, estimation_N, method, attribute, factor_predictability[attribute]]
        )
        baseline_attribute_predictability.append(
            [subspace_size, erased, estimation_N, method, attribute, baseline_scores[attribute]["accuracy"]]
        )


def _project_onto_subspace(Z: torch.Tensor, A: torch.Tensor, standardise: bool = True) -> torch.Tensor:
    """
    Utility function to project onto a subspace defined by `A` and optionally standardise the projections.
    """

    # Construct the representations by the projection VAE
    Z_p = Z @ A.detach().cpu()

    if standardise:
        Z_p = torch.from_numpy(StandardScaler().fit_transform(Z_p)).float()

    return Z_p


def _process_subspace(
    X: torch.Tensor,
    Z: torch.Tensor,
    Y: torch.Tensor,
    A: geoopt.ManifoldTensor,
    model: VAE,
    method: str,
    factors: List[str],
    alternative_factors: List[str],
    subspace_size: int,
    erased: bool,
    estimation_N: int,
    factor_mutual_informations: List[List[Any]],
    log_likelihoods: List[List[Any]],
    attribute_predictability: List[List[Any]],
    baseline_attribute_predictability: List[List[Any]],
    cfg: MISTConfig,
    device: str,
    rng: ds_utils.RandomGenerator,
) -> None:
    """
    Takes in a projection matrix onto a linear subspace of the full VAE representation space, construct a VAE which
    projects onto that space, and analyses the results by doing latent traversals, calculating the mutual information
    with regard to each of the axes, and sampling from the subspace.
    Args:
        X: Dictionary of the observations split into "train", "val", and "test" splits
        Z: Dictionary of the full VAE representations split into "train", "val", and "test" splits
        Y: Dictionary of the ground-truth factor values split into "train", "val", and "test" splits
        A: The projection matrix.
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
        cfg: The configuration object of the experiment.
        device: The device to load the data to.
        rng: The random generator.
    """
    rng = np.random.default_rng(rng)

    evaluation_ixs = rng.choice(len(Z), size=cfg.estimation_N, replace=False)
    latte_ksg_evaluation_ixs = rng.choice(len(Z), size=8000, replace=False)

    # `subspace_size` only refers to the dimensionality of the subspace the "maximisation matrix" projects onto
    # In case this function is called with the "erasing" matrix, we keep that dimensionality for logging,
    # but we need to actually use the entire space, wince the erasing matrix is square
    projection_subspace_size = A.shape[1]

    Z_p = _project_onto_subspace(Z, A)

    # Construct the VAE model which projects onto the subspace defined by `A_hat`
    # It has the same config as the original model, so we copy it, we just change the `latent_dim`
    projection_model = vae_utils.get_vae_class(cfg.model_class)(
        model_config=replace(model.model_config),
        encoder=ProjectionBaseEncoder(model.encoder, A),
        decoder=ProjectionBaseDecoder(model.decoder, A),
    )
    projection_model.model_config.latent_dim = projection_subspace_size

    # Generate images from the optimal subspace
    _fit_gmm(
        X=X,
        Z=Z_p,
        model=projection_model,
        cfg=cfg,
        model_type="projected",
        file_name=f"generated_images_projected_{method}_"
        f"{'erased' if erased else 'projected'}_{subspace_size}_{factors}_{estimation_N}.png",
        log_likelihoods=log_likelihoods,
    )

    # Evaluate the subspace graphically
    for M, model_type in zip([A, None], ["subspace", "projected"]):
        image_visualisation.graphically_evaluate_model(
            model,
            X,
            A=M,
            homotopy_n=cfg.plot_n_images,
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
        axis_mi = subspace_evaluation.axis_factor_mutual_information(
            Z_p, Y, factor_names=ds_utils.get_dataset_module(cfg.dataset).attribute_names
        )
        print(f"Average amount of information captured about:\n {axis_mi.mean(axis=1).sort_values(ascending=False)}.")
        print(f"Maximum amount of information captured about:\n {axis_mi.max(axis=1).sort_values(ascending=False)}.")
        axis_mi.to_csv(f"axis_factor_mi_{method}_{subspace_size}_{factors}_{estimation_N}.csv")

        if projection_subspace_size <= 2:
            # If dimensionality of the subspace is at most 2, also plot the heatmaps.
            space_evaluation.subspace_heatmaps(
                Z,
                Y[
                    :,
                    [ds_utils.get_dataset_module(cfg.dataset).attribute2idx[a] for a in factors + alternative_factors],
                ],
                A=A,
                factor_names=factors + alternative_factors,
                standardise=True,
                file_name=f"heatmaps_{method}_{subspace_size}_{factors}_{estimation_N}.png",
            )

        # Process the specified set of alternative factors to determine the degree to which they are captured
        # in the optimal subspace.
        _process_alternative_factors(
            Z=Z,
            Y=Y,
            A=A,
            ixs={
                # MIEstimationMethod.NAIVE_KSG: evaluation_ixs,
                MIEstimationMethod.KSG: latte_ksg_evaluation_ixs,
                MIEstimationMethod.MIST: evaluation_ixs,
                MIEstimationMethod.SKLEARN: evaluation_ixs,
            },
            method=method,
            factors=factors,
            alternative_factors=alternative_factors,
            subspace_size=subspace_size,
            estimation_N=estimation_N,
            cfg=cfg,
            factor_mutual_informations=factor_mutual_informations,
        )

    # Determine how much information is captured in the complement of the optimal subspace
    _determine_factor_mutual_information(
        Z=Z,
        Y=Y,
        A=A,
        ixs={
            # MIEstimationMethod.NAIVE_KSG: evaluation_ixs,
            MIEstimationMethod.KSG: latte_ksg_evaluation_ixs,
            MIEstimationMethod.MIST: evaluation_ixs,
            MIEstimationMethod.SKLEARN: evaluation_ixs,
        },
        factors=factors,
        prefix=[factors, method, subspace_size, factors, erased, estimation_N],
        factor_mutual_informations=factor_mutual_informations,
        cfg=cfg,
    )

    # Evaluate (the optimal subspace or its complement) with the (linear) predictability of the factors w.r.t. which
    # the subspace was estimated.
    _compute_predictability_of_attributes(
        Z=Z_p,
        Y=Y,
        method=method,
        factors=factors,
        subspace_size=subspace_size,
        erased=erased,
        estimation_N=estimation_N,
        cfg=cfg,
        rng=rng,
        attribute_predictability=attribute_predictability,
        baseline_attribute_predictability=baseline_attribute_predictability,
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
    Z: torch.Tensor,
    Y: torch.Tensor,
    factors: List[str],
    subspace_size: int,
    fitting_N: int,
    method: str,
    train_ixs: np.ndarray,
    optimal_subspace_mutual_information: List[List[Any]],
    cfg: MISTConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Determines the optimal subspace capturing the most information about the specified set of factors with the specified
    method.
    """

    factors_of_interest_idx = [ds_utils.get_dataset_module(cfg.dataset).attribute2idx[factor] for factor in factors]
    print(f"Determining the most informative {subspace_size}-dimensional subspace about {factors}.")
    if method == "MIST":
        result = subspace_search.find_subspace(
            Z=Z[train_ixs],
            Y=Y[train_ixs][:, factors_of_interest_idx],  # Train on the chosen subset of the data
            subspace_size=subspace_size,
            max_epochs=cfg.mist_max_epochs,
            batch_size=cfg.mist_batch_size,
            mine_network_width=cfg.mine_network_width,
            mine_learning_rate=cfg.mine_learning_rate,
            manifold_learning_rate=cfg.manifold_learning_rate,
            lr_scheduler_patience=cfg.lr_scheduler_patience,
            lr_scheduler_min_delta=cfg.lr_scheduler_min_delta,
            num_workers=cfg.num_workers,
            gpus=cfg.gpus,
        )
        optimal_subspace_mutual_information.append(
            [factors, subspace_size, method, fitting_N, result.mutual_information]
        )

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

        Y_c = LabelEncoder().fit_transform(Y[:, factors_of_interest_idx[0]])
        result = rlace_estimation.fit(Z[train_ixs], Y_c[train_ixs], rlace_params)

    elif method == "Linear Regression":
        assert len(factors_of_interest_idx) == 1
        result = linear_regression_estimation.fit(
            Z[train_ixs].numpy(), Y[train_ixs, factors_of_interest_idx[0]].numpy()
        )
    else:
        raise NotImplementedError

    print("\n --------------------------------------- \n\n")

    return result.A_1, result.E_1


def _fit_gmm(
    X: torch.Tensor,
    Z: torch.Tensor,
    model: VAE,
    cfg: MISTConfig,
    model_type: str,
    file_name: str,
    log_likelihoods: List[List[Any]],
) -> None:
    """
    Fits a GMM sampler on the representation space defined by the VAE model and generates images with it.
    """

    # Fit a GMM model on the produced latent space
    print("Fitting the GMM.")
    gmm_sampler = GaussianMixtureSampler(
        sampler_config=GaussianMixtureSamplerConfig(n_components=cfg.n_gmm_components), model=model
    )
    gmm_sampler.fit(X)

    # Generate a sample of images based on the fit sampler, plot, and save them.
    image_visualisation.generated_images(
        gmm_sampler.sample(num_samples=cfg.plot_nrows * cfg.plot_ncols),
        cfg.plot_nrows,
        cfg.plot_ncols,
        file_name=file_name,
    )

    # Compute the log likelihood of the validation data according to the sampler
    ll = gmm_sampler.gmm.score(Z)
    print(f"The log likelihood of the validation data is {ll}.")
    log_likelihoods.append([model_type, ll])


def _estimate_factor_entropies(
    Y: torch.Tensor,
    factors: List[str],
    factors_idx: List[int],
    factor_entropies: List[List[Any]],
) -> None:
    """
    Estimates the entropy of the specified set of factors.
    """

    # Estimate the entropy of each of the factors on the validation split
    for ii, factor_idx in enumerate(factors_idx):
        H = mutual_info_regression(Y[:, factor_idx].reshape((-1, 1)), Y[:, factor_idx].flatten())[0]
        print(f"H({factors[ii]}) = {H}.")
        factor_entropies.append([factors[ii], H])


def _determine_factor_mutual_information(
    Z: torch.Tensor,
    Y: torch.Tensor,
    ixs: Dict[MIEstimationMethod, List[int]],
    factors: List[str],
    factor_mutual_informations: List[List[Any]],
    prefix: List[Any],
    cfg: MISTConfig,
    A: Optional[torch.Tensor] = None,
) -> None:
    """
    Determines the mutual information between the specified set of factors in `Y` and the representations `Z`.
    """
    print(f"Training MIST to determine the information about {factors} in the space.")

    methods = (
        [
            # MIEstimationMethod.NAIVE_KSG,
            MIEstimationMethod.KSG,
            MIEstimationMethod.MIST,
            MIEstimationMethod.SKLEARN,
        ]
        if Z.shape[1] == 1
        # else [MIEstimationMethod.NAIVE_KSG, MIEstimationMethod.KSG, MIEstimationMethod.MIST]
        else [MIEstimationMethod.KSG, MIEstimationMethod.MIST]
    )

    for method in methods:
        result = space_evaluation.mutual_information(
            Z=Z,
            Y=Y,
            fitting_indices=ixs[method],
            dataset=cfg.dataset,
            factors=[tuple(factors)],
            A=A,
            method=method,
            standardise=True,
        )
        factor_mutual_informations.append(prefix + [method, result[tuple(factors)]])
        print(f"The information about {factors} by method {method} in the space is {result[tuple(factors)]:.3f}.")


def _process_factors_of_interest(
    X: Dict[str, torch.Tensor],
    Z: Dict[str, torch.Tensor],
    Y: Dict[str, torch.Tensor],
    model: VAE,
    factors: List[str],
    alternative_factors: List[str],
    factor_mutual_informations: List[List[Any]],
    attribute_predictability: List[List[Any]],
    baseline_attribute_predictability: List[List[Any]],
    factor_entropies: List[List[Any]],
    log_likelihoods: List[List[Any]],
    optimal_subspace_mutual_information: List[List[Any]],
    distances_to_permutations_matrix: List[List[Any]],
    cfg: MISTConfig,
    device: str,
    rng: ds_utils.RandomGenerator,
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
        cfg: The configuration object of the experiment.
        device: The device to load the data to.
        rng: The random generator.
    """

    # Estimate the entropy of the factors to determine the maximal amount of information which can be captured about
    # them.
    _estimate_factor_entropies(
        Y=Y["val"],
        factors=factors,
        factors_idx=[ds_utils.get_dataset_module(cfg.dataset).attribute2idx[factor] for factor in factors],
        factor_entropies=factor_entropies,
    )

    # To find out how much information about the factors of interest there is in the entire learned representation
    # space, we train a MIST model on the entire space
    evaluation_ixs = rng.choice(len(Z["val"]), size=cfg.estimation_N, replace=False)
    latte_ksg_evaluation_ixs = rng.choice(len(Z["val"]), size=8000, replace=False)
    _determine_factor_mutual_information(
        Z=Z["val"],
        Y=Y["val"],
        ixs={
            # MIEstimationMethod.NAIVE_KSG: evaluation_ixs,
            MIEstimationMethod.KSG: latte_ksg_evaluation_ixs,
            MIEstimationMethod.MIST: evaluation_ixs,
            MIEstimationMethod.SKLEARN: evaluation_ixs,
        },
        factors=factors,
        prefix=[factors, "N/A", "full", "N/A", "N/A", cfg.estimation_N],
        cfg=cfg,
        factor_mutual_informations=factor_mutual_informations,
    )

    # Get the number of data points the MIST model will be trained with
    fitting_Ns = _get_fitting_Ns(cfg, len(Z["train"]))
    print(f"Will fit the optimal subspaces with {[int(0.4 * N) for N in fitting_Ns]} data points.")

    # Get a random subset of the training dataset
    train_ixs = {N: rng.choice(int(Z["train"].shape[0]), size=N, replace=False) for N in fitting_Ns}

    # Next, we find, for each dimensionality specified in the config, the linear subspace capturing the most
    # information about these factors
    for subspace_size in cfg.subspace_sizes:

        if len(factors) > 1:
            methods = ["MIST"]
        else:
            # methods = ["MIST", "rLACE", "Linear Regression"] if subspace_size == 1 else ["MIST", "rLACE"]
            methods = ["MIST"]

        # The MIST model is trained in different number of data points to test the sensitivity to the dataset size
        for fitting_N in fitting_Ns:

            # Train MIST on the subspace of size `subspace_size` with `fitting_N` data points
            print(
                f"Looking into the subspace of size {subspace_size} for factors {factors} with "
                f"{fitting_N} data points ({0.4 * 100 * fitting_N / len(Z['train']):2.2f} % of the dataset)."
            )

            for method in methods:
                print(f">>> INVESTIGATING {method}")

                # Find the subspace capturing the specified set of factors using one of the methods.
                # Returns the projection matrix onto the subspace and the complement of the subspace.
                A, E = _find_subspace(
                    Z=Z["train"],
                    Y=Y["train"],
                    factors=factors,
                    subspace_size=subspace_size,
                    fitting_N=fitting_N,
                    method=method,
                    train_ixs=train_ixs[fitting_N],
                    cfg=cfg,
                    optimal_subspace_mutual_information=optimal_subspace_mutual_information,
                )

                permutation_dist = metrics.distance_to_permutation_matrix(A)
                distances_to_permutations_matrix.append([factors, subspace_size, method, fitting_N, permutation_dist])

                # Process and analyze the subspace defined by the projection matrix onto the optimal subspace and
                # the out defined by the projection matrix onto the *complement* of the optimal subspace,
                # i.e., the subspace with as much as possible of the information removed
                # Note that here, we pass the *entire* training dataset, since we will evaluate the amount of retained
                # information on the entire dataset
                for P, a_factors, erased in zip([A, E], [alternative_factors, []], [False, True]):
                    _process_subspace(
                        X=X["val"],
                        Z=Z["val"],
                        Y=Y["val"],
                        A=P.to(device),
                        model=model,
                        method=method,
                        factors=factors,
                        alternative_factors=a_factors,
                        subspace_size=subspace_size,
                        erased=erased,
                        estimation_N=fitting_N,
                        cfg=cfg,
                        device=device,
                        rng=rng,
                        attribute_predictability=attribute_predictability,
                        baseline_attribute_predictability=baseline_attribute_predictability,
                        log_likelihoods=log_likelihoods,
                        factor_mutual_informations=factor_mutual_informations,
                    )
                    print("\n --------------------------------------- \n\n")

                print("\n\n\n")


def _get_representations(model: VAE, X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Computes and normalises the representations given by the passed VAE model.
    """
    main_key = "train" if "train" in X else "val"

    # Get the latent representations given by the original VAE
    Z = {split: vae_utils.get_latent_representations(model, S) for split, S in X.items()}

    full_scaler = StandardScaler().fit(Z[main_key])
    Z = {split: torch.from_numpy(full_scaler.transform(S.numpy())).float() for split, S in Z.items()}

    return Z


def _save_results(
    log_likelihoods: List[List[Any]],
    factor_entropies: List[List[Any]],
    factor_mutual_informations: List[List[Any]],
    attribute_predictability: List[List[Any]],
    baseline_attribute_predictability: List[List[Any]],
    optimal_subspace_mutual_information: List[List[Any]],
    distances_to_permutations_matrix: List[List[Any]],
) -> None:
    results = [
        log_likelihoods,
        factor_entropies,
        factor_mutual_informations,
        attribute_predictability,
        baseline_attribute_predictability,
        optimal_subspace_mutual_information,
        distances_to_permutations_matrix,
    ]
    file_names = [
        "log_likelihoods.csv",
        "factor_entropies.csv",
        "factor_mutual_information.csv",
        "attribute_predictability.csv",
        "baseline_attribute_predictability.csv",
        "optimal_subspace_mutual_information.csv",
        "distances_to_permutations_matrix.csv",
    ]
    columnss = [
        ["Model Type", "Log-Likelihood"],
        ["Factor", "Entropy"],
        [
            "Factors",
            "Subspace Estimation Method",
            "Subspace Size",
            "Subspace Factors",
            "Erased",
            "Training Set Size",
            "Mutual Information Estimation Method",
            "Mutual Information",
        ],
        [
            "Subspace Size",
            "Erased",
            "Training Set Size",
            "Mutual Information Estimation Method",
            "Factor",
            "Predictability",
        ],
        [
            "Subspace Size",
            "Erased",
            "Training Set Size",
            "Mutual Information Estimation Method",
            "Factor",
            "Baseline Predictability",
        ],
        ["Factors", "Subspace Size", "Mutual Information Estimation Method", "Training Set Size", "Mutual Information"],
        ["Factors", "Subspace Size", "Mutual Information Estimation Method", "Training Set Size", "Distance"],
    ]
    for result, file_name, columns in zip(results, file_names, columnss):
        pd.DataFrame(
            result,
            columns=columns,
        ).to_csv(file_name)


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
    device = "cuda" if cfg.gpus > 0 else "cpu"

    # Load the trained VAE
    model = AutoModel.load_from_folder(cfg.vae_model_file_path).to(device)

    # Load the data from the pre-split dataset
    X, Y = ds_utils.load_split_data(
        {"train": cfg.file_paths_x[0], "val": cfg.file_paths_x[1], "test": cfg.file_paths_x[2]},
        {"train": cfg.file_paths_y[0], "val": cfg.file_paths_y[1], "test": cfg.file_paths_y[2]},
        cfg.dataset,
    )
    Z = _get_representations(model, X)

    # Generate samples from the model
    log_likelihoods = []
    _fit_gmm(
        X=X["val"],
        Z=Z["val"],
        model=model,
        cfg=cfg,
        model_type="full",
        file_name="full_model_generated_images.png",
        log_likelihoods=log_likelihoods,
    )

    # Visually inspect the reconstructions and traversals
    image_visualisation.graphically_evaluate_model(
        model,
        X["val"],
        homotopy_n=cfg.plot_n_images,
        n_rows=cfg.plot_nrows,
        n_cols=cfg.plot_ncols,
        rng=rng,
        device=device,
        file_prefix="full_model",
    )

    factor_entropies, factor_mutual_informations = [], []
    attribute_predictability, baseline_attribute_predictability = [], []
    optimal_subspace_mutual_information = []
    distances_to_permutations_matrix = []
    # Go through the specified factors of interest, and for each provided set, find the subspace capturing the most
    # information about then, and analyse it
    for ii in range(len(cfg.factors_of_interest)):
        print(f"Processing the set of factors {cfg.factors_of_interest[ii]}.")
        _process_factors_of_interest(
            X=X,
            Z=Z,
            Y=Y,
            model=model,
            factors=cfg.factors_of_interest[ii],
            alternative_factors=cfg.alternative_factors[ii],
            factor_mutual_informations=factor_mutual_informations,
            cfg=cfg,
            device=device,
            rng=rng,
            attribute_predictability=attribute_predictability,
            baseline_attribute_predictability=baseline_attribute_predictability,
            log_likelihoods=log_likelihoods,
            factor_entropies=factor_entropies,
            optimal_subspace_mutual_information=optimal_subspace_mutual_information,
            distances_to_permutations_matrix=distances_to_permutations_matrix,
        )

    _save_results(
        log_likelihoods,
        factor_entropies,
        factor_mutual_informations,
        attribute_predictability,
        baseline_attribute_predictability,
        optimal_subspace_mutual_information,
        distances_to_permutations_matrix,
    )

    print("Finished.")


if __name__ == "__main__":
    main()
