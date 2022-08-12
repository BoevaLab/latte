# (Possible) TODO (Anej): Generalise this to models other than the beta-VAE
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

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import geoopt

from sklearn.feature_selection import mutual_info_regression

from pythae.models import AutoModel
from pythae.models import BetaVAE
from pythae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig

from latte.dataset import celeba, shapes3d, utils as dsutils
from latte.models.vae import utils as vaeutils
from latte.utils import evaluation, visualisation
from latte.models.mist import estimation
from latte.models.vae.projection_vae import ProjectionBaseEncoder, ProjectionBaseDecoder
import latte.hydra_utils as hy


@hy.config
@dataclass
class MISTConfig:
    """
    The configuration of the script.

    Members:
        dataset: The dataset to analyse
        file_path_x: Either the path to the specific file of the observations or the path to the file to the entire
                     dataset, if the observations and the ground-truth factor values are in the same file.
        file_path_y: If the observations and ground-truth factor data are stored separately, path to the factor file.
        vae_model_file_path: Path to the file of the trained VAE model.
        factors_of_interest: A list of lists.
                             Each element of the outer list is a list of elements for which we want to find the linear
                             subspace capturing the most information about them *together*.
                             Multiple such lists can be provided to explore subspaces capturing information about
                             different sets of factors.
        subspace_sizes: A list of dimensionalities of the subspaces to consider.
                        For each set of factors of interest, a subspace of each size specified in `subspace_sizes` will
                        be computed.
        mist_fractions: The list of the different fractions of all the entire dataset to use to fit the MIST model.
                        This is useful to inspect the sensitivity of the model to the number of training points.
        mist_Ns: The list of the different numbers of data points to use to fit the MIST model.
                 This is useful to inspect the sensitivity of the model to the number of training points.
                 It overrides `mist_fractions`
        p_train: Fraction of the MIST data to use to train the model.
        p_val: Fraction of the MIST data to use as validation for the model.
        early_stopping_min_delta: Min delta parameter for the early stopping callback of the MIST model.
        early_stopping_patience: Patience parameter for the early stopping callback of the MIST model.
        mine_network_width: The script uses the default implemented MINE statistics network.
                            This specified the width to be used.
        mist_batch_size: Batch size for training MIST.
        mine_learning_rate: Learning rate for training MINE.
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
        seed: The random seed.
        gpus: The `gpus` parameter for Pytorch Lightning.
        num_workers:  The `num_workers` parameter for Pytorch Lightning data modules.
    """

    dataset: str
    file_path_x: str
    file_path_y: str
    vae_model_file_path: str
    factors_of_interest: List[List[str]] = field(
        default_factory=lambda: [
            ["No_Beard", "Goatee", "Mustache"],
            ["Male"],
            ["Bald"],
            ["Pale_Skin"],
            ["Black_Hair", "Brown_Hair", "Gray_Hair", "Blond_Hair"],
        ]
    )
    subspace_sizes: List[int] = field(default_factory=lambda: [1, 2])
    mist_fractions: Optional[List[float]] = None
    mist_Ns: Optional[List[int]] = field(default_factory=lambda: [55000, 16000])
    p_train: float = 0.85
    p_val: float = 0.1
    early_stopping_min_delta: float = 1e-3
    early_stopping_patience: int = 8
    mine_network_width: int = 256
    mist_batch_size: int = 128
    mine_learning_rate: float = 1e-4
    manifold_learning_rate: float = 1e-2
    mist_max_epochs: int = 256
    lr_scheduler_patience: int = 5
    lr_scheduler_min_delta: float = 1e-3
    n_gmm_components: int = 8
    plot_nrows: int = 6
    plot_ncols: int = 8
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

    if cfg.dataset == "celeba":
        X = torch.from_numpy(np.load(cfg.file_path_x))
        Y = torch.from_numpy(np.load(cfg.file_path_y))

        X /= 255

    elif cfg.dataset == "shapes3d":

        dataset = shapes3d.load_shapes3d(filepath=cfg.file_path_x)
        X = dataset.imgs[:]
        Y = torch.from_numpy(dataset.latents_values[:].astype(np.float32))
        del dataset  # Save space
        X = torch.from_numpy(np.transpose(X, (0, 3, 1, 2)).astype(np.float32) / 255)

    (X_train, X_val, X_test), (Y_train, Y_val, Y_test) = dsutils.split(
        [X, Y], p_train=cfg.p_train, p_val=cfg.p_val, seed=cfg.seed
    )

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
) -> estimation.MISTResult:
    """
    Trains a MIST model to find the optimal subspace of dimensionality `subspace_size` about the factors captured in `Y`
    Args:
        Z: Dictionary of the full VAE representations split into "train", "val", and "test" splits
        Y: Dictionary of the ground-truth factor values split into "train", "val", and "test" splits
        subspace_size: The current size of the subspace considered.
        fit_subspace: Whether subspace is to be fit (and the train/test fractions should be different)
        cfg: The configuration object of the experiment.

    Returns:
        The MIST estimation result

    """

    print("Training MIST.")

    early_stopping_callback = EarlyStopping(
        monitor="validation_mutual_information_mine",
        min_delta=cfg.early_stopping_min_delta,
        patience=cfg.early_stopping_patience,
        verbose=False,
        mode="max",
    )

    p_train, p_val = (0.4, 0.2) if fit_subspace else (0.7, 0.1)

    mi_estimation_result = estimation.find_subspace(
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
            callbacks=[early_stopping_callback],
            max_epochs=cfg.mist_max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            gpus=cfg.gpus,
        ),
    )

    print("Training MIST done.")

    return mi_estimation_result


def _process_subspace(
    X: Dict[str, torch.Tensor],
    Y: Dict[str, torch.Tensor],
    A_hat: geoopt.ManifoldTensor,
    model: BetaVAE,
    factors: List[str],
    subspace_size: int,
    erased: bool,
    mist_N: int,
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
        Y: Dictionary of the ground-truth factor values split into "train", "val", and "test" splits
        A_hat: The projection matrix.
        model: The original trained model
        factors: The factors of interest
        subspace_size: The current size of the subspace considered.
                       This is needed since in the case when looking at the "erasing" projection matrix, the matrix
                       will already be of dimensionality `n x n`, where `n` is the size of the full representation space
                       This means we can not get the dimensionality of the subspace it is erasing just from the size
                       of the matrix.
        erased: Whether the A_hat projects onto the complement of the maximum-information subspace or not.
        mist_N: Number of data points used for training.
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
    projection_model = BetaVAE(
        model_config=replace(model.model_config),
        encoder=ProjectionBaseEncoder(model.encoder, A_hat),
        decoder=ProjectionBaseDecoder(model.decoder, A_hat),
    )
    projection_model.model_config.latent_dim = projection_subspace_size

    # Construct the representations by the projection VAE
    Z_train_projected = vaeutils.get_latent_representations(projection_model, X["train"])
    Z_val_projected = vaeutils.get_latent_representations(projection_model, X["val"])
    # Z_test_projected = vaeutils.get_latent_representations(projection_model, X["test"])

    # Plot the latent traversals in the subspace
    visualisation.latent_traversals(
        vae_model=projection_model,
        Z=Z_train_projected,
        starting_x=X["val"][rng.choice(len(X["val"]))].to(device),
        n_values=cfg.plot_ncols,
        n_axes=projection_subspace_size,
        file_name=f"latent_traversals_{'erased' if erased else 'projected'}_{subspace_size}_{factors}_{mist_N}.png",
    )

    if not erased:
        # Intervene on the values of the subspace on a sample of the images to change their features
        for ii in range(cfg.plot_nrows):
            visualisation.latent_traversals(
                vae_model=model,
                Z=Z_train_projected,
                A_hat=A_hat,
                starting_x=X["val"][rng.choice(len(X["val"]))].to(device),
                n_values=cfg.plot_ncols,
                n_axes=projection_subspace_size,
                file_name=f"latent_traversals_intervened_{subspace_size}_{factors}_{mist_N}_{ii}.png",
            )

    # TODO (Anej): possibly add heatmaps or trends of the factors of interest against some dimensions

    if not erased:
        # Only calculate this for the projected subspace, since the number of dimensions it too large otherwise so
        # it takes too long
        print("Calculating the mutual information about the factors captured in the axes of the projected subspace.")
        axis_mi = evaluation.axis_factor_mutual_information(
            Z_val_projected, Y["val"], factor_names=_get_dataset_module(cfg).attribute_names
        )
        print(f"Average amount of information captured about:\n {axis_mi.mean(axis=1).sort_values(ascending=False)}.")
        print(f"Maximum amount of information captured about:\n {axis_mi.max(axis=1).sort_values(ascending=False)}.")
        axis_mi.to_csv(f"axis_factor_mi_{subspace_size}_{factors}_{mist_N}.csv")

    # Plot the reconstructions on the subspace
    # Just the reconstructions
    # visualisation.vae_reconstructions(
    #     dataset=X["val"],
    #     vae_model=projection_model,
    #     nrows=cfg.plot_nrows,
    #     ncols=cfg.plot_ncols,
    #     file_name=f"reconstructions_only"
    #               f"_{'erased' if erased else 'projected'}_{subspace_size}_{factors}_{mist_N}.png",
    # )
    # Reconstructions together with their original non-projected reconstructions
    visualisation.vae_reconstructions(
        dataset=X["val"],
        vae_model=model,
        latent_transformation=lambda z: z @ A_hat @ A_hat.T,
        nrows=cfg.plot_nrows,
        ncols=cfg.plot_ncols,
        file_name=f"reconstructions_transformation"
        f"_{'erased' if erased else 'projected'}_{subspace_size}_{factors}_{mist_N}.png",
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
        file_name=f"generated_images_{'erased' if erased else 'projected'}_{subspace_size}_{factors}_{mist_N}.png",
    )

    if erased:
        # If working with the model which erases information about the factors, also estimate how much information
        # is left in the VAE representation space
        # Since A_hat projects from the original representation space, its first dimension will be the original
        # representation space
        print(f"Training MIST on the erased {subspace_size}-dimensional subspace for {factors}.")
        erased_subspace_result = _train_mist(
            Z=Z_train_projected, Y=Y["train"], subspace_size=projection_subspace_size, fit_subspace=False, cfg=cfg
        )
        erased_mi = erased_subspace_result.mutual_information
        print(f"The information about {factors} in the erased space is {erased_mi}")
        experiment_results[f"I({factors}, {subspace_size}-erased-subspace | N={mist_N})"] = erased_mi


def _get_mist_Ns(cfg: MISTConfig, N: int) -> List[int]:
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
    if cfg.mist_Ns is not None:
        return [int(2.5 * N) for N in cfg.mist_Ns]
    else:
        return [int(2.5 * fraction * N) for fraction in cfg.mist_fractions]


def _process_factors_of_interest(
    X: Dict[str, torch.Tensor],
    Z: Dict[str, torch.Tensor],
    Y: Dict[str, torch.Tensor],
    model: BetaVAE,
    factors_of_interest: List[str],
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
        experiment_results: The dictionary in which to store the results of the experiment.
        cfg: The configuration object of the experiment.
        device: The device to load the data to.
        rng: The random generator.

    """

    # TODO (Anej): possibly add heatmaps or trends of the factors of interest against some dimensions

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
    mist_Ns = _get_mist_Ns(cfg, len(Z["train"]))
    print(
        f"Will fit the optimal subspaces with {mist_Ns} data points "
        f"(40 %, {[int(0.4 * N) for N in mist_Ns]}, of that for the training)."
    )

    # Next, we find, for each dimensionality specified in the config, the linear subspace capturing the most
    # information about these factors
    for subspace_size in cfg.subspace_sizes:

        # The MIST model is trained in different number of data points to test the sensitivity to the dataset size
        for mist_N in mist_Ns:

            # Get a random subset of the training dataset
            train_ixs = rng.choice(int(Z["train"].shape[0]), size=mist_N, replace=False)

            # Train MIST on the subspace of size `subspace_size` with `mist_N` data points
            print(
                f"Looking into the subspace of size {subspace_size} for factors {factors_of_interest} with "
                f"{mist_N} data points ({0.4 * 100 * mist_N / len(Z['train']):2.2f} % of the dataset)."
            )
            print(f"Determining the most informative {subspace_size}-dimensional subspace about {factors_of_interest}.")
            subspace_result = _train_mist(
                Z=Z["train"][train_ixs],
                Y=Y["train"][train_ixs][:, factors_of_interest_idx],  # Train on the chosen subset of the data
                subspace_size=subspace_size,
                fit_subspace=True,
                cfg=cfg,
            )
            mi = subspace_result.mutual_information
            print(f"The information about {factors_of_interest} in a {subspace_size}-dimensional space is {mi}")
            experiment_results[f"I({factors_of_interest}, {subspace_size}-subspace | N={mist_N})"] = mi

            # Process and analyze the subspace defined by the projection matrix onto the optimal subspace
            _process_subspace(
                X=X,
                Y=Y,
                A_hat=subspace_result.A.to(device),
                model=model,
                factors=factors_of_interest,
                subspace_size=subspace_size,
                erased=False,
                mist_N=mist_N,
                experiment_results=experiment_results,
                cfg=cfg,
                device=device,
                rng=rng,
            )
            # Process and analyze the subspace defined by the projection matrix onto the *complement* of the optimal
            # subspace, i.e., the subspace with as much as possible of the information removed
            # Note that here, we pass the *entire* training dataset, since we will evaluate the amount of retained
            # information on the entire dataset
            _process_subspace(
                X=X,
                Y=Y,
                A_hat=(torch.eye(model.model_config.latent_dim) - subspace_result.A @ subspace_result.A.T).to(device),
                model=model,
                factors=factors_of_interest,
                # subspace_size=trained_model.model_config.latent_dim,
                subspace_size=subspace_size,
                erased=True,
                mist_N=mist_N,
                experiment_results=experiment_results,
                cfg=cfg,
                device=device,
                rng=rng,
            )


@hy.main
def main(cfg: MISTConfig):

    assert cfg.dataset in ["celeba", "shapes3d"], "The dataset not supported."
    assert (
        cfg.mist_fractions is not None or cfg.mist_Ns is not None
    ), "At least one of `mist_fractions` and `mist_Ns` has to be specified."
    assert cfg.mist_fractions is None or all(fraction < 0.4 for fraction in cfg.mist_fractions), (
        "To get a good estimate of the mutual information, "
        "at most 40 % of the data can be used to fit the subspace with the MIST model."
    )

    # This will keep various metrics from the experiment
    experiment_results = dict()

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

    # Fit a GMM model on the produced latent space
    print("Fitting the GMM on the original latent space.")
    gmm_sampler = GaussianMixtureSampler(
        sampler_config=GaussianMixtureSamplerConfig(n_components=cfg.n_gmm_components), model=trained_model
    )
    gmm_sampler.fit(X_val)
    print("Fitting the GMM on the original latent space done.")

    # Generate a sample of images based on the fit sampler, plot, and save them.
    visualisation.generated_images(
        gmm_sampler.sample(num_samples=cfg.plot_nrows * cfg.plot_ncols),
        cfg.plot_nrows,
        cfg.plot_ncols,
        file_name="generated_images_full_model.png",
    )

    # Compute the log likelihood of the validation data according to the sampler
    org_ll = gmm_sampler.gmm.score(Z_val)
    print(f"The log likelihood of the validation data is {org_ll}.")
    experiment_results["Original model validation log likelihood"] = org_ll

    # Plot the latent traversals
    visualisation.latent_traversals(
        vae_model=trained_model,
        starting_x=X_val[rng.choice(len(X_val))].to(device),
        Z=Z_val,
        n_values=cfg.plot_ncols,
        n_axes=6,
        cmap=None,
        file_name="latent_traversals_full_model.png",
    )

    # Go through the specified factors of interest, and for each provided set, find the subspace capturing the most
    # information about then, and analyse it
    for factors_of_interest in cfg.factors_of_interest:
        print(f"Processing the set of factors {factors_of_interest}.")
        _process_factors_of_interest(
            {"train": X_train, "val": X_val, "test": X_test},
            {"train": Z_train, "val": Z_val, "test": Z_test},
            {"train": Y_train, "val": Y_val, "test": Y_test},
            trained_model,
            factors_of_interest,
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
