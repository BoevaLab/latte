"""
This scripts trains a (beta)VAE on the dSprites dataset and finds the optimal linear subspace capturing a selected set
of factor of variation.
It is similar to the `mist_benchmark_images.py` script, this one optionally *trains a VAE from scratch*,
and then fits MIST, and is specific to the `dSprites` data.
It is also older and thus less optimised.
TODO (Anej): This could be replaced by a more general script in the future.

To qualitatively evaluate the disentanglement, latent traversals of the original and transformed axes are plotted.
"""
# TODO (Anej): UPDATE; This should use the benchmark_VAE package
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import os

import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy import stats

from pythae.pipelines import TrainingPipeline
from pythae.models import BetaVAE, BetaVAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.models import AutoModel
from pythae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig

from latte.dataset.dsprites import DSpritesFactor
from latte.dataset import dsprites, utils as dsutils
from latte.models.mist import estimation
from latte.utils import visualisation, evaluation
from latte.models.vae import utils as vaeutils
from latte.models.benchmarks.dsprites import (
    DSpritesConvEncoder,
    DSpritesConvDecoder,
    DSpritesFCEncoder,
    DSpritesFCDecoder,
)
from latte.models.vae.projection_vae import ProjectionBaseEncoder, ProjectionBaseDecoder
import latte.hydra_utils as hy

import logging
import coloredlogs

coloredlogs.install()


@hy.config
@dataclass
class MISTdSpritesConfig:
    """
    The configuration of the experiment for training MIST on the dSprites data.

    Members:
        dataset_filepath: The path to the dSprites dataset file.
        latent_size: The size of the representation space of the VAE to use.
        beta: The beta value for the BetaVAE
        subspace_size: The size of the subspace capturing the factors of interest to consider.
        target_factors: The factors of interest.
                        A separate subspace will be found for each one of them.
        ignored_factors: Factors in the dataset to ignore.
        network_type: Chooses between the "conv" and "FC" implementations of the VAE components available.
        trained_vae_dir: The path to the directory where an optional pre-trained VAE is stored.
                         If this is provided, no VAE is trained from scratch.
        labelled_ratio: The fraction of the labelled examples to use for fitting the MIST model.
        noise_amount: The amount of salt and pepper noise to inject into the images when training the MIST model.
        network_width: The script uses the default implemented MINE statistics network.
                       This specified the width to be used.
        vae_max_epochs: Number of epochs to train the VAE for.
        mist_max_epochs: Maximum number of epochs for training MIST.
        vae_batch_size: Batch size for training the VAE.
        mist_batch_size: Batch size for training MIST.
        p_train: Fraction of the data used for training the VAE and MIST.
        p_val: Fraction of the data used for validation during training of the VAE and MIST.
        steps_saving: The number of epochs between each saving of the checkpoint.
        mist_learning_rate: Learning rate for training MIST.
        vae_learning_rate: Learning rate for training the VAE.
        manifold_learning_rate: Learning rate for optimising over the Stiefel manifold.
        n_gmm_components: Number of GMM components to use when fitting a GMM model to the marginal distributions
                          over the latent factors given the encoder.
                          This is used for sampling.
        model_path: The directory to save the VAE checkpoints to if training a new one.
        seed: The random seed.
        gpus: The `gpus` parameter for Pytorch Lightning.
        num_workers:  The `num_workers` parameter for Pytorch Lightning data modules.
    """

    dataset_filepath: Optional[str] = None
    latent_size: int = 8
    beta: float = 1
    subspace_size: int = 1
    target_factors: List[DSpritesFactor] = field(
        default_factory=lambda: [DSpritesFactor.X, DSpritesFactor.Y, DSpritesFactor.SCALE]
    )
    ignored_factors: List[DSpritesFactor] = field(
        default_factory=lambda: [DSpritesFactor.ORIENTATION, DSpritesFactor.SHAPE]
    )
    network_type: str = "conv"
    trained_vae_dir: Optional[str] = None
    labelled_ratio: float = 1.0
    noise_amount: float = 0.05
    network_width: int = 256
    vae_max_epochs: int = 64
    mist_max_epochs: int = 256
    vae_batch_size: int = 32
    mist_batch_size: int = 128
    p_train: float = 0.8
    p_val: float = 0.1
    steps_saving: int = 4
    mist_learning_rate: float = 1e-4
    vae_learning_rate: float = 2.5e-4
    manifold_learning_rate: float = 1e-2
    n_gmm_components: int = 16
    model_path: str = "mist_dSprites"
    seed: int = 42
    gpus: int = 1
    num_workers: int = 6


def _construct_pipeline(cfg: MISTdSpritesConfig) -> TrainingPipeline:
    """Constructs the `pythae` pipeline for training the VAE according to the config."""
    model_config = BetaVAEConfig(beta=cfg.beta, latent_dim=cfg.latent_size, reconstruction_loss="bce")

    # Initialise the BetaVAE
    model = BetaVAE(
        model_config=model_config,
        encoder=DSpritesConvEncoder(latent_size=cfg.latent_size)
        if cfg.network_type == "conv"
        else DSpritesFCEncoder(latent_size=cfg.latent_size),
        decoder=DSpritesConvDecoder(latent_size=cfg.latent_size)
        if cfg.network_type == "conv"
        else DSpritesFCDecoder(latent_size=cfg.latent_size),
    )

    # Set up the training configuration
    training_config = BaseTrainerConfig(
        output_dir=cfg.model_path,
        num_epochs=cfg.vae_max_epochs,
        learning_rate=cfg.vae_learning_rate,
        batch_size=cfg.vae_batch_size,
        steps_saving=cfg.steps_saving,
        steps_predict=None,
        no_cuda=cfg.gpus == 0,
    )

    # Build the Pipeline
    pipeline = TrainingPipeline(training_config=training_config, model=model)

    return pipeline


def _plot_traversals(
    vae_model: BetaVAE,
    X: torch.Tensor,
    Z: torch.Tensor,
    n_axes: int,
    version: str = "raw",
    factor_name: str = "",
    device: str = "cuda",
    rng: dsutils.RandomGenerator = 42,
) -> None:
    """A utility function to plot the traversals in the latent space of the `vae_model`."""
    rng = np.random.default_rng(rng)

    # Choose a starting point to traverse from
    starting_x = X[rng.choice(len(X))]

    factor_info = "" if version == "raw" else f"_factor_{factor_name}"

    visualisation.latent_traversals(
        vae_model,
        starting_x.to(device),
        n_axes=n_axes,
        Z=Z,
        file_name=f"{version}_vae_latent_traversals{factor_info}.png",
        cmap="Greys",
    )


def _train_unconstrained_mist(
    Z: torch.Tensor,
    Y: torch.Tensor,
    factor_ix: int,
    n_factors: int,
    labelled_ratio: float,
    latent_size: int,
    network_width: int,
    learning_rate: float,
    batch_size: int,
    num_workers: int,
    trainer_kwargs: Dict[str, Any],
    rng: dsutils.RandomGenerator,
) -> estimation.MISTResult:
    """Trains an unconstrained version of MIST to estimate the full amount of information captured about the factor
    at the index `factor_idx` in the full VAE representation space.
    Returns the MIST estimation result."""

    rng = np.random.default_rng(rng)
    # Select a *subset* of the dataset to be used for supervised training
    index = rng.choice(int(Y.shape[0]), size=int(labelled_ratio * Y.shape[0]), replace=False)

    # Train MIST on the entire representation space
    full_mi_estimation_result = estimation.find_subspace(
        X=Z[index],
        Z_max=Y[index, factor_ix].reshape((-1, 1)),
        Z_min=None,
        datamodule_kwargs=dict(num_workers=num_workers, batch_size=batch_size),
        model_kwargs=dict(
            x_size=latent_size,
            z_max_size=n_factors,
            z_min_size=None,
            mine_learning_rate=learning_rate,
            mine_network_width=network_width,
            club_network_width=None,
            subspace_size=latent_size,  # No projection;
            # this "needlessly" finds an orthogonal matrix, but we are not interested in it
            gamma=1.0,
        ),
        trainer_kwargs=dict(**trainer_kwargs),
    )
    return full_mi_estimation_result


def _plot_generated_samples(model: BetaVAE, X: torch.Tensor, Z: torch.Tensor, model_type: str, n_gmm_components: int):
    """A utility function to plot the generated images from the trained model."""

    # Train a GMM sampler on the marginal latent space according to the encoder.
    gmm_sampler = GaussianMixtureSampler(
        sampler_config=GaussianMixtureSamplerConfig(n_components=n_gmm_components), model=model
    )
    gmm_sampler.fit(X)

    # Calculate the log likelihood of the validation dataset according to this sampler.
    print(f"The log likelihood of the validation data is {gmm_sampler.gmm.score(Z)}.")

    # Generate and plot 3 * 8 images
    visualisation.generated_images(
        gmm_sampler.sample(num_samples=3 * 8),
        3,
        8,
        f"generated_images_{model_type}_model.png",
        cmap="Greys",
    )


def _train_manifold_mist(
    Z: torch.Tensor,
    Y: torch.Tensor,
    factor_ix: int,
    n_factors: int,
    labelled_ratio: float,
    latent_size: int,
    subspace_size: int,
    network_width: int,
    learning_rate: float,
    manifold_learning_rate: float,
    batch_size: int,
    num_workers: int,
    trainer_kwargs: Dict[str, Any],
    rng: dsutils.RandomGenerator,
) -> estimation.MISTResult:
    """Trains a "constrained" version of MIST to find the optimal linear subspace of the full representation space
    of dimensionality `subspace_size`.
    Returns the MIST estimation result."""

    rng = np.random.default_rng(rng)
    # Select a *subset* of the dataset to be used for supervised training
    index = rng.choice(int(Y.shape[0]), size=int(labelled_ratio * Y.shape[0]), replace=False)

    # Train the MIST model
    mi_estimation_result = estimation.find_subspace(
        X=Z[index],
        Z_max=Y[index, factor_ix].reshape((-1, 1)),
        Z_min=None,
        datamodule_kwargs=dict(num_workers=num_workers, batch_size=batch_size),
        model_kwargs=dict(
            x_size=latent_size,
            z_max_size=n_factors,
            z_min_size=None,
            mine_learning_rate=learning_rate,
            mine_network_width=network_width,
            manifold_learning_rate=manifold_learning_rate,
            club_network_width=None,
            subspace_size=subspace_size,
            gamma=1.0,
        ),
        trainer_kwargs=trainer_kwargs,
    )

    return mi_estimation_result


@hy.main
def main(cfg: MISTdSpritesConfig):

    pl.seed_everything(cfg.seed)

    device = "cuda" if cfg.gpus > 0 else "cpu"

    assert cfg.subspace_size <= cfg.latent_size, "The subspace size should be at most the size of the full space."

    # Load and split the dataset
    dataset = dsprites.load_dsprites_preprocessed(
        filepath=cfg.dataset_filepath,
        return_raw=True,
        factors=cfg.target_factors,
        ignored_factors=cfg.ignored_factors,
    )
    (X_train, X_val, X_test), (Y_train, Y_val, Y_test) = dsutils.split(
        [dataset.X, dataset.y], p_train=cfg.p_train, p_val=cfg.p_val, seed=cfg.seed
    )

    if cfg.trained_vae_dir is None:
        # Not using a trained model, so train a new one
        pipeline = _construct_pipeline(cfg)

        pipeline(
            train_data=X_train,
            eval_data=X_val,
        )

        # Load the best trained checkpoint
        last_training = sorted(os.listdir(cfg.model_path))[-1]
        trained_model = AutoModel.load_from_folder(os.path.join(cfg.model_path, last_training, "final_model")).to(
            device
        )

    else:
        # Using a trained model
        trained_model = AutoModel.load_from_folder(os.path.join(cfg.trained_vae_dir, "final_model")).to(device)

    # Get the latent representations given by the model
    Z_train = vaeutils.get_latent_representations(trained_model, X_train, cfg.latent_size)
    Z_val = vaeutils.get_latent_representations(trained_model, X_val, cfg.latent_size)

    _plot_generated_samples(trained_model, X_val, Z_val, "full", cfg.n_gmm_components)
    _plot_traversals(trained_model, X_val, Z_val, version="raw", n_axes=cfg.latent_size, device=device)
    visualisation.vae_reconstructions(
        dataset=X_val,
        vae_model=trained_model,
        nrows=3,
        ncols=11,
        file_name="original_reconstructions_only.png",
        cmap="Greys",
    )

    axes_factors_mi = evaluation.axis_factor_mutual_information(
        Z_val, Y_val, factor_names=[dsprites.factor_names[factor] for factor in cfg.target_factors]
    )
    axes_factors_mi.to_csv("raw_vae_factor_axes_mi.csv")

    logging.info("The mutual information between the target factors of variation and the original basis axes:")
    print(axes_factors_mi)

    variances = np.asarray([torch.var(Z_val[:, j]) for j in range(Z_val.shape[-1])])
    z1, z2 = np.argsort(-variances)[:2]

    for factor in cfg.target_factors:

        factor_ix, factor_name = dataset.factor2idx[factor], dsprites.factor_names[factor]

        logging.info(f"Training an unconstrained MINE model for the factor {factor_name}.")

        visualisation.factor_heatmap(
            Z_val[:, [z1, z2]],
            Y_val[:, factor_ix],
            target="ground-truth",
            file_name=f"factor_heatmap_{factor_name}.png",
        )

        # To calculate the full entropy of the factor of variation, we extract it from the dataset
        f = Z_train[:, factor_ix]
        h = stats.differential_entropy(f)
        logging.info(f"The entropy of the factor {factor_name} is {h: .3f}.")

        # Train an unconstrained MINE model to estimate the total amount of information
        # about the factor captured in the VAE latent space
        logging.info(f"Training unconstrained MIST for the factor {factor_name}.")
        early_stopping_callback = EarlyStopping(
            monitor="validation_mutual_information_mine",
            min_delta=1e-3,
            patience=25,
            verbose=False,
            mode="max",
        )
        unconstrained_training_result = _train_unconstrained_mist(
            Z=Z_train,
            Y=Y_train,
            factor_ix=factor_ix,
            n_factors=1,  # TODO (Anej): If we ever want to look into multiple factors at once, this can be changed
            labelled_ratio=cfg.labelled_ratio,
            latent_size=cfg.latent_size,
            network_width=cfg.network_width,
            learning_rate=cfg.mist_learning_rate,
            batch_size=cfg.mist_batch_size,
            num_workers=cfg.num_workers,
            trainer_kwargs=dict(
                max_epochs=cfg.mist_max_epochs,
                gpus=cfg.gpus,
                callbacks=[early_stopping_callback],
                enable_model_summary=False,
            ),
            rng=cfg.seed,
        )

        full_mi = unconstrained_training_result.mutual_information
        logging.info(
            f"The estimate of the captured mutual information between "
            f"the entire VAE space and {factor_name} is {full_mi: .3f}."
        )

        # Train a manifold MINE model to find the optimal lower-dimensional linear subspace capturing the
        # most information about the factor and estimate the amount of information about in captured in the subspace
        logging.info(f"Training constrained MIST for the factor {factor_name}.")
        early_stopping_callback = EarlyStopping(
            monitor="validation_mutual_information_mine",
            min_delta=1e-3,
            patience=25,
            verbose=False,
            mode="max",
        )
        constrained_training_result = _train_manifold_mist(
            Z=Z_train,
            Y=Y_train,
            factor_ix=factor_ix,
            n_factors=1,  # TODO (Anej): If we ever want to look into multiple factors at once, this can be changed
            labelled_ratio=cfg.labelled_ratio,
            latent_size=cfg.latent_size,
            subspace_size=cfg.subspace_size,
            network_width=cfg.network_width,
            learning_rate=cfg.mist_learning_rate,
            manifold_learning_rate=cfg.manifold_learning_rate,
            batch_size=cfg.mist_batch_size,
            num_workers=cfg.num_workers,
            trainer_kwargs=dict(
                max_epochs=cfg.mist_max_epochs,
                gpus=cfg.gpus,
                callbacks=[early_stopping_callback],
                enable_model_summary=False,
            ),
            rng=cfg.seed,
        )

        subspace_mi = constrained_training_result.mutual_information
        logging.info(
            f"The estimate of the captured mutual information between "
            f"the {cfg.subspace_size}-dimensional subspace and {factor_name} is {subspace_mi: .3f}."
        )

        A_hat = constrained_training_result.A.to(device)

        logging.info(f"The estimate projection matrix A_hat onto the linear subspace capturing {factor_name}:")
        print(A_hat)

        projection_model = BetaVAE(
            model_config=trained_model.model_config,
            encoder=ProjectionBaseEncoder(trained_model.encoder, A_hat),
            decoder=ProjectionBaseDecoder(trained_model.decoder, A_hat),
        )

        Z_val_projected = vaeutils.get_latent_representations(projection_model, X_val, cfg.subspace_size)
        Z_test_projected = vaeutils.get_latent_representations(projection_model, X_test, cfg.subspace_size)

        _plot_generated_samples(
            projection_model, X_val, Z_val_projected, f"projected_{factor_name}", cfg.n_gmm_components
        )
        _plot_traversals(
            projection_model,
            X_val,
            Z_val_projected,
            version="projected",
            n_axes=cfg.subspace_size,
            factor_name=dsprites.factor_names[factor],
            device=device,
        )
        visualisation.vae_reconstructions(
            dataset=X_val,
            vae_model=projection_model,
            nrows=3,
            ncols=11,
            file_name=f"projected_reconstructions_only_{factor_name}.png",
            cmap="Greys",
        )
        visualisation.vae_reconstructions(
            dataset=X_val,
            vae_model=trained_model,
            latent_transformation=lambda z: z @ A_hat @ A_hat.T,
            nrows=3,
            ncols=11,
            file_name=f"projected_reconstructions_together_{factor_name}.png",
            cmap="Greys",
        )

        axes_factors_mi = evaluation.axis_factor_mutual_information(
            Z_test_projected, Y_test, factor_names=[dsprites.factor_names[factor] for factor in cfg.target_factors]
        )
        axes_factors_mi.to_csv(f"projected_vae_factor_axes_mi_factor_{dsprites.factor_names[factor]}.csv")

        logging.info("The mutual information between the target factors of variation and the subspace basis axes:")
        print(axes_factors_mi)

        E_hat = torch.eye(cfg.latent_size).to(device) - A_hat @ A_hat.T
        erased_projection_model = BetaVAE(
            model_config=trained_model.model_config,
            encoder=ProjectionBaseEncoder(trained_model.encoder, E_hat),
            decoder=ProjectionBaseDecoder(trained_model.decoder, E_hat),
        )

        Z_train_erased = vaeutils.get_latent_representations(erased_projection_model, X_train, cfg.latent_size)
        Z_val_erased = vaeutils.get_latent_representations(erased_projection_model, X_val, cfg.latent_size)

        _plot_generated_samples(
            erased_projection_model, X_val, Z_val_erased, f"erased_{factor_name}", cfg.n_gmm_components
        )
        visualisation.vae_reconstructions(
            dataset=X_val,
            vae_model=erased_projection_model,
            nrows=3,
            ncols=11,
            file_name=f"erased_reconstructions_only_{factor_name}.png",
            cmap="Greys",
        )
        visualisation.vae_reconstructions(
            dataset=X_val,
            vae_model=trained_model,
            latent_transformation=lambda z: z @ E_hat @ E_hat.T,
            nrows=3,
            ncols=11,
            file_name=f"erased_reconstructions_together_{factor_name}.png",
            cmap="Greys",
        )

        logging.info(f"Training erased MIST model for the factor {factor_name}.")
        early_stopping_callback = EarlyStopping(
            monitor="validation_mutual_information_mine",
            min_delta=1e-3,
            patience=25,
            verbose=False,
            mode="max",
        )
        erased_training_result = _train_unconstrained_mist(
            Z=Z_train_erased,
            Y=Y_train,
            factor_ix=factor_ix,
            n_factors=1,  # TODO (Anej): If we ever want to look into multiple factors at once, this can be changed
            labelled_ratio=cfg.labelled_ratio,
            latent_size=cfg.latent_size,
            network_width=cfg.network_width,
            learning_rate=cfg.mist_learning_rate,
            batch_size=cfg.mist_batch_size,
            num_workers=cfg.num_workers,
            trainer_kwargs=dict(
                max_epochs=cfg.mist_max_epochs,
                gpus=cfg.gpus,
                callbacks=[early_stopping_callback],
                enable_model_summary=False,
            ),
            rng=cfg.seed,
        )
        erased_mi = erased_training_result.mutual_information
        logging.info(
            f"The estimate of the captured mutual information between "
            f"the erased subspace and {factor_name} is {erased_mi: .3f}."
        )


if __name__ == "__main__":
    main()
