"""
This scripts trains a (beta)VAE on the dSprites dataset and finds the optimal linear subspace capturing a selected set
of factor of variation.

To qualitatively evaluate the disentanglement, latent traversals of the original and transformed axes are plotted.

"""
import dataclasses
from typing import Any, Optional, Tuple, Dict

import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from sklearn.feature_selection import mutual_info_regression

from latte.models.datamodules import DSpritesDataModule
from latte.dataset.dsprites import DSpritesFactor
from latte.dataset import dsprites, utils as dsutils
from latte.models import vae
from latte.models.vae import BetaVAE, VAEReconstructionLossType, View
from latte.mine import estimation
from latte.utils import utils, visualisation, evaluation
import latte.hydra_utils as hy

import logging
import coloredlogs

coloredlogs.install()


def _construct_model(latent_size: int, beta: float) -> BetaVAE:
    encoder = nn.Sequential(
        nn.Conv2d(1, 32, 4, 2, 1),  # B,  32, 32, 32
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
        nn.ReLU(),
        nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
        nn.ReLU(),
        View((-1, 256)),  # B, 256
        nn.Linear(256, latent_size * 2),  # B, latent_size*2
    )

    decoder = nn.Sequential(
        nn.Linear(latent_size, 256),  # B, 256
        View((-1, 256, 1, 1)),  # B, 256,  1,  1
        nn.ReLU(),
        nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
        nn.ReLU(),
        nn.ConvTranspose2d(32, 1, 4, 2, 1),  # B, nc, 64, 64
    )

    beta = beta * latent_size / (64 * 64 * 1)

    vae_model = BetaVAE(
        latent_size=latent_size,
        encoder=encoder,
        decoder=decoder,
        loss_type=VAEReconstructionLossType.bernoulli,
        beta=beta,
    )

    return vae_model


def _train_vae_model(vae_model: BetaVAE, data: DSpritesDataModule, trainer_kwargs: Dict[str, Any]) -> None:
    logging.info("Training the VAE.")

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(vae_model, datamodule=data)

    trainer.test(model=vae_model, dataloaders=data)


def _plot_traversals(
    vae_model: BetaVAE,
    data: DSpritesDataModule,
    M: Dict[str, torch.Tensor],
    n_axes: int,
    A_hat: Optional[torch.Tensor] = None,
    factor_name: str = "",
    rng: dsutils.RandomGenerator = 42,
):
    rng = np.random.default_rng(rng)

    batch = next(iter(data.val_dataloader()))[0]
    starting_x = batch[rng.integers(0, len(batch))]

    version = "raw" if A_hat is None else "projected"
    factor_info = "" if A_hat is None else f"_factor_{factor_name}"

    visualisation.plot_latent_traversals(
        vae_model, starting_x, M, A=A_hat, n_axes=n_axes, file_name=f"{version}_vae_latent_traversals{factor_info}.png"
    )


def _train_unconstrained_mine(
    Z: Dict[str, torch.Tensor],
    F: Dict[str, torch.Tensor],
    factor_ix: int,
    n_factors: int,
    labelled_ratio: float,
    latent_size: int,
    network_width: int,
    learning_rate: float,
    num_workers: int,
    trainer_kwargs: Dict[str, Any],
    rng: dsutils.RandomGenerator,
) -> estimation.MutualInformationEstimationResult:

    rng = np.random.default_rng(rng)
    # Select a *subset* of the dataset to be used for supervised training
    index = {
        split: rng.choice(int(f.shape[0]), size=int(labelled_ratio * f.shape[0]), replace=False)
        for split, f in F.items()
    }

    full_mi_estimation_result = estimation.estimate_mutual_information(
        subspace_estimation=False,
        X={split: z[index[split], :] for split, z in Z.items()},
        Z={split: f[index[split], factor_ix].reshape((-1, 1)) for split, f in F.items()},
        data_presplit=True,
        datamodule_kwargs=dict(num_workers=num_workers),
        model_kwargs=dict(
            x_size=latent_size, z_size=n_factors, learning_rate=learning_rate, network_width=network_width
        ),
        trainer_kwargs=dict(**trainer_kwargs),
    )
    return full_mi_estimation_result


def _train_manifold_mine(
    Z: Dict[str, torch.Tensor],
    F: Dict[str, torch.Tensor],
    factor_ix: int,
    n_factors: int,
    labelled_ratio: float,
    latent_size: int,
    subspace_size: int,
    network_width: int,
    learning_rate: float,
    manifold_learning_rate: float,
    num_workers: int,
    trainer_kwargs: Dict[str, Any],
    rng: dsutils.RandomGenerator,
) -> estimation.MutualInformationEstimationResult:

    rng = np.random.default_rng(rng)
    # Select a *subset* of the dataset to be used for supervised training
    index = {
        split: rng.choice(int(f.shape[0]), size=int(labelled_ratio * f.shape[0]), replace=False)
        for split, f in F.items()
    }

    mi_estimation_result = estimation.estimate_mutual_information(
        subspace_estimation=True,
        X={split: z[index[split], :] for split, z in Z.items()},
        Z={split: f[index[split], factor_ix].reshape((-1, 1)) for split, f in F.items()},
        data_presplit=True,
        datamodule_kwargs=dict(num_workers=num_workers),
        model_kwargs=dict(
            x_size=latent_size,
            z_size=n_factors,
            network_width=network_width,
            learning_rate=learning_rate,
            manifold_learning_rate=manifold_learning_rate,
            subspace_size=subspace_size,
        ),
        trainer_kwargs=trainer_kwargs,
    )

    return mi_estimation_result


@hy.config
@dataclasses.dataclass
class DSpritesVAEConfig:
    """
    The configuration of the experiment for training a simple VAE on the dSprites data.

    Attributes:
        latent_size: The dimensionality of the VAE latent space
        beta: The beta value for the beta-VAE
        subspace_size: The dimensionality of the subspace ("k" above)
        target_factors: Which factors to keep the information about in the dataset
        ignored_factors: Which factors should be held fixed (all the variation in them will be removed)
        labelled_ratio: The ratio of the dataset that we assume is labelled.
                        This controls how many labelled points MINE model sees
                        to be able to reconstruct the linear subspace.
        noise_amount: The amount (ratio) of salt and pepper noise to add to the images
        network_width: The width of the MINE statistics network
        seed: The seed for the random generators
    """

    latent_size: int = 10
    beta: float = 1
    subspace_size: int = 1
    target_factors: Tuple = (
        DSpritesFactor.X,
        DSpritesFactor.Y,
        DSpritesFactor.SCALE,
    )
    ignored_factors: Tuple = (
        DSpritesFactor.ORIENTATION,
        DSpritesFactor.SHAPE,
    )
    labelled_ratio: float = 1.0
    noise_amount: float = 0.1
    network_width: int = 128
    vae_max_epochs: int = 128
    mine_max_epochs: int = 128
    batch_size: int = 128
    p_train: float = 0.7
    p_val: float = 0.1
    learning_rate: float = 1e-4
    manifold_learning_rate: float = 1e-2
    seed: int = 42
    gpus: int = 1
    num_workers: int = 6


@hy.main
def main(cfg: DSpritesVAEConfig):

    pl.seed_everything(cfg.seed)

    assert cfg.subspace_size <= cfg.latent_size, "The subspace size should be at most the size of the full space."

    data = DSpritesDataModule(
        ignored_factors=list(cfg.ignored_factors),
        target_factors=list(cfg.target_factors),
        raw=True,
        batch_size=cfg.batch_size,
        p_train=cfg.p_train,
        p_val=cfg.p_val,
        num_workers=cfg.num_workers,
        noise_amount=cfg.noise_amount,
    )

    vae_model = _construct_model(cfg.latent_size, cfg.beta)

    _train_vae_model(
        vae_model=vae_model,
        data=data,
        trainer_kwargs=dict(max_epochs=cfg.vae_max_epochs, gpus=cfg.gpus, enable_model_summary=False),
    )

    visualisation.plot_originals_and_reconstructions(
        data, vae_model, rng=32, file_name="raw_vae_dsprites_reconstructions.png"
    )

    Z, F = vae.get_representations(vae_model, data)

    _plot_traversals(vae_model, data, Z, n_axes=cfg.latent_size)

    axes_factors_mi = utils.axis_factor_mutual_information(
        Z["test"], F["test"], factor_names=[dsprites.factor_names[factor] for factor in cfg.target_factors]
    )
    axes_factors_mi.to_csv("raw_vae_factor_axes_mi.csv")

    logging.info("The mutual information between the target factors of variation and the original basis axes:")
    print(axes_factors_mi)

    for factor in cfg.target_factors:

        factor_ix, factor_name = data.dataset.factor2idx[factor], dsprites.factor_names[factor]

        logging.info(f"Training an unconstrained MINE model for the factor {factor_name}.")

        # To calculate the full entropy of the factor of variation, we extract it from the dataset
        f = F["train"][:, factor_ix]
        h = mutual_info_regression(f.reshape((-1, 1)), f)[0]
        logging.info(f"The entropy of the factor {factor_name} is {h: .3f}.")

        # Train an unconstrained MINE model to estimate the total amount of information
        # about the factor captured in the VAE latent space
        unconstrained_training_result = _train_unconstrained_mine(
            Z=Z,
            F=F,
            factor_ix=factor_ix,
            n_factors=1,  # TODO (Anej): If we ever want to look into multiple factors at once, this can be changed
            labelled_ratio=cfg.labelled_ratio,
            latent_size=cfg.latent_size,
            network_width=cfg.network_width,
            learning_rate=cfg.learning_rate,
            num_workers=cfg.num_workers,
            trainer_kwargs=dict(max_epochs=cfg.mine_max_epochs, gpus=cfg.gpus, enable_model_summary=False),
            rng=cfg.seed,
        )

        full_mi = unconstrained_training_result.mutual_information
        full_mi_val = unconstrained_training_result.validation_mutual_information
        logging.info(
            f"The estimate of the captured mutual information between "
            f"the entire VAE space and {factor_name} is {full_mi: .3f} (validation was {full_mi_val: .3f})."
        )

        # Train a manifold MINE model to find the optimal lower-dimensional linear subspace capturing the
        # most information about the factor and estimate the amount of information about in captured in the subspace
        logging.info(f"Training a manifold MINE model for the factor {factor_name}.")
        constrained_training_result = _train_manifold_mine(
            Z=Z,
            F=F,
            factor_ix=factor_ix,
            n_factors=1,  # TODO (Anej): If we ever want to look into multiple factors at once, this can be changed
            labelled_ratio=cfg.labelled_ratio,
            latent_size=cfg.latent_size,
            subspace_size=cfg.subspace_size,
            network_width=cfg.network_width,
            learning_rate=cfg.learning_rate,
            manifold_learning_rate=cfg.manifold_learning_rate,
            num_workers=cfg.num_workers,
            trainer_kwargs=dict(max_epochs=cfg.mine_max_epochs, gpus=cfg.gpus, enable_model_summary=False),
            rng=cfg.seed,
        )

        subspace_mi = constrained_training_result.mutual_information
        subspace_mi_val = constrained_training_result.validation_mutual_information
        logging.info(
            f"The estimate of the captured mutual information between "
            f"the {cfg.subspace_size}-dimensional subspace and {factor_name} is "
            f"{subspace_mi: .3f} (validation was {subspace_mi_val: .3f})."
        )

        A_hat = constrained_training_result.subspace

        logging.info(f"The estimate projection matrix A_hat onto the linear subspace capturing {factor_name}:")
        print(A_hat)

        T, _ = vae.get_representations(vae_model, data, A=A_hat)
        _plot_traversals(
            vae_model, data, T, n_axes=cfg.subspace_size, A_hat=A_hat, factor_name=dsprites.factor_names[factor]
        )

        alignment_evaluation = evaluation.evaluate_alignment_to_axes(A_hat.detach().cpu().numpy())
        print(alignment_evaluation)

        axes_factors_mi = utils.axis_factor_mutual_information(
            T["test"], F["test"], factor_names=[dsprites.factor_names[factor] for factor in cfg.target_factors]
        )
        axes_factors_mi.to_csv(f"projected_vae_factor_axes_mi_factor_{dsprites.factor_names[factor]}.csv")

        logging.info("The mutual information between the target factors of variation and the subspace basis axes:")
        print(axes_factors_mi)


if __name__ == "__main__":
    main()
