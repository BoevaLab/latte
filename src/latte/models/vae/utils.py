"""
Utils for various aspects of working with VAEs.
Implementations are meant to work with the `pythae` library.
"""
from typing import Any

import torch

from tqdm import trange

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
)


def get_vae_class(vae_flavour: str) -> Any:  # noqa: C901
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


def get_latent_representations(
    model: VAE, X: torch.Tensor, batch_size: int = 2048, device: str = "cuda"
) -> torch.Tensor:
    """
    A utility function to get the latent representations given by the encoder model of the VAE to the observations
    in X.
    Args:
        model: The VAE with to use for encoding.
        X: The observations to produce the latent representations for.
        batch_size: The batch size to use.
        device: The device to load the data to.

    Returns:
        The latent representations (the means of the posteriors) given by the encoder of the model.
    """
    model.eval()
    with torch.no_grad():
        Z = torch.zeros(size=(len(X), model.model_config.latent_dim))
        for ix in trange(0, len(X), batch_size):
            x = X[ix : ix + batch_size].to(device)
            y = model.encoder(x)
            Z[ix : ix + batch_size, :] = y.embedding.detach().cpu()
    model.train()

    return Z


def save_representations(
    trained_model: VAE,
    X: torch.Tensor,
    file_name: str,
    batch_size: int = 4096,
) -> None:
    """
    Constructs the representations of the data points in `X` (taken to be the means of their posteriors) as produced by
    the VAE `trained_model` and saves them as a `pytorch` tensor file under the name `file_name`.
    Args:
        trained_model: The VAE to use to construct the representations.
        X: The dataset of observations
        file_name: The file name under which to save the representations
        batch_size: The batch size to use for inference.

    """
    Z = get_latent_representations(trained_model, X, batch_size)
    torch.save(Z, open(file_name, "wb"))
