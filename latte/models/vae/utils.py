"""
Utils for various aspects of working with VAEs.
Implementations are meant to work with the `pythae` library.
"""

import torch

from tqdm import trange

from pythae.models.vae import vae_model


def get_latent_representations(
    model: vae_model.VAE, X: torch.Tensor, latent_size: int, batch_size: int = 2048, device: str = "cuda"
) -> torch.Tensor:
    """
    A utility function to get the latent representations given by the encoder model of the VAE to the observations
    in X.
    Args:
        model: The VAE with to use for encoding.
        X: The observations to produce the latent representations for.
        latent_size: The dimensionality of the latent space.
        batch_size: The batch size to use.
        device: The device to load the data to.

    Returns:
        The latent representations (the means of the posteriors) given by the encoder of the model.
    """
    model.eval()
    with torch.no_grad():
        Z = torch.zeros(size=(len(X), latent_size))
        for ix in trange(0, len(X), batch_size):
            x = X[ix : ix + batch_size].to(device)
            y = model.encoder(x)
            Z[ix : ix + batch_size, :] = y.embedding.detach().cpu()
    model.train()

    return Z
