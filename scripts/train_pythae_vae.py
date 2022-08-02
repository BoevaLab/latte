"""
This is a simple script that trains a `pythae` VAE model on one of the included datasets.
It is intentionally simple and customised to the included datasets since our goal is not to obtain state-of-the-art
VAE models but rather to analyse the representations which they construct with as few modification as possible

It currently works for the `CelebA` and the `Shapes3D` dataset, but it can easily be extended to other datasets as well.

Currently, this focuses on beta-VAE, beta-TCVAE and RHVAE models, but it can easily be modified to be more general.
"""
from typing import Optional, Tuple, Any
import dataclasses
import os

import torch
import numpy as np
import pandas as pd

from tqdm import trange

from pythae.pipelines import TrainingPipeline
from pythae.models import BetaVAE, BetaVAEConfig, BetaTCVAE, BetaTCVAEConfig, RHVAEConfig, RHVAE, VAE, AutoModel
from pythae.models.vae.vae_config import VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.models.nn.benchmarks.celeba.resnets import Encoder_ResNet_VAE_CELEBA, Decoder_ResNet_AE_CELEBA
from pythae.models.nn.benchmarks.celeba.convnets import Encoder_Conv_VAE_CELEBA, Decoder_Conv_AE_CELEBA

from latte.dataset import utils as dsutils, shapes3d
from latte.modules.callbacks import TensorboardCallback
import latte.hydra_utils as hy


@hy.config
@dataclasses.dataclass
class VAETrainConfig:
    """
    The configuration object for the training script.

    Members:
        dataset: The dataset to train on.
                 Currently, this can be "CelebA" or "Shapes3D"
        dataset_path: The path to the dataset file.
        vae_flavour: The flavour of the VAE to train.
                     Currently, this can be "BetaVAE" or "RHVAE".
        network_type: If there are multiple implementations of the VAE components available, this chooses one.
        seed: The seed for the training.
        p_train: The fraction of the data to use for training.
        p_val: The fraction of the data to use for validation.
        latent_size: The dimensionality of the latent space
        beta: The beta value for the BetaVAE
        model_path: The name of the directory to save the trained checkpoint to.
        loss: The type of loss to use in the `pythae` library.
              This can be "bce" or "mse".
        max_num_epochs: The maximum number of epochs to train the VAE for.
        learning_rate: The VAE learning rate.
        batch_size: The batch size to use in the VAE training.
        steps_saving: The number of epochs between each saving of the checkpoint.
        no_cuda: Whether to not use cuda and just train on the CPU
    """

    dataset: str
    dataset_path: str
    vae_flavour: str = "BetaVAE"
    network_type: str = "resnet"
    seed: int = 42
    p_train: float = 0.8
    p_val: float = 0.1
    latent_size: int = 10
    beta: float = 1
    model_path: str = "vae_train"
    loss: str = "mse"
    max_num_epochs: int = 40
    learning_rate: float = 1e-4
    batch_size: int = 128
    steps_saving: Optional[int] = 4
    no_cuda: bool = False


def get_dataset(dataset: str, path: str) -> torch.Tensor:
    """
    Loads the data (the observation only) for the specified dataset.
    Args:
        dataset: The dataset to load.
        path: The path to the data file.

    Returns:
        The data in the form of a Tensor.
    """
    print("Loading the dataset.")
    if dataset == "dsprites":
        ...
    elif dataset == "shapes3d":
        dataset = shapes3d.load_shapes3d(filepath=path)
        X = dataset.imgs[:]
        del dataset
        X = torch.from_numpy(np.transpose(X, (0, 3, 1, 2)).astype(np.float32) / 255)
    elif dataset == "celeba":
        X = torch.from_numpy(np.load(path))
        X /= 255

    print("Dataset loaded.")
    return X


def _get_model_classes(dataset: str, network_type: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Returns the classes of the VAE components to be used in by the VAE according to the dataset.
    Args:
        dataset: The dataset specified.
        network_type: Applicable when multiple implementations are available for the particular dataset.
                      Chooses the desired option.

    Returns:
        The encoder and decoder class.
    """
    if dataset == "dsprites":
        return None, None  # Not implemented yet
    elif dataset in ["shapes3d", "celeba"]:
        assert network_type in ["conv", "resnet"]
        if network_type == "resnet":
            return Encoder_ResNet_VAE_CELEBA, Decoder_ResNet_AE_CELEBA
        else:
            return Encoder_Conv_VAE_CELEBA, Decoder_Conv_AE_CELEBA
    else:  # Not implemented yet
        return None, None


def evaluate_trained_model(model: VAE, X_test: torch.Tensor, batch_size: int, device: str) -> pd.DataFrame:
    """
    Evaluates a trained model by looking at the reconstruction and regularisation (KL) values on held-out data, as well
    as the total value of the ELBO on this data.
    Args:
        model: The trained model.
        X_test: Held-out data to test the model on.
        batch_size: Batch size to use.
        device: Device to load the data to.

    Returns:
        A dataframe holding the results of the evaluation.
    """

    print("Evaluating the model.")

    model_outputs = dict(reconstruction_losses=[], reg_losses=[], losses=[])

    # Run the entire dataset `X_test` through the model
    model.eval()
    with torch.no_grad():
        for ix in trange(0, len(X_test), batch_size):
            x = X_test[ix : ix + batch_size].to(device)
            y = model({"data": x})
            model_outputs["reconstruction_losses"].append(y.reconstruction_loss.detach().cpu().numpy())
            model_outputs["reg_losses"].append(y.reg_loss.detach().cpu().numpy())
            model_outputs["losses"].append(y.loss.detach().cpu().numpy())
    model.train()

    results = {
        "Value": {
            "Mean reconstruction loss": np.mean(model_outputs["reconstruction_losses"]),
            "Mean KL loss": np.mean(model_outputs["reg_losses"]),
            "ELBO": np.sum(model_outputs["losses"]),
        }
    }

    return pd.DataFrame(results)


def _get_input_dim(dataset: str) -> Tuple[int, int, int]:
    """
    Returns the appropriate data shape according to the dataset.
    Args:
        dataset: The specified dataset.

    Returns:
        The input shape
    """
    if dataset == "dsprites":
        return 1, 64, 64
    elif dataset in ["shapes3d", "celeba"]:
        return 3, 64, 64
    else:  # Not implemented yet
        return -1, -1, -1


def _get_model_config(cfg: VAETrainConfig) -> VAEConfig:
    """
    Constructs the `pythae` model configuration according to the config of the script.
    Args:
        cfg: The script configuration.

    Returns:
        The `pythae` model configuration.
    """

    if cfg.vae_flavour == "RHVAE":
        model_config = RHVAEConfig(
            input_dim=_get_input_dim(cfg.dataset),
            latent_dim=cfg.latent_size,
            n_lf=1,
            eps_lf=0.001,
            beta_zero=0.3,
            temperature=1.5,
            regularization=0.001,
        )
    elif cfg.vae_flavour == "BetaVAE":
        model_config = BetaVAEConfig(
            beta=cfg.beta,
            latent_dim=cfg.latent_size,
            reconstruction_loss=cfg.loss,
        )
    elif cfg.vae_flavour == "BetaTCVAE":
        model_config = BetaTCVAEConfig(
            beta=cfg.beta,
            latent_dim=cfg.latent_size,
            reconstruction_loss=cfg.loss,
        )

    return model_config


def _get_vae_class(vae_flavour: str) -> Any:
    """Returns the appropriate model class according to the specification."""
    if vae_flavour == "RHVAE":
        return RHVAE
    elif vae_flavour == "BetaVAE":
        return BetaVAE
    elif vae_flavour == "BetaTCVAE":
        return BetaTCVAE


def _get_model(cfg: VAETrainConfig) -> VAE:
    """
    Constructs the model to be trained based on the script configuration.
    Args:
        cfg: The script configuration.

    Returns:
        The configured model
    """

    # Construct the model configuration object
    model_config = _get_model_config(cfg)
    # Get the component classes
    Encoder, Decoder = _get_model_classes(cfg.dataset, cfg.network_type)
    # Get the full model class
    Vae = _get_vae_class(cfg.vae_flavour)

    # Initialise and return the model
    return Vae(
        model_config=model_config,
        encoder=Encoder(model_config),
        decoder=Decoder(model_config),
    )


@hy.main
def main(cfg: VAETrainConfig):
    assert cfg.dataset in ["dsprites", "celeba", "shapes3d"], f"Dataset {cfg.dataset} is not supported."
    assert cfg.dataset_path is not None
    assert cfg.vae_flavour in ["BetaVAE", "RHVAE", "BetaTCVAE"]

    device = "cpu" if cfg.no_cuda else "cuda"

    # Load and split the data
    X = get_dataset(cfg.dataset, cfg.dataset_path)
    X_train, X_val, X_test = dsutils.split(D=[X], p_train=cfg.p_train, p_val=cfg.p_val, seed=cfg.seed)[0]
    del X  # Save memory

    # Construct the model
    model = _get_model(cfg)

    # Construct the training config according to the script configuration
    training_config = BaseTrainerConfig(
        output_dir=cfg.model_path,
        num_epochs=cfg.max_num_epochs,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        steps_saving=cfg.steps_saving,
        steps_predict=None,
        no_cuda=cfg.no_cuda,
    )

    # Train the model with the `pythae` pipeline
    pipeline = TrainingPipeline(training_config=training_config, model=model)

    print("Training the model.")
    pipeline(train_data=X_train, eval_data=X_val, callbacks=[TensorboardCallback()])
    print("Model trained.")

    # Load the best model according to the validation data
    last_training = sorted(os.listdir(cfg.model_path))[-1]
    trained_model = AutoModel.load_from_folder(os.path.join(cfg.model_path, last_training, "final_model"))

    # Evaluate the final model
    evaluation_results = evaluate_trained_model(trained_model, X_test, batch_size=2048, device=device)
    print(">>>>>>>>>>>>>>> Evaluation results <<<<<<<<<<<<<<")
    print(evaluation_results)
    evaluation_results.to_csv(f"trained_vae_evaluation_results_{cfg.dataset}.csv")

    print("DONE!")


if __name__ == "__main__":
    main()
