"""
This is a simple script that trains a `pythae` VAE model on one of the included datasets and saves the
constructed representations for the entire dataset.
It is intentionally simple and customised to the included datasets since our goal is not to obtain state-of-the-art
VAE models but rather to analyse the representations which they construct with as few modification as possible

It currently works for the `CelebA` and the `Shapes3D` dataset, but it can easily be extended to other datasets as well.

Notes:
    For reproducibility purposes, the script assumes that the dataset is already split into the "train", "val",
    and "test" splits.
    The VAE will be trained on the train split and the validation split will be used for validation.
    Representations will be saved for all splits individually.
"""
from typing import Optional, Tuple, Any
import dataclasses
import os

import torch
import numpy as np
import pandas as pd
from torchvision import transforms

from tqdm import trange

from pythae.pipelines import TrainingPipeline
from pythae.models import (
    BetaVAEConfig,
    BetaTCVAEConfig,
    RHVAEConfig,
    AEConfig,
    FactorVAEConfig,
    IWAEConfig,
    INFOVAE_MMD_Config,
    VAMPConfig,
    DisentangledBetaVAEConfig,
    VAE,
    AutoModel,
    BaseAEConfig,
)
from pythae.trainers import BaseTrainerConfig, AdversarialTrainerConfig
from pythae.models.nn.benchmarks.celeba.resnets import Encoder_ResNet_VAE_CELEBA, Decoder_ResNet_AE_CELEBA
from pythae.models.nn.benchmarks.celeba.convnets import Encoder_Conv_VAE_CELEBA, Decoder_Conv_AE_CELEBA

from latte.models.vae import utils as vaeutils
from latte.models.benchmarks import conv_models
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
        dataset_paths: The paths to the dataset split files.
                       It should contain the paths to the train, validation, and test splits in that order.
        vae_flavour: The flavour of the VAE to train.
                     Currently, this can be "BetaVAE", "BetaTCVAE" or "RHVAE".
        network_type: If there are multiple implementations of the VAE components available, this chooses one.
        seed: The seed for the training.
        latent_size: The dimensionality of the latent space
        alpha, beta, gamma, lbd, C, warmup_epoch, number_samples, number_components, kernel_choice, kernel_width:
            The parameters for the specified VAEs
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
    dataset_paths: Tuple[str, str, str]
    vae_flavour: str = "BetaVAE"
    network_type: str = "resnet"
    seed: int = 42
    latent_size: int = 10
    alpha: float = 1
    beta: float = 1
    gamma: float = 1
    lbd: float = 10
    C: float = 30
    warmup_epoch: int = 25
    number_samples: int = 3
    kernel_choice: str = "imq"
    kernel_bandwidth: float = 1
    number_components: int = 50
    model_path: str = "vae_train"
    loss: str = "mse"
    max_num_epochs: int = 40
    learning_rate: float = 1e-4
    batch_size: int = 128
    steps_saving: Optional[int] = 4
    no_cuda: bool = False


def get_dataset(dataset: str, paths: Tuple[str, str, str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads the data (the observation only) for the specified dataset.
    Args:
        dataset: The dataset to load.
        paths: The paths to the data train, val, and test datasets in that order.

    Returns:
        The data in the form of a Tensor.
    """
    print("Loading the dataset.")
    X_train = torch.from_numpy(np.load(paths[0]))
    X_val = torch.from_numpy(np.load(paths[0]))
    X_test = torch.from_numpy(np.load(paths[1]))

    if dataset in ["morphomnist"]:
        resize = transforms.Resize((32, 32))
        X_train, X_val, X_test = (resize(x) for x in [X_train, X_val, X_test])

    if dataset in ["celeba", "shapes3d", "morphomnist"]:
        X_train, X_val, X_test = (x.float() / 255 for x in [X_train, X_val, X_test])
    elif dataset in ["dsprites"]:
        X_train, X_val, X_test = (x.float() for x in [X_train, X_val, X_test])

    print("Dataset loaded.")
    return X_train, X_val, X_test


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
        return lambda c: conv_models.ConvEncoder(c, n_channels=1), lambda c: conv_models.ConvDecoder(c, n_channels=1)
    elif dataset in ["shapes3d", "celeba"]:
        assert network_type in ["conv", "resnet"]
        if network_type == "conv":
            return conv_models.ConvEncoder, conv_models.ConvDecoder
        if network_type == "resnet":
            return Encoder_ResNet_VAE_CELEBA, Decoder_ResNet_AE_CELEBA
        else:
            return Encoder_Conv_VAE_CELEBA, Decoder_Conv_AE_CELEBA
    elif dataset == "morphomnist":
        return conv_models.ConvEncoderMNIST, conv_models.ConvDecoderMNIST
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

    model_outputs = dict(reconstruction_losses=0.0, reg_losses=0.0, losses=0.0)

    # Run the entire dataset `X_test` through the model
    model.eval()
    with torch.no_grad():
        for ix in trange(0, len(X_test), batch_size):
            x = X_test[ix : ix + batch_size].to(device)
            y = model({"data": x})
            model_outputs["reconstruction_losses"] += y.reconstruction_loss.detach().sum().cpu().numpy()
            model_outputs["reg_losses"] += y.reg_loss.detach().sum().cpu().numpy()
            model_outputs["losses"] += y.loss.detach().sum().cpu().numpy()
    model.train()

    results = {
        "Value": {
            "Mean reconstruction loss": model_outputs["reconstruction_losses"] / len(X_test),
            "Mean KL loss": model_outputs["reg_losses"] / len(X_test),
            "ELBO": model_outputs["losses"],
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


def _get_model_config(cfg: VAETrainConfig) -> BaseAEConfig:  # noqa: C901
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
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=cfg.gamma,
            latent_dim=cfg.latent_size,
            reconstruction_loss=cfg.loss,
        )
    elif cfg.vae_flavour == "AE":
        return AEConfig(
            latent_dim=cfg.latent_size,
            reconstruction_loss=cfg.loss,
        )
    elif cfg.vae_flavour == "FactorVAE":
        return FactorVAEConfig(
            latent_dim=cfg.latent_size,
            reconstruction_loss=cfg.loss,
            gamma=cfg.gamma,
        )
    elif cfg.vae_flavour == "IWAE":
        return IWAEConfig(
            latent_dim=cfg.latent_size,
            reconstruction_loss=cfg.loss,
            number_samples=cfg.number_samples,
        )
    elif cfg.vae_flavour == "INFOVAE_MMD":
        return INFOVAE_MMD_Config(
            latent_dim=cfg.latent_size,
            reconstruction_loss=cfg.loss,
            kernel_choice=cfg.kernel_choice,
            alpha=cfg.alpha,
            lbd=cfg.lbd,
            kernel_bandwidth=cfg.kernel_bandwidth,
        )
    elif cfg.vae_flavour == "VAMP":
        return VAMPConfig(
            input_dim=_get_input_dim(cfg.dataset),
            latent_dim=cfg.latent_size,
            reconstruction_loss=cfg.loss,
            number_components=cfg.number_components,
        )
    elif cfg.vae_flavour == "DisentangledBetaVAE":
        return DisentangledBetaVAEConfig(
            latent_dim=cfg.latent_size,
            reconstruction_loss=cfg.loss,
            beta=cfg.beta,
            C=cfg.C,
            warmup_epoch=cfg.warmup_epoch,
        )
    else:
        raise NotImplementedError

    return model_config


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
    Vae = vaeutils.get_vae_class(cfg.vae_flavour)

    # Initialise and return the model
    return Vae(
        model_config=model_config,
        encoder=Encoder(model_config),
        decoder=Decoder(model_config),
    )


def _save_representations(
    trained_model: VAE, X_train: torch.Tensor, X_val: torch.Tensor, X_test: torch.Tensor, cfg: VAETrainConfig
) -> None:
    base_fname = (
        f"{cfg.dataset}_{cfg.vae_flavour}_b{cfg.beta}_C{cfg.C}"
        f"_dim{cfg.latent_size}_l{cfg.loss}_m{cfg.network_type}_s{cfg.seed}"
    )
    for X, split in zip([X_train, X_val, X_test], ["train", "val", "test"]):
        vaeutils.save_representations(trained_model, X, f"{base_fname}_{split}.pt", 4096)


@hy.main
def main(cfg: VAETrainConfig):
    assert cfg.dataset in ["dsprites", "celeba", "shapes3d", "morphomnist"], f"Dataset {cfg.dataset} is not supported."
    assert cfg.dataset_paths is not None

    device = "cpu" if cfg.no_cuda else "cuda"

    # Load and split the data
    X_train, X_val, X_test = get_dataset(cfg.dataset, cfg.dataset_paths)

    # Construct the model
    model = _get_model(cfg)

    # Construct the training config according to the script configuration
    TrainerConfig = BaseTrainerConfig if cfg.vae_flavour != "FactorVAE" else AdversarialTrainerConfig
    training_config = TrainerConfig(
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
    trained_model = AutoModel.load_from_folder(os.path.join(cfg.model_path, last_training, "final_model")).to(device)

    # Evaluate the final model
    evaluation_results = evaluate_trained_model(trained_model, X_test, batch_size=4096, device=device)
    print(">>>>>>>>>>>>>>> Evaluation results <<<<<<<<<<<<<<")
    print(evaluation_results)
    evaluation_results.to_csv(f"trained_vae_evaluation_results_{cfg.dataset}.csv")

    print("Saving the representations.")
    _save_representations(trained_model, X_train, X_val, X_test, cfg)

    print("DONE!")


if __name__ == "__main__":
    main()
