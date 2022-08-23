"""
A module containing utility functions for the MIST model for estimating and optimising the mutual information over
linear subspaces using MINE and CLUB.
"""

from typing import Optional, Dict, Union, Any
from dataclasses import dataclass

import geoopt
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from latte.models.mine import mine

from latte.models.mist import supervised_mist
from latte.modules.data import datamodules as dm
from latte.manifolds import utils as mutils


@dataclass
class MISTResult:
    """A dataclass for holding the result of mutual information estimation using MINE

    Members:
        loss (float): The estimate of the mutual information as calculated on the test dataset
        validation_loss (float): The estimate of the mutual information
                                                as calculated on the validation dataset
        estimator (supervised_mist.SupervisedMIST): The model trained and used for estimation of the mutual information
        A: In case of also estimating the linear subspace capturing the most information, this holds the
                  d-frame defining the estimated subspace
    """

    mutual_information: float
    loss: float
    estimator: supervised_mist.SupervisedMIST
    A: geoopt.ManifoldTensor


def _construct_supervised_mist(
    x_size: int,
    z_max_size: int,
    subspace_size: int,
    mine_network_width: int,
    club_network_width: Optional[int] = None,
    z_min_size: Optional[int] = None,
    statistics_network: Optional[mine.ManifoldStatisticsNetwork] = None,
    club_density_estimators: Optional[Dict[str, nn.Module]] = None,
    mine_learning_rate: float = 1e-4,
    club_learning_rate: float = 1e-4,
    manifold_learning_rate: float = 1e-3,
    lr_scheduler_patience: int = 8,
    lr_scheduler_min_delta: float = 1e-3,
    alpha: float = 0.01,
    gamma: float = 1.0,
    n_density_updates: int = 0,
    verbose: bool = False,
    checkpoint_path: Optional[str] = None,
) -> supervised_mist.SupervisedMIST:
    """
    Constructs the Supervised MIST model to be used for finding the optimal subspace.
    It can either construct it from scratch or load a trained model.

    Args:
        x_size: Dimensionality of the observed distribution.
        z_max_size: Dimensionality of the distribution in regard to which mutual information should be maximised.
        z_min_size: Dimensionality of the distribution in regard to which mutual information should be minimised.
        subspace_size: Dimensionality of the subspace onto which the first distribution should be projected.
        mine_network_width: Size of the hidden layers of the standard `MINE` network.
                            Will be used if the `statistics_network` is not provided.
        club_network_width: Size of the hidden layers of the standard `MINE` network.
                            Will be used if the `club_density_estimators` is not provided.
        statistics_network: Statistics network for the MINE model. It will override `mine_network_width`
        club_density_estimators: A dictionary containing the two components of the density estimator of the CLUB model.
                                 If provided, it should contain an entry for "mean" and an entry for "log_variance".
        mine_learning_rate: Learning rate for the parameters of the MINE model.
        club_learning_rate: Learning rate for the parameters of the CLUB model.
        manifold_learning_rate: Learning rate for the projection matrix.
        lr_scheduler_patience: Patience for the learning rate schedulers.
        lr_scheduler_min_delta: Min delta for the learning rate schedulers.
        alpha: The alpha value of for the MINE loss function.
        gamma: Gamma value for the MIST model.
               It determines the contribution of the MINE and CLUB losses.
        n_density_updates: Number of updates to the density estimator of the CLUB model to perform on each batch.
        verbose: Verbosity of the learning rate schedulers.
        checkpoint_path: Path to a saved version of the model if loading a trained network.

    Returns:
        The constructed MIST model.
    """

    if statistics_network is None:
        statistics_network = mine.StatisticsNetwork(
            S=nn.Sequential(
                nn.Linear(subspace_size + z_max_size, mine_network_width),
                nn.ReLU(),
                nn.Linear(mine_network_width, mine_network_width),
                nn.ReLU(),
            ),
            out_dim=mine_network_width,
        )
    if club_density_estimators is None and gamma < 1.0:
        # If the CLUB density estimator is not provided and minimisation is intended, we construct the default networks
        club_density_estimators = {
            "mean": nn.Sequential(
                nn.Linear(subspace_size, club_network_width),
                nn.ReLU(),
                nn.Linear(club_network_width, club_network_width),
                nn.ReLU(),
                nn.Linear(club_network_width, club_network_width),
                nn.ReLU(),
                nn.Linear(club_network_width, z_min_size),
            ),
            "log_variance": nn.Sequential(
                nn.Linear(subspace_size, club_network_width),
                nn.ReLU(),
                nn.Linear(club_network_width, club_network_width),
                nn.ReLU(),
                nn.Linear(club_network_width, club_network_width),
                nn.ReLU(),
                nn.Linear(club_network_width, z_min_size),
            ),
        }

    # The arguments for the MINE model passed down to MIST
    mine_args = {
        "T": statistics_network,
        "kind": mine.MINEObjectiveType.MINE,
        "alpha": alpha,
    }
    # The arguments for the club model passed down to MIST
    # If no minimisation is to be done, they are left empty
    club_args = (
        {
            "mean_estimator": club_density_estimators["mean"],
            "log_variance_estimator": club_density_estimators["log_variance"],
        }
        if gamma < 1.0
        else None
    )

    # This is a convenience thing to enable using the same function to both construct a new model or load a trained
    # (best) one with the same setting of the hyperparameters.
    # It allows the function to be used to load the best checkpoint model based on the validation data.
    if checkpoint_path is None:
        model = supervised_mist.SupervisedMIST(
            n=x_size,
            d=subspace_size,
            mine_args=mine_args,
            club_args=club_args,
            gamma=gamma,
            mine_learning_rate=mine_learning_rate,
            club_learning_rate=club_learning_rate,
            manifold_learning_rate=manifold_learning_rate,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_min_delta=lr_scheduler_min_delta,
            n_density_updates=n_density_updates if gamma < 1.0 else 0,
            verbose=verbose,
        )
    else:
        model = supervised_mist.SupervisedMIST.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            n=x_size,
            d=subspace_size,
            mine_args=mine_args,
            club_args=club_args,
            gamma=gamma,
            mine_learning_rate=mine_learning_rate,
            club_learning_rate=club_learning_rate,
            manifold_learning_rate=manifold_learning_rate,
            n_density_updates=n_density_updates if gamma < 1.0 else 0,
            verbose=verbose,
        )

    return model


def _train_supervised_mist(
    model: supervised_mist.SupervisedMIST,
    data: dm.MISTDataModule,
    trainer_kwargs: Dict[str, Any],
    log_to_wb: bool = False,
    wb_run_name: Optional[str] = None,
) -> Dict[str, Union[str, Dict[str, float]]]:
    """
    Trains a constructed Supervised MIST model to find the projection matrix onto the optimal subspace.
    Returns:
        The path to the best trained model checkpoint (best on the validation set),
        and the results of evaluating the model on the validation and the test splits of the data.
    """

    # Optional Weights and Biases callback
    wandb_logger = None
    if log_to_wb:
        wandb_logger = WandbLogger(project="latte", name=wb_run_name)

    callbacks = trainer_kwargs.get("callbacks", [])
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="validation_loss", mode="min")
    callbacks.append(checkpoint_callback)
    trainer_kwargs["callbacks"] = callbacks

    # If progress should be logged to W&B, pass the logger, else use the default (True)
    trainer = pl.Trainer(**trainer_kwargs, logger=wandb_logger if log_to_wb else True)

    # Train the model to find the subspace
    trainer.fit(model, datamodule=data)

    # Validate the model
    # On the validation split
    validation_results = trainer.validate(ckpt_path="best", dataloaders=data, verbose=False)
    # On the test split
    test_results = trainer.test(ckpt_path="best", dataloaders=data, verbose=False)

    return {
        "best_model_path": checkpoint_callback.best_model_path,
        "test_results": test_results[0],  # We only use 1 dataloader, so we return the first element of the results
        "validation_results": validation_results[0],
    }


def find_subspace(
    X: Union[torch.Tensor, Dict[str, torch.Tensor]],
    Z_max: Union[torch.Tensor, Dict[str, torch.Tensor]],
    Z_min: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
    data_presplit: bool = False,
    log_to_wb: Optional[bool] = False,
    wb_run_name: Optional[str] = None,
    datamodule_kwargs: Optional[Dict] = None,
    model_kwargs: Optional[Dict] = None,
    trainer_kwargs: Optional[Dict] = None,
) -> MISTResult:
    """
    Main function of the module.
    It uses the Supervised MIST model to find the optimal linear subspace (dimensionality specified in `model_kwargs`)
    which captures as much information as possible about the random variables `Z_max` and as little information as
    possible about `Z_min`.
    See the implementation of the `SupervisedMIST` model for more details.
    Args:
        X: Samples of the first distribution.
        Z_max: Corresponding samples of the distribution in regard to which the mutual information should be maximised.
        Z_min: Corresponding samples of the distribution in regard to which the mutual information should be minimised.
        data_presplit: A flag indicating that the data is already split into train, val, and test splits and is of the
                       form suitable for the SplitMISTDataModule.
                       This is useful for when the same data was used to train an unsupervised representations learning
                       model (e.g., a VAE), and you want to preserve that split.
        log_to_wb: Whether to optionally log to W&B
        wb_run_name: The optional manually set name for the W&B project
        datamodule_kwargs: Arguments passed on to the utility functions to prepare the data.
        model_kwargs: Arguments passed on to the utility functions to construct the MIST model.
        trainer_kwargs: Arguments passed on to the utility functions to train the model.

    Returns:
        A MISTResult object containing the results of the estimation.
    """

    data = (
        dm.SplitMISTDataModule(X, Z_max, Z_min, **datamodule_kwargs if datamodule_kwargs is not None else dict())
        if data_presplit
        else dm.MISTDataModule(X, Z_max, Z_min, **datamodule_kwargs if datamodule_kwargs is not None else dict())
    )

    # Construct the model
    model = _construct_supervised_mist(
        **model_kwargs if model_kwargs is not None else dict(),
    )

    # Train the model
    training_results = _train_supervised_mist(
        model,
        data,
        trainer_kwargs if trainer_kwargs is not None else dict(),
        log_to_wb=log_to_wb,
        wb_run_name=wb_run_name,
    )

    # Load the best model on the validation data
    best_model = _construct_supervised_mist(
        checkpoint_path=training_results["best_model_path"],
        **model_kwargs if model_kwargs is not None else dict(),
    )

    # Extract the found projection matrix
    A_hat = best_model.projection_layer.A.detach().cpu()

    # Assert we get a valid orthogonal matrix
    assert mutils.is_orthonormal(A_hat, atol=1e-2), (
        f"A_hat.T @ A_hat = {A_hat.T @ A_hat}, "
        f"distance from orthogonal = {torch.linalg.norm(A_hat.T @ A_hat - torch.eye(A_hat.shape[1]))}"
    )

    # Construct the result
    mi_estimate = MISTResult(
        mutual_information=training_results["test_results"]["test_mutual_information_mine_epoch"],
        loss=training_results["test_results"]["test_loss_epoch"],
        estimator=best_model,
        A=A_hat,
    )

    return mi_estimate
