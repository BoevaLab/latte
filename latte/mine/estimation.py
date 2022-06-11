from typing import Optional, Dict, Union, Any, Tuple
from dataclasses import dataclass

import geoopt
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from latte.mine import mine
from latte.models import datamodules as dm
from latte.manifolds import utils as mutils


@dataclass
class MutualInformationEstimationResult:
    """A dataclass for holding the result of mutual information estimation using MINE

    Members:
        mutual_information (float): The estimate of the mutual information as calculated on the test dataset
        validation_mutual_information (float): The estimate of the mutual information
                                                as calculated on the validation dataset
        estimator (mine.MINE): The model trained and used for estimation of the mutual information
        subspace: In case of also estimating the linear subspace capturing the most information, this holds the
                  k-frame defining the estimated subspace
    """

    mutual_information: float
    validation_mutual_information: float
    estimator: mine.MINE
    subspace: Optional[geoopt.ManifoldTensor] = None


def _construct_mine_model(
    manifold: bool,
    x_size: int,
    z_size: int,
    network_width: int,
    subspace_size: Optional[int] = 0,
    learning_rate: float = 1e-4,
    manifold_learning_rate: float = 1e-3,
    alpha: float = 0.01,
) -> mine.MINE:
    """A utility function for creating the MINE model used for estimation of mutual information and possibly searching
    for the optimal linear subspace.

    Args:
        manifold: Whether the model should also estimate the linear subspace of the space which X lives in of dimension
                  `subspace_size` capturing the most mutual information about Z
        x_size: The dimension of X
        z_size: The dimension of Z
        network_width: The side of the hidden layers in the statistics network
        subspace_size: If `manifold` is True, this controls the dimension of the linear subspace in the space of X
                       over which the mutual information should be optimised
        learning_rate: The learning_rate of the MINE model
        manifold_learning_rate: The learning rate for the k-frame manifold optimiser
        alpha: The bias-compensating alpha value for the MINE estimator

    Returns:
        The constructed model
    """
    if manifold:
        assert subspace_size != 0, "No target subspace size provided"
        S = mine.ManifoldStatisticsNetwork(
            S=nn.Sequential(
                nn.Linear(subspace_size + z_size, network_width),
                nn.ReLU(),
                nn.Linear(network_width, network_width),
                nn.ReLU(),
            ),
            k=subspace_size,
            d=x_size,
            out_dim=network_width,
        )
        mine_model = mine.ManifoldMINE(
            T=S,
            kind=mine.MINEObjectiveType.MINE,
            alpha=alpha,
            learning_rate=learning_rate,
            manifold_learning_rate=manifold_learning_rate,
        )

    else:
        S = mine.StatisticsNetwork(
            S=nn.Sequential(
                nn.Linear(x_size + z_size, network_width), nn.ReLU(), nn.Linear(network_width, network_width), nn.ReLU()
            ),
            out_dim=network_width,
        )
        mine_model = mine.MINE(T=S, kind=mine.MINEObjectiveType.MINE, alpha=alpha, learning_rate=learning_rate)

    return mine_model


def _train_mine(
    mine_model: mine.MINE,
    data: dm.GenericMINEDataModule,
    trainer_kwargs: Dict[str, Any],
    log_to_wb: Optional[bool] = False,
    wandb_run_name: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Trains a constructed MINE model and returns the testing result on the test split
    Args:
        mine_model: The model to train
        data: The prepared data in a GenericMINEDataModule data module
        trainer_kwargs: Arguments passed on to the utility functions to prepare the data
        log_to_wb: Whether to optionally log to W&B
        wandb_run_name: The optional manually set name for the W&B project

    Returns:
        The results of evaluating the model on the validation and the test splits of the data
    """

    wandb_logger = WandbLogger(project="latte", name=wandb_run_name)

    # If progress should be logged to W&B, pass the logger, else use the default (True)
    trainer = pl.Trainer(**trainer_kwargs, logger=wandb_logger if log_to_wb else True)
    trainer.fit(mine_model, datamodule=data)

    validation_results = trainer.validate(ckpt_path="best", dataloaders=data)
    test_results = trainer.test(ckpt_path="best", dataloaders=data)

    return (
        test_results[0],
        validation_results[0],
    )  # We only use 1 dataloader, so we return the first element of the results


def estimate_mutual_information(
    subspace_estimation: bool,
    X: Union[torch.Tensor, Dict[str, torch.Tensor]],
    Z: Union[torch.Tensor, Dict[str, torch.Tensor]],
    data_presplit: bool = False,
    log_to_wb: Optional[bool] = False,
    wandb_run_name: Optional[str] = None,
    datamodule_kwargs: Optional[Dict] = None,
    model_kwargs: Optional[Dict] = None,
    trainer_kwargs: Optional[Dict] = None,
) -> MutualInformationEstimationResult:
    """
    Compute the estimate of mutual information between two random variables based on the MINE estimate by
    training a MINE network and reporting the estimate on the test set.
    Optionally, find the estimate of the k-dimensional linear subspace which captures
    the most mutual information in X about Z.
    This optimisation is performed by simultaneously minimising the loss of the MINE model (to maximise the lower bound
    on the mutual information) and the loss over the manifold of the k-frames representing the linear subspace.

    Args:
        subspace_estimation: Whether to also estimate the linear subspace or not
        X: Empirical sample of distribution 1
        Z: Empirical sample of distribution 2
        data_presplit: A flog indicating that the data is already split into train, val, and test splits and is of the
                       form suitable for the SplitMINEDataModule.
                       This is useful for when the same data was used to train an unsupervised representations learning
                       model (e.g., a VAE), and you want to preserve that split.
        log_to_wb: Whether to optionally log to W&B
        wandb_run_name: The optional manually set name for the W&B project
        trainer_kwargs: Arguments passed on to the utility functions to prepare the data
        model_kwargs: Arguments passed on to the utility functions to construct the MINE model
        datamodule_kwargs: Arguments passed on to the utility functions to train the model

    Returns:
        A MutualInformationEstimationResult
    """

    data = (
        dm.SplitMINEDataModule(X, Z, **datamodule_kwargs if datamodule_kwargs is not None else dict())
        if data_presplit
        else dm.GenericMINEDataModule(X, Z, **datamodule_kwargs if datamodule_kwargs is not None else dict())
    )

    mine_model = _construct_mine_model(
        manifold=subspace_estimation, **model_kwargs if model_kwargs is not None else dict()
    )

    test_results, validation_results = _train_mine(
        mine_model, data, trainer_kwargs if trainer_kwargs is not None else dict()
    )

    mi_estimate = MutualInformationEstimationResult(
        mutual_information=test_results["test_mutual_information_epoch"],
        validation_mutual_information=validation_results["validation_mutual_information_epoch"],
        estimator=mine_model,
    )

    if subspace_estimation:
        A_hat = mine_model.T.projection_layer.A.detach().cpu()

        # Assert we get a valid k-frame
        assert mutils.is_orthonormal(A_hat)
        mi_estimate.subspace = A_hat

    return mi_estimate
