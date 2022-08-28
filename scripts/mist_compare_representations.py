# TODO (Anej): Documentation
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.preprocessing import StandardScaler

from pythae.models import AutoModel
from pythae.models import BetaVAE, BetaTCVAE, FactorVAE

from latte.dataset import celeba, shapes3d, utils as dsutils
from latte.models.vae import utils as vae_utils
from latte.models.mist import estimation as mist_estimation
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
        model_class: The VAE flavour to use.
                     Currently supported: "BetaVAE", "BetaTCVAE", "FactorVAE".
        early_stopping_min_delta: Min delta parameter for the early stopping callback of the MIST model.
        early_stopping_patience: Patience parameter for the early stopping callback of the MIST model.
        mine_estimation_network_width: The script uses the default implemented MINE statistics network for estimating
                                       the mutual information.
                                       This specified the width to be used.
        mine_subspace_network_width: The script uses a simpler network for determining the optimal subspace.
                                     This specified the width to be used.
        mist_batch_size: Batch size for training MIST.
        mine_estimation_learning_rate: Learning rate for training `MINE` when used for estimation of the mutual
                                       information or the entropy.
                                       Should be around `1e-4`.
        mine_subspace_learning_rate: Learning rate for training `MINE` when used for estimation of the optimal subspace
                                     capturing the most information about a given factor.
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
    factors_of_interest: List[str]
    file_paths_x: Tuple[str, str, str]
    file_paths_y: Tuple[str, str, str]
    vae_model_file_paths: List[str]
    subspace_sizes: Tuple[List[int], List[int]] = ([1, 2], [1, 2])
    model_class: str = "BetaVAE"
    early_stopping_min_delta: float = 1e-2
    early_stopping_patience: int = 12
    mine_estimation_network_width: int = 128
    mine_subspace_network_width: int = 128
    mist_batch_size: int = 128
    mine_estimation_learning_rate: float = 1e-4
    mine_subspace_learning_rate: float = 1e-5
    manifold_learning_rate: float = 1e-2
    mist_max_epochs: int = 256
    lr_scheduler_patience: int = 6
    lr_scheduler_min_delta: float = 1e-2
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

    X_train = torch.from_numpy(np.load(cfg.file_paths_x[0])).float()
    Y_train = torch.from_numpy(np.load(cfg.file_paths_y[0])).float()
    X_val = torch.from_numpy(np.load(cfg.file_paths_x[1])).float()
    Y_val = torch.from_numpy(np.load(cfg.file_paths_y[1])).float()
    X_test = torch.from_numpy(np.load(cfg.file_paths_x[2])).float()
    Y_test = torch.from_numpy(np.load(cfg.file_paths_y[2])).float()

    if cfg.dataset in ["celeba", "shapes3d"]:
        X_train /= 255
        X_val /= 255
        X_test /= 255

    print("Data loaded.")
    return (X_train, X_val, X_test), (Y_train, Y_val, Y_test)


def _get_dataset_module(cfg: MISTConfig) -> Any:
    """Returns the module to use for various utilities according to the dataset."""
    if cfg.dataset == "celeba":
        return celeba
    elif cfg.dataset == "shapes3d":
        return shapes3d


def _train_mist(
    Z: Dict[str, torch.Tensor],
    Y: Dict[str, torch.Tensor],
    subspace_size: Tuple[int, int],
    fit_subspace: bool,
    cfg: MISTConfig,
) -> mist_estimation.MISTResult:
    """
    Trains a MIST model to find the optimal subspace of dimensionality `subspace_size` about the factors captured in `Y`
    Args:
        Z: Dictionary of the full VAE representations split into "train", "val", and "test" splits
        Y: Dictionary of the ground-truth factor values split into "train", "val", and "test" splits
        subspace_size: The current size of the subspace considered.
        cfg: The configuration object of the experiment.

    Returns:
        The MIST estimation result

    """

    print("Training MIST.")

    # If we are also estimating the subspaces, follow a different training regime
    # if (Z["train"].shape[1], Y["train"].shape[1]) == subspace_size:
    if fit_subspace:
        mine_learning_rate = cfg.mine_subspace_learning_rate
        # statistics_network = mine.StatisticsNetwork(
        #     S=nn.Sequential(
        #         nn.Linear(subspace_size[0] + subspace_size[1], cfg.mine_subspace_network_width),
        #         nn.ReLU(),
        #     ),
        #     out_dim=cfg.mine_subspace_network_width,
        # )
        statistics_network = None
    else:
        mine_learning_rate = cfg.mine_estimation_learning_rate
        statistics_network = None

    early_stopping_callback = EarlyStopping(
        monitor="validation_mutual_information_mine",
        min_delta=cfg.early_stopping_min_delta,
        patience=cfg.early_stopping_patience,
        verbose=False,
        mode="max",
    )

    mi_estimation_result = mist_estimation.find_subspace(
        X=Z,
        Z_max=Y,
        data_presplit=True,
        datamodule_kwargs=dict(num_workers=cfg.num_workers, batch_size=cfg.mist_batch_size),
        model_kwargs=dict(
            x_size=Z["train"].shape[1],
            z_max_size=Y["train"].shape[1],
            subspace_size=subspace_size,
            model_comparison=True,
            statistics_network=statistics_network,
            mine_network_width=cfg.mine_estimation_network_width,
            mine_learning_rate=mine_learning_rate,
            manifold_learning_rate=cfg.manifold_learning_rate,
            lr_scheduler_patience=cfg.lr_scheduler_patience,
            lr_scheduler_min_delta=cfg.lr_scheduler_min_delta,
            verbose=False,
        ),
        trainer_kwargs=dict(
            callbacks=[early_stopping_callback],
            max_epochs=cfg.mist_max_epochs,
            enable_progress_bar=False,
            enable_model_summary=True,
            gpus=cfg.gpus,
        ),
    )

    print("Training MIST done.\n\n")

    return mi_estimation_result


def _process_subspace(
    Z_1: Dict[str, torch.Tensor],
    Z_2: Dict[str, torch.Tensor],
    pair_id: Tuple[int, int],
    Y: Dict[str, torch.Tensor],
    experiment_results: Dict[str, float],
    subspace_result: mist_estimation.MISTResult,
    cfg: MISTConfig,
):

    A_hat = subspace_result.A.detach().cpu()
    B_hat = subspace_result.B.detach().cpu()

    subspace_size = (A_hat.shape[1], B_hat.shape[1])

    print(f"Processing the {subspace_size}-dimensional subspaces.")

    Z_1_projected = {split: Z_1[split] @ A_hat.detach().cpu() for split in Z_1}
    Z_2_projected = {split: Z_2[split] @ B_hat.detach().cpu() for split in Z_2}

    """"
    if (
        Z_1_projected["train"].shape[1] != Z_1["train"].shape[1]
        or Z_2_projected["train"].shape[1] != Z_2["train"].shape[1]
    ):
        print("Re-calculating the value of MI on the subspaces.")
        comparison_result = _train_mist(
            Z={"train": Z_1_projected["train"], "val": Z_1_projected["val"], "test": Z_1_projected["test"]},
            Y={"train": Z_2_projected["train"], "val": Z_2_projected["val"], "test": Z_2_projected["test"]},
            subspace_size=(Z_1_projected["train"].shape[1], Z_2_projected["train"].shape[1]),
            fit_subspace=False,
            cfg=cfg,
        )
        mi = comparison_result.mutual_information
        print(f"The mutual information in the projected subspaces is {mi:.3f}.")
        experiment_results[f"I(R{pair_id[0]}, R{pair_id[1]} | subspace_sizes={subspace_size})"] = mi
    """

    for factor_of_interest in cfg.factors_of_interest:
        for kk, Z in enumerate([Z_1_projected, Z_2_projected]):
            print(f"Estimating the information about {factor_of_interest} in the subspaces for model {kk}.")
            factor_result_1 = _train_mist(
                Z={"train": Z["train"], "val": Z["val"], "test": Z["test"]},
                Y={
                    "train": Y["train"][:, [_get_dataset_module(cfg).attribute2idx[factor_of_interest]]],
                    "val": Y["val"][:, [_get_dataset_module(cfg).attribute2idx[factor_of_interest]]],
                    "test": Y["test"][:, [_get_dataset_module(cfg).attribute2idx[factor_of_interest]]],
                },
                subspace_size=(Z["train"].shape[1], 1),
                fit_subspace=False,
                cfg=cfg,
            )
            mi = factor_result_1.mutual_information
            experiment_results[f"I(R{pair_id[kk]}, {factor_of_interest} | subspace_sizes={subspace_size})"] = mi
            print(f"I(R{pair_id[kk]}, {factor_of_interest} | subspace_sizes={subspace_size}) = {mi:.3f}.")


def _get_model_class(model_class: str) -> Any:
    if model_class == "BetaVAE":
        return BetaVAE
    elif model_class == "FactorVAE":
        return FactorVAE
    elif model_class == "BetaTCVAE":
        return BetaTCVAE
    else:
        raise NotImplementedError


def _process_latent_representations(
    Z: Dict[str, List[torch.Tensor]],
    Y: Dict[str, torch.Tensor],
    experiment_results: Dict[str, float],
    subspace_sizes: Tuple[List[int], List[int]],
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
        Z: Dictionary of the full VAE representations split into "train", "val", and "test" splits
        Y: Dictionary of the ground-truth factor values split into "train", "val", and "test" splits
        experiment_results: The dictionary in which to store the results of the experiment.
        cfg: The configuration object of the experiment.
        device: The device to load the data to.
        rng: The random generator.
    """

    # To find out how much information about the factors of interest there is in the entire learned representation
    # space, we train a MIST model on the entire space
    # This is done on the *entire* dataset
    print("Training MIST to determine the mutual information between the representations.")
    for ii, jj in product(range(len(Z["train"])), range(len(Z["train"]))):
        if ii >= jj:
            continue
        # Consider all specified subspaces sizes and the full representation spaces
        for s1, s2 in product(
            [Z["train"][ii].shape[1]] + subspace_sizes[0], [Z["train"][jj].shape[1]] + subspace_sizes[1]
        ):
            print(f"Looking into subspaces of size {(s1, s2)}.")

            comparison_result = _train_mist(
                Z={"train": Z["train"][ii], "val": Z["val"][ii], "test": Z["test"][ii]},
                Y={"train": Z["train"][jj], "val": Z["val"][jj], "test": Z["test"][jj]},
                subspace_size=(s1, s2),
                fit_subspace=(Z["train"][ii].shape[1], Z["train"][jj].shape[1]) != (s1, s2),
                cfg=cfg,
            )
            mi = comparison_result.mutual_information
            print(f"The information between the representations of model {ii} and {jj} is {mi}")
            experiment_results[f"I(R{ii}, R{jj} | subspace_sizes={(s1, s2)}, train)"] = mi
            _process_subspace(
                Z_1={"train": Z["train"][ii], "val": Z["val"][ii], "test": Z["test"][ii]},
                Z_2={"train": Z["train"][jj], "val": Z["val"][jj], "test": Z["test"][jj]},
                Y=Y,
                pair_id=(ii, jj),
                experiment_results=experiment_results,
                subspace_result=comparison_result,
                cfg=cfg,
            )

            print("\n---------------------------------------------------------------\n\n\n")


@hy.main
def main(cfg: MISTConfig):

    assert cfg.dataset in ["celeba", "shapes3d"], "The dataset not supported."

    # This will keep various metrics from the experiment
    experiment_results = dict()

    pl.seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    (X_train, X_val, X_test), (Y_train, Y_val, Y_test) = _load_data(cfg)

    device = "cuda" if cfg.gpus > 0 else "cpu"

    # Load the trained VAE
    trained_models = [
        AutoModel.load_from_folder(vae_model_file_path).to(device) for vae_model_file_path in cfg.vae_model_file_paths
    ]

    # Get the latent representations given by the original VAE
    Z_trains = [vae_utils.get_latent_representations(trained_model, X_train) for trained_model in trained_models]
    Z_vals = [vae_utils.get_latent_representations(trained_model, X_val) for trained_model in trained_models]
    Z_tests = [vae_utils.get_latent_representations(trained_model, X_test) for trained_model in trained_models]

    for ii in range(len(Z_trains)):
        full_scaler = StandardScaler().fit(Z_trains[ii].numpy())
        Z_trains[ii] = torch.from_numpy(full_scaler.transform(Z_trains[ii].numpy())).float()
        Z_vals[ii] = torch.from_numpy(full_scaler.transform(Z_vals[ii].numpy())).float()
        Z_tests[ii] = torch.from_numpy(full_scaler.transform(Z_tests[ii].numpy())).float()

    _process_latent_representations(
        {"train": Z_trains, "val": Z_vals, "test": Z_tests},
        {"train": Y_train, "val": Y_val, "test": Y_test},
        experiment_results,
        cfg.subspace_sizes,
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
