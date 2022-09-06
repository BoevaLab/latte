# TODO (Anej): Documentation
from typing import Tuple, Dict, List
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
import pytorch_lightning as pl
import torch

from sklearn.preprocessing import StandardScaler

from pythae.models import AutoModel, VAE

from latte.dataset import utils as dsutils
from latte.models.vae import utils as vae_utils
from latte.tools import model_comparison, space_evaluation
from latte.tools.space_evaluation import MIEstimationMethod
from latte.utils import visualisation
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
        vae_model_file_paths: Path to the file of the trained VAE model.
        factors_of_interest: A list of lists.
                             Each element of the outer list is a list of elements for which we want to find the linear
                             subspace capturing the most information about them *together*.
                             Multiple such lists can be provided to explore subspaces capturing information about
                             different sets of factors.
        subspace_sizes: A list of dimensionalities of the subspaces to consider.
                        For each set of factors of interest, a subspace of each size specified in `subspace_sizes` will
                        be computed.
        n_runs: Number of times to repeat the estimation with MIST to ensure better stability.
                If n_runs > 1, the median result will be taken.
        mist_max_epochs: Maximum number of epochs for training MIST.
        plot_nrows: Number of rows to plot when plotting generated images and reconstructions by the VAE.
        plot_ncols: Number of columns to plot when plotting generated images and reconstructions by the VAE.
        seed: The random seed.
        gpus: The `gpus` parameter for Pytorch Lightning.
    """

    dataset: str
    factors_of_interest: List[str]
    file_paths_x: Tuple[str, str, str]
    file_paths_y: Tuple[str, str, str]
    vae_model_file_paths: List[str]
    subspace_sizes: Tuple[List[int], List[int]] = ([1, 2], [1, 2])
    fitting_N: int = 8000
    n_runs: int = 5
    mist_max_epochs: int = 256
    seed: int = 1
    gpus: int = 1


def _evaluate_subspace_factor_information(
    Z: Dict[str, torch.Tensor],
    Y: Dict[str, torch.Tensor],
    A: torch.Tensor,
    train_ixs: Dict[MIEstimationMethod, List[int]],
    pair_id: Tuple[int, int],
    ix: int,
    subspace_size: int,
    subspace_method: str,
    experiment_results: Dict[str, float],
    cfg: MISTConfig,
) -> Dict[MIEstimationMethod, Dict[str, float]]:

    methods = (
        [MIEstimationMethod.KSG, MIEstimationMethod.LATTE_KSG, MIEstimationMethod.MIST, MIEstimationMethod.SKLEARN]
        if subspace_size == 1
        else [MIEstimationMethod.KSG, MIEstimationMethod.LATTE_KSG, MIEstimationMethod.MIST]
    )
    result = {
        method: space_evaluation.evaluate_space_with_mutual_information(
            Z=Z["val"],
            Y=Y["val"],
            A=A,
            ixs=train_ixs[method],
            dataset=cfg.dataset,
            factors=[(factor,) for factor in cfg.factors_of_interest],
            method=method,
            standardise=False,
        )
        for method in methods
    }
    # Correct for the enforced tuples
    result = {method: {factors[0]: value for factors, value in r.items()} for method, r in result.items()}

    for method, factor in product(methods, cfg.factors_of_interest):
        experiment_results[
            f"I(R{ix}, {factor} | pair={pair_id}, subspace_sizes={subspace_size}, {method}, {subspace_method})"
        ] = result[method][factor]
        print(
            f"I(R{ix}, {factor} | pair={pair_id}, subspace_sizes={subspace_size}, {method}, {subspace_method}) = "
            f"{result[method][factor]}"
        )

    return result


def _compute_factor_information_spearman_correlation(
    pair_id: Tuple[int, int],
    results: Tuple[Dict[MIEstimationMethod, Dict[str, float]], Dict[MIEstimationMethod, Dict[str, float]]],
    subspace_method: str,
    subspace_size: Tuple[int, int],
    factor_information: List[Dict[MIEstimationMethod, Dict[str, float]]],
    experiment_results: Dict[str, float],
) -> None:

    for method in [MIEstimationMethod.KSG, MIEstimationMethod.LATTE_KSG, MIEstimationMethod.MIST]:
        full_space_mis = np.asarray(
            list(factor_information[pair_id[0]][method].values())
            + list(factor_information[pair_id[1]][method].values())
        )
        subspace_mis = np.asarray(list(results[0][method].values()) + list(results[1][method].values()))

        rho = stats.spearmanr(full_space_mis, subspace_mis).correlation
        experiment_results[f"rho({subspace_method} | pair={pair_id}, method={str(method)}, size={subspace_size})"] = rho
        print(f"rho({subspace_method} | pair={pair_id}, method={str(method)}, size={subspace_size}) = {rho}")


def _plot_latent_traversals(
    X: torch.Tensor,
    Zs: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    pair_id: Tuple[int, int],
    As: Tuple[torch.Tensor, torch.Tensor],
    models: Tuple[VAE, VAE],
    subspace_method: str,
    subspace_size: Tuple[int, int],
    device: str,
    seed: int,
) -> None:

    for ix, Z, A, model in zip(pair_id, Zs, As, models):
        space_evaluation.subspace_latent_traversals(
            Z=Z["val"],
            A=A,
            trained_model=model,
            X=X,
            device=device,
            standardise=False,
            rng=seed,
            file_name=f"subspace_latent_traversals_m{ix}_p{pair_id}_size{subspace_size}_{subspace_method}.png",
        )


def _plot_heatmaps_and_trends(
    Zs: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    pair_id: Tuple[int, int],
    Y: Dict[str, torch.Tensor],
    As: Tuple[torch.Tensor, torch.Tensor],
    factors_of_interest: List[str],
    dataset: str,
    subspace_method: str,
    subspace_size: Tuple[int, int],
) -> None:

    for ix, Z, A in zip(pair_id, Zs, As):
        if A.shape[1] <= 2:
            space_evaluation.subspace_heatmaps(
                Z["val"],
                Y["val"][:, [dsutils.get_dataset_module(dataset).attribute2idx[a] for a in factors_of_interest]],
                A=A,
                attribute_names=factors_of_interest,
                interpolate=True,
                file_name=f"factor_heatmap_m{ix}_p{pair_id}_size{subspace_size}_{subspace_method}.png",
            )


def _process_subspace(
    X: Dict[str, torch.Tensor],
    Z_1: Dict[str, torch.Tensor],
    Z_2: Dict[str, torch.Tensor],
    pair_id: Tuple[int, int],
    models: Tuple[VAE, VAE],
    Y: Dict[str, torch.Tensor],
    A_1: torch.Tensor,
    A_2: torch.Tensor,
    subspace_method: str,
    factor_information: List[Dict[MIEstimationMethod, Dict[str, float]]],
    experiment_results: Dict[str, float],
    cfg: MISTConfig,
    rng: dsutils.RandomGenerator,
):

    print(f"Processing the {(A_1.shape[1], A_2.shape[1])}-dimensional subspaces found by {subspace_method}.")

    factor_mutual_information_results = []

    # Select a new set of indexes for evaluation
    train_ixs = rng.choice(len(Z_1["val"]), size=cfg.fitting_N, replace=False)  # len(Z_1["val"]) == len(Z_2["val"])
    latte_ksg_train_ixs = rng.choice(len(Z_1["val"]), size=2000, replace=False)  # len(Z_1["val"]) == len(Z_2["val"])
    for ix, Z, A, subspace_size in zip(pair_id, [Z_1, Z_2], [A_1, A_2], (A_1.shape[1], A_2.shape[1])):
        print(f"Estimating the information about factors of interest in the subspaces for model {ix}.")

        if subspace_size < Z["val"].shape[1]:
            # Estimate the mutual information with each of the available methods
            # but only if the subspace is not the entire representation space (otherwise, it was already computed)
            factor_mutual_information_results.append(
                _evaluate_subspace_factor_information(
                    Z=Z,
                    Y=Y,
                    A=A,
                    train_ixs={
                        MIEstimationMethod.KSG: train_ixs,
                        MIEstimationMethod.LATTE_KSG: latte_ksg_train_ixs,
                        MIEstimationMethod.MIST: train_ixs,
                        MIEstimationMethod.SKLEARN: train_ixs,
                    },
                    pair_id=pair_id,
                    ix=ix,
                    subspace_size=subspace_size,
                    subspace_method=subspace_method,
                    experiment_results=experiment_results,
                    cfg=cfg,
                )
            )

            subspace_mutual_information = space_evaluation.subspace_mutual_information(
                Z["val"], train_ixs, A, standardise=False
            )
            subspace_entropy = space_evaluation.subspace_entropy(Z["val"], train_ixs, A, standardise=False)
            experiment_results[
                f"MI(R{ix} | pair={pair_id}, subspace_sizes={subspace_size}, {subspace_method})"
            ] = subspace_mutual_information
            experiment_results[
                f"H(R{ix} | pair={pair_id}, subspace_sizes={subspace_size}, {subspace_method})"
            ] = subspace_entropy
            print(
                f"MI(R{ix} | pair={pair_id}, subspace_sizes={subspace_size}, {subspace_method}) = "
                f"{subspace_mutual_information}."
            )
            print(f"H(R{ix} | pair={pair_id}, subspace_sizes={subspace_size}, {subspace_method}) = {subspace_entropy}.")

        else:
            # Just use the already-computed values
            factor_mutual_information_results.append(factor_information[ix])

    _compute_factor_information_spearman_correlation(
        pair_id=pair_id,
        results=(factor_mutual_information_results[0], factor_mutual_information_results[1]),
        subspace_method=subspace_method,
        subspace_size=(A_1.shape[1], A_2.shape[1]),
        factor_information=factor_information,
        experiment_results=experiment_results,
    )

    _plot_latent_traversals(
        Zs=(Z_1, Z_2),
        pair_id=pair_id,
        X=X["val"],
        As=(A_1, A_2),
        models=models,
        subspace_method=subspace_method,
        subspace_size=(A_1.shape[1], A_2.shape[1]),
        seed=cfg.seed,
        device="cuda" if cfg.gpus > 0 else "cpu",
    )

    _plot_heatmaps_and_trends(
        Zs=(Z_1, Z_2),
        pair_id=pair_id,
        Y=Y,
        As=(A_1, A_2),
        factors_of_interest=cfg.factors_of_interest,
        dataset=cfg.dataset,
        subspace_method=subspace_method,
        subspace_size=(A_1.shape[1], A_2.shape[1]),
    )


def _compute_information_about_factors(
    Z: List[torch.Tensor],
    Y: torch.Tensor,
    cfg: MISTConfig,
    rng: dsutils.RandomGenerator,
) -> List[Dict[MIEstimationMethod, Dict[str, float]]]:
    """
    Computes the mutual information about all factors of interest contained in each of the representations in Z
    with MIST.
    Args:
        Z: Representations constructed by each of the models in consideration.
        Y: Values of the factors of interest.
        cfg: The script configuration.
        rng: Random generator.
    Returns:
        A list of dictionaries.
        Each element of the list is a dictionary specifying how much information about each of the factors is contained
        in the representation space at that index.
    """

    train_ixs = rng.choice(len(Y), size=cfg.fitting_N, replace=False)
    latte_ksg_train_ixs = rng.choice(len(Y), size=2000, replace=False)

    results = [
        {
            method: space_evaluation.evaluate_space_with_mutual_information(
                Z=z,
                Y=Y,
                ixs=train_ixs if method != MIEstimationMethod.LATTE_KSG else latte_ksg_train_ixs,
                dataset=cfg.dataset,
                factors=[(factor,) for factor in cfg.factors_of_interest],
                method=method,
                standardise=False,
            )
            for method in [MIEstimationMethod.KSG, MIEstimationMethod.LATTE_KSG, MIEstimationMethod.MIST]
        }
        for z in Z
    ]

    # Correct for the enforced tuples
    results = [
        {method: {factors[0]: value for factors, value in r.items()} for method, r in result.items()}
        for result in results
    ]

    return results


def _process_latent_representations(
    X: Dict[str, torch.Tensor],
    Z: Dict[str, List[torch.Tensor]],
    Y: Dict[str, torch.Tensor],
    models: List[VAE],
    experiment_results: Dict[str, float],
    factor_information: List[Dict[MIEstimationMethod, Dict[str, float]]],
    cfg: MISTConfig,
    rng: dsutils.RandomGenerator,
) -> None:
    """

    Args:
        Z: Dictionary of the full VAE representations split into "train", "val", and "test" splits
        Y: Dictionary of the ground-truth factor values split into "train", "val", and "test" splits
        experiment_results: The dictionary in which to store the results of the experiment.
        cfg: The configuration object of the experiment.
        rng: The random generator.
    """

    train_ixs = rng.choice(len(Z["val"][0]), size=cfg.fitting_N, replace=False)
    cca_subspaces_done = set([])

    print("Training MIST to determine the mutual information between the representations.")
    for ii, jj in product(range(len(Z["train"])), range(len(Z["train"]))):
        if ii >= jj:
            continue
        # Consider all specified subspaces sizes and the full representation spaces
        for s1, s2 in product(
            [Z["train"][ii].shape[1]] + cfg.subspace_sizes[0], [Z["train"][jj].shape[1]] + cfg.subspace_sizes[1]
        ):
            print(f"Looking into subspaces of size {(s1, s2)}.")

            A_1, _, A_2, _, mi = model_comparison.find_common_subspaces_with_mutual_information(
                Z_1=Z["val"][ii],
                Z_2=Z["val"][jj],
                ixs=train_ixs,
                subspace_size=(s1, s2),
                standardise=False,
                max_epochs=cfg.mist_max_epochs,
                gpus=cfg.gpus,
            )
            print(f"The information between the representations of model {ii} and {jj} is {mi}")
            experiment_results[f"I(R{ii}, R{jj} | subspace_sizes={(s1, s2)} )"] = mi

            _process_subspace(
                X=X,
                Z_1={"train": Z["train"][ii], "val": Z["val"][ii], "test": Z["test"][ii]},
                Z_2={"train": Z["train"][jj], "val": Z["val"][jj], "test": Z["test"][jj]},
                Y=Y,
                pair_id=(ii, jj),
                models=(models[ii], models[jj]),
                A_1=A_1,
                A_2=A_2,
                subspace_method="mist",
                factor_information=factor_information,
                experiment_results=experiment_results,
                cfg=cfg,
                rng=rng,
            )

            if s1 != s2 or s1 in cca_subspaces_done:
                continue

            cca_subspaces_done.add(s1)

            A_1, _, A_2, _ = model_comparison.find_common_subspaces_with_correlation(
                Z_1=Z["val"][ii].numpy(),
                Z_2=Z["val"][jj].numpy(),
                ixs=train_ixs,
                subspace_size=s1,
                standardise=False,
            )

            _process_subspace(
                X=X,
                Z_1={"train": Z["train"][ii], "val": Z["val"][ii], "test": Z["test"][ii]},
                Z_2={"train": Z["train"][jj], "val": Z["val"][jj], "test": Z["test"][jj]},
                Y=Y,
                pair_id=(ii, jj),
                models=(models[ii], models[jj]),
                A_1=A_1,
                A_2=A_2,
                subspace_method="CCA",
                factor_information=factor_information,
                experiment_results=experiment_results,
                cfg=cfg,
                rng=rng,
            )

            print("\n---------------------------------------------------------------\n\n\n")


@hy.main
def main(cfg: MISTConfig):

    assert cfg.dataset in ["celeba", "shapes3d"], "The dataset not supported."

    # This will keep various metrics from the experiment
    experiment_results = dict()

    pl.seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    X, Y = dsutils.load_split_data(
        {"train": cfg.file_paths_x[0], "val": cfg.file_paths_x[1], "test": cfg.file_paths_x[2]},
        {"train": cfg.file_paths_y[0], "val": cfg.file_paths_y[1], "test": cfg.file_paths_y[2]},
        cfg.dataset,
    )

    device = "cuda" if cfg.gpus > 0 else "cpu"

    # Load the trained VAE
    models = [
        AutoModel.load_from_folder(vae_model_file_path).to(device) for vae_model_file_path in cfg.vae_model_file_paths
    ]

    # Get the latent representations given by the original VAE
    Zs = {split: [vae_utils.get_latent_representations(model, S) for model in models] for split, S in X}

    for ii in range(len(Zs["train"])):
        full_scaler = StandardScaler().fit(Zs["train"][ii].numpy())
        Zs["train"][ii] = torch.from_numpy(full_scaler.transform(Zs["train"][ii].numpy())).float()
        Zs["val"][ii] = torch.from_numpy(full_scaler.transform(Zs["val"][ii].numpy())).float()
        Zs["test"][ii] = torch.from_numpy(full_scaler.transform(Zs["test"][ii].numpy())).float()

    for ii in range(len(Zs["train"])):
        visualisation.graphically_evaluate_model(
            models[ii], X["val"], Zs["val"][ii], file_prefix=f"model_{ii}", device=device, repeats=5
        )

    factor_information = _compute_information_about_factors(Zs["val"], Y["val"], cfg, rng)

    _process_latent_representations(
        X=X,
        Z=Zs,
        Y=Y,
        models=models,
        experiment_results=experiment_results,
        factor_information=factor_information,
        cfg=cfg,
        rng=rng,
    )

    print("Finished.")

    # Save the experiment results in a data frame
    results_df = pd.DataFrame({"Value": experiment_results})
    print("The final experiment results:")
    print(results_df)
    results_df.to_csv("experiment_results.csv")


if __name__ == "__main__":
    main()
