# TODO (Anej): documentation
import torch

from latte.models.mist import estimation as mist_estimation


def find_subspace(
    Z: torch.Tensor,
    Y: torch.Tensor,
    subspace_size: int,
    mist_max_epochs: int,
    mist_batch_size: int,
    mine_network_width: int,
    mine_learning_rate: float,
    manifold_learning_rate: float,
    lr_scheduler_patience: int,
    lr_scheduler_min_delta: float,
    num_workers: int,
    gpus: int,
) -> mist_estimation.MISTResult:
    """
    Trains a MIST model to find the optimal subspace of dimensionality `subspace_size` about the factors captured in `Y`
    Args:
        Z: Full VAE representations
        Y: Ground-truth factor values
        subspace_size: The current size of the subspace considered.

    Returns:
        The MIST estimation result

    """

    print("Training MIST.")

    mi_estimation_result = mist_estimation.fit(
        X=Z,
        Z_max=Y,
        datamodule_kwargs=dict(num_workers=num_workers, batch_size=mist_batch_size, p_train=0.4, p_val=0.2),
        model_kwargs=dict(
            x_size=Z.shape[1],
            z_max_size=Y.shape[1],
            subspace_size=subspace_size,
            mine_network_width=mine_network_width,
            mine_learning_rate=mine_learning_rate,
            manifold_learning_rate=manifold_learning_rate,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_min_delta=lr_scheduler_min_delta,
            verbose=False,
        ),
        trainer_kwargs=dict(
            max_epochs=mist_max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            gpus=gpus,
        ),
    )

    print("Training MIST done.")

    return mi_estimation_result
