import torch


def is_orthonormal(A: torch.Tensor) -> bool:
    """A simple utility function to assert that the learned projection k-frame is column-orthonormal.

    Args:
        A (torch.Tensor): The (learned) projection matrix
    """
    return torch.linalg.norm(A.T @ A - torch.eye(A.shape[1])) < 1e-3


def correct_solution(A: torch.Tensor, A_hat: torch.Tensor) -> bool:
    """Asserts that the learned projection k-frame is close to the solution in artificial datasets.

    Args:
        A (torch.Tensor): The ground-truth projection matrix
        A_hat (torch.Tensor): The learned projection matrix"""
    return torch.linalg.norm(A_hat @ A_hat.T - A @ A.T) < 1e-3
