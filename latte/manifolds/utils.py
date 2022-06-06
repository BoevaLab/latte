import torch


def almost_equal(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-3) -> bool:
    """Utility function to test for numerical equality.

    Args:
        a: Tensor a
        b: Tensor b
        atol:  The numerical tolerance for equality

    Returns:
        Whether the tensors are considered equal
    """
    return torch.linalg.norm(a - b) < atol


def is_orthonormal(A: torch.Tensor, atol: float = 1e-3) -> bool:
    """A simple utility function to assert that the learned projection k-frame is column-orthonormal.

    Args:
        A (torch.Tensor): The (learned) projection matrix
        atol (float): The numerical tolerance for equality
    """
    return almost_equal(A.T @ A, torch.eye(A.shape[1]), atol)


def correct_solution(A: torch.Tensor, A_hat: torch.Tensor, atol: float = 1e-3) -> bool:
    """Asserts that the learned projection k-frame is close to the solution in artificial datasets.

    Args:
        A (torch.Tensor): The ground-truth projection matrix
        A_hat (torch.Tensor): The learned projection matrix
        atol (float): The numerical tolerance for equality"""
    return almost_equal(A_hat @ A_hat.T, A @ A.T, atol)
