import torch
import geoopt


from latte.manifolds import utils as mutils


def test_orthonormality() -> None:

    A_orth = geoopt.Stiefel().random(10, 3)

    assert mutils.is_orthonormal(A_orth)

    A_nonorth = 2 * geoopt.Stiefel().random(10, 3)

    assert not mutils.is_orthonormal(A_nonorth)


def test_close() -> None:
    A = geoopt.Stiefel().random(10, 3)

    assert mutils.correct_solution(A, A)
    assert not mutils.correct_solution(A, A + torch.ones_like(A))
    assert not mutils.correct_solution(A, A + 0.1 * torch.ones_like(A))
    assert not mutils.correct_solution(A, A + 0.01 * torch.ones_like(A))
