from math import sqrt

import torch
from torch import Tensor

from .qr import qr
from .utils import scaled_matmul


def randomized_qb(
    A: Tensor,
    rank: int,
    niter: int = 0,
    test_matrix: str = "subs",
    left: bool | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Randomzied QB decomposition, defined as `A = Q @ B` or `A = B @ Q`.

    Args:
        A (Tensor):
            &#45; with shape `(*, m, n)`.
        rank (int):
            &#45; target `rank`.
        niter (int, optional):
            &#45; number of power iterations.
            Default: `0`.
        test_matrix (str, optional):
            &#45; the type of test matrix.
            Optional: `'gauss'` | `'subs'`.
            Default: `'subs'`.
        left (bool or None, optional):
            &#45; the order of the orthogonal matrix Q.
            If `True`, return `(Q, B)` where `A = Q @ B`.
            If `False`, return `(B, Q)` where `A = B @ Q`.
            If `None`, return the one with lower computational complexity.
            Default: `None`.

    Returns:
        (Tensor, Tensor):
            &#45; with shape `(*, m, k)`, `(*, k, n)`, where `k = min(m, n, rank)`.
    """
    assert isinstance(A, Tensor)
    assert isinstance(rank, int)
    assert isinstance(niter, int)
    assert isinstance(test_matrix, str)
    assert isinstance(left, bool | None)

    m, n = A.shape[-2], A.shape[-1]

    if (left is True) or (left is None and m <= n):
        k = min(m, n, rank)

        # Generate a test matrix Ohm
        if test_matrix == "gauss":
            Ohm = torch.randn(*A.shape[:-2], n, k, dtype=A.dtype, device=A.device)  # (*, n, k)
        elif test_matrix == "subs":
            idx = torch.randperm(m)[:k]
            Ohm = A[..., idx, :].mT  # (*, n, k)
        else:
            raise ValueError("Invalid value of `test_matrix`.")

        # Sample column space of A with Ohm matrix, and generate the approximate orthonormal basis Q
        Y = (A @ Ohm).div_(sqrt(n))  # (*, m, k)
        for _ in range(niter):
            Q, _ = qr(Y)  # (*, m, k)
            Ohm = scaled_matmul(A.mT, Q)  # (*, n, k)
            Y = (A @ Ohm).div_(sqrt(n))  # (*, m, k)
        Q, _ = qr(Y)  # (*, m, k)

        # Compute projected B
        B = scaled_matmul(Q.mT, A) # (*, k, n)
        return Q, B

    elif (left is False) or (left is None and m > n):
        QT, BT = randomized_qb(A.mT, rank, niter=niter, test_matrix=test_matrix, left=True)
        return BT.mT, QT.mT

    else:
        raise ValueError("Invalid value of `left`, must be `True`, `False` or `None`.")


def qb_reconstruct(
    Q: Tensor,
    B: Tensor,
) -> Tensor:
    """
    Reconstruction of QB decomposition.

    Args:
        Q (Tensor):
            &#45; with shape `(*, m, k)`.
        B (Tensor):
            &#45; with shape `(*, k, n)`.

    Returns:
        Tensor:
            &#45; with shape `(*, m, n)`.
    """
    assert isinstance(Q, Tensor)
    assert isinstance(B, Tensor)

    return Q @ B
