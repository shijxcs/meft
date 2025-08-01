from functools import reduce
from math import sqrt
from operator import mul

import torch
from torch import Tensor

from . import config
from .eigh import truncated_eigh
from .qb import randomized_qb
from .utils import scaled_matmul


def truncated_svd(
    A: Tensor,
    rank: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Truncated singular value decomposition,
    defined as `A = U @ S.diag_embed() @ V.mT`.

    Args:
        A (Tensor):
            &#45; with shape `(*, m, n)`.
        rank (int or None, optional):
            &#45; target `rank`. If `None`, then `k = min(m, n)`.
            Default: `None`.

    Returns:
        (U, S, V) (Tensor, Tensor, Tensor):
            &#45; with shape `(*, m, k)`, `(*, k)`, `(*, n, k)`, where `k = min(m, n, rank)`.
    """
    assert isinstance(A, Tensor)
    assert isinstance(rank, int | None)

    m, n = A.shape[-2], A.shape[-1]

    if m <= n:
        AAT = (A @ A.mT).div_(sqrt(n))  # (*, m, m)
        L, U = truncated_eigh(AAT, rank)  # (*, k), (*, m, k)  ## A = U @ S @ VT ==> AAT = U @ L @ UT = U @ S^2 @ UT
        L.mul_(sqrt(n))

        S = L.sqrt()  # (*, k)
        V = scaled_matmul(A.mT, U).mul_(L.rsqrt().nan_to_num_(0).unsqueeze_(-2))  # (*, n, k)  ## A = U @ S @ VT ==> V = AT @ U @ L^(-1/2)

        if config.SCALING_UNIT:
            S.div_(sqrt(n))
            V.mul_(sqrt(n))
    else:
        V, S, U = truncated_svd(A.mT, rank)  # A = U @ S @ VT ==> AT = V @ S @ UT

    return U, S, V


def randomized_svd(
    A: Tensor,
    rank: int,
    nover: int = 0,
    niter: int = 0,
    test_matrix: str = "subs",
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Randomzied singular value decomposition,
    defined as `A = U @ S.diag_embed() @ V.mT`.

    Args:
        A (Tensor):
            &#45; with shape `(*, m, n)`.
        rank (int):
            &#45; target `rank`.
        nover (int, optional):
            &#45; number of overestimated rank.
            Default: `0`.
        niter (int, optional):
            &#45; number of power iterations.
            Default: `0`.
        test_matrix (str, optional):
            &#45; the type of test matrix.
            Optional: `'gauss'` | `'subs'`.
            Default: `'subs'`.

    Returns:
        (U, S, V) (Tensor, Tensor, Tensor):
            &#45; with shape `(*, m, k)`, `(*, k)`, `(*, n, k)`, where `k = min(m, n, rank)`.
    """
    assert isinstance(A, Tensor)
    assert isinstance(rank, int)
    assert isinstance(nover, int)
    assert isinstance(niter, int)
    assert isinstance(test_matrix, str)

    m, n = A.shape[-2], A.shape[-1]

    if m <= n:
        k = min(m, n, rank)
        p = min(nover, m - k)

        # Compute SVD on projected B
        Q, B = randomized_qb(A, k+p, niter=niter, test_matrix=test_matrix, left=True)
        Ub, S, V = truncated_svd(B, k)  # (*, k+p, k), (*, k), (*, n, k)
        U = scaled_matmul(Q, Ub)   # (*, m, k)

        if config.SCALING_UNIT:
            U.mul_(sqrt(k+p))
            S.mul_(sqrt(k+p))
    else:
        V, S, U = randomized_svd(A.mT, rank, nover=nover, niter=niter)

    return U, S, V


def nystrom_svd(
    A: Tensor,
    rank: int,
    nover: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Nyström singular value decomposition,
    defined as `A = U @ S.diag_embed() @ V.mT`.

    Args:
        A (Tensor):
            &#45; with shape `(*, m, n)`.
        rank (int):
            &#45; target `rank`.
        nover (int, optional):
            &#45; number of overestimated rank.
            Default: `0`.

    Returns:
        (U, S, V) (Tensor, Tensor, Tensor):
            &#45; with shape `(*, m, k)`, `(*, k)`, `(*, n, k)`, where `k = min(m, n, rank)`.
    """
    assert isinstance(A, Tensor)
    assert isinstance(rank, int)
    assert isinstance(nover, int)

    m, n = A.shape[-2], A.shape[-1]

    if m <= n:
        k = min(m, n, rank)
        p = min(nover, m - k)

        # Sample subspace of A @ AT such that A @ AT = C @ W^-1 @ CT
        idx = torch.randperm(m)[:k+p]
        Ak = A[..., idx, :]  # (*, k+p, n)
        C = (A @ Ak.mT).div_(sqrt(n))  # (*, m, k+p)  ## C = A @ AkT
        W = C[..., idx, :]  # (*, k+p, k+p)  ## W = Ak @ AkT
        Lw, Uw = truncated_eigh(W)  # (*, k+p), (*, k+p, k+p)  ## A @ AT = C @ W^-1 @ CT = C @ Uw @ Lw^-1 @ UwT @ CT
        Lw.mul_(sqrt(n))

        # Compute SVD on low-rank approximation E where A @ AT = E @ ET
        E = scaled_matmul(C, Uw).mul_(Lw.rsqrt().nan_to_num_(0).unsqueeze_(-2))  # (*, m, k+p)  ## E = C @ Uw @ Lw^(-1/2) ==> A @ AT = E @ ET
        U, S, _ = truncated_svd(E, k)  # (*, m, k), (*, k)  ## A @ AT = E @ ET = U @ S^2 @ UT
        V = scaled_matmul(A.mT, U).mul_(S.reciprocal().nan_to_num_(0).unsqueeze_(-2))  # (*, n, k)  ## A = U @ S @ VT ==> V = AT @ U @ S^-1

        if config.SCALING_UNIT:
            S.mul_(sqrt(k+p))
            V.div_(sqrt(k+p))
    else:
        V, S, U = nystrom_svd(A.mT, rank, nover=nover)

    return U, S, V


def svd_reconstruct(
    U: Tensor,
    S: Tensor,
    V: Tensor,
) -> Tensor:
    """
    Reconstruction of singular value decomposition.

    Args:
        U (Tensor):
            &#45; with shape `(*, m, k)`.
        S (Tensor):
            &#45; with shape `(*, k)`.
        V (Tensor):
            &#45; with shape `(*, n, k)`.

    Returns:
        Tensor:
            &#45; with shape `(*, m, n)`.
    """
    assert isinstance(U, Tensor)
    assert isinstance(S, Tensor)
    assert isinstance(V, Tensor)

    if reduce(mul, U.shape) < reduce(mul, V.shape):
        return (U * S.unsqueeze(-2)) @ V.mT
    else:
        return U @ (V * S.unsqueeze(-2)).mT
