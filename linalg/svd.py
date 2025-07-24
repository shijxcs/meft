from functools import reduce
from math import sqrt
from operator import mul
from typing import *

import torch
from torch import Tensor

from ._config import CONFIG
from ._utils import scaled_matmul
from .eigh import truncated_eigh
from .qr import qr


def truncated_svd(
    A: Tensor,
    rank: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Truncated singular value decomposition.

    Args:
        A (Tensor):
            &#45; with shape `(*, m, n)`.
        rank (int, optional):
            &#45; target `rank`. If `None`, then `k = min(m, n)`.
            Default: `None`.

    Returns:
        (U, S, V) (Tensor, Tensor, Tensor):
            &#45; with shape `(*, m, k)`, `(*, k)`, `(*, n, k)`, where `k = min(m, n, rank)`.
    """
    assert isinstance(A, Tensor)
    assert isinstance(rank, Optional[int])

    m, n = A.shape[-2], A.shape[-1]

    if m <= n:
        AAT = scaled_matmul(A, A.mT)  # (*, m, m)
        L, U = truncated_eigh(AAT, rank)  # (*, k), (*, m, k)  ## A = U @ S @ VT ==> AAT = U @ L @ UT (L = S^2)

        if CONFIG.SCALING_UNIT:
            S = L.sqrt()  # (*, k)
            V = scaled_matmul(A.mT, U).mul_(L.rsqrt().nan_to_num_(0).unsqueeze_(-2))  # (*, n, k)
        else:
            S = L.sqrt().mul_(sqrt(n))  # (*, k)
            V = (A.mT @ U).mul_(L.rsqrt().nan_to_num_(0).unsqueeze_(-2)).div_(sqrt(n))  # (*, n, k)  ## A = U @ S @ VT ==> V = AT @ U @ L^(-1/2)

    else:
        V, S, U = truncated_svd(A.mT, rank)  # A = U @ S @ VT ==> AT = V @ S @ UT

    return U, S, V


def randomized_svd(
    A: Tensor,
    rank: int,
    n_oversamples: int = 0,
    n_iters: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Randomzied singular value decomposition.

    Args:
        A (Tensor):
            &#45; with shape `(*, m, n)`.
        rank (int):
            &#45; target `rank`.
        n_oversamples (int, optional):
            &#45; number of oversampled vectors.
            Default: `0`.
        n_iters (int, optional):
            &#45; number of power iterations.
            Default: `0`.

    Returns:
        (U, S, V) (Tensor, Tensor, Tensor):
            &#45; with shape `(*, m, k)`, `(*, k)`, `(*, n, k)`, where `k = min(m, n, rank)`.
    """
    assert isinstance(A, Tensor)
    assert isinstance(rank, int)
    assert isinstance(n_oversamples, int)
    assert isinstance(n_iters, int)

    m, n = A.shape[-2], A.shape[-1]

    k = min(m, n, rank)
    p = n_oversamples

    if m <= n:
        # Sample column space of A with Ohm matrix, and generate the approximate orthonormal basis Q
        Ohm = torch.randn(n, k + p, dtype=A.dtype, device=A.device)  # (*, n, k+p)
        Y = scaled_matmul(A, Ohm)  # (*, m, k+p)
        for _ in range(n_iters):
            Q, R = qr(Y)  # (*, m, k+p)
            Ohm = A.mT @ Q  # (*, n, k+p)
            Y = scaled_matmul(A, Ohm)  # (*, m, k+p)
        Q, R = qr(Y)  # (*, m, k+p)

        # Compute SVD on projected B
        if CONFIG.SCALING_UNIT:
            B = scaled_matmul(Q.mT, A) * sqrt(k+p) # (*, k+p, n)
            Ub, S, V = truncated_svd(B, k)  # (*, k+p, k), (*, k), (*, n, k)
            U = scaled_matmul(Q, Ub) * sqrt(k+p)   # (*, m, k)
        else:
            B = Q.mT @ A  # (*, k+p, n)
            Ub, S, V = truncated_svd(B, k)  # (*, k+p, k), (*, k), (*, n, k)
            U = Q @ Ub  # (*, m, k)

    else:
        V, S, U = randomized_svd(A.mT, rank, n_oversamples=n_oversamples, n_iters=n_iters)

    return U, S, V


def row_aware_randomized_svd(
    A: Tensor,
    rank: int,
    n_oversamples: int = 0,
    n_iters: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Row-aware randomized singular value decomposition.
    https://arxiv.org/abs/2408.04503

    Args:
        A (Tensor):
            &#45; with shape `(*, m, n)`.
        rank (int):
            &#45; target `rank`.
        n_oversamples (int, optional):
            &#45; number of oversampled vectors.
            Default: `0`.
        n_iters (int, optional):
            &#45; number of power iterations.
            Default: `0`.

    Returns:
        (U, S, V) (Tensor, Tensor, Tensor):
            &#45; with shape `(*, m, k)`, `(*, k)`, `(*, n, k)`, where `k = min(m, n, rank)`.
    """
    assert isinstance(A, Tensor)
    assert isinstance(rank, int)
    assert isinstance(n_oversamples, int)
    assert isinstance(n_iters, int)

    m, n = A.shape[-2], A.shape[-1]

    k = min(m, n, rank)
    p = n_oversamples

    if m <= n:
        # Sample subspace of A with Ak matrix, and Generate the approximate orthonormal basis Q
        idx = torch.randperm(m)[:k+p]
        Ak = A[..., idx, :]  # (*, k+p, n)
        C = scaled_matmul(A, Ak.mT)  # (*, m, k+p)
        for _ in range(n_iters):
            Q, R = qr(C)  # (*, m, k+p)
            Ohm = A.mT @ Q  # (*, n, k+p)
            C = scaled_matmul(A, Ohm)  # (*, m, k+p)
        Q, R = qr(C)  # (*, m, k+p)

        # Compute SVD on projected B
        if CONFIG.SCALING_UNIT:
            B = scaled_matmul(Q.mT, A).mul_(sqrt(k+p)) # (*, k+p, n)
            Ub, S, V = truncated_svd(B, k)  # (*, k+p, k), (*, k), (*, n, k)
            U = scaled_matmul(Q, Ub).mul_(sqrt(k+p))   # (*, m, k)
        else:
            B = Q.mT @ A  # (*, k+p, n)
            Ub, S, V = truncated_svd(B, k)  # (*, k+p, k), (*, k), (*, n, k)
            U = Q @ Ub  # (*, m, k)

    else:
        V, S, U = row_aware_randomized_svd(A.mT, rank, n_oversamples=n_oversamples, n_iters=n_iters)

    return U, S, V


def nystrom_svd(
    A: Tensor,
    rank: int,
    n_oversamples: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Nyström singular value decomposition.

    Args:
        A (Tensor):
            &#45; with shape `(*, m, n)`.
        rank (int):
            &#45; target `rank`.
        n_oversamples (int, optional):
            &#45; number of oversampled vectors.
            Default: `0`.

    Returns:
        (U, S, V) (Tensor, Tensor, Tensor):
            &#45; with shape `(*, m, k)`, `(*, k)`, `(*, n, k)`, where `k = min(m, n, rank)`.
    """
    assert isinstance(A, Tensor)
    assert isinstance(rank, int)
    assert isinstance(n_oversamples, int)

    m, n = A.shape[-2], A.shape[-1]

    k = min(m, n, rank)
    p = n_oversamples

    if m <= n:
        # Sample subspace of A @ AT such that A @ AT = C @ W^-1 @ CT
        idx = torch.randperm(m)[:k+p]
        Ak = A[..., idx, :]  # (*, k+p, n)  ## P @ A
        C = scaled_matmul(A, Ak.mT)  # (*, m, k+p)  ## C = A @ AT @ PT
        W = C[..., idx, :]  # (*, k+p, k+p)  ## W = P @ A @ AT @ PT
        Lw, Qw = truncated_eigh(W)  # (*, k+p), (*, k+p, k+p)  ## A @ AT = C @ W^-1 @ CT = C @ Qw @ Lw^-1 @ QwT @ CT

        # Compute SVD on low-rank approximation E where A @ AT = E @ ET
        if CONFIG.SCALING_UNIT:
            E = scaled_matmul(C, Qw).mul_(Lw.rsqrt().nan_to_num_(0).unsqueeze_(-2)).mul_(sqrt(len(idx))) # (*, m, k+p)
            U, S, _ = truncated_svd(E, k)  # (*, m, k), (*, k)
            V = scaled_matmul(A.mT, U).mul_(S.reciprocal().nan_to_num_(0).unsqueeze_(-2)) # (*, n, k)
        else:
            E = (C @ Qw).mul_(Lw.rsqrt().nan_to_num_(0).unsqueeze_(-2)).mul_(sqrt(n))  # (*, m, k+p)  ## E = C @ Qw @ Lw^(-1/2) ==> A @ AT = E @ ET
            U, S, _ = truncated_svd(E, k)  # (*, m, k), (*, k)  ## A @ AT = E @ ET = U @ S^2 @ UT
            V = (A.mT @ U).mul_(S.reciprocal().nan_to_num_(0).unsqueeze_(-2))  # (*, n, k)  ## A = U @ S @ VT ==> V = AT @ U @ S^-1

    else:
        V, S, U = nystrom_svd(A.mT, rank, n_oversamples=n_oversamples)

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
        return (U @ S.diag_embed()) @ V.mT
    else:
        return U @ (V @ S.diag_embed()).mT
