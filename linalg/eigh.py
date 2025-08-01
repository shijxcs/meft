from math import sqrt

import torch
from torch import Tensor

from . import config


def truncated_eigh(
    A: Tensor,
    rank: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Truncated eigenvalue decomposition of a real symmetric matrix,
    defined as `A = U @ L.diag_embed() @ U.mT`.

    Args:
        A (Tensor):
            &#45; with shape `(*, n, n)`.
        rank (int or None, optional):
            &#45; target `rank`. If `None`, no truncation is applied.
            Default: `None`.

    Returns:
        (L, U) (Tensor, Tensor):
            &#45; with shape `(*, k)`, `(*, n, k)`, where `k = min(n, rank)`.
    """
    assert isinstance(A, Tensor)
    assert isinstance(rank, int | None)

    n = A.shape[-1]

    L, U = torch.linalg.eigh(A.float().nan_to_num_(0))

    if config.SCALING_UNIT:
        U.mul_(sqrt(n))
        L.div_(n)

    L = L.to(A.dtype).flip(-1)  # (*, n)
    U = U.to(A.dtype).flip(-1)  # (*, n, n)

    if rank is not None:
        L = L[..., :rank]  # (*, k)
        U = U[..., :, :rank]  # (*, n, k)

    L.mul_(L.gt(0))  # (*, k)
    U.mul_(L.gt(0).unsqueeze_(-2))  # (*, n, k)
    return L, U


def eigh_reconstruct(
    L: Tensor,
    U: Tensor,
) -> Tensor:
    """
    Reconstruction of eigenvalue decomposition.

    Args:
        L (Tensor):
            &#45; with shape `(*, k)`.
        U (Tensor):
            &#45; with shape `(*, n, k)`.

    Returns:
        Tensor:
            &#45; with shape `(*, n, n)`.
    """
    assert isinstance(L, Tensor)
    assert isinstance(U, Tensor)

    return U @ L.diag_embed() @ U.mT
