from math import sqrt
from typing import *

import torch
from torch import Tensor

from ._config import CONFIG


def truncated_eigh(
    A: Tensor,
    rank: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Truncated eigenvalue decomposition of a real symmetric matrix.

    Args:
        A (Tensor):
            &#45; with shape `(*, n, n)`.
        rank (int, optional):
            &#45; target `rank`. If `None`, no truncation is applied.
            Default: `None`.

    Returns:
        (L, Q) (Tensor, Tensor):
            &#45; with shape `(*, k)`, `(*, n, k)`, where `k = min(n, rank)`.
    """
    assert isinstance(A, Tensor)
    assert isinstance(rank, Optional[int])

    n = A.shape[-1]

    L, Q = torch.linalg.eigh(A.float().nan_to_num_(0))

    if CONFIG.SCALING_UNIT:
        Q *= sqrt(n)
        L /= n

    L = L.to(A.dtype).flip(-1)  # (*, n)
    Q = Q.to(A.dtype).flip(-1)  # (*, n, n)

    if rank is not None:
        L = L[..., :rank]  # (*, k)
        Q = Q[..., :, :rank]  # (*, n, k)

    L.mul_(L.gt(0))  # (*, k)
    Q.mul_(L.gt(0).unsqueeze_(-2))  # (*, n, k)
    return L, Q


def eigh_reconstruct(
    L: Tensor,
    Q: Tensor,
) -> Tensor:
    """
    Reconstruction of eigenvalue decomposition.

    Args:
        L (Tensor):
            &#45; with shape `(*, k)`.
        Q (Tensor):
            &#45; with shape `(*, n, k)`.

    Returns:
        Tensor:
            &#45; with shape `(*, n, n)`.
    """
    assert isinstance(L, Tensor)
    assert isinstance(Q, Tensor)

    return Q @ L.diag_embed() @ Q.mT
