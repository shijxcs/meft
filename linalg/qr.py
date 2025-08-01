from math import sqrt

import torch
from torch import Tensor

from . import config


def qr(
    A: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    QR decomposition, defined as `A = Q @ R`.

    Args:
        A (Tensor):
            &#45; with shape `(*, m, n)`.

    Returns:
        (Q, R) (Tensor, Tensor):
            &#45; with shape `(*, m, min(m, n))`, `(*, min(m, n), n)`.
    """
    assert isinstance(A, Tensor)

    m, n = A.shape[-2], A.shape[-1]

    Q, R = torch.linalg.qr(A.float().nan_to_num_(0))

    if config.SCALING_UNIT:
        Q.mul_(sqrt(m))
        R.div_(sqrt(m))

    Q = Q.to(A.dtype)
    R = R.to(A.dtype)
    return Q, R


def qr_reconstruct(
    Q: Tensor,
    R: Tensor,
) -> Tensor:
    """
    Reconstruction of QR decomposition.

    Args:
        Q (Tensor):
            &#45; with shape `(*, m, k)`.
        R (Tensor):
            &#45; with shape `(*, k, n)`.

    Returns:
        Tensor:
            &#45; with shape `(*, m, n)`.
    """
    assert isinstance(Q, Tensor)
    assert isinstance(R, Tensor)

    return Q @ R
