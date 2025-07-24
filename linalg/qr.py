from math import sqrt
from typing import *

import torch
from torch import Tensor

from ._config import CONFIG


def qr(
    A: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    QR decomposition.

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

    if CONFIG.SCALING_UNIT:
        Q *= sqrt(m)
        R /= sqrt(m)

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
