import inspect
from typing import *

import torch
from torch import Tensor

from .. import compress_cache, compress_processor, CompressedTensor
from .._config import CONFIG
from ...linalg import *

SVD_FUNC_MAPPING = {
    "tsvd": truncated_svd,
    "rsvd": randomized_svd,
    "rrsvd": row_aware_randomized_svd,
    "nyssvd": nystrom_svd,
}


class SingularValueDecomposedTensor(CompressedTensor):
    """
    This class implements **SingularValueDecomposedTensor** as a **CompressedTensor** subclass.

    Args:
        tensor (Tensor):
            &#45; the original tensor, with shape: `(*, m, n)`.
            If `CompressedTensor`, it will first call `.reconstruct()` to recover the original tensor.
        method (str, optional):
            &#45; the svd method.
            Optional: `'tsvd'` | `'rsvd'` | `'nyssvd'` | `'rrsvd'`.
            Default: `'rrsvd'`.
        rank (Union[int, float]):
            &#45; the target rank of the compressed tensor.
            If `float`, it indicates the ratio to hidden size and should be in range `(0,1)`.
        **kwargs (Mapping[str, Any], optional):
            &#45; additional keyword arguments used by compression.

    Returns:
        SingularValueDecomposedTensor:
            &#45; storaged with three tensors: **U** `(*, m, k)`, **S** `(k)`, and **V** `(n, k)`.
    """

    def __new__(cls, tensor: Tensor, **kwargs) -> Self:

        if not isinstance(tensor, Tensor):
            raise TypeError("Invalid type of `tensor`, must be `torch.Tensor`.")

        if isinstance(tensor, CompressedTensor):
            tensor = tensor.reconstruct()

        method = kwargs.pop("method", "rrsvd")
        if method in SVD_FUNC_MAPPING:
            svd_func = SVD_FUNC_MAPPING[method]
        else:
            raise ValueError("Invalid value of `method`.")

        rank = kwargs.pop("rank")
        if isinstance(rank, int):
            kwargs.update({"rank": rank})
        elif isinstance(rank, float) and 0 < rank < 1:
            kwargs.update({"rank": int(tensor.shape[-1] * rank)})
        else:
            raise TypeError("Invalid type of `rank`, must be `int` or `float` in range `(0,1)`.")

        all_kwargs = {name: param.default for name, param in inspect.signature(svd_func).parameters.items()}
        all_kwargs.update(kwargs)
        all_kwargs.update({"method": method})
        kwargs_key = tuple(sorted(all_kwargs.items()))
        tensor_key = tensor

        if CONFIG.CACHE_COMPRESS and tensor_key in compress_cache[kwargs_key]:
            U, S, V = compress_cache[kwargs_key][tensor_key]

        else:
            if CONFIG.ASYNC_COMPRESS and compress_processor.running:
                U = torch.zeros(*tensor.shape[:-1], rank, dtype=tensor.dtype, device=tensor.device)
                S = torch.zeros(rank, dtype=tensor.dtype, device=tensor.device)
                V = torch.zeros(tensor.shape[-1], rank, dtype=tensor.dtype, device=tensor.device)
                compress_processor.submit(
                    cls.svd_decompose,
                    args=(svd_func, tensor, kwargs),
                    outputs=(U, S, V),
                )
            else:
                U, S, V = cls.svd_decompose(svd_func, tensor, kwargs)

            if CONFIG.CACHE_COMPRESS:
                compress_cache[kwargs_key][tensor_key] = (U, S, V)  # Use orginal `tensor_key` instead of `tensor`, which may be transformed.

        compressed_tensor = super().__new__(cls)
        compressed_tensor.U = U  # (*l, m, k)
        compressed_tensor.S = S  # (k)
        compressed_tensor.V = V  # (n, k)
        compressed_tensor.requires_grad = tensor.requires_grad
        return compressed_tensor

    @torch.no_grad
    @staticmethod
    def svd_decompose(svd_func: Callable, tensor: Tensor, kwargs: Mapping[str, Any]):
        tensor_shape = tensor.shape  # (*l, m, n)
        tensor = tensor.flatten(0, -2)  # (*l, m, n) -> (lm, n)
        U, S, V = svd_func(tensor, **kwargs)  # (lm, n) -> (lm, k), (k), (n, k)
        U = U.unflatten(-2, tensor_shape[:-1])  # (lm, k) -> (*l, m, k)
        return U, S, V

    @torch.no_grad
    def reconstruct(self) -> Tensor:
        return svd_reconstruct(self.U, self.S, self.V)

    @classmethod
    def from_usv(cls, U: Tensor, S: Tensor, V: Tensor) -> Self:
        """
        Returns a **SingularValueDecomposedTensor** created from
        **U** (left singular vectors), **S** (singular values), **V** (right singular vectors).

        Args:
            U (Tensor):
                &#45; with shape `(*, m, k)`.
            S (Tensor):
                &#45; with shape `(k)`.
            V (Tensor):
                &#45; with shape `(*, n, k)`.

        Returns:
            SingularValueDecomposedTensor:
                &#45; with shape `(*, m, n)` and rank `k`.
        """

        if not (isinstance(U, Tensor) and isinstance(S, Tensor) and isinstance(V, Tensor)):
            raise TypeError("Invalid combination of arguments, must be (Tensor, Tensor, Tensor).")

        compressed_tensor = super().__new__(cls)
        compressed_tensor.U = U
        compressed_tensor.S = S
        compressed_tensor.V = V
        compressed_tensor.requires_grad = False
        return compressed_tensor

    @property
    def rank(self) -> int:
        return self.S.shape[-1]

    def __repr__(self) -> str:
        data = self.reconstruct().detach()
        data.requires_grad = self.requires_grad
        return f"{data.__str__()[:-1]}, svd_rank={self.rank})"
