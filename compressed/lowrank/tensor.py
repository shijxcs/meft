import torch
from torch import Tensor

from ..tensor import CompressedTensor
from ..utils import COMPRESS_FUNC_MAPPING, RECONSTRUCT_FUNC_MAPPING

LOWRANK_COMPRESS_FUNC_MAPPING = COMPRESS_FUNC_MAPPING["lowrank"]
LOWRANK_RECONSTRUCT_FUNC_MAPPING = RECONSTRUCT_FUNC_MAPPING["lowrank"]


class LowRankDecomposedTensor(CompressedTensor):
    """
    This class implements **LowRankDecomposedTensor** as a **CompressedTensor** subclass.

    Args:
        tensor (Tensor):
            &#45; the original tensor, with shape: `(*, m, n)`.
            If `CompressedTensor`, it will first call `.reconstruct()` to recover the original tensor.
        method (str, optional):
            &#45; the compressing method.
            Default: `'rqb'`.
        rank (int or float):
            &#45; the target rank of the compressed tensor.
            If `float`, it indicates the ratio to hidden size and should be in range `(0,1)`.
        **kwargs:
            &#45; additional keyword arguments used by compression.

    Returns:
        LowRankDecomposedTensor:
            &#45; storaged with tensor factors.
    """

    def __new__(cls, tensor: Tensor, **kwargs):
        method = kwargs.pop("method", "rqb")
        if method not in LOWRANK_COMPRESS_FUNC_MAPPING:
            raise ValueError("Invalid value of `method`.")
        kwargs.update({"method": method})

        rank = kwargs.pop("rank")
        if isinstance(rank, int):
            pass
        elif isinstance(rank, float) and 0 < rank < 1:
            rank = int(tensor.shape[-1] * rank)
        else:
            raise TypeError("Invalid type of `rank`, must be `int` or `float` in range `(0,1)`.")
        kwargs.update({"rank": rank})

        compressed_tensor = super().__new__(cls, tensor, **kwargs)
        compressed_tensor.rank = rank
        return compressed_tensor

    @property
    def rank(self) -> int:
        return self._rank

    @rank.setter
    def rank(self, value):
        self._rank = value

    @torch.no_grad
    @staticmethod
    def compress(tensor: Tensor, **kwargs) -> tuple[Tensor, ...]:
        method = kwargs.pop("method")
        compress_func = LOWRANK_COMPRESS_FUNC_MAPPING[method]
        tensor = tensor.flatten(0, -2)  # (*l, m, n) -> (lm, n)
        factors = compress_func(tensor, **kwargs)  # (lm, n) -> ...
        return factors

    @torch.no_grad
    def reconstruct(self) -> Tensor:
        reconstruct_func = LOWRANK_RECONSTRUCT_FUNC_MAPPING[self.method]
        tensor = reconstruct_func(*self.factors)  # ... -> (lm, n)
        tensor = tensor.reshape(*self.shape)  # (lm, n) -> (*l, m, n)
        tensor.requires_grad = self.requires_grad
        return tensor

    def __repr__(self) -> str:
        data = self.reconstruct()
        data.requires_grad = self.requires_grad
        return f"{data.__repr__()[:-1]}, rank={self.rank})"
