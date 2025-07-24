from abc import abstractmethod
from collections import defaultdict
from typing import *

from torch import Tensor

from ..utils.threading import TaskProcessor
from ..utils.weakref import WeakHashKeyDictionary

compress_cache = defaultdict(WeakHashKeyDictionary)
compress_processor = TaskProcessor()


class CompressedTensor(Tensor):
    """
    This class implements **CompressedTensor** as a **Tensor** subclass.

    Args:
        tensor (Tensor, optional):
            &#45; the original tensor.
        method (str, optional):
            &#45; the compression method.
            Optional: `'tsvd'` | `'rsvd'` | `'nyssvd'` | `'rrsvd'`.
            Default: `'rrsvd'`.
        **kwargs (Mapping[str, Any], optional):
            &#45; additional keyword arguments used by compression.

    Returns:
        CompressedTensor:
            *-* return as a subclass type (e.g. `SingularValueDecomposedTensor`).
    """

    def __new__(cls, tensor: Tensor = None, **kwargs: Optional[Mapping[str, Any]]) -> Self:
        # The `CompressedTensor` class will not be instantiated directly;
        # it will return as a subclass type based on the argument `method`.
        if cls is CompressedTensor:
            method = kwargs.get("method", "rrsvd")
            if method in ("tsvd", "rsvd", "nyssvd", "rrsvd"):
                from .svd import SingularValueDecomposedTensor
                return SingularValueDecomposedTensor(tensor, **kwargs)
            else:
                raise ValueError("Invalid value of `method`.")
        else:
            return super().__new__(cls)

    @abstractmethod
    def reconstruct(self) -> Tensor:
        pass

    @property
    def requires_grad(self) -> bool:
        return self._fake_requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._fake_requires_grad = value
