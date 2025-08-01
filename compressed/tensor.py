from abc import abstractmethod

from torch import Size, Tensor

from . import config
from .utils import COMPRESS_FUNC_MAPPING, get_compress_cache, get_compress_processor


class CompressedTensor(Tensor):
    """
    This class implements **CompressedTensor** as a **Tensor** subclass.

    Args:
        tensor (Tensor):
            &#45; the original tensor.
        method (str, optional):
            &#45; the compressing method.
            Default: `'rqb'`.
        **kwargs:
            &#45; additional keyword arguments used by compression.

    Returns:
        CompressedTensor:
            *-* return as a subclass type (e.g. `SingularValueDecomposedTensor`).
    """

    def __new__(cls, tensor: Tensor, **kwargs):
        # The `CompressedTensor` class will not be instantiated directly;
        # it will return as a subclass type based on the argument `method`.
        if cls is CompressedTensor:
            method = kwargs.get("method", "rqb")
            if method in COMPRESS_FUNC_MAPPING["lowrank"]:
                from .lowrank import LowRankDecomposedTensor
                return LowRankDecomposedTensor(tensor, **kwargs)
            else:
                raise ValueError("Invalid value of `method`.")

        else:
            if not isinstance(tensor, Tensor):
                raise TypeError("Invalid type of `tensor`, must be `torch.Tensor`.")

            kwargs_key = tuple(sorted(kwargs.items()))
            tensor_key = tensor

            compress_cache = get_compress_cache()

            if config.CACHE_COMPRESS and tensor_key in compress_cache[kwargs_key]:
                factors = compress_cache[kwargs_key][tensor_key]

            else:
                if isinstance(tensor, CompressedTensor):
                    tensor = tensor.reconstruct()

                compress_processor = get_compress_processor()

                if config.ASYNC_COMPRESS and compress_processor.running:
                    factors = []
                    compress_processor.submit(
                        cls.compress,
                        args=(tensor,),
                        kwargs=kwargs,
                        outputs=factors,
                    )
                else:
                    factors = [*cls.compress(tensor, **kwargs)]

                if config.CACHE_COMPRESS:
                    compress_cache[kwargs_key][tensor_key] = factors  # Use orginal `tensor_key` instead of `tensor`, which may be transformed.

            compressed_tensor = super().__new__(cls)
            compressed_tensor.factors = factors
            compressed_tensor.method = kwargs["method"]
            compressed_tensor.requires_grad = tensor_key.requires_grad
            compressed_tensor.shape = tensor_key.shape
            return compressed_tensor

    @staticmethod
    @abstractmethod
    def compress(tensor: Tensor, **kwargs) -> tuple[Tensor, ...]:
        pass

    @abstractmethod
    def reconstruct(self) -> Tensor:
        pass

    @property
    def factors(self) -> list:
        return self._factors

    @factors.setter
    def factors(self, value):
        self._factors = value

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    @property
    def requires_grad(self) -> bool:
        return self._fake_requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._fake_requires_grad = value

    @property
    def shape(self) -> Size:
        return self._original_shape

    @shape.setter
    def shape(self, value):
        self._original_shape = value
