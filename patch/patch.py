import warnings
from collections.abc import Callable
from types import MethodType
from functools import partial

from torch import nn
from transformers.utils.import_utils import is_peft_available

from .functions import checkpoint, nn_linear_forward


def _checkpoint_module(
    module: nn.Module,
    compress_kwargs: dict | None = None,
) -> None:
    requires_grad = any(param.requires_grad for param in module.parameters())
    module.forward = MethodType(partial(checkpoint, module.forward.__func__, requires_grad=requires_grad, compress_kwargs=compress_kwargs), module)


def _patch_module(
    module: nn.Module,
    forward: Callable,
    compress_kwargs: dict | None = None,
) -> None:
    if is_peft_available():
        from peft.tuners.tuners_utils import BaseTunerLayer
        if isinstance(module, BaseTunerLayer):
            _patch_module(module.get_base_layer(), forward, compress_kwargs=compress_kwargs)
            _patch_peft_module(module, compress_kwargs=compress_kwargs)
            return
        elif hasattr(module, "__module__") and module.__module__.startswith("peft."):
            warnings.warn(f"No patch supported for module type: {type(module)}.")
            return
    module.forward = MethodType(partial(forward, compress_kwargs=compress_kwargs), module)


def _patch_peft_module(
    module: nn.Module,
    compress_kwargs: dict | None = None,
) -> None:
    from peft.tuners import lora
    if isinstance(module, lora.Linear):
        for adapter in module.lora_A.values():
            _patch_module(adapter, nn_linear_forward, compress_kwargs=compress_kwargs)
    else:
        warnings.warn(f"No patch supported for module type: {type(module)}.")
