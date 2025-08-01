from collections.abc import Callable

import torch
from torch import nn, Tensor

from ...ops.checkpoint import CheckpointFunction


def checkpoint(
    function: Callable,
    self: nn.Module,
    hidden_states: Tensor,
    *args,
    preserve_rng_state: bool = True,
    requires_grad: bool = True,
    compress_kwargs: dict | None = None,
    **kwargs,
):
    """
    This method is an improved version of `torch.utils.checkpoint.checkpoint`
    with `use_reentrant=True`, which supports recording the autograd graph
    when setting `requires_grad=True`.

    The checkpointed `function` is allowed to receive multiple arguments, but
    `hidden_states` must be the first arguments or in the keyword arguments.
    And only `hidden_states` can be compressed.
    """

    if requires_grad:
        dummy = torch.empty((0,), requires_grad=True)
    else:
        dummy = None

    if not self.training:
        compress_kwargs = None

    n_args = len(args)
    n_kwargs = len(kwargs)
    kwargs_keys = tuple(kwargs.keys())
    kwargs_vals = tuple(kwargs.values())

    return CheckpointFunction.apply(
        function,
        self,
        hidden_states,
        preserve_rng_state,
        dummy,
        compress_kwargs,
        n_args,
        n_kwargs,
        *args,
        *kwargs_keys,
        *kwargs_vals
    )
