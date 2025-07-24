import contextlib
from typing import *

import torch
from torch import Tensor
from torch.utils.checkpoint import (
    _get_device_module, _infer_device_type, get_device_states, set_device_states, _get_autocast_kwargs
)

from ..compressed import CompressedTensor


def detach_variable(
    hidden_states: Tensor,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Tuple[Tensor, Tuple[Any, ...], Mapping[str, Any]]:

    if isinstance(hidden_states, CompressedTensor):
        detached_hidden_states = hidden_states.reconstruct().detach()
    else:
        detached_hidden_states = hidden_states.detach()
    detached_hidden_states.requires_grad = hidden_states.requires_grad

    detached_args = []
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            detached_args.append(arg)
        else:
            x = arg.detach()
            x.requires_grad = arg.requires_grad
            detached_args.append(x)

    detached_kwargs = {}
    for key, val in kwargs.items():
        if not isinstance(val, torch.Tensor):
            detached_kwargs[key] = val
        else:
            x = val.detach()
            x.requires_grad = val.requires_grad
            detached_kwargs[key] = x
        
    return detached_hidden_states, detached_args, detached_kwargs


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        run_function: Any,
        self: Any,
        hidden_states: Tensor,
        preserve_rng_state: bool,
        dummy: Optional[Tensor],
        compress_kwargs: Optional[Mapping[str, Any]],
        n_args: int,
        n_kwargs: int,
        *args_kwargs: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        args, kwargs_keys, kwargs_vals = \
            args_kwargs[:n_args], args_kwargs[n_args:-n_kwargs], args_kwargs[-n_kwargs:]
        kwargs = dict(zip(kwargs_keys, kwargs_vals))

        outputs = run_function(self, hidden_states, *args, **kwargs)
        return outputs

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        run_function, self, hidden_states, preserve_rng_state, dummy, compress_kwargs, n_args, n_kwargs, *args_kwargs = inputs
        args, kwargs_keys, kwargs_vals = \
            args_kwargs[:n_args], args_kwargs[n_args:-n_kwargs], args_kwargs[-n_kwargs:]
        kwargs = dict(zip(kwargs_keys, kwargs_vals))

        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device_type = _infer_device_type(hidden_states, *args, *kwargs_vals)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device_type
        )
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(hidden_states, *args, *kwargs_vals)

        ctx.run_function = run_function
        ctx.self = self

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.input_args = []
        ctx.input_kwargs = {}
        ctx.tensor_keys = []
        saved_tensors = []

        ctx.tensor_keys.append(None)
        if compress_kwargs is not None:
            saved_tensors.append(CompressedTensor(hidden_states, **compress_kwargs))
        else:
            saved_tensors.append(hidden_states)

        for key, val in enumerate(args):
            if not torch.is_tensor(val):
                ctx.input_args.append(val)
            else:
                ctx.input_args.append(None)
                ctx.tensor_keys.append(key)  # int
                saved_tensors.append(val)

        for key, val in kwargs.items():
            if not torch.is_tensor(val):
                ctx.input_kwargs[key] = val
            else:
                ctx.input_kwargs[key] = None
                ctx.tensor_keys.append(key)  # str
                saved_tensors.append(val)

        ctx.save_for_backward(*saved_tensors)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tuple[Tensor, ...]) -> Tuple[Any, ...]:
        # Copy the list to avoid modifying original list.
        input_args = ctx.input_args
        input_kwargs = ctx.input_kwargs

        # Fill in inputs with appropriate saved tensors.
        for key, tensor in zip(ctx.tensor_keys, ctx.saved_tensors):
            if isinstance(key, int):
                input_args[key] = tensor
            elif isinstance(key, str):
                input_kwargs[key] = tensor
            else:
                hidden_states = tensor

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device_type
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states, device_type=ctx.device_type)
            detached_hidden_states, detached_args, detached_kwargs = detach_variable(hidden_states, input_args, input_kwargs)

            device_autocast_ctx = torch.amp.autocast(
                device_type=ctx.device_type, **ctx.device_autocast_kwargs
            ) if torch.amp.is_autocast_available(ctx.device_type) else contextlib.nullcontext()
            with torch.enable_grad(), device_autocast_ctx, torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
                outputs = ctx.run_function(ctx.self, detached_hidden_states, *detached_args, **detached_kwargs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        grad_outputs_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                grad_outputs_with_grad.append(grad_outputs[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "None of output has requires_grad=True, this checkpoint() is not necessary."
            )
        torch.autograd.backward(outputs_with_grad, grad_outputs_with_grad)

        grad_hidden_states = detached_hidden_states.grad
        grads_args = tuple(
            arg.grad if isinstance(arg, torch.Tensor) else None
            for arg in detached_args
        )
        grads_kwargs_keys = tuple(
            None for _ in detached_kwargs.keys()
        )
        grads_kwargs_vals = tuple(
            val.grad if isinstance(val, torch.Tensor) else None
            for val in detached_kwargs.values()
        )
        return (None, None, grad_hidden_states, None, None, None, None, None) + grads_args + grads_kwargs_keys + grads_kwargs_vals
