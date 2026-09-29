"""Cohere parallel residual/MLP/attention hooks with guaranteed cleanup.

Patching runs without a KV cache. At every autoregressive step the full prefix is
recomputed, and only the fixed original prompt position is replaced.
"""
from contextlib import contextmanager
from dataclasses import dataclass


def tensor_output(output):
    return output[0] if isinstance(output, tuple) else output


def replace_output(output, tensor):
    return (tensor,) + output[1:] if isinstance(output, tuple) else tensor


def layer_module(model, layer, component="residual"):
    if model.config.model_type != "cohere":
        raise ValueError("Intervention adapter is verified only for the Cohere (Aya-23) architecture")
    layers = model.model.layers
    if not 0 <= layer < len(layers):
        raise ValueError("Layer outside the model")
    if component == "residual":
        return layers[layer]
    if component == "mlp":
        return layers[layer].mlp
    if component == "attention":
        return layers[layer].self_attn
    raise ValueError("Unknown component")


@dataclass
class Patch:
    layer: int
    component: str
    position: int
    vector: object


@contextmanager
def patch_context(model, patches=()):
    handles = []
    try:
        for patch in patches:
            if patch.position < 0:
                raise ValueError("Patch positions must be fixed nonnegative prompt indices")

            def hook(_module, _inputs, output, patch=patch):
                value = tensor_output(output)
                if value.ndim != 3 or patch.position >= value.shape[1]:
                    raise ValueError("Patch position/shape mismatch")
                result = value.clone()
                result[:, patch.position, :] = patch.vector.to(device=value.device, dtype=value.dtype)
                return replace_output(output, result)

            handles.append(layer_module(model, patch.layer, patch.component).register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_context(model, requests):
    """requests are (layer, component, fixed_position); vectors are detached to CPU."""
    import torch
    handles, cache = [], {}
    try:
        for layer, component, position in requests:
            key = (layer, component, position)

            def hook(_module, _inputs, output, key=key):
                value = tensor_output(output)
                vector = value[:, key[2], :].detach().clone().cpu()
                if not torch.isfinite(vector).all():
                    raise ValueError("Nonfinite captured activation")
                cache[key] = vector

            handles.append(layer_module(model, layer, component).register_forward_hook(hook))
        yield cache
    finally:
        for handle in handles:
            handle.remove()


def logit_lens(model, vector):
    """Whole residual readout only; projecting an isolated MLP output is a different diagnostic."""
    value = vector.to(device=model.model.norm.weight.device, dtype=model.model.norm.weight.dtype)
    return (model.lm_head(model.model.norm(value)) * model.logit_scale).float()
