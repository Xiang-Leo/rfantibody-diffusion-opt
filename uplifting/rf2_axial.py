"""Opt-in RF2 attention and triangular multiplication experiments.

The baseline is RFantibody 3e5a5bb1. Install once before loading the model.
These paths are restricted to the unstriped FP32 CUDA inference calls.
"""

from functools import wraps

import torch
from torch.nn import functional as F


TRITON_MODES = ("triton", "triton-fp16", "triton-fp16-value", "triton-fp16-both")
AXIAL_MODES = ("reuse", "packed", "direct", *TRITON_MODES,
               "row-reuse", "row-direct", *(f"row-{mode}" for mode in TRITON_MODES))


def _supported(module, pair, stride):
    length = pair.shape[1]
    return (
        pair.ndim == 4
        and pair.shape[2] == length
        and pair.is_cuda
        and pair.dtype == torch.float32
        and not torch.is_grad_enabled()
        and not torch.is_autocast_enabled()
        and (module.training or stride <= 0 or stride >= length)
    )


def _packed_weights(module, name, layers):
    # Store the cache outside the state_dict. Tensor versions change on a
    # normal parameter update; identities and devices also catch .to()/reload.
    parameters = tuple(tensor for layer in layers for tensor in (layer.weight, layer.bias)
                       if tensor is not None)
    try:
        signature = tuple((id(tensor), tensor._version, tensor.device, tensor.dtype)
                          for tensor in parameters)
    except RuntimeError:
        # Tensors created inside inference_mode do not have a version counter.
        return _concatenate_weights(layers)
    cache = getattr(module, name, None)
    if cache is None or cache[0] != signature:
        cache = (signature, _concatenate_weights(layers))
        setattr(module, name, cache)
    return cache[1]


def _concatenate_weights(layers):
    weight = torch.cat([layer.weight for layer in layers], dim=0)
    bias = None
    if layers[0].bias is not None:
        bias = torch.cat([layer.bias for layer in layers])
    return weight, bias


def _torch_pack_qk(query, key, scale, length, *, key_divisor=None, output_dtype=None):
    _, _, _, heads, hidden = query.shape
    if key_divisor is None:
        key_divisor = length
    dtype = output_dtype or query.dtype
    q = query.new_empty((1, heads, length, hidden, length), dtype=dtype)
    k = key.new_empty((1, heads, hidden, length, length), dtype=dtype)
    torch.mul(query.permute(0, 3, 2, 4, 1), scale, out=q)
    torch.div(key.permute(0, 3, 4, 1, 2), key_divisor, out=k)
    return q.view(heads, length, hidden * length), k.view(heads, hidden * length, length)


def _select_pack(mode):
    base_mode = mode.removeprefix("row-")
    if base_mode == "direct":
        return _torch_pack_qk
    if base_mode in TRITON_MODES:
        from rf2_axial_pack import pack_qk

        return pack_qk
    return None


def _half_bmm(left, right):
    matmul = torch.backends.cuda.matmul
    reduced = matmul.allow_fp16_reduced_precision_reduction
    if not reduced:
        return torch.bmm(left, right)
    matmul.allow_fp16_reduced_precision_reduction = False
    try:
        return torch.bmm(left, right)
    finally:
        matmul.allow_fp16_reduced_precision_reduction = reduced


def _half_weighted_value(attention, value, equation):
    batch, length, _, heads, hidden = value.shape
    attn_matrix = attention.new_empty((batch, heads, length, length), dtype=torch.float16)
    value_matrix = value.new_empty((batch, heads, length, length, hidden), dtype=torch.float16)
    # Cast and pack directly into the BMM operands.
    torch.mul(attention.permute(0, 3, 1, 2), 1.0, out=attn_matrix)
    torch.mul(value.permute(0, 3, 2, 1, 4), 1.0, out=value_matrix)
    output = _half_bmm(
        attn_matrix.view(batch * heads, length, length),
        value_matrix.view(batch * heads, length, length * hidden),
    ).view(batch, heads, length, length, hidden)
    if equation == "bijh,bnjhd->bnihd":
        return output.permute(0, 3, 2, 1, 4)
    if equation == "bijh,bkjhd->bikhd":
        return output.permute(0, 2, 3, 1, 4)
    raise ValueError(f"Unsupported weighted-value equation {equation}")


def _attention(module, pair, bias, mode, pack_qk=None, *, key_divisor=None,
               value_equation="bijh,bnjhd->bnihd"):
    batch, length = pair.shape[:2]
    if key_divisor is None:
        key_divisor = length
    base_mode = mode.removeprefix("row-")
    half_qk = base_mode in ("triton-fp16", "triton-fp16-both")
    half_value = base_mode in ("triton-fp16-value", "triton-fp16-both")
    heads, hidden = module.h, module.dim
    project_before_transpose = module.is_row and mode.startswith("row-")
    transpose_input = module.is_row and not project_before_transpose
    if transpose_input:
        pair = pair.transpose(1, 2)
        bias = bias.transpose(1, 2)
    pair = module.norm_pair(pair)
    bias = module.to_b(module.norm_bias(bias))
    if project_before_transpose:
        bias = bias.transpose(1, 2)
    shape = (batch, length, length, heads, hidden)

    if base_mode == "packed":
        weights, _ = _packed_weights(module, "_uplifting_qkv",
                                     (module.to_q, module.to_k, module.to_v))
        qkv = F.linear(pair, weights).view(batch, length, length, 3, heads, hidden)
        query, key, value = qkv.unbind(dim=3)
    else:
        query = module.to_q(pair).view(shape)
        key = module.to_k(pair).view(shape)
        if project_before_transpose:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
        value = None

    # Preserve the stock D-then-N reduction order. The Triton path only packs
    # projection layouts and scales values, then calls the same shape of BMM.
    if pack_qk is None or batch != 1:
        query.mul_(module.scaling)
        key.div_(key_divisor)
        attention = torch.einsum("bnihk,bnjhk->bijh", query, key)
    else:
        q_matrix, k_matrix = pack_qk(query, key, module.scaling, length,
                                     key_divisor=key_divisor,
                                     output_dtype=torch.float16 if half_qk else None)
        if half_qk:
            logits = _half_bmm(q_matrix, k_matrix).float()
        else:
            logits = torch.bmm(q_matrix, k_matrix)
        attention = logits.permute(1, 2, 0).unsqueeze(0)
        del q_matrix, k_matrix
    del query, key
    attention = F.softmax(attention + bias, dim=-2)
    if value is None:
        value = module.to_v(pair).view(shape)
        if project_before_transpose:
            value = value.transpose(1, 2)
    if half_value:
        weighted = _half_weighted_value(attention, value, value_equation)
    else:
        weighted = torch.einsum(value_equation, attention, value)
    del value, attention
    if project_before_transpose:
        weighted = weighted.transpose(1, 2)

    # Materialize the standard contiguous output directly, instead of
    # materializing the noncontiguous einsum view and then multiplying it.
    gate = module.to_g(pair).sigmoid_().view(shape)
    torch.mul(gate, weighted, out=gate)
    result = module.to_out(gate.view(batch, length, length, heads * hidden))
    return result.transpose(1, 2) if transpose_input else result


def _triangle(module, pair, mode):
    length = pair.shape[1]
    pair = module.norm(pair)
    if mode == "packed":
        weights, biases = _packed_weights(module, "_uplifting_triangle",
                                          (module.left_proj, module.left_gate,
                                           module.right_proj, module.right_gate))
        left, left_gate, right, right_gate = F.linear(pair, weights, biases).split(
            module.d_hidden, dim=-1
        )
        # Leave the activation outputs contiguous for the original contractions.
        left = left_gate.sigmoid().mul_(left)
        right = right_gate.sigmoid().mul_(right)
    else:
        left = module.left_gate(pair).sigmoid_().mul_(module.left_proj(pair))
        right = module.right_gate(pair).sigmoid_().mul_(module.right_proj(pair))
    right.div_(float(length))
    if module.outgoing:
        result = torch.einsum("bikd,bjkd->bijd", left, right)
    else:
        result = torch.einsum("bkid,bkjd->bijd", left, right)
    del left, right
    result = module.out_proj(module.norm_out(result))
    return module.gate(pair).sigmoid_().mul_(result)


def install(mode="reuse", *, triangle="off"):
    """Enable a candidate path, leaving other RF2 calls on the original code."""
    from rfantibody.rf2.network.Attention_module import BiasedAxialAttention, TriangleMultiplication

    if mode != "off" and mode not in AXIAL_MODES:
        raise ValueError(f"Unknown axial mode {mode}")
    if triangle not in ("off", "reuse", "packed"):
        raise ValueError(f"Unknown triangle mode {triangle}")
    applied = {"axial": 0, "triangle": 0}
    if mode != "off":
        pack_qk = _select_pack(mode)

        old_attention = BiasedAxialAttention.forward

        @wraps(old_attention)
        def attention(module, pair, bias, stride):
            if not _supported(module, pair, stride):
                return old_attention(module, pair, bias, stride)
            applied["axial"] += 1
            return _attention(module, pair, bias, mode, pack_qk)

        BiasedAxialAttention.forward = attention

    if triangle != "off":
        old_triangle = TriangleMultiplication.forward

        @wraps(old_triangle)
        def multiplication(module, pair, stride):
            if not _supported(module, pair, stride):
                return old_triangle(module, pair, stride)
            applied["triangle"] += 1
            return _triangle(module, pair, triangle)

        TriangleMultiplication.forward = multiplication
    return applied
