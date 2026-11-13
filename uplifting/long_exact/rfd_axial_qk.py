"""Opt-in FP32 axial Q/K projection and packing trial for RFantibody."""

from functools import wraps
import math
import struct

import torch
from torch.nn import functional as F
import triton
import triton.language as tl


RFD_AV = "bijh,bkjhd->bikhd"
RF2_AV = "bijh,bnjhd->bnihd"
QK_EQUATION = "bnihk,bnjhk->bijh"


@triton.jit
def _pack_query(source, dest, scale, length: tl.constexpr, heads: tl.constexpr,
                hidden: tl.constexpr, rows: tl.constexpr, cols: tl.constexpr):
    n = tl.program_id(0) * rows + tl.arange(0, rows)
    col = tl.program_id(1) * cols + tl.arange(0, cols)
    hd = heads * hidden
    i = col // hd
    component = col % hd
    valid = (i[:, None] < length) & (n[None, :] < length)
    src = (n[None, :] * length + i[:, None]) * (2 * hd) + component[:, None]
    q = tl.load(source + src, valid, other=0) * scale
    head, d = component // hidden, component % hidden
    row = (head * length + i) * hidden + d
    dst = row[:, None] * length + n[None, :]
    tl.store(dest + dst, q, valid)


@triton.jit
def _pack_key(source, dest, scale, length: tl.constexpr, hd: tl.constexpr,
              rows: tl.constexpr, cols: tl.constexpr):
    nj = tl.program_id(0) * rows + tl.arange(0, rows)
    col = tl.program_id(1) * cols + tl.arange(0, cols)
    square = length * length
    valid = (col[:, None] < hd) & (nj[None, :] < square)
    src = nj[None, :] * (2 * hd) + hd + col[:, None]
    k = tl.load(source + src, valid, other=0) * scale
    dst = col[:, None] * square + nj[None, :]
    tl.store(dest + dst, k, valid)


def projection_weight(module):
    """Cache concatenated read-only projection weights by parameter version."""
    query, key = module.to_q.weight, module.to_k.weight
    if query.is_inference() or key.is_inference():
        return torch.cat((query, key), dim=0)
    versions = tuple((weight._version, weight.data_ptr(), weight.device, weight.dtype)
                     for weight in (query, key))
    cached = getattr(module, "_uplifting_qk_projection", None)
    if cached is None or cached[0] is not query or cached[1] is not key or cached[2] != versions:
        cached = (query, key, versions, torch.cat((query, key), dim=0))
        module._uplifting_qk_projection = cached
    return cached[3]


def pack_combined(qk, heads, hidden, scale, divisor):
    """Scale/pack the [1,L,L,2*H*D] linear output without copying Q/K slices."""
    length = qk.shape[1]
    hd = heads * hidden
    if (qk.shape != (1, length, length, 2 * hd) or not qk.is_contiguous()
            or not qk.is_cuda or qk.dtype != torch.float32 or qk.numel() >= 2**31):
        raise ValueError("packed Q/K requires a contiguous singleton FP32 CUDA projection")
    divisor = struct.unpack("f", struct.pack("f", divisor))[0]
    query = qk.new_empty((heads, length, hidden * length))
    key = qk.new_empty((heads, hidden * length, length))
    with torch.cuda.device(qk.device):
        _pack_query[(triton.cdiv(length, 32), triton.cdiv(length * hd, 64))](
            qk, query, scale, length, heads, hidden, 32, 64,
        )
        _pack_key[(triton.cdiv(length * length, 64), triton.cdiv(hd, 32))](
            qk, key, 1.0 / divisor, length, hd, 64, 32,
        )
    return query, key


def key_divisor(mode, length):
    return math.sqrt(length) if mode == "rfd" else length


def attention(module, pair, bias, mode):
    length = pair.shape[1]
    heads, hidden = module.h, module.dim
    shape = (1, length, length, heads, hidden)
    if module.is_row:
        pair = pair.transpose(1, 2)
        bias = bias.transpose(1, 2)
    pair = module.norm_pair(pair)
    bias = module.to_b(module.norm_bias(bias))
    qk = F.linear(pair, projection_weight(module))
    query, key = pack_combined(qk, heads, hidden, module.scaling, key_divisor(mode, length))
    logits = torch.bmm(query, key).permute(1, 2, 0).unsqueeze(0)
    del qk, query, key
    weights = F.softmax(logits + bias, dim=-2)
    del logits, bias

    value = module.to_v(pair).view(shape)
    weighted = torch.einsum(RFD_AV if mode == "rfd" else RF2_AV, weights, value)
    del weights, value
    gate = module.to_g(pair)
    if mode == "rfd":
        gate.sigmoid_()
        torch.mul(gate.view(shape), weighted, out=gate.view(shape))
    else:
        gate = torch.sigmoid(gate) * weighted.reshape(1, length, length, -1)
    result = module.to_out(gate)
    return result.transpose(1, 2) if module.is_row else result


def eligible(module, pair, bias, stride, mode):
    if pair.ndim != 4 or pair.shape[0] != 1 or pair.shape[1] != pair.shape[2]:
        return False
    length = pair.shape[1]
    return (
        length > 1 and module.h == 4 and module.dim == 32
        and pair.shape[-1] == 128 and pair.is_cuda and pair.dtype == torch.float32
        and bias.device == pair.device and bias.dtype == torch.float32
        and not torch.is_grad_enabled() and not torch.is_autocast_enabled()
        and not torch.backends.cuda.matmul.allow_tf32
        and 2 * length**2 * module.h * module.dim < 2**31
        and all(type(layer) is torch.nn.Linear and layer.bias is None
                and layer.weight.device == pair.device and layer.weight.dtype == pair.dtype
                and layer.out_features == module.h * module.dim
                for layer in (module.to_q, module.to_k))
        and (mode == "rf2" or not module.training)
        and (module.training or stride <= 0 or stride >= length)
    )


def activate(mode, calls):
    if mode == "rfd":
        from rfantibody.rfdiffusion.Attention_module import BiasedAxialAttention
    else:
        from rfantibody.rf2.network.Attention_module import BiasedAxialAttention

    original = BiasedAxialAttention.forward

    @wraps(original)
    def forward(module, pair, bias, *strides, **kwargs):
        stride = strides[0] if strides else kwargs.get("stride", 0)
        if not eligible(module, pair, bias, stride, mode):
            calls["fallback"] += 1
            return original(module, pair, bias, *strides, **kwargs)
        calls["fused_qk"] += 1
        return attention(module, pair, bias, mode)

    BiasedAxialAttention.forward = forward


def install(torch, mode):
    if mode not in ("rfd", "rf2"):
        raise ValueError(f"unknown RFantibody axial Q/K mode {mode}")
    calls = {"fused_qk": 0, "fallback": 0}
    if mode == "rf2":
        activate(mode, calls)
        return calls

    import rfd_axial

    install_p4 = rfd_axial.install

    @wraps(install_p4)
    def install_then_activate(mode="reuse"):
        rfd_axial.install = install_p4
        if mode != "triton":
            raise ValueError("the RFdiffusion Q/K trial requires the P4 triton mode")
        counts = install_p4(mode)
        activate("rfd", calls)
        return counts

    rfd_axial.install = install_then_activate
    return calls
