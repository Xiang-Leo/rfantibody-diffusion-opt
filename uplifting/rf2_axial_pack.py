"""Fused scale and layout copy for the two tied-attention BMM operands."""

import struct

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_q(source, dest, scale, length: tl.constexpr, heads: tl.constexpr,
            hidden: tl.constexpr, rows: tl.constexpr, cols: tl.constexpr,
            transposed_source: tl.constexpr):
    n = tl.program_id(0) * rows + tl.arange(0, rows)
    col = tl.program_id(1) * cols + tl.arange(0, cols)
    hd = heads * hidden
    width = length * hd
    valid = (col[:, None] < width) & (n[None, :] < length)
    i = col // hd
    if transposed_source:
        src = (i[:, None] * length + n[None, :]) * hd + (col % hd)[:, None]
    else:
        src = n[None, :] * width + col[:, None]
    values = tl.load(source + src, valid, other=0) * scale
    head = (col // hidden) % heads
    d = col % hidden
    target_row = (head * length + i) * hidden + d
    target = target_row[:, None] * length + n[None, :]
    tl.store(dest + target, values, valid)


@triton.jit
def _pack_k(source, dest, scale, length: tl.constexpr, hd: tl.constexpr,
            rows: tl.constexpr, cols: tl.constexpr, transposed_source: tl.constexpr):
    nj = tl.program_id(0) * rows + tl.arange(0, rows)
    col = tl.program_id(1) * cols + tl.arange(0, cols)
    length_squared = length * length
    valid = (col[:, None] < hd) & (nj[None, :] < length_squared)
    if transposed_source:
        physical_nj = (nj % length) * length + nj // length
    else:
        physical_nj = nj
    src = physical_nj[None, :] * hd + col[:, None]
    values = tl.load(source + src, valid, other=0) * scale
    target = col[:, None] * length_squared + nj[None, :]
    tl.store(dest + target, values, valid)


def pack_qk(query, key, scale, length, *, key_divisor=None, output_dtype=None):
    """Return the dense matrices that ATen normally materializes in einsum."""
    _, _, _, heads, hidden = query.shape
    transposed_source = not query.is_contiguous()
    if transposed_source:
        if not (query.transpose(1, 2).is_contiguous() and key.transpose(1, 2).is_contiguous()):
            raise ValueError("The fused pack expects contiguous or row-transposed projections")
    elif not key.is_contiguous():
        raise ValueError("The fused pack requires matching projection layouts")
    if key_divisor is None:
        key_divisor = length
    # PyTorch CUDA scalar division rounds the divisor to the opmath dtype
    # before forming its reciprocal. RFdiffusion supplies sqrt(length).
    divisor = struct.unpack("f", struct.pack("f", key_divisor))[0]
    dtype = output_dtype or query.dtype
    q = torch.empty((heads, length, hidden * length), dtype=dtype, device=query.device)
    k = torch.empty((heads, hidden * length, length), dtype=dtype, device=key.device)
    width = length * heads * hidden
    square = length * length
    hd = heads * hidden
    with torch.cuda.device(query.device):
        _pack_q[(triton.cdiv(length, 32), triton.cdiv(width, 64))](
            query, q, scale, length, heads, hidden, 32, 64, transposed_source
        )
        _pack_k[(triton.cdiv(square, 64), triton.cdiv(hd, 32))](
            key, k, 1.0 / divisor, length, hd, 64, 32, transposed_source
        )
    return q, k
