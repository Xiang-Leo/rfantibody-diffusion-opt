"""Test two separately rounded FP32 partial sums for the RFdiffusion radial BMM."""

import triton
import triton.language as tl


@triton.jit
def _fma(a, b, value):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", (a, b, value),
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _grouped(left, right, dest, total, rows: tl.constexpr, inner: tl.constexpr,
             columns: tl.constexpr, serial_tail: tl.constexpr, tile: tl.constexpr):
    index = tl.program_id(0) * tile + tl.arange(0, tile)
    edge = index // (rows * columns)
    row = index // columns % rows
    column = index % columns
    valid = index < total
    head = tl.full((tile,), 0.0, tl.float32)
    half: tl.constexpr = (inner + 1) // 2
    for k in tl.static_range(half):
        a = tl.load(left + (edge * rows + row) * inner + k, valid, other=0.0)
        b = tl.load(right + (edge * inner + k) * columns + column, valid, other=0.0)
        head = _fma(a, b, head)
    tail = tl.full((tile,), 0.0, tl.float32)
    if serial_tail:
        tail = tl.where(edge == 65535, head, tail)
    for k in tl.static_range(half, inner):
        a = tl.load(left + (edge * rows + row) * inner + k, valid, other=0.0)
        b = tl.load(right + (edge * inner + k) * columns + column, valid, other=0.0)
        tail = _fma(a, b, tail)
    value = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=f,f,f", (head, tail),
        dtype=tl.float32, is_pure=True, pack=1,
    )
    if serial_tail:
        value = tl.where(edge == 65535, tail, value)
    tl.store(dest + index, value, valid)


def contract(left, right, *, serial_tail=False):
    edges, rows, inner = left.shape
    columns = right.shape[-1]
    output = left.new_empty((edges, rows, columns))
    tile = 256
    _grouped[(triton.cdiv(output.numel(), tile),)](
        left, right, output, output.numel(), rows, inner, columns, serial_tail, tile,
        num_warps=4, enable_fp_fusion=False,
    )
    return output


def contract_with_cublas_tail(torch, left, right):
    output = contract(left, right)
    # The 65,536-edge cuBLAS dispatch uses a different kernel on its last
    # edge. Preserve its arithmetic with a single stock contraction.
    torch.bmm(left[-1:], right[-1:], out=output[-1:])
    return output
