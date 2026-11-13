"""Triton candidates for the angular projection's one-to-four term BMM."""

import triton
import triton.language as tl


@triton.jit
def _project(left, right, dest, total, rows: tl.constexpr, inner: tl.constexpr,
             columns: tl.constexpr, first_mul: tl.constexpr, tile: tl.constexpr):
    index = tl.program_id(0) * tile + tl.arange(0, tile)
    edge = index // (rows * columns)
    row = index // columns % rows
    col = index % columns
    valid = index < total
    value = tl.full((tile,), 0.0, tl.float32)
    for k in tl.static_range(inner):
        a = tl.load(left + (edge * rows + row) * inner + k, valid, other=0.0)
        b = tl.load(right + (edge * inner + k) * columns + col, valid, other=0.0)
        if first_mul and k == 0:
            value = a * b
        else:
            value = tl.inline_asm_elementwise(
                "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", (a, b, value),
                dtype=tl.float32, is_pure=True, pack=1,
            )
    tl.store(dest + index, value, valid)


def project(left, right, *, first_mul=False):
    edges, rows, inner = left.shape
    columns = right.shape[-1]
    output = left.new_empty((edges, rows, columns))
    total = output.numel()
    tile = 256
    _project[(triton.cdiv(total, tile),)](
        left, right, output, total, rows, inner, columns, first_mul, tile,
        num_warps=4, enable_fp_fusion=False,
    )
    return output
