"""Stream exact-first RFdiffusion radial tiles into the convolution output."""

from functools import wraps

import triton


from .rfd_radial_grouped import _grouped


CHUNK_EDGES = 65536
TILE_SIZES = (8192, 16384, 32768)
FEATURE_SHAPES = {(3, 3): (12, (3, 4, 4)), (8, 1): (16, (1, 2, 4))}


def _output(torch, vectors, out):
    expected = (CHUNK_EDGES, 16, 4)
    if out is None:
        return vectors.new_empty(expected)
    if (out.shape != expected or out.device != vectors.device or out.dtype != torch.float32
            or not out.is_contiguous()):
        raise ValueError("Expected a contiguous FP32 destination with the stock layout")
    return out


def _write(torch, projected, vectors, output, *, stock_last_edge):
    edges, _, inner = projected.shape
    count = (edges - int(stock_last_edge)) * 16 * 4
    if count:
        _grouped[(triton.cdiv(count, 256),)](
            projected, vectors, output, count, 16, inner, 4, False, 256,
            num_warps=4, enable_fp_fusion=False,
        )
    if stock_last_edge:
        # The final edge of the original 65,536-edge CUDA BMM is evaluated
        # by a separate cuBLAS dispatch, irrespective of the tile boundary.
        torch.bmm(projected[-1:], vectors[-1:], out=output[-1:])


def _validate(torch, vectors, input_tensor):
    inner = vectors.shape[1] if vectors.ndim == 3 else 0
    if (inner not in (12, 16) or vectors.shape != (CHUNK_EDGES, inner, 4)
            or not vectors.is_cuda or vectors.dtype != torch.float32
            or input_tensor.dtype != vectors.dtype or input_tensor.device != vectors.device
            or not input_tensor.is_contiguous() or not vectors.is_contiguous()
            or torch.is_grad_enabled() or torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()
            or torch.backends.cuda.matmul.allow_tf32):
        raise ValueError("Expected a full RFdiffusion FP32 CUDA chunk and precomputed angular projection")
    return inner


def direct(torch, projected, vectors, *, out=None):
    """Write a full projected chunk directly, bypassing the P7 result copy."""
    inner = _validate(torch, vectors, projected)
    if projected.shape != (CHUNK_EDGES, 16, inner):
        raise ValueError("Expected a full contiguous radial projection")
    output = _output(torch, vectors, out)
    with torch.cuda.device(vectors.device):
        _write(torch, projected, vectors, output, stock_last_edge=True)
    return output


def stream(torch, hidden, weight, vectors, tile_edges, *, out=None, inspect_projection=None):
    """Project tiles with stock Linear and contract directly into the final layout."""
    inner = _validate(torch, vectors, hidden)
    if (tile_edges not in TILE_SIZES or hidden.shape != (CHUNK_EDGES, 32)
            or weight.shape != (16 * inner, 32) or weight.device != hidden.device
            or weight.dtype != hidden.dtype or not weight.is_contiguous()):
        raise ValueError("Expected validated radial Linear dimensions and tile size")
    output = _output(torch, vectors, out)
    linear = torch.nn.functional.linear
    with torch.cuda.device(vectors.device):
        for start in range(0, CHUNK_EDGES, tile_edges):
            stop = start + tile_edges
            selection = slice(start, stop)
            projected = linear(hidden[selection], weight)
            if inspect_projection is not None:
                inspect_projection(selection, projected)
            _write(torch, projected.view(tile_edges, 16, inner), vectors[selection], output[selection],
                   stock_last_edge=stop == CHUNK_EDGES)
            del projected
    return output


def install(torch, mode):
    """Compose P7 baseline kernels and replace eligible K=12/16 full chunks."""
    if mode not in ("dense-direct", *(str(size) for size in TILE_SIZES)):
        raise ValueError(f"Unknown RFdiffusion radial stream mode: {mode}")
    from . import radial_norm, rfd_se3_node_pack, se3_bmm_out
    from .rfd_radial_grouped import contract_with_cublas_tail
    from se3_transformer.model.layers.convolution import ConvSE3FuseLevel, VersatileConvSE3

    baseline_node = rfd_se3_node_pack.install(torch, "both")
    baseline_basis = se3_bmm_out.install(torch, "basis-k4", radial_kernel=contract_with_cublas_tail)
    baseline_norm, prefix = radial_norm.install(torch, "fused", return_prefix=True)
    fallback = VersatileConvSE3.forward
    tile_edges = None if mode == "dense-direct" else int(mode)
    counts = {
        "direct_chunks": 0, "streamed_chunks": 0, "k12": 0, "k16": 0,
        "node": baseline_node, "basis": baseline_basis, "norm": baseline_norm,
    }

    @wraps(fallback)
    def forward(module, features, invariant, basis):
        edges, _, input_components = features.shape
        spec = FEATURE_SHAPES.get(tuple(features.shape[1:]))
        chunk_size = getattr(module, "inference_edge_chunk_size", 0)
        if (spec is None or basis is None or edges < CHUNK_EDGES or module.training
                or chunk_size not in (0, CHUNK_EDGES) or (chunk_size and edges <= chunk_size)
                or module.fuse_level != ConvSE3FuseLevel.FULL or module.channels_out != 16
                or not features.is_cuda or features.dtype != torch.float32 or not features.is_contiguous()
                or not basis.is_contiguous() or basis.dtype != features.dtype or basis.device != features.device
                or invariant.dtype != features.dtype or invariant.device != features.device
                or torch.is_grad_enabled() or torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()
                or torch.backends.cuda.matmul.allow_tf32):
            return fallback(module, features, invariant, basis)
        inner, basis_shape = spec
        if (basis.shape != (edges, *basis_shape) or module.channels_in * module.freq_sum != inner):
            return fallback(module, features, invariant, basis)

        profile = module.radial_func
        final = profile.net[-1]
        if (tile_edges is not None and
                (type(final) is not torch.nn.Linear or final.bias is not None
                 or final.weight.shape != (16 * inner, 32) or final.weight.dtype != features.dtype
                 or final.weight.device != features.device or not final.weight.is_contiguous()
                 or any(layer._forward_hooks or layer._forward_pre_hooks for layer in (profile, final)))):
            return fallback(module, features, invariant, basis)

        output = features.new_empty((edges, 16, 4))
        for start in range(0, edges, CHUNK_EDGES):
            stop = min(start + CHUNK_EDGES, edges)
            count = stop - start
            selection = slice(start, stop)
            chunk_input = invariant[selection]
            if count == CHUNK_EDGES and tile_edges is not None:
                hidden = prefix(profile, chunk_input)
                if hidden is None:
                    return fallback(module, features, invariant, basis)
            else:
                projected = profile(chunk_input).view(count, 16, inner)
            angular = torch.bmm(
                features[selection], basis[selection].view(count, input_components, -1),
            ).view(count, inner, 4)
            if count != CHUNK_EDGES:
                output[selection].copy_(torch.bmm(projected, angular))
                del projected
            elif tile_edges is None:
                direct(torch, projected, angular, out=output[selection])
                del projected
                counts["direct_chunks"] += 1
            else:
                stream(torch, hidden, final.weight, angular, tile_edges, out=output[selection])
                del hidden
                counts["streamed_chunks"] += 1
            del angular
            if count == CHUNK_EDGES:
                counts[f"k{inner}"] += 1
        return output

    VersatileConvSE3.forward = forward
    return counts
