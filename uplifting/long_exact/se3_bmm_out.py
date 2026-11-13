"""Opt-in FP32 experiments in the chunked SE(3) convolution."""

from functools import wraps


def install(torch, mode, *, radial_kernel=None):
    from se3_transformer.model.layers.convolution import ConvSE3FuseLevel, VersatileConvSE3

    if mode not in ("bmm-out", "basis-k4"):
        raise ValueError(f"Unknown SE(3) contraction mode: {mode}")
    basis_kernel = None
    if mode == "basis-k4":
        from .se3_small_bmm import project

        basis_kernel = project
    original = VersatileConvSE3.forward
    stats = {"calls": 0, "chunks": 0, "direct_chunks": 0, "k4_chunks": 0, "radial_chunks": 0, "edges": 0}

    @wraps(original)
    def forward(module, features, invariant, basis):
        edges, _, in_dim = features.shape
        chunk_size = getattr(module, "inference_edge_chunk_size", 0)
        dense = edges <= chunk_size if chunk_size else module.training or edges <= 4096
        if (dense or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or torch.is_autocast_cpu_enabled() or torch.backends.cuda.matmul.allow_tf32
                or not features.is_cuda or features.dtype != torch.float32
                or invariant.device != features.device or invariant.dtype != features.dtype
                or (basis is not None and (basis.dtype != features.dtype or basis.device != features.device))):
            return original(module, features, invariant, basis)

        padded_dim = in_dim if basis is None else basis.shape[-1]
        out_dim = padded_dim
        if basis is not None and module.fuse_level != ConvSE3FuseLevel.FULL:
            out_dim += padded_dim % 2 - 1
        contiguous_basis = basis is not None and features.is_contiguous() and basis.is_contiguous()
        use_basis_kernel = (basis_kernel is not None and contiguous_basis
                            and module.fuse_level == ConvSE3FuseLevel.PARTIAL
                            and features.shape[1:] == (32, 4) and basis.shape[1:] in ((4, 2, 1), (4, 4, 3)))
        use_radial_kernel = (radial_kernel is not None and contiguous_basis
                             and module.fuse_level == ConvSE3FuseLevel.FULL and module.channels_out == 16
                             and padded_dim == 4 and module.channels_in * module.freq_sum in (12, 16))
        if basis_kernel is not None and not (use_basis_kernel or use_radial_kernel):
            return original(module, features, invariant, basis)
        output = features.new_empty((edges, module.channels_out, out_dim))
        stride = chunk_size or 65536
        stats["calls"] += 1
        stats["edges"] += edges
        for start in range(0, edges, stride):
            stop = min(start + stride, edges)
            edge_slice = slice(start, stop)
            count = stop - start
            radial = module.radial_func(invariant[edge_slice]).view(
                count, module.channels_out, module.channels_in * module.freq_sum,
            )
            vectors = features[edge_slice]
            if basis is not None:
                basis_view = basis[edge_slice].view(count, in_dim, -1)
                if use_basis_kernel and count == 65536:
                    vectors = basis_kernel(vectors, basis_view)
                    stats["k4_chunks"] += 1
                else:
                    vectors = torch.bmm(vectors, basis_view)
                vectors = vectors.view(count, -1, padded_dim)
            stats["chunks"] += 1
            if use_radial_kernel and count == 65536:
                output[edge_slice].copy_(radial_kernel(torch, radial, vectors))
                stats["radial_chunks"] += 1
            elif mode == "bmm-out" and padded_dim == out_dim:
                torch.bmm(radial, vectors, out=output[edge_slice])
                stats["direct_chunks"] += 1
            else:
                output[edge_slice].copy_(torch.bmm(radial, vectors)[:, :, :out_dim])
        return output

    VersatileConvSE3.forward = forward
    return stats
