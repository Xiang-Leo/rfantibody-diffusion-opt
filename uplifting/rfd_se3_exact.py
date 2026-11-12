"""FP32 node reuse and four-term basis projection for antibody inference."""

from functools import wraps

import triton
import triton.language as tl


@triton.jit
def _project_four(left, right, dest, total, columns: tl.constexpr, tile: tl.constexpr):
    index = tl.program_id(0) * tile + tl.arange(0, tile)
    edge = index // (32 * columns)
    row = index // columns % 32
    column = index % columns
    valid = index < total
    value = tl.full((tile,), 0.0, tl.float32)
    for k in tl.static_range(4):
        a = tl.load(left + (edge * 32 + row) * 4 + k, valid, other=0.0)
        b = tl.load(right + (edge * 4 + k) * columns + column, valid, other=0.0)
        value = tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", (a, b, value),
            dtype=tl.float32, is_pure=True, pack=1,
        )
    tl.store(dest + index, value, valid)


def _basis_project(left, right):
    edges, rows, _ = left.shape
    columns = right.shape[-1]
    output = left.new_empty((edges, rows, columns))
    tile = 256
    _project_four[(triton.cdiv(output.numel(), tile),)](
        left, right, output, output.numel(), columns, tile,
        num_warps=4, enable_fp_fusion=False,
    )
    return output


def _self_contribution(kernel, node, dst):
    projected = kernel @ node.contiguous()
    if projected.stride(-2) == 1 and projected.stride(-1) == projected.shape[-2]:
        return projected.transpose(-2, -1).index_select(0, dst).transpose(-2, -1)
    return projected.index_select(0, dst)


def _fp32_inference(torch, module, tensor):
    return (tensor.is_cuda and tensor.dtype == torch.float32 and not module.training
            and not torch.is_grad_enabled() and not torch.is_autocast_enabled()
            and not torch.is_autocast_cpu_enabled() and not torch.backends.cuda.matmul.allow_tf32)


def install(torch):
    from dgl.ops import copy_e_sum
    from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel, VersatileConvSE3

    stock_head = ConvSE3.forward
    stock_projection = VersatileConvSE3.forward
    counts = {"node_convolutions": 0, "projection_chunks": 0}
    partial = ConvSE3FuseLevel.PARTIAL
    chunk_size = 65536

    @wraps(stock_head)
    def head(module, node_feats, edge_feats, graph, basis):
        if (module.used_fuse_level != partial or not hasattr(module, "conv_out")
                or not module.self_interaction or not module.pool or not module.sum_over_edge
                or module.fiber_in.degrees != [0, 1] or module.fiber_in.channels != [32, 32]
                or module.fiber_out.degrees != [0, 1] or module.fiber_out.channels not in ([8, 2], [64, 2])
                or set(node_feats) != {"0", "1"} or set(edge_feats) != {"0"}):
            return stock_head(module, node_feats, edge_feats, graph, basis)
        node0, node1, edge = node_feats["0"], node_feats["1"], edge_feats["0"]
        nodes, edges = graph.num_nodes(), graph.num_edges()
        edge_width = 33 if module.fiber_out[0] == 8 else 65
        if (not _fp32_inference(torch, module, node0)
                or node0.device != node1.device or node0.device != edge.device or node0.device != graph.device
                or node1.dtype != node0.dtype or edge.dtype != node0.dtype
                or node0.shape != (nodes, 32, 1) or node1.shape != (nodes, 32, 3)
                or edge.shape != (edges, edge_width, 1) or not graph.is_homogeneous
                or graph.idtype != torch.int64 or edges < 4096):
            return stock_head(module, node_feats, edge_feats, graph, basis)

        # Pack at the node level, then preserve the stock edge order and DGL sum.
        src, dst = graph.edges()
        features = torch.cat((node0, node1), dim=-1).index_select(0, src)
        invariant = edge.squeeze(-1)
        output = {degree: module.conv_out[degree](features, invariant, basis[f"out{degree}_fused"])
                  for degree in ("0", "1")}
        for degree, result in output.items():
            result += _self_contribution(module.to_kernel_self[degree], node_feats[degree], dst)
            output[degree] = copy_e_sum(graph, result)
        counts["node_convolutions"] += 1
        return output

    @wraps(stock_projection)
    def projection(module, features, invariant, basis):
        edges = features.shape[0]
        if (edges < chunk_size or getattr(module, "inference_edge_chunk_size", 0) != 0
                or not _fp32_inference(torch, module, features) or module.fuse_level != partial
                or features.shape[1:] != (32, 4) or not features.is_contiguous()
                or invariant.dtype != features.dtype or invariant.device != features.device
                or basis is None or basis.shape[1:] not in ((4, 2, 1), (4, 4, 3))
                or basis.dtype != features.dtype or basis.device != features.device or not basis.is_contiguous()):
            return stock_projection(module, features, invariant, basis)

        # Match the cuBLAS FP32 accumulation for the measured full chunks.
        # Keep stock BMM for tails and for the subsequent radial contraction.
        width = basis.shape[-1]
        output = features.new_empty((edges, module.channels_out, width))
        for start in range(0, edges, chunk_size):
            stop = min(start + chunk_size, edges)
            count = stop - start
            radial = module.radial_func(invariant[start:stop]).view(
                count, module.channels_out, module.channels_in * module.freq_sum,
            )
            node_rows = features[start:stop]
            basis_rows = basis[start:stop].view(count, 4, -1)
            if count == chunk_size:
                vectors = _basis_project(node_rows, basis_rows)
                counts["projection_chunks"] += 1
            else:
                vectors = torch.bmm(node_rows, basis_rows)
            output[start:stop].copy_(torch.bmm(radial, vectors.view(count, -1, width)))
        return output

    ConvSE3.forward = head
    VersatileConvSE3.forward = projection
    return counts
