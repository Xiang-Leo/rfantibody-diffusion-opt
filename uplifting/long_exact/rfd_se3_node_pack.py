"""Opt-in node packing and self-interaction reuse for the RFdiffusion SE(3) head."""

from functools import wraps


def _pack_nodes(torch, node_feats, src):
    # The stock path gathers both degrees before concatenating. Packing on
    # the node axis produces the same edge rows with one materialization.
    return torch.cat((node_feats["0"], node_feats["1"]), dim=-1).index_select(0, src)


def _self_edges(kernel, node, dst):
    # Keep the matrix product for each destination and gather its result.
    # The caller still adds the self term to each edge before DGL reduction.
    projected = kernel @ node.contiguous()
    # matmul can leave the last two dimensions in transposed storage order.
    # Select from that order to keep the stock edge contribution's strides.
    if projected.stride(-2) == 1 and projected.stride(-1) == projected.shape[-2]:
        return projected.transpose(-2, -1).index_select(0, dst).transpose(-2, -1)
    return projected.index_select(0, dst)


def _eligible(torch, partial, module, node_feats, edge_feats, graph):
    if (torch.is_grad_enabled() or torch.is_autocast_enabled()
            or torch.backends.cuda.matmul.allow_tf32
            or module.used_fuse_level != partial or not hasattr(module, "conv_out")
            or not module.self_interaction or not module.pool or not module.sum_over_edge
            or module.fiber_in.degrees != [0, 1] or module.fiber_in.channels != [32, 32]
            or module.fiber_out.degrees != [0, 1] or module.fiber_out.channels not in ([8, 2], [64, 2])
            or set(node_feats) != {"0", "1"} or set(edge_feats) != {"0"}):
        return False

    node0, node1, edge = node_feats["0"], node_feats["1"], edge_feats["0"]
    nodes = graph.num_nodes()
    edges = graph.num_edges()
    edge_width = 33 if module.fiber_out[0] == 8 else 65
    return (
        node0.is_cuda and node0.device == node1.device == edge.device == graph.device
        and node0.dtype == node1.dtype == edge.dtype == torch.float32
        and node0.shape == (nodes, 32, 1) and node1.shape == (nodes, 32, 3)
        and edge.shape == (edges, edge_width, 1)
        and graph.is_homogeneous and graph.idtype == torch.int64 and edges >= 4096
    )


def _final_convolution(torch, copy_e_sum, module, node_feats, edge_feats, graph, basis,
                       *, pack, reuse_self, edge_add=None):
    # Adapted from ConvSE3, NVIDIA CORPORATION & AFFILIATES (2021), MIT.
    src, dst = graph.edges()
    if pack:
        features = _pack_nodes(torch, node_feats, src)
    else:
        features = torch.cat((node_feats["0"][src], node_feats["1"][src]), dim=-1)

    invariant = edge_feats["0"].squeeze(-1)
    output = {}
    for degree in ("0", "1"):
        output[degree] = module.conv_out[degree](features, invariant, basis[f"out{degree}_fused"])

    for degree in ("0", "1"):
        kernel = module.to_kernel_self[degree]
        node = node_feats[degree]
        if edge_add is not None and output[degree].is_contiguous():
            # Keep edge ID and operand order, then use the stock reduction.
            output[degree] = edge_add(graph, output[degree], kernel @ node.contiguous())
        else:
            contribution = _self_edges(kernel, node, dst) if reuse_self else kernel @ node[dst]
            output[degree] += contribution
        output[degree] = copy_e_sum(graph, output[degree])
    return output


def install(torch, mode):
    """Install an isolated CUDA experiment and return its invocation counters."""
    if mode not in ("pack", "self", "both", "self-add"):
        raise ValueError(f"Unknown RFdiffusion SE(3) node mode: {mode}")

    from dgl.ops import copy_e_sum
    from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel

    original = ConvSE3.forward
    pack = mode in ("pack", "both")
    reuse_self = mode in ("self", "both", "self-add")
    counts = {"pack": 0, "self": 0, "fallback": 0}
    edge_add = None
    if mode == "self-add":
        from dgl.ops import e_add_v

        counts["self-add"] = 0

        def edge_add(graph, edge, projected):
            result = e_add_v(graph, edge, projected)
            counts["self-add"] += 1
            return result

    @wraps(original)
    def forward(module, node_feats, edge_feats, graph, basis):
        if not _eligible(torch, ConvSE3FuseLevel.PARTIAL, module, node_feats, edge_feats, graph):
            counts["fallback"] += 1
            return original(module, node_feats, edge_feats, graph, basis)

        counts["pack"] += int(pack)
        counts["self"] += int(reuse_self)
        return _final_convolution(torch, copy_e_sum, module, node_feats, edge_feats, graph, basis,
                                  pack=pack, reuse_self=reuse_self, edge_add=edge_add)

    ConvSE3.forward = forward
    return counts
