"""Pool RFdiffusion SE(3) values without a weighted edge activation tensor."""

from functools import wraps
from importlib.metadata import version

import triton
import triton.language as tl


TORCH_VERSION = '2.3.1+cu118'
DGL_VERSION = '2.4.0+cu118'
TRITON_VERSION = '2.3.1'
HEADS = 4
CHANNELS = 8
COMPONENTS = 4
FEATURES = CHANNELS * COMPONENTS


@triton.jit
def _rn_add(left, right):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', '=f,f,f',
                                     (left, right), dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _rn_mul(left, right):
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', '=f,f,f',
                                     (left, right), dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _message(values, weights, edge_ids, index, active, feature, head,
             ROW_STRIDE: tl.constexpr, HAS_EDGE_IDS: tl.constexpr):
    if HAS_EDGE_IDS:
        edge = tl.load(edge_ids + index, active, other=0).to(tl.int32)
    else:
        edge = index
    value = tl.load(values + edge * ROW_STRIDE + feature, active, other=0.0)
    weight = tl.load(weights + edge * 4 + head, active, other=0.0)
    return _rn_mul(weight, value)


@triton.jit
def _weighted_csc(values, weights, indptr, edge_ids, output,
                  ROW_STRIDE: tl.constexpr, HAS_EDGE_IDS: tl.constexpr):
    destination = tl.program_id(0)
    feature = tl.arange(0, 32)
    head = feature // 8
    start = tl.load(indptr + destination).to(tl.int32)
    stop = tl.load(indptr + destination + 1).to(tl.int32)
    zero = tl.full((32,), 0.0, tl.float32)
    acc = zero
    # DGL SpMMCsrKernel accumulates each channel in increasing CSC position.
    # Load four messages ahead while preserving the order of the FP32 adds.
    for index in range(start, stop, 4):
        valid1 = index + 1 < stop
        valid2 = index + 2 < stop
        valid3 = index + 3 < stop
        p0 = _message(values, weights, edge_ids, index, True, feature, head, ROW_STRIDE, HAS_EDGE_IDS)
        p1 = _message(values, weights, edge_ids, index + 1, valid1, feature, head, ROW_STRIDE, HAS_EDGE_IDS)
        p2 = _message(values, weights, edge_ids, index + 2, valid2, feature, head, ROW_STRIDE, HAS_EDGE_IDS)
        p3 = _message(values, weights, edge_ids, index + 3, valid3, feature, head, ROW_STRIDE, HAS_EDGE_IDS)
        acc = _rn_add(acc, p0)
        acc = tl.where(valid1, _rn_add(acc, p1), acc)
        acc = tl.where(valid2, _rn_add(acc, p2), acc)
        acc = tl.where(valid3, _rn_add(acc, p3), acc)
    # The stock kernel adds its local sum to a positive-zero output tensor.
    tl.store(output + destination * 32 + feature, _rn_add(zero, acc))


@triton.jit
def _strided_edge_dot(keys, queries, destinations, output, items,
                      ROW_STRIDE: tl.constexpr, FMA: tl.constexpr, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = index < items
    edge = index // 4
    head = index % 4
    node = tl.load(destinations + edge, valid, other=0).to(tl.int32)
    key_row = keys + edge * ROW_STRIDE + head * 8
    query_row = queries + node * 32 + head * 8
    acc = tl.full((TILE,), 0.0, tl.float32)
    for component in tl.static_range(8):
        left = tl.load(key_row + component, valid, other=0.0)
        right = tl.load(query_row + component, valid, other=0.0)
        if FMA:
            acc = tl.inline_asm_elementwise('fma.rn.f32 $0, $1, $2, $3;', '=f,f,f,f',
                                            (left, right, acc), dtype=tl.float32, is_pure=True, pack=1)
        else:
            acc = _rn_add(acc, _rn_mul(left, right))
    tl.store(output + index, acc, valid)


def check_versions(torch, dgl):
    if ((torch.__version__, version('dgl'), triton.__version__) != (TORCH_VERSION, DGL_VERSION, TRITON_VERSION)
            or dgl.__version__ not in (DGL_VERSION, DGL_VERSION.split('+')[0])):
        raise RuntimeError(f'Expected Torch {TORCH_VERSION}, DGL {DGL_VERSION} and Triton {TRITON_VERSION}')


def graph_supports_csc(graph):
    if not graph.is_homogeneous or graph.is_block:
        return False
    formats = graph.formats()
    return 'csc' in formats['created'] or 'csc' in formats['not created']


def csc_inputs(torch, graph, value):
    indptr, _, edge_ids = graph.adj_tensors('csc')
    if (indptr.shape != (graph.num_nodes() + 1,) or edge_ids.numel() not in (0, value.shape[0])
            or indptr.dtype not in (torch.int32, torch.int64) or indptr.device != value.device
            or (edge_ids.numel() and (edge_ids.dtype != indptr.dtype or edge_ids.device != value.device))
            or not indptr.is_contiguous() or not edge_ids.is_contiguous()):
        raise ValueError('Unexpected DGL CSC tensor layout')
    return indptr, edge_ids


def eligible_value(torch, value):
    return (isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[1:] == (CHANNELS, COMPONENTS)
            and value.stride() in ((FEATURES, COMPONENTS, 1), (2 * FEATURES, COMPONENTS, 1))
            and 0 < value.shape[0] * value.stride(0) < 2**31 and value.is_cuda and value.dtype == torch.float32)


def weighted_pool(torch, value, weights, indptr, edge_ids):
    if (not eligible_value(torch, value) or weights.shape != (value.shape[0], HEADS)
            or weights.device != value.device or weights.dtype != value.dtype or not weights.is_contiguous()):
        raise ValueError('Expected FP32 `(E,8,4)` values and dense `(E,4)` weights')
    result = value.new_empty((indptr.numel() - 1, CHANNELS, COMPONENTS))
    with torch.cuda.device(value.device):
        _weighted_csc[(result.shape[0],)](
            value, weights, indptr, edge_ids if edge_ids.numel() else indptr, result,
            value.stride(0), bool(edge_ids.numel()), num_warps=1, enable_fp_fusion=False,
        )
    return result


def strided_edge_dot(torch, key, query, destination, *, fma):
    if (not eligible_value(torch, key) or query.shape[1:] != (HEADS, FEATURES // HEADS)
            or query.device != key.device or query.dtype != key.dtype or not query.is_contiguous()
            or destination.shape != (key.shape[0],) or destination.device != key.device
            or destination.dtype not in (torch.int32, torch.int64) or not destination.is_contiguous()):
        raise ValueError('Expected contiguous destination IDs and the FP32 four-head edge-dot layout')
    result = key.new_empty((key.shape[0], HEADS))
    tile = 128
    with torch.cuda.device(key.device):
        _strided_edge_dot[(triton.cdiv(result.numel(), tile),)](
            key, query, destination, result, result.numel(), key.stride(0), fma, tile,
            num_warps=4, enable_fp_fusion=False,
        )
    return result


def stock_pool(dgl, graph, value, weights):
    shaped = value.view(value.shape[0], HEADS, CHANNELS // HEADS, COMPONENTS)
    weighted = weights[..., None, None] * shaped
    return dgl.ops.copy_e_sum(graph, weighted).view(graph.num_nodes(), CHANNELS, COMPONENTS)


def check_bytes(torch, expected, observed, label):
    if expected.shape != observed.shape or expected.stride() != observed.stride() or expected.dtype != observed.dtype:
        raise RuntimeError(f'{label}: tensor layout differs')
    stock = expected.contiguous().view(torch.int32)
    trial = observed.contiguous().view(torch.int32)
    if not torch.equal(stock, trial):
        different = int((stock != trial).sum().item())
        raise RuntimeError(f'{label}: {different}/{expected.numel()} FP32 values differ at the bit level')


def install(torch, mode):
    """Attach independent pool/key trials or combine them with P8 dense-direct."""
    modes = ('pool', 'audit', 'key', 'key-audit', 'p8', 'p8-audit', 'p8-key', 'p8-key-audit',
             'p8-both', 'p8-both-audit')
    if mode not in modes:
        raise ValueError(f'Unknown RFdiffusion SE(3) attention mode: {mode}')
    import dgl
    import numpy as np
    from se3_transformer.model.layers.attention import AttentionSE3
    from se3_transformer.runtime.utils import unfuse_features

    check_versions(torch, dgl)
    counts = dict(calls=0, audited=0, fallback=0, edges=0, dense_calls=0,
                  removed_weighted_bytes=0, removed_key_bytes=0, key_calls=0,
                  key_audited=0, module_audited=0)
    if mode.startswith('p8'):
        from . import rfd_next

        counts['p8'] = rfd_next.install(torch, 'dense-direct')
    auditing = mode.endswith('audit')
    key_enabled = 'key' in mode or '-both' in mode
    pool_enabled = 'key' not in mode
    original = AttentionSE3.forward

    @wraps(original)
    def forward(module, value, key, query, graph):
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or torch.is_autocast_cpu_enabled() or not eligible_value(torch, value)
                or not eligible_value(torch, key) or key.shape != value.shape or key.device != value.device
                or module.num_heads != HEADS or module.key_fiber.degrees != [0, 1]
                or module.value_fiber.degrees != [0, 1] or module.key_fiber.channels != [CHANNELS, CHANNELS]
                or module.value_fiber.channels != [CHANNELS, CHANNELS] or graph.device != value.device
                or graph.num_edges() != value.shape[0] or graph.num_nodes() == 0
                or not isinstance(query, dict) or not graph_supports_csc(graph)
                or torch.cuda.get_device_capability(value.device) != (8, 9)):
            counts['fallback'] += 1
            return original(module, value, key, query, graph)
        nodes = graph.num_nodes()
        for degree, size in (('0', 1), ('1', 3)):
            tensor = query.get(degree)
            if (tensor is None or tensor.shape != (nodes, CHANNELS, size)
                    or tensor.device != value.device or tensor.dtype != value.dtype):
                counts['fallback'] += 1
                return original(module, value, key, query, graph)

        head_key = key.reshape(value.shape[0], HEADS, -1)
        head_query = torch.cat((query['0'], query['1']), dim=-1).reshape(nodes, HEADS, -1)
        direct_key = key_enabled and key.stride(0) == 2 * FEATURES and 'coo' in graph.formats()['created']
        if not direct_key and not pool_enabled:
            counts['fallback'] += 1
            return original(module, value, key, query, graph)
        with torch.cuda.amp.autocast(False):
            if direct_key:
                _, destination = graph.edges(order='eid')
                weights = strided_edge_dot(torch, key, head_query, destination, fma=True)
                counts['key_calls'] += 1
                counts['removed_key_bytes'] += key.numel() * key.element_size()
                if auditing:
                    reference = dgl.ops.e_dot_v(graph, head_key, head_query).squeeze(-1)
                    check_bytes(torch, reference, weights, f'actual RFdiffusion edge logits N={nodes}')
                    counts['key_audited'] += 1
            else:
                weights = dgl.ops.e_dot_v(graph, head_key, head_query).squeeze(-1)
            weights /= np.sqrt(module.key_fiber.num_features)
            weights = dgl.ops.edge_softmax(graph, weights)
        if pool_enabled:
            indptr, edge_ids = csc_inputs(torch, graph, value)
            pooled = weighted_pool(torch, value, weights, indptr, edge_ids)
            counts['removed_weighted_bytes'] += value.numel() * value.element_size()
            if auditing:
                reference = stock_pool(dgl, graph, value, weights)
                check_bytes(torch, reference, pooled, f'actual RFdiffusion N={nodes}, E={value.shape[0]}')
                counts['audited'] += 1
        else:
            pooled = stock_pool(dgl, graph, value, weights)
        outputs = unfuse_features(pooled, module.value_fiber.degrees)
        if auditing and key_enabled:
            reference_outputs = original(module, value, key, query, graph)
            for degree in outputs:
                check_bytes(torch, reference_outputs[degree], outputs[degree],
                            f'actual RFdiffusion attention output N={nodes}, degree={degree}')
            counts['module_audited'] += 1
        counts['calls'] += 1
        counts['dense_calls'] += int(value.shape[0] == nodes * (nodes - 1))
        counts['edges'] += value.shape[0]
        return outputs

    AttentionSE3.forward = forward
    return counts
