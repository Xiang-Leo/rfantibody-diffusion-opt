"""FP32 width-128 LayerNorm that reads an axial row transpose directly."""

import ctypes
from functools import wraps
import math
import os
from pathlib import Path


WIDTH = 128
DEFAULT_LIBRARY = Path(__file__).resolve().parents[1].joinpath("native-build", "axial_norm_kernel.so")
TORCH_VERSION = "2.3.1+cu118"


class AxialNorm:
    def __init__(self, torch, library):
        if torch.__version__ != TORCH_VERSION:
            raise RuntimeError(f"axial normalization requires Torch {TORCH_VERSION}")
        self.torch = torch
        self.library = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.launch = self.library.axial_norm_launch
        self.launch.argtypes = [ctypes.c_void_p] * 6 + [
            ctypes.c_uint32, ctypes.c_float, ctypes.c_void_p,
        ]
        self.launch.restype = ctypes.c_int

    def eligible(self, x, norm):
        torch = self.torch
        if not x.is_cuda or x.dtype != torch.float32 or x.ndim != 4:
            return False
        batch, length, columns, width = x.shape
        return (
            batch == 1 and length == columns and 1 < length and width == WIDTH
            and length**2 * WIDTH < 2**32
            and x.stride()[1:] == (WIDTH, length * WIDTH, 1)
            and x.data_ptr() % 16 == 0
            and norm.normalized_shape == (WIDTH,) and math.isfinite(norm.eps) and norm.eps > 0
            and all(t is not None and t.device == x.device and t.dtype == x.dtype
                    and t.shape == (WIDTH,) and t.is_contiguous() and t.data_ptr() % 16 == 0
                    for t in (norm.weight, norm.bias))
            and torch.cuda.get_device_capability(x.device) == (8, 9)
        )

    def __call__(self, x, norm, *, moments=False):
        torch = self.torch
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        stat_shape = (*x.shape[:-1], 1)
        mean = x.new_empty(stat_shape) if moments else None
        rstd = x.new_empty(stat_shape) if moments else None
        with torch.cuda.device(x.device):
            status = self.launch(
                x.data_ptr(), norm.weight.data_ptr(), norm.bias.data_ptr(), out.data_ptr(),
                None if mean is None else mean.data_ptr(), None if rstd is None else rstd.data_ptr(),
                x.shape[1], norm.eps, torch.cuda.current_stream(x.device).cuda_stream,
            )
        if status:
            raise RuntimeError(f"CUDA axial normalization launch failed ({status})")
        return (out, mean, rstd) if moments else out


def assert_bytes(torch, label, stock, trial):
    if (stock.shape, stock.stride(), stock.dtype) != (trial.shape, trial.stride(), trial.dtype):
        raise RuntimeError(f"axial normalization {label} layout differs")
    a = stock.contiguous().view(torch.int32).reshape(-1)
    b = trial.contiguous().view(torch.int32).reshape(-1)
    if not torch.equal(a, b):
        different = torch.nonzero(a != b).flatten()
        first = different[0].item()
        raise RuntimeError(
            f"axial normalization {label} differs at {different.numel()} values, "
            f"first flat index {first}: stock bits 0x{a[first].item() & 0xffffffff:08x}, "
            f"trial bits 0x{b[first].item() & 0xffffffff:08x}"
        )


def audit_norm(torch, runner, x, norm):
    expected = torch.ops.aten.native_layer_norm(x, [WIDTH], norm.weight, norm.bias, norm.eps)
    trial = runner(x, norm, moments=True)
    for label, stock, actual in zip(("output", "mean", "rstd"), expected, trial):
        assert_bytes(torch, f"L={x.shape[1]} {label}", stock, actual)
    return trial[0]


def install(torch, mode, *, field="norm_pair"):
    """Attach to one row BiasedAxialAttention LayerNorm field."""
    if mode not in ("rfd", "rf2", "rfd-audit", "rf2-audit"):
        raise ValueError("axial normalization mode must be rfd, rf2, rfd-audit or rf2-audit")
    if field not in ("norm_pair", "norm_bias"):
        raise ValueError("axial normalization field must be norm_pair or norm_bias")
    if mode.startswith("rfd"):
        from rfantibody.rfdiffusion.Attention_module import BiasedAxialAttention
    else:
        from rfantibody.rf2.network.Attention_module import BiasedAxialAttention

    runner = AxialNorm(torch, os.environ.get("UPLIFTING_AXIAL_NORM_LIBRARY", DEFAULT_LIBRARY))
    audit = mode.endswith("-audit")
    counts = {"fused_norms": 0, "fused_rows": 0, "audited_norms": 0, "fallback": 0}
    original = BiasedAxialAttention.__init__

    @wraps(original)
    def init(module, *args, **kwargs):
        original(module, *args, **kwargs)
        norm = getattr(module, field)
        if not module.is_row or type(norm) is not torch.nn.LayerNorm or norm.normalized_shape != (WIDTH,):
            return
        stock = norm.forward
        audited_lengths = set()

        @wraps(stock)
        def forward(x):
            if (torch.is_grad_enabled() or torch.is_autocast_enabled()
                    or torch.is_autocast_cpu_enabled() or not runner.eligible(x, norm)):
                counts["fallback"] += 1
                return stock(x)
            length = x.shape[1]
            if audit and length not in audited_lengths:
                result = audit_norm(torch, runner, x, norm)
                audited_lengths.add(length)
                counts["audited_norms"] += 1
            else:
                result = runner(x, norm)
            counts["fused_norms"] += 1
            counts["fused_rows"] += length * length
            return result

        norm.forward = forward

    BiasedAxialAttention.__init__ = init
    return counts
