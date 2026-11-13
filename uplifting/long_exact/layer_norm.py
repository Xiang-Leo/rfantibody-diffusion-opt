"""Contiguous FP32 LayerNorm trial for large inference tensors."""

import ctypes
from functools import wraps
import math
import os
from pathlib import Path


WIDTHS = (32, 64, 128)
MIN_ROWS = 1024
DEFAULT_LIBRARY = Path(__file__).resolve().parents[1].joinpath("native-build", "layer_norm_kernel.so")
TORCH_VERSION = "2.3.1+cu118"


class ContiguousLayerNorm:
    def __init__(self, torch, library):
        if torch.__version__ != TORCH_VERSION:
            raise RuntimeError(f"contiguous LayerNorm requires Torch {TORCH_VERSION}")
        self.torch = torch
        self.library = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.launch = self.library.layer_norm_launch
        self.launch.argtypes = [ctypes.c_void_p] * 6 + [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float, ctypes.c_void_p,
        ]
        self.launch.restype = ctypes.c_int

    def eligible(self, x, norm):
        torch = self.torch
        if type(x) is not torch.Tensor or not x.is_cuda or x.dtype != torch.float32 or x.ndim < 2:
            return False
        width = x.shape[-1]
        return (
            width in WIDTHS and 0 < x.numel() < 2**32
            and x.is_contiguous() and x.data_ptr() % 16 == 0
            and norm.normalized_shape == (width,) and math.isfinite(norm.eps) and norm.eps > 0
            and all(t is not None and t.device == x.device and t.dtype == x.dtype
                    and t.shape == (width,) and t.is_contiguous() and t.data_ptr() % 16 == 0
                    for t in (norm.weight, norm.bias))
            and torch.cuda.get_device_capability(x.device) == (8, 9)
        )

    def __call__(self, x, norm, *, moments=False):
        torch = self.torch
        width = x.shape[-1]
        rows = x.numel() // width
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        stat_shape = (*x.shape[:-1], 1)
        mean = x.new_empty(stat_shape) if moments else None
        rstd = x.new_empty(stat_shape) if moments else None
        with torch.cuda.device(x.device):
            status = self.launch(
                x.data_ptr(), norm.weight.data_ptr(), norm.bias.data_ptr(), out.data_ptr(),
                None if mean is None else mean.data_ptr(), None if rstd is None else rstd.data_ptr(),
                rows, width, norm.eps, torch.cuda.current_stream(x.device).cuda_stream,
            )
        if status:
            raise RuntimeError(f"CUDA contiguous LayerNorm launch failed ({status})")
        return (out, mean, rstd) if moments else out


def assert_bytes(torch, label, stock, trial):
    if (stock.shape, stock.stride(), stock.dtype) != (trial.shape, trial.stride(), trial.dtype):
        raise RuntimeError(f"contiguous LayerNorm {label} layout differs")
    a = stock.contiguous().view(torch.int32).reshape(-1)
    b = trial.contiguous().view(torch.int32).reshape(-1)
    if not torch.equal(a, b):
        different = torch.nonzero(a != b).flatten()
        first = different[0].item()
        raise RuntimeError(
            f"contiguous LayerNorm {label} differs at {different.numel()} values, "
            f"first flat index {first}: stock bits 0x{a[first].item() & 0xffffffff:08x}, "
            f"trial bits 0x{b[first].item() & 0xffffffff:08x}"
        )


def audit_norm(torch, runner, x, norm):
    width = x.shape[-1]
    expected = torch.ops.aten.native_layer_norm(x, [width], norm.weight, norm.bias, norm.eps)
    trial = runner(x, norm, moments=True)
    for label, stock, actual in zip(("output", "mean", "rstd"), expected, trial):
        assert_bytes(torch, f"shape={tuple(x.shape)} {label}", stock, actual)
    return trial[0]


def install(torch, mode):
    """Attach to stock LayerNorm calls while keeping every ineligible call on ATen."""
    if mode not in ("fused", "audit"):
        raise ValueError("contiguous LayerNorm mode must be fused or audit")
    cls = torch.nn.LayerNorm
    original = cls.forward
    if original.__module__ != cls.__module__ or hasattr(original, "__wrapped__"):
        raise RuntimeError("contiguous LayerNorm requires the stock LayerNorm.forward")
    runner = ContiguousLayerNorm(torch, os.environ.get("UPLIFTING_LAYER_NORM_LIBRARY", DEFAULT_LIBRARY))
    counts = {"fused_norms": 0, "fused_rows": 0, "audited_norms": 0, "fallback": 0,
              **{f"width{width}": 0 for width in WIDTHS}}

    @wraps(original)
    def forward(norm, input):
        x = input
        if (
            type(norm) is not cls or torch.is_grad_enabled() or torch.is_autocast_enabled()
            or not runner.eligible(x, norm)
            or x.numel() // x.shape[-1] < MIN_ROWS
        ):
            counts["fallback"] += 1
            return original(norm, x)
        width = x.shape[-1]
        if mode == "audit":
            key = (x.device, x.shape, norm.eps)
            if getattr(norm, "_uplifting_contiguous_norm_audit", None) != key:
                result = audit_norm(torch, runner, x, norm)
                norm._uplifting_contiguous_norm_audit = key
                counts["audited_norms"] += 1
            else:
                result = runner(x, norm)
        else:
            result = runner(x, norm)
        counts["fused_norms"] += 1
        counts["fused_rows"] += x.numel() // width
        counts[f"width{width}"] += 1
        return result

    cls.forward = forward
    return counts
