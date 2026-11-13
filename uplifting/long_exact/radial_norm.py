"""Experimental fused FP32 width-32 LayerNorm + ReLU for the SE(3) radial MLP."""

import ctypes
from functools import wraps
import math
import os
from pathlib import Path


WIDTH = 32
DEFAULT_LIBRARY = Path(__file__).resolve().parents[1].joinpath("native-build", "radial_norm_kernel.so")


class RadialNorm:
    def __init__(self, torch, library):
        self.torch = torch
        self.library = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.launch = self.library.radial_norm_launch
        self.launch.argtypes = ([ctypes.c_void_p] * 6
                                + [ctypes.c_uint32, ctypes.c_float, ctypes.c_void_p, ctypes.c_int])
        self.launch.restype = ctypes.c_int

    def eligible(self, x, norm):
        torch = self.torch
        weight, bias = norm.weight, norm.bias
        return (
            x.is_cuda and x.ndim == 2 and 0 < x.shape[0] <= (2**32 - 1) // WIDTH
            and x.shape[1] == WIDTH and x.dtype == torch.float32 and x.is_contiguous()
            and norm.normalized_shape == (WIDTH,) and math.isfinite(norm.eps) and norm.eps > 0
            and all(t is not None and t.device == x.device and t.dtype == x.dtype
                    and t.shape == (WIDTH,) and t.is_contiguous() and t.data_ptr() % 16 == 0
                    for t in (weight, bias))
            and x.data_ptr() % 16 == 0
        )

    def __call__(self, x, norm, *, activate=True, moments=False):
        if not self.eligible(x, norm):
            raise ValueError("The radial normalization kernel requires aligned width-32 CUDA FP32 tensors")
        torch = self.torch
        y = torch.empty_like(x)
        mean = x.new_empty((x.shape[0], 1)) if moments else None
        rstd = x.new_empty((x.shape[0], 1)) if moments else None
        with torch.cuda.device(x.device):
            status = self.launch(
                x.data_ptr(), norm.weight.data_ptr(), norm.bias.data_ptr(), y.data_ptr(),
                None if mean is None else mean.data_ptr(), None if rstd is None else rstd.data_ptr(),
                x.shape[0], norm.eps, torch.cuda.current_stream(x.device).cuda_stream, int(activate),
            )
        if status:
            raise RuntimeError(f"CUDA radial normalization launch failed ({status})")
        return (y, mean, rstd) if moments else y


def _assert_bytes(torch, label, stock, trial):
    if stock.shape != trial.shape or stock.stride() != trial.stride() or stock.dtype != trial.dtype:
        raise RuntimeError(f"Radial normalization {label} layout differs")
    lhs = stock.reshape(-1).view(torch.int32)
    rhs = trial.reshape(-1).view(torch.int32)
    if not torch.equal(lhs, rhs):
        differences = torch.nonzero(lhs != rhs).flatten()
        index = differences[0].item()
        raise RuntimeError(
            f"Radial normalization {label} differs at {differences.numel()} elements; "
            f"first flat index {index}: stock bits 0x{lhs[index].item() & 0xffffffff:08x}, "
            f"trial bits 0x{rhs[index].item() & 0xffffffff:08x}"
        )


def install(torch, mode, *, return_prefix=False):
    """P6 run_pair-compatible hook. Audit arithmetic before enabling fused timing."""
    if mode not in ("fused", "audit"):
        raise ValueError("Radial normalization mode must be fused or audit")
    if torch.__version__ != "2.3.1+cu118":
        raise RuntimeError("Radial normalization is pinned to Torch 2.3.1+cu118")

    from se3_transformer.model.layers.convolution import RadialProfile

    runner = RadialNorm(torch, os.environ.get("UPLIFTING_RADIAL_NORM_LIBRARY", DEFAULT_LIBRARY))
    original = RadialProfile.forward
    counts = {"radial_profiles": 0, "fused_norms": 0, "fused_rows": 0, "stock_profiles": 0}
    nn = torch.nn
    signature = (nn.Linear, nn.LayerNorm, nn.ReLU, nn.Linear, nn.LayerNorm, nn.ReLU, nn.Linear)

    def prefix(module, features):
        layers = tuple(module.net) if type(module.net) is nn.Sequential else ()
        if (torch.is_grad_enabled() or torch.is_autocast_enabled() or not features.is_cuda
                or features.dtype != torch.float32 or features.ndim != 2 or features.shape[0] < 1024
                or tuple(type(layer) for layer in layers) != signature
                or nn.modules.module._global_forward_hooks or nn.modules.module._global_forward_pre_hooks
                or any(layer._forward_hooks or layer._forward_pre_hooks
                       for layer in (module.net, *(layers[index] for index in (1, 2, 4, 5))))):
            return None

        if torch.cuda.get_device_capability(features.device) != (8, 9):
            return None

        def apply_norm(x, norm, relu):
            if not runner.eligible(x, norm):
                return relu(norm(x))
            trial = runner(x, norm)
            if mode == "audit":
                _assert_bytes(torch, "radial activation", relu(norm(x)), trial)
            counts["fused_norms"] += 1
            counts["fused_rows"] += x.shape[0]
            return trial

        counts["radial_profiles"] += 1
        x = apply_norm(layers[0](features), layers[1], layers[2])
        return apply_norm(layers[3](x), layers[4], layers[5])

    @wraps(original)
    def forward(module, features):
        hidden = prefix(module, features)
        if hidden is None:
            counts["stock_profiles"] += 1
            return original(module, features)
        return module.net[6](hidden)

    RadialProfile.forward = forward
    return (counts, prefix) if return_prefix else counts
