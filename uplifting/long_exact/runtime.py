"""Assemble the validated RFdiffusion full-length FP32 inference stack."""

import os
from pathlib import Path


NATIVE_LIBRARIES = {
    "UPLIFTING_AXIAL_NORM_LIBRARY": "axial_norm_kernel.so",
    "UPLIFTING_LAYER_NORM_LIBRARY": "layer_norm_kernel.so",
    "UPLIFTING_RADIAL_NORM_LIBRARY": "radial_norm_kernel.so",
    "UPLIFTING_RFD_RESIDUAL_NORM_LIBRARY": "rfd_residual_norm_kernel.so",
}
LINEAR_LIBRARY = "linear_cublaslt.so"
SUPPORTED_GPUS = {"NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090 D"}


def native_directory(path):
    path = Path(path).expanduser().resolve()
    missing = [name for name in (*NATIVE_LIBRARIES.values(), LINEAR_LIBRARY) if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Missing native files in {path}: {', '.join(missing)}. Run uplifting/build_exact.py first.")
    return path


def install(torch, path, *, audit=False):
    if not torch.cuda.is_available():
        raise RuntimeError("RFdiffusion long-exact requires a CUDA GPU")
    gpu = torch.cuda.get_device_name()
    if torch.cuda.get_device_capability() != (8, 9) or gpu not in SUPPORTED_GPUS:
        return {"enabled": False, "fallback": f"unverified GPU {gpu}"}
    if torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("RFdiffusion long-exact requires FP32 matmul with TF32 disabled")
    native = native_directory(path)
    for key, name in NATIVE_LIBRARIES.items():
        os.environ[key] = str(native / name)

    from . import axial_norm, linear_dispatch, rfd_residual_norm, rfd_se3_attention

    return {
        "enabled": True,
        "se3": rfd_se3_attention.install(torch, "p8-both"),
        "residual_norm": rfd_residual_norm.install(torch, "p8", install_base=False),
        "bias_norm": axial_norm.install(torch, "rfd", field="norm_bias"),
        "dispatch": linear_dispatch.install(torch, native, audit=audit),
    }
