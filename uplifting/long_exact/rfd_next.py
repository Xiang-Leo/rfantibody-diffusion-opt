"""Install the measured RFdiffusion axial and dense SE(3) kernels."""

from . import axial_norm, layer_norm, rfd_axial_qk, rfd_radial_stream


def install(torch, mode):
    return {
        "se3": rfd_radial_stream.install(torch, mode),
        "contiguous_norm": layer_norm.install(torch, "fused"),
        "axial_norm": axial_norm.install(torch, "rfd"),
        "qk": rfd_axial_qk.install(torch, "rfd"),
    }
