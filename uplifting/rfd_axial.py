"""Opt-in RFdiffusion axial layout experiments for the pinned antibody model."""

from functools import wraps
import math

from rf2_axial import AXIAL_MODES, _attention, _select_pack, _supported


def install(mode="reuse"):
    from rfantibody.rfdiffusion.Attention_module import BiasedAxialAttention

    if mode not in AXIAL_MODES:
        raise ValueError(f"Unknown diffusion axial mode {mode}")
    pack_qk = _select_pack(mode)

    original = BiasedAxialAttention.forward
    usage = {"rfd_axial": 0}

    @wraps(original)
    def forward(module, pair, bias):
        if not _supported(module, pair, 0):
            return original(module, pair, bias)
        usage["rfd_axial"] += 1
        return _attention(module, pair, bias, mode, pack_qk,
                          key_divisor=math.sqrt(pair.shape[1]),
                          value_equation="bijh,bkjhd->bikhd")

    BiasedAxialAttention.forward = forward
    return usage
