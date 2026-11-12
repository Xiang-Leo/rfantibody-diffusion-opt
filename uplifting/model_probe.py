"""Run a pinned inference entry with optional forward instrumentation."""

import argparse
from collections import defaultdict
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import time


MODULES = {
    "Attention", "AttentionWithBias", "BiasedAxialAttention", "ConvSE3",
    "FeedForwardLayer", "IterBlock", "IterativeSimulator", "MSA2Pair", "MSA_emb",
    "MSAPairStr2MSA", "PairStr2Pair", "RadialProfile", "Recycling", "SE3Transformer",
    "SE3TransformerWrapper", "Str2Str", "TemplatePairStack", "Templ_emb", "TriangleMultiplication",
}


class ModuleEvents:
    def __init__(self, torch, model):
        self.torch = torch
        self.active = {}
        self.events = defaultdict(list)
        self.modules = {}
        for name, module in model.named_modules():
            kind = type(module).__name__
            if kind in MODULES:
                self.modules[module] = (name, kind)
                module.register_forward_pre_hook(self.start)
                module.register_forward_hook(self.finish)

    def start(self, module, inputs):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        self.active[module] = event

    def finish(self, module, inputs, result):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        self.events[module].append((self.active.pop(module), event))

    def collect(self):
        grouped = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
        for module, events in self.events.items():
            name, kind = self.modules[module]
            grouped[kind]["calls"] += len(events)
            grouped[kind]["seconds"] += sum(start.elapsed_time(end) / 1000 for start, end in events)
        self.events.clear()
        return dict(grouped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("rf2", "rfd"))
    parser.add_argument("--entry", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--precision", default="stock",
                        choices=("stock", "tf32", "model-fp16", "model-bf16",
                                 "pair-tf32", "pair-bf16", "pair-fp16", "pair-einsum"))
    parser.add_argument("--modules", action="store_true")
    parser.add_argument("--capture-forward", type=int)
    parser.add_argument("--stop-after", type=int, help="diagnostic only, stop after this forward count")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    entry = args.entry.resolve(strict=True)
    if args.metadata.exists():
        parser.error("choose a fresh metadata path")
    overrides = args.overrides
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    if not overrides:
        parser.error("supply the original Hydra overrides")
    if args.stop_after is not None and (args.stop_after < 1 or not (args.modules or args.capture_forward)):
        parser.error("stopping early requires an instrumented diagnostic")

    sys.dont_write_bytecode = True
    import torch
    if args.model == "rf2":
        from rfantibody.rf2.network.RoseTTAFoldModel import RoseTTAFoldModule
    else:
        from rfantibody.rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
        basis = os.environ.get("NABLOOP_RFD_SE3_BASIS") == "1"
        edge_chunk = int(os.environ.get("NABLOOP_RFD_SE3_EDGE_CHUNK", "0"))
        if basis or edge_chunk:
            from se3_transformer.model import SE3Transformer
            initialize = SE3Transformer.__init__

            @wraps(initialize)
            def configure(transformer, *positional, **keyword):
                return initialize(transformer, *positional, **keyword,
                                  omit_unused_basis_layouts=basis,
                                  inference_edge_chunk_size=edge_chunk)

            SE3Transformer.__init__ = configure
    plugin_counts = {}
    use_tf32 = args.precision == "tf32" or os.environ.get("NABLOOP_MATMUL_TF32") == "1"
    if args.model == "rf2":
        axial = os.environ.get("NABLOOP_RF2_AXIAL", "off")
        triangle = os.environ.get("NABLOOP_RF2_TRIANGLE", "off")
        if axial != "off" or triangle != "off":
            from rf2_axial import install
            plugin_counts["rf2_layout"] = install(axial, triangle=triangle)
        amp = os.environ.get("NABLOOP_RF2_AMP")
        if amp:
            from rf2_amp import install
            plugin_counts["rf2_amp"] = install(torch, *amp.split(":"))
        gate_rows = os.environ.get("NABLOOP_RF2_RBF_GATE_ROWS")
        if gate_rows is not None:
            from rf2_rbf_gate import install
            plugin_counts["rf2_rbf_gate"] = install(int(gate_rows))
        if os.environ.get("NABLOOP_RF2_FULL_GRAPH") == "1":
            from rf2_full_graph import install
            plugin_counts["rf2_full_graph"] = install(torch)
        if os.environ.get("NABLOOP_RF2_BATCH") == "1":
            from rf2_batch_rng import install
            plugin_counts["rf2_batch"] = install(torch)
    else:
        axial = os.environ.get("NABLOOP_RFD_AXIAL")
        if axial:
            from rfd_axial import install
            plugin_counts["rfd_axial"] = install(axial)
        if os.environ.get("NABLOOP_RFD_MSA_PAIR") == "1":
            from rfd_msa_pair import install
            plugin_counts["rfd_msa_pair"] = install(torch)
    if args.precision.startswith("pair-"):
        if args.model != "rfd":
            parser.error("pair precision modes require RFdiffusion")
        from rfd_pair import install
        plugin_counts["rfd_pair"] = install(torch, args.precision)
    radial_amp = int(os.environ.get("NABLOOP_SE3_RADIAL_FP16_CHUNK", "0"))
    if radial_amp:
        from se3_radial_amp import install
        plugin_counts["se3_radial_amp"] = install(torch, radial_amp)
    fused_radial = os.environ.get("NABLOOP_SE3_FUSED_RADIAL")
    if fused_radial:
        if radial_amp:
            parser.error("select one experimental SE(3) radial implementation")
        from rfd_se3 import install
        plugin_counts["fused_radial"] = install(torch, fused_radial)
    if os.environ.get("NABLOOP_SE3_DENSE_ATTENTION") == "1":
        from rfd_se3_attention import install
        plugin_counts["dense_se3_attention"] = install(torch, model=args.model)
    fast_layer_norm = os.environ.get("NABLOOP_FAST_LAYER_NORM")
    if fast_layer_norm:
        from fast_layer_norm import install
        plugin_counts["fast_layer_norm"] = install(torch, fast_layer_norm)

    original = RoseTTAFoldModule.forward
    amp_dtype = {"model-fp16": torch.float16, "model-bf16": torch.bfloat16}.get(args.precision)
    counter = 0
    module_events = None
    first_forward = {}
    output = args.metadata.open("x")
    start_wall = time.perf_counter()

    def emit(record):
        print(json.dumps(record), file=output, flush=True)

    def restore_float32(value):
        if isinstance(value, torch.Tensor) and value.dtype in (torch.float16, torch.bfloat16):
            return value.float()
        if isinstance(value, (tuple, list)):
            return type(value)(restore_float32(item) for item in value)
        return value

    class DiagnosticComplete(BaseException):
        pass

    @wraps(original)
    def forward(model, *positional, **keyword):
        nonlocal counter, module_events
        counter += 1
        original_tf32 = torch.backends.cuda.matmul.allow_tf32
        if counter == 1:
            first_forward.update(kind="model", model=args.model, mode=args.precision,
                                 training=model.training, dtype=str(next(model.parameters()).dtype),
                                 stock_matmul_allow_tf32=original_tf32,
                                 forward_allow_tf32=use_tf32 or original_tf32,
                                 source=inspect.getfile(RoseTTAFoldModule),
                                 mode_flags={key: value for key, value in sorted(os.environ.items())
                                             if key.startswith(("NABLOOP_RF2_", "NABLOOP_RFD_", "NABLOOP_SE3_"))
                                             and key != "NABLOOP_RF2_INPUT"},
                                 cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                 gpu=torch.cuda.get_device_name())
            emit(first_forward)
        if args.modules and module_events is None:
            module_events = ModuleEvents(torch, model)
        capture = counter == args.capture_forward
        if args.modules or capture:
            torch.cuda.synchronize()
        capture_hooks = []
        if capture:
            for submodule in model.modules():
                kind = type(submodule).__name__
                if kind in MODULES:
                    def push(module, inputs, label=kind):
                        torch.cuda.nvtx.range_push(label)

                    def pop(module, inputs, result):
                        torch.cuda.nvtx.range_pop()

                    capture_hooks.extend((submodule.register_forward_pre_hook(push),
                                          submodule.register_forward_hook(pop)))
            torch.cuda.cudart().cudaProfilerStart()
        start = time.perf_counter()
        try:
            if use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
            if amp_dtype is None:
                result = original(model, *positional, **keyword)
            else:
                with torch.autocast("cuda", dtype=amp_dtype):
                    result = original(model, *positional, **keyword)
                result = restore_float32(result)
            if args.modules or capture:
                torch.cuda.synchronize()
        finally:
            if use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = original_tf32
            if capture:
                torch.cuda.cudart().cudaProfilerStop()
                for handle in capture_hooks:
                    handle.remove()
        if module_events:
            emit({"kind": "forward", "ordinal": counter, "wall_seconds": time.perf_counter() - start,
                  "classes": module_events.collect()})
        if counter == args.stop_after:
            raise DiagnosticComplete()
        return result

    RoseTTAFoldModule.forward = forward
    sys.argv = [str(entry), *overrides]
    success = False
    try:
        runpy.run_path(str(entry), run_name="__main__")
        success = True
    except DiagnosticComplete:
        pass
    finally:
        emit({"kind": "total", "success": success, "forward_count": counter,
              "diagnostic_partial": args.stop_after is not None,
              "wall_seconds": time.perf_counter() - start_wall,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "plugin_calls": plugin_counts})
        output.close()


if __name__ == "__main__":
    main()
