"""Dispatch only FP32 Linear shapes screened on the RFdiffusion model."""

import atexit
import ctypes as c
from functools import wraps
import json
import os
from pathlib import Path


MANIFEST = Path(__file__).with_name("linear_manifest.json")


def library(path):
    api = c.CDLL(str(path))
    pointer, integer = c.c_void_p, c.c_int
    api.linear_lt_create.argtypes = [c.c_int64, c.c_int64, c.c_int64, c.c_int32, pointer,
                                     c.c_int32, c.c_uint64, c.POINTER(integer)]
    api.linear_lt_create.restype = pointer
    api.linear_lt_set_bias.argtypes = [pointer, pointer]
    api.linear_lt_set_bias.restype = integer
    api.axial_lt_version.argtypes = []
    api.axial_lt_version.restype = c.c_size_t
    api.axial_lt_count.argtypes = [pointer]
    api.axial_lt_count.restype = integer
    api.axial_lt_info.argtypes = [pointer, integer, c.POINTER(c.c_uint32), c.POINTER(c.c_uint32),
                                  c.POINTER(c.c_uint32), c.POINTER(c.c_uint64)]
    api.axial_lt_info.restype = integer
    api.axial_lt_run.argtypes = [pointer, integer, pointer, pointer, pointer, pointer, c.c_uint64, pointer]
    api.axial_lt_run.restype = integer
    api.axial_lt_destroy.argtypes = [pointer]
    api.axial_lt_destroy.restype = None
    return api


def matches(api, plan, choice):
    index, *expected = choice
    if index >= api.axial_lt_count(plan):
        return False
    identifier, tile, stages, workspace = c.c_uint32(), c.c_uint32(), c.c_uint32(), c.c_uint64()
    status = api.axial_lt_info(plan, index, c.byref(identifier), c.byref(tile), c.byref(stages), c.byref(workspace))
    return status == 0 and expected == [identifier.value, tile.value, stages.value, workspace.value]


class Dispatcher:
    def __init__(self, torch, native):
        self.torch = torch
        self.stock = torch.nn.functional.linear
        self.api = library(native / "linear_cublaslt.so")
        manifest = json.loads(MANIFEST.read_text())
        self.enabled = (manifest["torch"], manifest["cublaslt"]) == (torch.__version__, self.api.axial_lt_version())
        self.workspace_bytes = manifest["workspace_bytes"]
        self.choices = {(*row["shape"], row["bias"], row["relu"]): (row["layout"], row["algorithm"])
                        for row in manifest["choices"]} if self.enabled else {}
        self.plans = {}
        self.packs = {}
        self.workspaces = {}
        atexit.register(self.close)

    def close(self):
        for plan in self.plans.values():
            if plan:
                self.api.axial_lt_destroy(plan)
        self.plans.clear()

    def packed(self, weight, stream):
        if weight.is_inference():
            return weight.t().contiguous()
        key = (id(weight), weight.device, stream)
        old = self.packs.get(key)
        if old is not None and old[0] is weight and old[1] == weight._version:
            return old[2]
        transposed = weight.t().contiguous()
        self.packs[key] = (weight, weight._version, transposed)
        return transposed

    def choose(self, source, weight, bias, relu):
        torch = self.torch
        if (not self.enabled or type(source) is not torch.Tensor
                or type(weight) not in (torch.Tensor, torch.nn.Parameter) or source.ndim < 2 or weight.ndim != 2
                or source.dtype != torch.float32 or not source.is_cuda or not source.is_contiguous()
                or source.shape[-1] != weight.shape[1] or weight.shape[1] == 0
                or torch.is_grad_enabled() or torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()
                or torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest"
                or os.environ.get("DISABLE_ADDMM_CUDA_LT") == "1"):
            return None
        n, k = weight.shape
        key = (source.numel() // k, k, n, bias is not None, relu)
        if (key not in self.choices or torch.cuda.get_device_capability(source.device) != (8, 9)
                or any(type(t) not in (torch.Tensor, torch.nn.Parameter) or t.device != source.device
                       or t.dtype != source.dtype or not t.is_contiguous() or t.is_neg() or t.data_ptr() % 16
                       for t in (source, weight, bias) if t is not None)
                or (bias is not None and bias.shape != (n,))):
            return None
        return key

    def run(self, source, weight, bias, key):
        rows, k, n, with_bias, relu = key
        torch = self.torch
        layout, choice = self.choices[key]
        with torch.cuda.device(source.device):
            stream = torch.cuda.current_stream(source.device).cuda_stream
            packed = self.packed(weight, stream)
            if layout == "torch-packed":
                result = self.stock(source, packed.t(), bias)
                return result.relu_() if relu else result
            stream_key = (source.device, stream)
            workspace = self.workspaces.get(stream_key)
            if workspace is None:
                workspace = torch.empty(self.workspace_bytes, dtype=torch.uint8, device=source.device)
                self.workspaces[stream_key] = workspace
            plan_key = (stream_key, key)
            if plan_key not in self.plans:
                status = c.c_int()
                plan = self.api.linear_lt_create(rows, k, n, 1, bias.data_ptr() if with_bias else None,
                                                 relu, workspace.nbytes, c.byref(status))
                if plan and not matches(self.api, plan, choice):
                    self.api.axial_lt_destroy(plan)
                    plan = None
                self.plans[plan_key] = plan
            else:
                plan = self.plans[plan_key]
                if plan and with_bias and self.api.linear_lt_set_bias(plan, bias.data_ptr()):
                    return None
            if not plan:
                return None
            output = source.new_empty((*source.shape[:-1], n))
            status = self.api.axial_lt_run(plan, choice[0], source.data_ptr(), packed.data_ptr(),
                                           output.data_ptr(), workspace.data_ptr(), workspace.nbytes, stream)
            if status:
                raise RuntimeError(f"FP32 cuBLASLt Linear launch {key} failed ({status})")
            return output


def install(torch, native, *, audit):
    from rfantibody.rfdiffusion.Attention_module import FeedForwardLayer

    dispatch = Dispatcher(torch, native)
    counts = {"enabled": dispatch.enabled, "linear": 0, "relu": 0, "audited": 0,
              "plan_fallback": 0, "shapes": {}}
    audited = set()

    def candidate(source, weight, bias, key, category):
        result = dispatch.run(source, weight, bias, key)
        if result is None:
            counts["plan_fallback"] += 1
            return None
        if audit and key not in audited:
            reference = dispatch.stock(source, weight, bias)
            if key[-1]:
                reference.relu_()
            if (result.stride() != reference.stride()
                    or not torch.equal(reference.view(torch.int32), result.view(torch.int32))):
                raise RuntimeError(f"RFdiffusion FP32 Linear {key} differs from the stock output")
            audited.add(key)
            counts["audited"] += 1
        counts[category] += 1
        label = str(key)
        counts["shapes"][label] = counts["shapes"].get(label, 0) + 1
        return result

    @wraps(dispatch.stock)
    def linear(input, weight, bias=None):
        key = dispatch.choose(input, weight, bias, False)
        result = candidate(input, weight, bias, key, "linear") if key is not None else None
        return dispatch.stock(input, weight, bias) if result is None else result

    original = FeedForwardLayer.forward

    @wraps(original)
    def forward(module, src):
        layer = module.linear1
        if (type(src) is not torch.Tensor or src.ndim != 4 or src.shape[0] != 1
                or src.shape[1] != src.shape[2] or src.shape[-1] != 128 or module.training
                or type(layer) is not torch.nn.Linear or layer._forward_hooks or layer._forward_pre_hooks
                or torch.nn.modules.module._global_forward_hooks or torch.nn.modules.module._global_forward_pre_hooks
                or (src.shape[1] ** 2, 128, 256, True, True) not in dispatch.choices
                or torch.is_grad_enabled() or torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()
                or torch.backends.cuda.matmul.allow_tf32 or not src.is_cuda or src.dtype != torch.float32
                or layer.weight.shape != (256, 128) or layer.bias is None):
            return original(module, src)
        normalized = module.norm(src)
        key = dispatch.choose(normalized, layer.weight, layer.bias, True)
        hidden = candidate(normalized, layer.weight, layer.bias, key, "relu") if key is not None else None
        if hidden is None:
            hidden = layer(normalized).relu_()
        return module.linear2(module.dropout(hidden))

    if dispatch.enabled:
        torch.nn.functional.linear = linear
        FeedForwardLayer.forward = forward
    return counts
