"""Fingerprint RFdiffusion sample states and RNG separately from wall timing."""

import argparse
from functools import wraps
from hashlib import sha256
import json
from pathlib import Path
import pickle
import random
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("probe_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import numpy as np
    import torch
    from rfantibody.rfdiffusion.inference.model_runners import AbSampler

    def fingerprint(value):
        if isinstance(value, torch.Tensor):
            data = value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy()
            return {"sha256": sha256(data).hexdigest(), "dtype": str(value.dtype),
                    "shape": list(value.shape)}
        if isinstance(value, (list, tuple)):
            return [fingerprint(item) for item in value]
        if isinstance(value, dict):
            return {str(key): fingerprint(item) for key, item in value.items()}
        if isinstance(value, (int, str, float, bool)) or value is None:
            return value
        raise TypeError(f"Unexpected sample value: {type(value)!r}")

    def rng():
        return {
            "torch_cpu": fingerprint(torch.get_rng_state()),
            "torch_cuda": fingerprint(torch.cuda.get_rng_state_all()),
            "python": sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
            "numpy": sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
        }

    designs = []
    original_init, original_step = AbSampler.sample_init, AbSampler.sample_step

    @wraps(original_init)
    def init(sampler, *a, **kw):
        initial = original_init(sampler, *a, **kw)
        designs.append({"initial": fingerprint(initial), "initial_rng": rng(), "steps": []})
        return initial

    @wraps(original_step)
    def step(sampler, *a, **kw):
        result = original_step(sampler, *a, **kw)
        designs[-1]["steps"].append({"t": int(kw["t"]), "sample": fingerprint(result), "rng": rng()})
        return result

    AbSampler.sample_init, AbSampler.sample_step = init, step
    probe_args = args.probe_args[1:] if args.probe_args[:1] == ["--"] else args.probe_args
    sys.argv = [str(args.probe), *probe_args]
    runpy.run_path(str(args.probe), run_name="__main__")
    if not designs or any([row["t"] for row in design["steps"]] != list(range(50, 0, -1))
                          for design in designs):
        raise RuntimeError("Trace did not cover every diffusion step for each design")
    with args.trace.open("x") as file:
        json.dump({"designs": designs}, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
