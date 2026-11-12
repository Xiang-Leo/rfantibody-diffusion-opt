#!/usr/bin/env python3
"""Generate antibody backbones with the RFdiffusion P4 exact configuration."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


PLUGINS = Path(__file__).resolve().parent
REPO = PLUGINS.parent
SCHEDULE = "T_50_omega_1000_min_sigma_0_02_min_b_1_5_max_b_2_5_schedule_linear.pkl"
VERSIONS = {"torch": "2.3.1+cu118", "dgl": "2.4.0+cu118", "triton": "2.3.1"}
FLAGS = {
    "NABLOOP_RFD_LEVERS": "indices,rbf,msa-index,full-graph",
    "NABLOOP_RFD_QUIET_STEPS": "1",
    "NABLOOP_RFD_SE3_BASIS": "1",
    "NABLOOP_RFD_AXIAL": "triton",
    "NABLOOP_RFD_ISOLATE_TRITON_RNG": "1",
    "NABLOOP_RFD_EXACT_PREPROCESS": "1",
}


def input_file(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise argparse.ArgumentTypeError(f"expected a nonempty file: {path}")
    return path


def hydra_path(key, path):
    return f"{key}={json.dumps(str(path), ensure_ascii=False)}"


def verify_run(metadata, result, entry, designs):
    first, *_, last = (json.loads(line) for line in metadata.read_text().splitlines())
    source = REPO / "src/rfantibody/rfdiffusion/RoseTTAFoldModel.py"
    if (
        first["kind"] != "model" or first["model"] != "rfd" or first["mode"] != "stock"
        or first["dtype"] != "torch.float32" or first["training"] is not False
        or first["forward_allow_tf32"] is not False
        or Path(first["source"]).resolve() != source.resolve()
    ):
        raise RuntimeError(f"{entry.name} did not use the expected source and precision")
    if (
        last["kind"] != "total" or last["success"] is not True
        or last["forward_count"] != designs * 50
        or last["plugin_calls"].get("rfd_axial", {}).get("rfd_axial", 0) == 0
    ):
        raise RuntimeError("the expected number of RFdiffusion forwards or axial calls was not recorded")
    with result.open("rb") as handle:
        tags = [line.split()[1] for line in handle if line.startswith(b"QV_TAG ")]
    if len(tags) != designs or len(set(tags)) != designs:
        raise RuntimeError(f"expected {designs} distinct Quiver records, found {len(tags)}")
    return last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=input_file, help="full antigen PDB")
    parser.add_argument("--framework", required=True, type=input_file, help="CDR-annotated antibody framework PDB")
    parser.add_argument("--loops", required=True, help="design loops and lengths, e.g. H3:7,L3:9")
    parser.add_argument("--hotspots", required=True, help="target PDB chain and residue IDs, e.g. T305,T456")
    parser.add_argument("--weights", required=True, type=input_file, help="external RFdiffusion_Ab.pt checkpoint")
    parser.add_argument("--out", required=True, type=Path, help="fresh output directory")
    parser.add_argument("--designs", default=1, type=int, help="number of backbones, default 1")
    parser.add_argument("--seed", default=0, type=int, help="seed of first backbone, default 0")
    parser.add_argument("--gpu", help="CUDA device index or UUID, default respects CUDA_VISIBLE_DEVICES")
    parser.add_argument("--schedule-dir", type=Path, help="trusted directory containing the 50-step table and cached_schedules")
    parser.add_argument("--triton-cache", type=Path, help="reusable Triton cache directory, default under --out")
    args = parser.parse_args()

    if args.designs < 1 or not 0 <= args.seed <= args.seed + args.designs - 1 < 2**32:
        parser.error("designs must be positive and all per-design seeds must be unsigned 32-bit integers")
    if not args.loops.strip() or not args.hotspots.strip():
        parser.error("supply the design loops and hotspot residue IDs")
    out = args.out.expanduser().resolve()
    if out.exists():
        parser.error(f"choose a fresh output directory: {out}")
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
        parser.error("activate a Linux Python 3.10 environment with the pinned CUDA packages")
    for package, required in VERSIONS.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "missing"
        if installed != required:
            parser.error(f"{package} must be {required} in the active environment, found {installed}")

    entry = REPO / "scripts/rfdiffusion_inference.py"
    if not entry.is_file() or not (PLUGINS / "nabloop_rfd_runtime.py").is_file():
        parser.error("run from the patched RFantibody checkout with the bundled uplifting files")
    schedule_dir = args.schedule_dir.expanduser().resolve() if args.schedule_dir else None
    if schedule_dir and not ((schedule_dir / SCHEDULE).is_file() and (schedule_dir / "cached_schedules").is_dir()):
        parser.error(f"--schedule-dir must contain {SCHEDULE} and cached_schedules/")

    os.umask(0o077)
    out.mkdir(parents=True)
    work = out / "work"
    work.mkdir()
    if schedule_dir:
        shutil.copy2(schedule_dir / SCHEDULE, work / SCHEDULE)
        shutil.copytree(schedule_dir / "cached_schedules", work / "cached_schedules")
    else:
        (work / "cached_schedules").mkdir()
    cache = args.triton_cache.expanduser().resolve() if args.triton_cache else out / "triton-cache"
    cache.mkdir(parents=True, exist_ok=True)

    env = {key: value for key, value in os.environ.items() if not key.startswith("NABLOOP_")}
    pythonpath = [str(REPO / "src"), str(REPO / "include/SE3Transformer"), str(PLUGINS)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env.update(FLAGS)
    env.update(
        PYTHONPATH=os.pathsep.join(pythonpath), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
        RFANTIBODY_ROOT=str(work), RFANTIBODY_WEIGHTS=str(args.weights.parent),
        RFANTIBODY_SCRIPTS=str(REPO / "scripts"), TRITON_CACHE_DIR=str(cache),
        DGLBACKEND="pytorch", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
    )
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    result = out / "result.qv"
    metadata = out / "metadata.jsonl"
    log = out / "run.log"
    command = [
        sys.executable, str(PLUGINS / "strict_probe.py"), str(PLUGINS / "model_probe.py"),
        "--model", "rfd", "--entry", str(entry), "--metadata", str(metadata),
        "--precision", "stock", "--", "--config-name", "antibody",
        hydra_path("antibody.target_pdb", args.target),
        hydra_path("antibody.framework_pdb", args.framework),
        f"antibody.design_loops=[{args.loops}]", f"ppi.hotspot_res=[{args.hotspots}]",
        hydra_path("inference.ckpt_override_path", args.weights),
        hydra_path("inference.quiver", result), f"inference.num_designs={args.designs}",
        "diffuser.T=50", "inference.final_step=1", "inference.deterministic=True",
        "inference.write_trajectory=False", f"inference.design_startnum={args.seed}",
        "inference.output_prefix=backbone", hydra_path("hydra.run.dir", out / "hydra"),
    ]
    print(f"RFdiffusion log: {log}", flush=True)
    started = time.perf_counter()
    with log.open("x") as handle:
        status = subprocess.run(command, cwd=work, env=env, stdout=handle, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    if status.returncode:
        parser.exit(1, f"RFdiffusion exited with status {status.returncode}. See {log}\n")
    try:
        completion = verify_run(metadata, result, entry, args.designs)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"RFdiffusion output verification failed ({error}). See {log}\n")
    (out / "summary.json").write_text(json.dumps({
        "flags": FLAGS, "argv": command, "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "wall_seconds": wall, "designs": args.designs, "metadata": completion,
    }, indent=2) + "\n")
    print(f"Generated {args.designs} backbone(s) in {wall:.2f}s. Output: {result}")


if __name__ == "__main__":
    main()
