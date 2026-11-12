# RFdiffusion P4 exact mode

This checkout uses RFantibody commit `3e5a5bb1cb98befde9fc049ae3ff58cabb491f67` with the P4 inference optimizations. The launcher uses the full target, 50 diffusion steps, the original FP32 matmul setting, and one deterministic seed per backbone. P4 produced identical normalized Quiver output to the pinned original on short and typical inputs on the same GPU. The short test also matched all 50 sampling steps and RNG states. Exactness and performance for a new GPU or target require a paired comparison.

Activate a Linux Python 3.10 environment with a CUDA-capable NVIDIA GPU, PyTorch `2.3.1+cu118`, DGL `2.4.0+cu118`, Triton `2.3.1`, and the upstream RFantibody dependencies. P4 used Python 3.10.22. Supply the antibody checkpoint from outside the checkout. The tested `RFdiffusion_Ab.pt` has SHA-256 `736c18ae8e867ed505d19853fa5380514ff439675d002ff3a73c7287acf07e3c`.

Run the launcher from this checkout.

```bash
python uplifting/run_exact.py \
  --target /path/to/full-target.pdb \
  --framework /path/to/cdr-annotated-framework.pdb \
  --loops 'H3:7,L3:9' --hotspots 'T305,T456' \
  --weights /path/to/RFdiffusion_Ab.pt \
  --designs 1 --seed 0 --gpu 0 \
  --out /path/to/new-run
```

Use the target PDB's chain letters and residue numbers for hotspots. Match loop names and lengths to the annotated framework. With multiple designs, seeds advance by one from `--seed`. Set `--gpu` to an available CUDA device index or UUID, or omit it to use the existing CUDA device assignment.

The launcher writes `result.qv`, `run.log`, `metadata.jsonl`, `summary.json`, and Hydra configuration in the new output directory. `--schedule-dir` can reuse a trusted upstream cache containing `T_50_omega_1000_min_sigma_0_02_min_b_1_5_max_b_2_5_schedule_linear.pkl` and `cached_schedules/`. Without it, upstream inference generates its tables as needed. Pass `--triton-cache` to reuse a compilation cache across runs. First-run table generation and compilation affect wall time.

Use `python uplifting/run_exact.py --help` for argument details. The upstream `scripts/rfdiffusion_inference.py` also remains available. The P4 launcher installs the axial packing and SE(3) basis optimizations and isolates the Triton cache RNG before running that script.

The `opt/long-2x` branch adds two optional settings to the same launcher.

- `--se3-exact` packs node features before expanding them onto edges, reuses self-interaction at each destination, and uses a four-term FP32 Triton projection for the measured full SE(3) chunks. The node and projection invocation counts appear in `summary.json`.
- `--reuse-ca-rbf` reuses the Cα distance features between the pair update and structure update in each eligible inference block.

The SE(3) experiment on an RTX 4090 D took 418.48 seconds for one backbone on the frozen 893-residue target at 50 steps. The same card took 645.96 seconds with the pinned original and 507.31 seconds with P4. The normalized Quiver matched the original. These are single observations. Use a same-card comparison for a different checkpoint, input, or GPU.

The `baseline/upstream` branch points to the pristine upstream commit. The `opt/exact` branch contains the source changes and this launcher. The [P4 benchmark summary](benchmarks/p4-summary.json) records input sizes, wall times, and the scope of the comparison. Use `python uplifting/compare_exact.py --help` to compare artifacts after running both versions on the same GPU with the same inputs and settings.
