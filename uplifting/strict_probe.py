"""Run the probe with Triton cache filenames isolated from model Python RNG."""

from pathlib import Path
import random
import runpy
import sys


def main():
    path = Path(sys.argv[1]).resolve(strict=True)
    import triton
    from triton.runtime import cache

    if triton.__version__ != "2.3.1" or cache.random is not random:
        raise RuntimeError("Triton cache RNG isolation requires the verified Triton 2.3.1 runtime")
    # Triton uses this source only to choose a temporary cache filename.
    # Keep model randomness independent of whether a kernel is already cached.
    cache.random = random.SystemRandom()
    try:
        sys.argv = [str(path), *sys.argv[2:]]
        runpy.run_path(str(path), run_name="__main__")
    finally:
        cache.random = random


if __name__ == "__main__":
    main()
