"""Runtime helpers loaded only by the prepared RFantibody command."""

from collections import defaultdict
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import time

import torch


LEVERS = frozenset(os.environ.get("NABLOOP_RFD_LEVERS", "").split(","))
PROFILE = os.environ.get("NABLOOP_RFD_PROFILE")
TIMINGS = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
CACHE_STATS = defaultdict(lambda: {"hits": 0, "misses": 0})
CACHE = {}


def reset_design_cache():
    CACHE.clear()


def cached(name, idx, compute, variant=None):
    """Keep one entry per operation and release it at each new design."""
    entry = CACHE.get(name)
    if entry is not None and entry[0] is idx and entry[1] == variant:
        CACHE_STATS[name]["hits"] += 1
        return entry[2]
    value = compute()
    CACHE[name] = idx, variant, value
    CACHE_STATS[name]["misses"] += 1
    return value


@contextmanager
def timed_phase(name):
    start = start_phase()
    try:
        yield
    finally:
        finish_phase(name, start)


def start_phase():
    if not PROFILE:
        return None
    # Synchronization is only enabled for diagnostics. Use the normal run's
    # wall time to measure production throughput.
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    return time.perf_counter()


def finish_phase(name, start):
    if start is None:
        return
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    TIMINGS[name]["calls"] += 1
    TIMINGS[name]["seconds"] += time.perf_counter() - start


def phase(name):
    return timed_phase(name) if PROFILE else nullcontext()


def write_profile():
    if PROFILE:
        profile = {"phases": dict(TIMINGS), "cache": dict(CACHE_STATS)}
        Path(PROFILE).write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
