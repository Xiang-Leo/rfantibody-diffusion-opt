"""Compare complete RFdiffusion or RF2 internal traces from separate diagnostic runs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


def load(folder, model, expected):
    key = "designs" if model == "rfd" else "candidates"
    paths = sorted(folder.glob("trace-*.json"))
    rows = [row for path in paths for row in json.loads(path.read_text())[key]]
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} {key}, found {len(rows)} at {folder}")
    for row in rows:
        if model == "rfd":
            if [item["t"] for item in row["steps"]] != list(range(50, 0, -1)):
                raise ValueError("Incomplete diffusion trace")
        elif [item["cycle"] for item in row["cycles"]] != list(range(1, 12)) or row["selected"] is None:
            raise ValueError("Incomplete RF2 recycle trace")
    return rows


def first_difference(a, b, path="$"):
    if type(a) is not type(b):
        return path
    if isinstance(a, list):
        if len(a) != len(b):
            return path
        for i, (x, y) in enumerate(zip(a, b)):
            diff = first_difference(x, y, f"{path}[{i}]")
            if diff:
                return diff
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            return path
        for key in a:
            diff = first_difference(a[key], b[key], f"{path}.{key}")
            if diff:
                return diff
    elif a != b:
        return path
    return None


def digest(rows):
    return sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("rfd", "rf2"))
    parser.add_argument("reference", type=Path)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference, trial = (load(folder, args.model, args.count) for folder in (args.reference, args.trial))
    difference = first_difference(reference, trial)
    result = {"model": args.model, "items": args.count, "exact": difference is None,
              "reference_sha256": digest(reference), "trial_sha256": digest(trial),
              "first_difference": difference}
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")
    return int(difference is not None)


if __name__ == "__main__":
    raise SystemExit(main())
