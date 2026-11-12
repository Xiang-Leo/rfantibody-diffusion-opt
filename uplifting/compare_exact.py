"""Compare published RFantibody bytes and audit completed inference work."""

import argparse
import csv
from hashlib import sha256
from itertools import zip_longest
import json
import math
from pathlib import Path
import re


FILES = {
    "rfd": ("1_rfdiffusion.qv", "rfd/result.qv", "result.qv", "backbones.qv"),
    "mpnn": ("2_proteinmpnn.qv", "mpnn/candidates.qv", "candidates.qv", "result.qv"),
    "rf2": ("3_rf2.qv", "rf2/result.qv", "result.qv", "rf2.qv"),
}
STAGES = {"rfd": "RFdiffusion", "mpnn": "ProteinMPNN", "rf2": "antibody_RoseTTAFold2"}
FIELD = re.compile(rb"(QV_TAG|QV_SCORE)([ \t]+)(\S+)")
DESIGN = re.compile(rb"(?:^|[/_])(backbone_\d+(?:_dldesign_\d+)?(?:_(?:best|cycle_?\d+|\d+))?)\Z")
PATH_PREFIX = (b"/", b"./", b"../", b"_home_", b"_tmp_", b"_private_tmp_",
               b"_private_var_", b"_var_", b"_Users_", b"_mnt_", b"_opt_", b"_srv_")
STEP = re.compile(rb"\bTimestep (\d+), input to next step:")
CYCLE = re.compile(rb"\[RF2\]\s+Cycle (\d+)/(\d+) - pLDDT: (\S+)")
JSON_TAG = re.compile(rb'("(?:tag|backbone_tag|rf2_score|structure|folder)"\s*:\s*)("(?:[^"\\]|\\.)*")')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical_tag(tag):
    match = DESIGN.search(tag)
    if match and (match.start(1) == 0 or tag.startswith(PATH_PREFIX)):
        return match[1]
    return tag


def replace_tag(line, field):
    return line[:field.start(3)] + canonical_tag(field[3]) + line[field.end(3):]


def quiver(path, stage):
    raw = path.read_bytes()
    lines, atoms, scores = [], [], []
    residues = {}
    ids = []
    tag = None
    current_atoms = current_scores = 0
    for line in raw.splitlines(keepends=True):
        field = FIELD.match(line)
        if field and field[1] == b"QV_TAG":
            if tag is not None:
                require(current_atoms and (stage != "rf2" or current_scores == 1), "incomplete Quiver record")
            tag = field[3]
            ids.append(canonical_tag(tag))
            current_atoms = current_scores = 0
        require(tag is not None, "Quiver content precedes first tag")
        if field:
            require(field[3] == tag, "QV_SCORE does not refer to the current tag")
            line = replace_tag(line, field)
            if field[1] == b"QV_SCORE":
                current_scores += 1
                require(current_scores == 1, "duplicate QV_SCORE for one record")
                require(len(line.split(maxsplit=2)) == 3, "QV_SCORE payload is empty")
                scores.append(line)
        elif line.startswith(b"ATOM  "):
            require(len(line) >= 54, "truncated ATOM record")
            current_atoms += 1
            atoms.append(line)
            key = (len(ids), line[21:27])
            name = line[17:20]
            require(residues.setdefault(key, name) == name, "inconsistent residue identity")
        lines.append(line)
    require(ids and current_atoms and (stage != "rf2" or current_scores == 1), "incomplete final Quiver record")
    require(len(set(ids)) == len(ids), "duplicate canonical Quiver identifiers")
    return {
        "canonical_lines": lines,
        "ids": ids,
        "digests": {
            "raw_sha256": sha256(raw).hexdigest(),
            "canonical_sha256": sha256(b"".join(lines)).hexdigest(),
            "atoms_sha256": sha256(b"".join(atoms)).hexdigest(),
            "scores_sha256": sha256(b"".join(scores)).hexdigest(),
            "ordered_residue_identities_sha256": sha256(repr(list(residues.items())).encode()).hexdigest(),
        },
    }


def locate(directory, stage):
    if directory.is_file():
        return directory
    for name in FILES[stage]:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"missing {stage} Quiver artifact")


def read_json(path):
    return json.loads(path.read_text())


def logged_work(directory, artifact, stage, expected):
    if stage == "rfd":
        logs = sorted(directory.glob("1_rfdiffusion_*.log"))
        markers = (b" - Making design ", b" - Finished design in ")
        pattern = STEP
        sequence = [(str(index).encode(),) for index in range(expected, 1, -1)]
    elif stage == "rf2":
        shard = re.fullmatch(r"rf2_output_(\d+)", artifact.stem)
        logs = [directory / f"3_rf2_{shard[1]}.log"] if shard else sorted(directory.glob("3_rf2_*.log"))
        markers = (b"[RF2] Processing:", b"[RF2] Completed:")
        pattern = CYCLE
        sequence = [(str(index).encode(), str(expected).encode()) for index in range(1, expected + 1)]
    else:
        return None
    if not logs:
        primary = directory / ("1_rfdiffusion.log" if stage == "rfd" else "3_rf2.log")
        logs = [primary if primary.is_file() else directory / "run.log"]
    if not all(log.is_file() for log in logs):
        return None
    completed = []
    active = None
    for log in logs:
        for line in log.read_bytes().splitlines():
            if markers[0] in line:
                require(active is None, "overlapping candidate starts in log")
                active = []
            elif markers[1] in line:
                require(active is not None and [row[:len(sequence[0])] for row in active] == sequence,
                        "logged inference steps or cycles differ")
                completed.append(active)
                active = None
            else:
                match = pattern.search(line)
                if match:
                    require(active is not None, "logged forward outside an active candidate")
                    active.append(match.groups())
        require(active is None, "unfinished candidate in log")
    return completed or None


def completed_run(root, artifact, stage, count, expected, require_complete):
    directory = root if root.is_dir() else artifact.parent
    metadata_candidates = [directory / f"{Path(FILES[stage][0]).stem}.metadata.jsonl",
                           artifact.parent / "metadata.jsonl"]
    source = None
    if stage != "mpnn":
        metadata = next((candidate for candidate in metadata_candidates if candidate.is_file()), None)
        if metadata:
            with metadata.open() as handle:
                entries = [json.loads(line) for line in handle if line.strip()]
            require(bool(entries), "empty model metadata")
            total = entries[-1]
            require(total.get("kind") == "total" and total.get("success") is True
                    and total.get("forward_count") == count * expected,
                    "model metadata does not prove all required forwards")
            source = "model metadata"
    stage_file = directory / "stages.json"
    if stage_file.is_file():
        matches = [row for row in read_json(stage_file) if row["name"] == STAGES[stage]]
        require(len(matches) == 1 and matches[0]["status"] == "succeeded", "pipeline stage is incomplete")
        progress = matches[0].get("progress")
        if stage == "rf2" and progress and not re.fullmatch(r"rf2_output_\d+", artifact.stem):
            require(progress.get("attempted") == progress.get("completed") == count and progress.get("failed") == 0,
                    "pipeline RF2 progress differs from the output count")
        if stage == "mpnn":
            source = "pipeline stage"
    timing = directory / "timing.json"
    if timing.is_file():
        timing_data = read_json(timing)
        require(timing_data.get("exit_code") == 0, "timing record reports failed command")
    logs = logged_work(artifact.parent, artifact, stage, expected)
    if logs is not None:
        require(len(logs) == count, "logged completed candidate count differs from Quiver")
        source = source or "complete inference log"
    require(not require_complete or source is not None, f"missing completion evidence for {stage}")
    result = {"records": count, "completion_evidence": source}
    if source and stage != "mpnn":
        result["verified_work_items"] = count * expected
        result["work_kind"] = "diffusion steps" if stage == "rfd" else "RF2 forwards"
    return result, logs


def compare_lines(before, after):
    for index, (a, b) in enumerate(zip_longest(before, after), 1):
        if a == b:
            continue
        sample = a if a is not None else b
        kind = "atom" if sample.startswith(b"ATOM  ") else "score" if sample.startswith(b"QV_SCORE ") else "other"
        return {"line": index, "kind": kind}
    return None


def stage_result(reference, trial, stage, count, steps, cycles, require_complete):
    work = steps if stage == "rfd" else cycles
    reference_file = locate(reference, stage)
    original = quiver(reference_file, stage)
    n = len(original["ids"])
    require(count is None or n == count, f"reference {stage} record count is {n}, expected {count}")
    reference_status, original_logs = completed_run(reference, reference_file, stage, n, work, require_complete)
    result = {"reference": reference_status | original["digests"]}
    if trial is None:
        return result, original["ids"]
    trial_file = locate(trial, stage)
    candidate = quiver(trial_file, stage)
    new_count = len(candidate["ids"])
    trial_status, candidate_logs = completed_run(trial, trial_file, stage, new_count, work, require_complete)
    result["trial"] = trial_status | candidate["digests"]
    result["record_count_equal"] = n == new_count
    result["canonical_ids_and_order_equal"] = original["ids"] == candidate["ids"]
    mismatch = compare_lines(original["canonical_lines"], candidate["canonical_lines"])
    result["first_mismatch"] = mismatch
    result["byte_equal_after_output_tag_normalization"] = mismatch is None
    result["atoms_equal"] = original["digests"]["atoms_sha256"] == candidate["digests"]["atoms_sha256"]
    result["scores_equal"] = original["digests"]["scores_sha256"] == candidate["digests"]["scores_sha256"]
    result["sequence_and_numbering_equal"] = (
        original["digests"]["ordered_residue_identities_sha256"]
        == candidate["digests"]["ordered_residue_identities_sha256"]
    )
    result["logged_cycles_and_plddt_equal"] = (
        original_logs == candidate_logs if stage == "rf2" and original_logs is not None and candidate_logs is not None
        else None
    )
    return result, original["ids"]


def normalize_json_output_paths(path, root):
    raw = path.read_bytes()

    def rewrite(match):
        key = match[1].split(b'"')[1]
        value = json.loads(match[2]).encode()
        if key in (b"tag", b"backbone_tag"):
            changed = canonical_tag(value)
        elif key == b"rf2_score":
            field = FIELD.match(value)
            changed = replace_tag(value, field) if field else value
        else:
            target = Path(value.decode())
            changed = str(target.relative_to(root)).encode() if target.is_absolute() and target.is_relative_to(root) else value
        return match[1] + (match[2] if changed == value else json.dumps(changed.decode()).encode())

    return JSON_TAG.sub(rewrite, raw)


def route_context(directory, ids):
    root = directory.resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    require(manifest.get("success") is True and manifest.get("partial") is False, "route manifest incomplete")
    stages = read_json(root / "stages.json")
    stage_names = [row["name"] for row in stages]
    require(set(STAGES.values()) | {"preflight", "QC_deduplication", "Boltz2"} <= set(stage_names)
            and all(row.get("status") == "succeeded" for row in stages), "route stages incomplete or reused")
    require(ids["rf2"] == [tag + b"_best" for tag in ids["mpnn"]], "RF2 candidates differ from ProteinMPNN lineage")
    parents = []
    for tag in ids["mpnn"]:
        match = re.fullmatch(rb"(.+)_dldesign_\d+", tag)
        require(match is not None, "ProteinMPNN candidate lacks a backbone identifier")
        parents.append(match[1])
    require(set(parents) == set(ids["rfd"]), "ProteinMPNN candidates do not cover the RFdiffusion backbones")
    qc = read_json(root / "qc.json")["candidates"]
    require([canonical_tag(row["tag"].encode()) for row in qc] == ids["rf2"], "QC candidate order differs from RF2")
    with (root / "candidates.csv").open(newline="") as handle:
        require([row["candidate_id"] for row in csv.DictReader(handle)] == [row["candidate_id"] for row in qc],
                "candidate CSV differs from the QC candidate order")
    selected = [row["candidate_id"] for row in qc if row["selected_for_boltz"]]
    boltz = read_json(root / "boltz_summary.json")["candidates"]
    require([row["candidate_id"] for row in boltz] == selected
            and all(row.get("status") == "succeeded" for row in boltz), "Boltz2 output is incomplete")
    digests = {
        "candidates_csv_sha256": sha256((root / "candidates.csv").read_bytes()).hexdigest(),
        "qc_canonical_sha256": sha256(normalize_json_output_paths(root / "qc.json", root)).hexdigest(),
        "boltz_summary_canonical_sha256": sha256(normalize_json_output_paths(root / "boltz_summary.json", root)).hexdigest(),
    }
    boltz_structures = {}
    for row in boltz:
        artifact = (root / row["structure"]).resolve(strict=True)
        require(artifact.is_relative_to(root), "Boltz2 structure is outside its run directory")
        boltz_structures[row["candidate_id"]] = sha256(artifact.read_bytes()).hexdigest()
    return {"succeeded_stages": stage_names, "qc_candidates": len(qc), "boltz_completed": len(boltz),
            "selected_for_boltz": selected, "artifact_digests": digests | {"boltz_structures": boltz_structures}}


def raw_trace(path, cycles, quiver_digest):
    original = path.read_bytes()
    candidates = json.loads(original)["candidates"]
    require(bool(candidates), "empty raw RF2 trace")
    trace_output = quiver(locate(path.parent, "rf2"), "rf2")
    require(len(candidates) == len(trace_output["ids"]) and trace_output["digests"]["canonical_sha256"] == quiver_digest,
            "raw RF2 trace does not match the corresponding candidate Quiver")
    for row in candidates:
        entries = row["cycles"]
        require([entry["cycle"] for entry in entries] == list(range(1, cycles + 1))
                and 1 <= row["best_cycle"] <= cycles
                and all({"cpu_rng", "cuda_rng", "metrics", "xyz_prev"} <= entry.keys() for entry in entries),
                "raw RF2 trace is missing cycles, state, scores, or RNG")
    return {"candidate_count": len(candidates), "raw_sha256": sha256(original).hexdigest(),
            "associated_quiver_canonical_sha256": quiver_digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "route"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trial", type=Path, help="omit to inventory original artifacts")
    parser.add_argument("--count", type=int, help="expected Quiver records for a standalone stage")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cycles", type=int, default=11, help="RF2 forwards per candidate, including initial forward")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--raw-reference", type=Path)
    parser.add_argument("--raw-trial", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 2 or args.cycles < 1 or (args.count is not None and (args.stage == "route" or args.count < 1)):
        parser.error("use positive expected workload; --count is for standalone stages")
    if args.raw_trial and (not args.raw_reference or not args.trial):
        parser.error("--raw-trial requires --trial and --raw-reference")
    if args.raw_reference and args.stage not in ("rf2", "route"):
        parser.error("raw RF2 traces require --stage rf2 or --stage route")

    stages = tuple(STAGES) if args.stage == "route" else (args.stage,)
    report = {"schema_version": 1, "stages": {}, "errors": []}
    ids = {}
    for stage in stages:
        try:
            result, ids[stage] = stage_result(args.reference, args.trial, stage, args.count,
                                              args.steps, args.cycles, args.require_complete or args.stage == "route")
            report["stages"][stage] = result
            if args.trial and (not result["byte_equal_after_output_tag_normalization"]
                               or result["logged_cycles_and_plddt_equal"] is False):
                report["errors"].append(f"{stage}: exact comparison failed")
        except (OSError, KeyError, TypeError, ValueError) as error:
            report["errors"].append(f"{stage}: {error}")
    if args.stage == "route" and len(ids) == len(stages):
        try:
            before = route_context(args.reference, ids)
            report["route"] = {"reference": before}
            if args.trial:
                trial_ids = {stage: quiver(locate(args.trial, stage), stage)["ids"] for stage in stages}
                after = route_context(args.trial, trial_ids)
                report["route"].update(trial=after, exact=before == after)
                if before != after:
                    report["errors"].append("route stage order, screening, or Boltz2 outputs differ")
        except (OSError, KeyError, TypeError, ValueError) as error:
            report["errors"].append(f"route: {error}")
    if args.raw_reference:
        try:
            rf2 = report["stages"]["rf2"]
            before = raw_trace(args.raw_reference, args.cycles, rf2["reference"]["canonical_sha256"])
            report["rf2_raw_trace"] = {"reference": before}
            if args.raw_trial:
                after = raw_trace(args.raw_trial, args.cycles, rf2["trial"]["canonical_sha256"])
                report["rf2_raw_trace"].update(trial=after, exact=before == after)
                if before != after:
                    report["errors"].append("raw RF2 trace differs")
        except (OSError, KeyError, TypeError, ValueError) as error:
            report["errors"].append(f"raw RF2 trace: {error}")
    report["valid"] = not report["errors"]
    report["exact"] = report["valid"] if args.trial else None
    report["raw_internal_equality_checked"] = bool(args.raw_trial and report.get("rf2_raw_trace", {}).get("exact"))
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
