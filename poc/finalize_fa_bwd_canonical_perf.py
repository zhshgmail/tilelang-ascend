#!/usr/bin/env python3
"""Fail-closed raw-device-time finalizer for central Cannbot perf output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import statistics
from pathlib import Path


DEVICE_DOMAIN = "msprof.task_time.kernel_device_us"
META_KERNEL_TYPES = {"PROFILING_ENABLE", "PROFILING_DISABLE", "TASK_TIMEOUT_SET", ""}


class FinalizeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _kernel_rows(profile_root: Path) -> list[tuple[float, str, float]]:
    files = sorted(profile_root.rglob("task_time_*.csv"))
    if len(files) != 1:
        raise FinalizeError(f"expected one task_time CSV below {profile_root}, observed {len(files)}")
    kernels = []
    with files[0].open("r", encoding="utf-8", errors="replace") as stream:
        for row in csv.DictReader(stream):
            if row.get("kernel_type", "") in META_KERNEL_TYPES:
                continue
            try:
                start = float((row.get("task_start(us)", "") or "0").strip())
                duration = float(row.get("task_time(us)", "") or 0)
            except ValueError:
                continue
            if duration > 0:
                kernels.append((start, row.get("kernel_name", "unknown"), duration))
    kernels.sort(key=lambda item: item[0])
    if not kernels:
        raise FinalizeError(f"no positive device kernel rows below {profile_root}")
    return kernels


def exact_active_runs(profile_root: Path, warmup: int = 3, repeats: int = 5) -> dict:
    """Extract the unique widest repeated suffix; never estimate a task split."""
    kernels = _kernel_rows(profile_root)
    count = warmup + repeats
    valid = []
    for width in range(1, len(kernels) // count + 1):
        suffix = kernels[-width * count:]
        names = [item[1] for item in suffix[:width]]
        if all(
            [item[1] for item in suffix[index * width:(index + 1) * width]] == names
            for index in range(count)
        ):
            valid.append(width)
    if not valid:
        raise FinalizeError(f"no exact {count}-run repeated suffix below {profile_root}")
    width = max(valid)
    suffix = kernels[-width * count:]
    runs = [
        [item[2] for item in suffix[index * width:(index + 1) * width]]
        for index in range(count)
    ]
    active = [sum(run) for run in runs[warmup:]]
    if len(active) != repeats or any(value <= 0 for value in active):
        raise FinalizeError("active device run extraction is incomplete")
    return {
        "task_time_csv_sha256": sha256_file(sorted(profile_root.rglob("task_time_*.csv"))[0]),
        "kernel_rows_total": len(kernels),
        "excluded_preamble_kernel_rows": len(kernels) - width * count,
        "kernel_sequence": [item[1] for item in suffix[:width]],
        "warmup_device_us": [sum(run) for run in runs[:warmup]],
        "active_device_us": active,
        "active_mean_device_us": statistics.mean(active),
        "attribution": "exact_active_runs",
    }


def validate_mapping_receipts(op_dir: Path, binding: dict) -> list[dict]:
    expected_files = binding["mapped_candidate_files"]
    expected_hashes = binding["candidate_files"]
    by_case: dict[int, list[dict]] = {index: [] for index in range(50)}
    for path in sorted((op_dir / "mapping_receipts").glob("case_*_pid_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        case_index = value.get("case_index")
        if case_index not in by_case:
            raise FinalizeError(f"mapping receipt has invalid case: {path}")
        rows = value.get("mapped_bundle_owned_shared_objects")
        if [row.get("relative_path") for row in rows or []] != expected_files:
            raise FinalizeError(f"mapped set differs: {path}")
        for row in rows:
            relative = row["relative_path"]
            if row.get("sha256") != expected_hashes[f"candidate/bundle/{relative}"]:
                raise FinalizeError(f"mapped DSO hash differs: {path}:{relative}")
        by_case[case_index].append({"path": str(path), "sha256": sha256_file(path)})
    missing = [index for index, values in by_case.items() if not values]
    if missing:
        raise FinalizeError(f"mapped-set coverage incomplete: {missing}")
    return [{"case_index": index, "receipts": by_case[index]} for index in range(50)]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FinalizeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _aggregate(rows: list[dict]) -> dict:
    refs = [row["ref_us"] for row in rows]
    candidates = [row["asc_us"] for row in rows]
    speedups = [row["speedup"] for row in rows]
    return {
        "n_cases_valid": len(rows),
        "geomean_speedup": statistics.geometric_mean(speedups),
        "mean_speedup": statistics.mean(speedups),
        "median_speedup": statistics.median(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
        "mean_ref_us": statistics.mean(refs),
        "median_ref_us": statistics.median(refs),
        "geomean_ref_us": statistics.geometric_mean(refs),
        "total_ref_us": sum(refs),
        "mean_asc_us": statistics.mean(candidates),
        "median_asc_us": statistics.median(candidates),
        "geomean_asc_us": statistics.geometric_mean(candidates),
        "total_asc_us": sum(candidates),
        "total_speedup": sum(refs) / sum(candidates),
    }


def finalize(args: argparse.Namespace) -> None:
    op_dir = args.op_dir.resolve(strict=True)
    central = json.loads((op_dir / "performance.json").read_text(encoding="utf-8"))
    binding = json.loads((op_dir / "PERF_INPUT_BINDING.json").read_text(encoding="utf-8"))
    if central.get("n_cases_total") != 50 or len(central.get("per_case", [])) != 50:
        raise FinalizeError("central helper did not cover exact fixed50")
    if central.get("profiling_mode") != "quick" or central.get("asc_mode") != "per_case":
        raise FinalizeError("central helper did not use quick per-case mode")
    if central.get("warmup") != 3 or central.get("repeats") != 5:
        raise FinalizeError("central warmup/repeat profile differs")
    if central.get("device_select_source") != "cli" or central.get("device_id") != args.device:
        raise FinalizeError("central helper device binding differs")
    output = args.output.resolve()
    if output.exists():
        raise FinalizeError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    raw_root = output / "raw_profiles"
    exact_rows = []
    raw_rows = []
    for expected_case, row in enumerate(central["per_case"]):
        if row.get("case") != expected_case or row.get("ref_error") is not None or row.get("asc_error") is not None:
            raise FinalizeError(f"invalid central row {expected_case}")
        sides = {}
        for side, key in (("reference", "ref_prof_dir"), ("candidate", "asc_prof_dir")):
            source = Path(row[key]).resolve(strict=True)
            parsed = exact_active_runs(source)
            target = raw_root / side / f"case_{expected_case:02d}"
            shutil.copytree(source, target)
            sides[side] = {**parsed, "archived_profile_root": str(target)}
        ref_us = sides["reference"]["active_mean_device_us"]
        asc_us = sides["candidate"]["active_mean_device_us"]
        speedup = ref_us / asc_us
        exact_rows.append({
            "case": expected_case,
            "shape": row["shape"],
            "dtype": row["dtype"],
            "ref_us": ref_us,
            "asc_us": asc_us,
            "speedup": speedup,
            "ref_error": None,
            "asc_error": None,
            "ref_prof_dir": str(raw_root / "reference" / f"case_{expected_case:02d}"),
            "asc_prof_dir": str(raw_root / "candidate" / f"case_{expected_case:02d}"),
            "ref_timing_domain": DEVICE_DOMAIN,
            "asc_timing_domain": DEVICE_DOMAIN,
            "ref_attribution": "exact_active_runs",
            "asc_attribution": "exact_active_runs",
            "ref_device": args.device,
            "asc_device": args.device,
        })
        raw_rows.append({
            "case": expected_case,
            "timing_domain": DEVICE_DOMAIN,
            "reference": sides["reference"],
            "candidate": sides["candidate"],
            "speedup": speedup,
        })
    aggregates = _aggregate(exact_rows)
    exact = {
        "task": central["task"],
        "task_dir": central["task_dir"],
        "n_cases_total": 50,
        **aggregates,
        "warmup": 3,
        "repeats": 5,
        "seed": 0,
        "device_id": args.device,
        "device_select_source": "cli",
        "timing_method": "msprof.quick.Task_Duration",
        "profiling_mode": "quick",
        "asc_mode": "per_case",
        "per_case": exact_rows,
    }
    write_json(output / "performance_exact.json", exact)
    with (output / "per_case_device_time_raw.jsonl").open("w", encoding="utf-8") as stream:
        for row in raw_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    mappings = validate_mapping_receipts(op_dir, binding)
    write_json(output / "mapped_set_receipts.json", mappings)

    adapter_path = op_dir / "central_perf_profile_adapter.py"
    adapter = _load_module(adapter_path, "central_perf_profile_adapter")
    receipt = adapter.adapt_performance_json(
        output / "performance_exact.json",
        profile_path=op_dir / "CENTRAL_PERFORMANCE_PROFILE.json",
        expected_device=args.device,
        upstream_binding=binding["central_provider"]["upstream"],
    )
    if receipt.get("status") not in {"PASS", "FAIL"} or receipt.get("reasons"):
        raise FinalizeError(f"central profile adapter rejected exact raw: {receipt.get('reasons')}")
    write_json(output / "CENTRAL_PROFILE_RECEIPT.json", receipt)
    write_json(output / "RESULT.json", {
        "status": receipt["status"],
        "product_authority": False,
        "authority": "central_cannbot_perf_probe_v1_shadow_only",
        "fixed_case_count": 50,
        "geomean_speedup": aggregates["geomean_speedup"],
        "threshold": 1.0,
        "timing_domain": DEVICE_DOMAIN,
        "source_commit": binding["source_commit"],
        "bundle_manifest_sha256": binding["bundle_manifest_sha256"],
    })
    manifest_lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "EVIDENCE_MANIFEST.sha256":
            manifest_lines.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    manifest = output / "EVIDENCE_MANIFEST.sha256"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "geomean_speedup": aggregates["geomean_speedup"],
        "manifest_sha256": sha256_file(manifest),
    }, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    finalize(parse_args())
