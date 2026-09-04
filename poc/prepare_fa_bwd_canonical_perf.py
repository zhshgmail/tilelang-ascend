#!/usr/bin/env python3
"""Prepare, but never launch, canonical device-time profiling for FA-bwd."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath


SOURCE_COMMIT = "e28825ac9af5264b85a97e8ec0e25f3d238c37a3"
SOURCE_TREE = "3ca646473522839e4a4d0cece1441cedba03520b"
BUNDLE_MANIFEST_SHA256 = "eca6576a5d6e3ee0c86034d52ff9dbeced32742a1449ddcb926f8a04069f2a89"
CANONICAL_MODEL_SHA256 = "71a29698c475bb66844308eb96429fe8300f2fd6bacfb32a3f5c338038e72793"
CANONICAL_CASES_SHA256 = "63b545dd67682935c51910b42c4324c4d375559aff12b44b469c1aaecd97c253"
CONSUMER_SHA256 = "5aaf9d8c9266a497ecd75a00fa8ea2161efebd44e99c4b12cfa2a167de6906d5"
PROFILE_SHA256 = "637336eb9a1128bc5f12ba4aab8937e191f74040c98bd25b6b00ef6bb18a7079"
PROFILE_ADAPTER_SHA256 = "2f4bb596010f6a2183b91d9607cbebfce88284e0ee2c64d94bc13838ab90d5c3"
HELPER_SHA256 = "a37714fff7046d7a344baa1037e5f734ffc0723f29ac75f53b3314e5152ada37"

EXPECTED_CANDIDATE_HASHES = {
    "candidate/bundle/generated/libtilelang_fa_bwd_dispatch.so":
        "50031c8a349728c934c37c3a4d858cd951962e2f4ded082e7cb8062d0119bc83",
    "candidate/bundle/generated/kernel/fa_bwd_fp16.cpp.so":
        "29772d133f56be1915baa01499cc58a84feb11a592e40c8d359c666eda5c651a",
    "candidate/bundle/generated/kernel/fa_bwd_bf16.cpp.so":
        "4836feda36b0cad48134f2d3ce0c9113c50a72caf7ddac7fd78644e47de3137f",
    "candidate/bundle/generated/kernel/fa_bwd_fp32.cpp.so":
        "65d037a992781cce5f5ef0044d7f651502fcdbb7a62115a6d4c73c31f9a8e68e",
}


class PrepError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PrepError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise PrepError(f"SHA-256 mismatch for {path}: expected {expected}, observed {actual}")


def _case_contract(case: dict) -> dict:
    specs = {item["name"]: item for item in case["inputs"]}
    q = specs["q"]["shape"]
    k = specs["k"]["shape"]
    return {
        "B": q[0],
        "Sq": q[1],
        "Sk": k[1],
        "Hq": q[2],
        "Hk": k[2],
        "D": q[3],
        "dtype": specs["q"]["dtype"],
        "causal": bool(specs["causal"]["value"]),
        "window_left": int(specs["window_left"]["value"]),
        "window_right": int(specs["window_right"]["value"]),
        "softcap": float(specs["softcap"]["value"]),
    }


def load_case_contracts(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]
    if len(rows) != 50:
        raise PrepError(f"frozen descriptor denominator differs: {len(rows)}")
    contracts = [_case_contract(row) for row in rows]
    encoded = [json.dumps(row, sort_keys=True) for row in contracts]
    if len(encoded) != len(set(encoded)):
        raise PrepError("frozen50 contracts are not uniquely identifiable at runtime")
    return contracts


def verify_source(source_repo: Path) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"], cwd=source_repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=source_repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if (head, tree) != (SOURCE_COMMIT, SOURCE_TREE):
        raise PrepError(f"source identity differs: head={head}, tree={tree}")


def write_manifest(root: Path) -> Path:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "PREP_MANIFEST.sha256":
            relative = path.relative_to(root).as_posix()
            if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
                raise PrepError(f"unsafe manifest path: {relative}")
            lines.append(f"{sha256_file(path)}  {relative}")
    target = root / "PREP_MANIFEST.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def prepare(args: argparse.Namespace) -> None:
    source_repo = args.source_repo.resolve(strict=True)
    candidate_bundle = args.candidate_bundle.resolve(strict=True)
    canonical_model = args.canonical_model.resolve(strict=True)
    canonical_cases = args.canonical_cases.resolve(strict=True)
    consumer = args.consumer.resolve(strict=True)
    provider_binding = args.provider_binding.resolve(strict=True)
    provider_profile = args.provider_profile.resolve(strict=True)
    provider_adapter = args.provider_adapter.resolve(strict=True)
    locked_helper = args.locked_helper.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise PrepError(f"refusing existing output: {output}")

    verify_source(source_repo)
    require_hash(candidate_bundle / "MANIFEST.sha256", BUNDLE_MANIFEST_SHA256)
    require_hash(canonical_model, CANONICAL_MODEL_SHA256)
    require_hash(canonical_cases, CANONICAL_CASES_SHA256)
    require_hash(consumer, CONSUMER_SHA256)
    require_hash(provider_profile, PROFILE_SHA256)
    require_hash(provider_adapter, PROFILE_ADAPTER_SHA256)
    require_hash(locked_helper, HELPER_SHA256)
    provider = read_json(provider_binding)
    if provider.get("status") != "READY" or provider.get("selected_capabilities") != ["performance"]:
        raise PrepError("central provider binding is not READY for performance only")
    stage_files = provider.get("stage_files", {})
    if stage_files.get("ops/ops-profiling/scripts/msprof_perf_summary.py") != HELPER_SHA256:
        raise PrepError("provider binding helper identity differs")

    case_contracts = load_case_contracts(canonical_cases)
    output.mkdir(parents=True)
    shutil.copytree(candidate_bundle, output / "candidate" / "bundle")
    shutil.copy2(canonical_model, output / "model.py")
    shutil.copy2(canonical_cases, output / "29_FlashAttentionBwd.json")
    shutil.copy2(canonical_cases, output / "fa_bwd_perf_cases.jsonl")
    shutil.copy2(consumer, output / "fa_bwd_consumer.py")
    shutil.copy2(Path(__file__).with_name("fa_bwd_perf_model_new.py"), output / "model_new_ascendc.py")
    shutil.copy2(Path(__file__).with_name("finalize_fa_bwd_canonical_perf.py"), output / "finalize_perf.py")

    for relative, expected in EXPECTED_CANDIDATE_HASHES.items():
        require_hash(output / relative, expected)
    mapped = [
        "generated/kernel/fa_bwd_fp16.cpp.so",
        "generated/kernel/fa_bwd_bf16.cpp.so",
        "generated/kernel/fa_bwd_fp32.cpp.so",
        "generated/libtilelang_fa_bwd_dispatch.so",
    ]
    binding = {
        "schema": "tilelang.fa_bwd_canonical_perf_input/1",
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "kernel_count": 3,
        "dispatcher_count": 1,
        "bundle_manifest_sha256": BUNDLE_MANIFEST_SHA256,
        "candidate_files": EXPECTED_CANDIDATE_HASHES,
        "mapped_candidate_files": mapped,
        "fixed_case_count": 50,
        "case_contracts": case_contracts,
        "canonical_model_sha256": CANONICAL_MODEL_SHA256,
        "canonical_cases_sha256": CANONICAL_CASES_SHA256,
        "consumer_sha256": CONSUMER_SHA256,
        "central_provider": {
            "binding_path": str(provider_binding),
            "binding_sha256": sha256_file(provider_binding),
            "binding_receipt_sha256": provider["receipt_sha256"],
            "upstream": read_json(Path(provider["lock_path"]))["upstream"],
            "helper_sha256": HELPER_SHA256,
            "profile_id": "cannbot_perf_probe_v1",
            "profile_definition_sha256": PROFILE_SHA256,
            "profile_adapter_sha256": PROFILE_ADAPTER_SHA256,
        },
        "measurement": {
            "timing_domain": "msprof.task_time.kernel_device_us",
            "timing_method": "msprof.quick.Task_Duration",
            "ordering": "reference_then_candidate_per_case_same_device_same_session",
            "warmup": 3,
            "active_repeats": 5,
            "per_case": True,
            "aggregate": "geomean_i(reference_device_us_i/candidate_device_us_i)",
            "threshold": 1.0,
            "wall_clock_fallback": "forbidden",
        },
        "required_post_run": [
            "central performance.json",
            "50 reference and 50 candidate raw msprof profile roots",
            "50 exact candidate mapped-set receipts",
            "strict active-run extraction with no proportional fallback",
            "central profile adapter PASS or FAIL receipt",
            "complete evidence manifest",
        ],
    }
    write_json(output / "PERF_INPUT_BINDING.json", binding)
    shutil.copy2(provider_binding, output / "CENTRAL_PROVIDER_BINDING.json")
    shutil.copy2(provider_profile, output / "CENTRAL_PERFORMANCE_PROFILE.json")
    shutil.copy2(provider_adapter, output / "central_perf_profile_adapter.py")
    shutil.copy2(locked_helper, output / "central_msprof_helper.py")
    manifest = write_manifest(output)
    print(json.dumps({
        "status": "PREPARED_DEVICE_FREE_NOT_RUN",
        "output": str(output),
        "prep_manifest_sha256": sha256_file(manifest),
        "fixed_case_count": 50,
        "dispatcher_count": 1,
        "kernel_count": 3,
    }, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--canonical-model", type=Path, required=True)
    parser.add_argument("--canonical-cases", type=Path, required=True)
    parser.add_argument("--consumer", type=Path, required=True)
    parser.add_argument("--provider-binding", type=Path, required=True)
    parser.add_argument("--provider-profile", type=Path, required=True)
    parser.add_argument("--provider-adapter", type=Path, required=True)
    parser.add_argument("--locked-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
