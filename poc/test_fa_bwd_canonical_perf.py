from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import finalize_fa_bwd_canonical_perf as finalize
import prepare_fa_bwd_canonical_perf as prepare


def _write_task_time(path: Path, names: list[str]) -> None:
    output = path / "PROF_1" / "mindstudio_profiler_output"
    output.mkdir(parents=True)
    target = output / "task_time_1.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["kernel_type", "kernel_name", "task_start(us)", "task_time(us)"],
        )
        writer.writeheader()
        for index, name in enumerate(names):
            writer.writerow({
                "kernel_type": "AI_CORE",
                "kernel_name": name,
                "task_start(us)": index,
                "task_time(us)": index + 1,
            })


def test_exact_active_runs_excludes_preamble_and_keeps_five_active(tmp_path: Path) -> None:
    sequence = ["matmul", "softmax", "matmul"]
    _write_task_time(tmp_path, ["input_prep"] + sequence * 8)
    result = finalize.exact_active_runs(tmp_path)
    assert result["excluded_preamble_kernel_rows"] == 1
    assert result["kernel_sequence"] == sequence
    assert len(result["warmup_device_us"]) == 3
    assert len(result["active_device_us"]) == 5
    assert result["attribution"] == "exact_active_runs"


def test_exact_active_runs_rejects_nonrepeated_raw(tmp_path: Path) -> None:
    _write_task_time(tmp_path, [f"kernel_{index}" for index in range(8)])
    with pytest.raises(finalize.FinalizeError, match="no exact"):
        finalize.exact_active_runs(tmp_path)


def test_fixed50_contracts_are_unique() -> None:
    cases = Path(__file__).with_name("inputs") / "op29_fa_bwd_fixed50_shapes.csv"
    # The CSV is the compiler support matrix; the exact canonical JSON is staged
    # at packaging time.  Verify the fixed denominator here without inventing
    # any shape.
    rows = list(csv.DictReader(cases.open(encoding="utf-8")))
    assert len(rows) == 50


def test_mapping_receipts_reject_incomplete_case_coverage(tmp_path: Path) -> None:
    mapping_dir = tmp_path / "mapping_receipts"
    mapping_dir.mkdir()
    binding = {
        "mapped_candidate_files": ["generated/kernel/fa_bwd_fp16.so"],
        "candidate_files": {
            "candidate/bundle/generated/kernel/fa_bwd_fp16.so": "1" * 64,
        },
    }
    (mapping_dir / "case_00_pid_1.json").write_text(json.dumps({
        "case_index": 0,
        "mapped_bundle_owned_shared_objects": [{
            "relative_path": "generated/kernel/fa_bwd_fp16.so",
            "sha256": "1" * 64,
        }],
    }), encoding="utf-8")
    with pytest.raises(finalize.FinalizeError, match="coverage incomplete"):
        finalize.validate_mapping_receipts(tmp_path, binding)


def test_mapping_receipts_accept_normalized_loaded_order(tmp_path: Path) -> None:
    mapping_dir = tmp_path / "mapping_receipts"
    mapping_dir.mkdir()
    binding_order = [
        "generated/kernel/fa_bwd_fp16.cpp.so",
        "generated/kernel/fa_bwd_bf16.cpp.so",
        "generated/kernel/fa_bwd_fp32.cpp.so",
        "generated/libtilelang_fa_bwd_dispatch.so",
    ]
    hashes = {
        f"candidate/bundle/{relative}": f"{index + 1:x}" * 64
        for index, relative in enumerate(binding_order)
    }
    normalized = sorted(binding_order)
    for case_index in range(50):
        (mapping_dir / f"case_{case_index:02d}_pid_1.json").write_text(
            json.dumps({
                "case_index": case_index,
                "mapped_bundle_owned_shared_objects": [
                    {
                        "relative_path": relative,
                        "sha256": hashes[f"candidate/bundle/{relative}"],
                    }
                    for relative in normalized
                ],
            }),
            encoding="utf-8",
        )
    result = finalize.validate_mapping_receipts(
        tmp_path,
        {
            "mapped_candidate_files": binding_order,
            "candidate_files": hashes,
        },
    )
    assert len(result) == 50


def test_candidate_contract_is_exact_e288() -> None:
    assert prepare.SOURCE_COMMIT == "e28825ac9af5264b85a97e8ec0e25f3d238c37a3"
    assert prepare.SOURCE_TREE == "3ca646473522839e4a4d0cece1441cedba03520b"
    assert len(prepare.EXPECTED_CANDIDATE_HASHES) == 4
    assert prepare.PROFILE_SHA256 == (
        "637336eb9a1128bc5f12ba4aab8937e191f74040c98bd25b6b00ef6bb18a7079"
    )
