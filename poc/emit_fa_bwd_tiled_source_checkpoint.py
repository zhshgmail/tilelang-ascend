#!/usr/bin/env python3
"""Emit a device-free Route-A IR/source receipt for tiled FA-Bwd."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_lowering_failure(error_text: str) -> str:
    """Return a stable fail-closed boundary label for a Route-A failure."""

    if (
        "Mismatch in sync points between cube and vec" in error_text
        and "ascend_combinecv.cc" in error_text
    ):
        return "ASCEND_COMBINE_CV_SYNC_POINT_MISMATCH"
    return "UNCLASSIFIED_ROUTE_A_LOWERING_FAILURE"


def write_manifest(output: Path) -> None:
    rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name != "MANIFEST.sha256":
            rows.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "MANIFEST.sha256").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    identity = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD", "HEAD^{tree}"],
        text=True,
    ).splitlines()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    )
    if status:
        raise RuntimeError(f"source worktree is dirty:\n{status}")

    sys.path.insert(0, str(repo))
    from poc.fa_bwd_tiled_source_validator import validate_tiled_source
    from poc.fa_bwd_tiled_symbolic_lowering import (
        TILED_FA_BWD_PASS_CONFIGS,
        make_fa_bwd_tiled,
    )
    from poc.run_fa_bwd_real_lowering import lower_variant
    from tilelang.jit.adapter import ascendc_dispatch

    cases_path = repo / "poc/inputs/op29_fa_bwd_fixed50_shapes.csv"
    cases = ascendc_dispatch.load_fa_bwd_cases(cases_path)
    plan = ascendc_dispatch.plan_fa_bwd_dispatch(cases)
    if len(cases) != 50 or len(plan.variants) != 3:
        raise AssertionError("tiled checkpoint requires fixed50 and three dtype variants")

    output.mkdir(parents=True)
    source_dir = output / "generated_sources"
    ir_dir = output / "route_a_ir"
    source_dir.mkdir()
    ir_dir.mkdir()
    variants: dict[str, object] = {}
    for variant in plan.variants:
        suffix = ascendc_dispatch.DTYPE_SUFFIXES[variant.dtype]
        function = make_fa_bwd_tiled(
            variant.dtype, variant.host_entry, variant.kernel_symbol
        )
        ir_path = ir_dir / f"fa_bwd_{suffix}.tir"
        ir_text = function.script()
        ir_path.write_text(ir_text, encoding="utf-8")
        gemm_intrinsics = ir_text.count("T.ascend_gemm_v0")
        if gemm_intrinsics < 5:
            raise AssertionError(
                f"{variant.dtype}: expected five GEMM roles in Route-A IR, "
                f"found {gemm_intrinsics}"
            )

        variants[variant.dtype] = {
            "host_entry": variant.host_entry,
            "kernel_entry": variant.kernel_symbol,
            "case_ids": list(variant.case_ids),
            "route_a_ir": {
                "path": str(ir_path.relative_to(output)),
                "sha256": sha256(ir_path),
                "bytes": ir_path.stat().st_size,
                "gemm_intrinsic_count": gemm_intrinsics,
            },
        }

    result = {
        "status": "IN_PROGRESS",
        "authority": "DEVICE_FREE_ROUTE_A_IR_ONLY",
        "npu_used": False,
        "bisheng_invoked": False,
        "source_head": identity[0],
        "source_tree": identity[1],
        "route_a": {"target": "ascendc", "platform": "A5"},
        "pass_configs": TILED_FA_BWD_PASS_CONFIGS,
        "fixed50": {
            "path": str(cases_path),
            "sha256": sha256(cases_path),
            "case_count": len(cases),
        },
        "dispatch": {
            "host_dispatcher_count": 1,
            "dtype_kernel_count": len(plan.variants),
            "symbolic_extents": list(plan.symbolic_extents),
            "variant_key": "dtype",
            "per_shape_kernels": 0,
        },
        "variants": variants,
        "remaining_gates": [
            "CANN_9_2_BISHENG_15_DAV3510_COMPILE_ONLY",
            "FRESH_NPU_FIXED50_PRECISION_AND_KNOWN_BAD",
            "CANONICAL_SAME_CANDIDATE_MSPROF_GE_1X",
        ],
    }

    # Build every dtype IR before entering the shared CombineCV pass.  A
    # compiler failure must retain the complete symbolic-family receipt while
    # refusing to claim that generated source exists.
    for variant in plan.variants:
        suffix = ascendc_dispatch.DTYPE_SUFFIXES[variant.dtype]
        try:
            artifact = lower_variant(
                variant.dtype,
                variant.host_entry,
                variant.kernel_symbol,
                kernel_path="tiled",
            )
        except Exception:
            error_text = traceback.format_exc()
            error_path = output / "LOWERING_ERROR.txt"
            error_path.write_text(error_text, encoding="utf-8")
            result.update(
                {
                    "status": "LOWERING_BLOCKED",
                    "authority": "DEVICE_FREE_ROUTE_A_IR_BOUNDARY_ONLY",
                    "lowering_boundary": {
                        "dtype": variant.dtype,
                        "classification": classify_lowering_failure(error_text),
                        "error_path": str(error_path.relative_to(output)),
                        "error_sha256": sha256(error_path),
                        "generated_source_admitted": False,
                        "unsafe_bypass_rejected": (
                            "Disabling tl.ascend_auto_cross_core_sync emits "
                            "source without the required Cube/Vector workspace "
                            "handoff and is not a product candidate."
                        ),
                    },
                }
            )
            result["remaining_gates"].insert(
                0, "MANUAL_EXPERT_CV_STATE_MACHINE_OR_COMPILER_CFG_SUPPORT"
            )
            result_path = output / "RESULT.json"
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            write_manifest(output)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2

        source_path = source_dir / f"fa_bwd_{suffix}.cpp"
        source_path.write_text(artifact.kernel_source, encoding="utf-8")
        guard = validate_tiled_source(
            artifact.kernel_source,
            variant.host_entry,
            variant.kernel_symbol,
            variant.dtype,
        )
        variants[variant.dtype].update(
            {
                "generated_source": {
                    "path": str(source_path.relative_to(output)),
                    "sha256": sha256(source_path),
                    "bytes": source_path.stat().st_size,
                },
                "guard": guard,
            }
        )

    result.update(
        {
            "status": "PASS",
            "authority": "DEVICE_FREE_ROUTE_A_IR_AND_SOURCE_ONLY",
        }
    )
    result_path = output / "RESULT.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_manifest(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
