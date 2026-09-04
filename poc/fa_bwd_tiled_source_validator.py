"""Fail-closed structural checks for generated tiled FA-Bwd AscendC source."""

from __future__ import annotations

import re


def validate_tiled_source(
    source: str,
    host_entry: str,
    kernel_entry: str,
    dtype: str,
) -> dict[str, object]:
    """Validate identity and Cube+Vector structure without claiming NPU proof."""

    required = {
        "kernel_entry": kernel_entry,
        "host_entry": host_entry,
        "runtime_B": "int64_t B",
        "runtime_Sq": "int64_t Sq",
        "runtime_Sk": "int64_t Sk",
        "runtime_Hq": "int64_t Hq",
        "runtime_Hk": "int64_t Hk",
        "runtime_D": "int64_t D",
    }
    missing = [name for name, token in required.items() if token not in source]
    if missing:
        raise AssertionError(f"{dtype}: missing tiled source identity: {missing}")

    forbidden = [
        token
        for token in (
            "case_index",
            "fixed50",
            "aclnn",
            "torch",
            "DataCacheCleanAndInvalid",
        )
        if token in source
    ]
    if forbidden:
        raise AssertionError(f"{dtype}: forbidden tiled source tokens: {forbidden}")

    cube_tokens = re.findall(r"(?:Mmad|Gemm|gemm)", source, flags=re.IGNORECASE)
    vector_tokens = re.findall(
        r"(?:Exp|Mul|Sub|Div|Select|ReduceSum)", source, flags=re.IGNORECASE
    )
    if len(cube_tokens) < 5:
        raise AssertionError(
            f"{dtype}: expected five Cube GEMM roles, saw {len(cube_tokens)}"
        )
    if not vector_tokens:
        raise AssertionError(f"{dtype}: no Vector probability/dS operations found")
    if "AtomicAdd" in source:
        raise AssertionError(f"{dtype}: output ownership regressed to atomics")

    return {
        "authority": "DEVICE_FREE_GENERATED_SOURCE_ONLY",
        "kernel_path": "symbolic_tiled_cube_vector",
        "dtype": dtype,
        "host_entry": host_entry,
        "kernel_entry": kernel_entry,
        "cube_operation_token_count": len(cube_tokens),
        "vector_operation_token_count": len(vector_tokens),
        "runtime_symbol_count": 6,
        "case_specialization_absent": True,
        "atomic_output_dependency_absent": True,
    }
