"""Focused device-free tests for the owned-tile generated-source validator."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from poc.run_fa_bwd_real_lowering import (
    CPP_DATA_TYPES,
    FLAT_BUFFER_HANDLES,
    INPUT_FLAT_BUFFERS,
    OUTPUT_FLAT_BUFFERS,
    SYMBOLIC_EXTENTS,
    validate_real_source,
)


def _entry_names(dtype: str) -> tuple[str, str]:
    return f"host_fa_bwd_{dtype}", f"kernel_fa_bwd_{dtype}"


def _valid_source(dtype: str) -> str:
    """Build the smallest source text satisfying the generated-source contract."""

    data_type = CPP_DATA_TYPES[dtype]
    host_entry, kernel_entry = _entry_names(dtype)
    extent_args = ", ".join(f"int64_t {name}" for name in SYMBOLIC_EXTENTS)
    lines = [
        f'extern "C" __global__ __aicore__ void {kernel_entry}({extent_args}) {{',
    ]
    for buffer_name, handle_name in FLAT_BUFFER_HANDLES.items():
        buffer_type = "float" if buffer_name in {"max_flat", "sum_flat"} else data_type
        lines.extend(
            [
                f"AscendC::GlobalTensor<{buffer_type}> {buffer_name};",
                f"{buffer_name}.SetGlobalBuffer((__gm__ {buffer_type}*){handle_name});",
            ]
        )
    for buffer_name in (*INPUT_FLAT_BUFFERS, *("q_flat",) * 12):
        buffer_type = "float" if buffer_name in {"max_flat", "sum_flat"} else data_type
        lines.append(
            f"tl::ascend::copy_gm_to_ub<{buffer_type}, 32>("
            f"input_ub[0], {buffer_name}[0], 32, 1, 32, 0);"
        )
    lines.extend(
        [
            "scratch.SetValue(0, 0.000000e+00f);",
            "scratch.SetValue(0, 0.000000e+00f);",
            "scratch.SetValue(0, 0.000000e+00f);",
            "scratch.SetValue(1, scratch.GetValue(1));",
            "AscendC::Exp(exp_out[0], exp_in[0], 32);",
        ]
    )
    output_source = "acc_tile" if dtype == "float32" else "out_tile"
    for buffer_name in OUTPUT_FLAT_BUFFERS:
        if dtype != "float32":
            lines.append(
                "AscendC::Cast(out_tile[0], acc_tile[0], "
                "AscendC::RoundMode::CAST_RINT, 32);"
            )
        lines.append(
            f"tl::ascend::copy_ub_to_gm<{data_type}, 32>("
            f"{buffer_name}[0], {output_source}[0], 32, 1, 32);"
        )
    lines.extend(
        [
            "}",
            f'extern "C" void {host_entry}({extent_args}) {{',
            f"{kernel_entry}<<<24, nullptr, stream>>>();",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_valid_owned_tile_contract_is_accepted(dtype: str) -> None:
    host_entry, kernel_entry = _entry_names(dtype)
    result = validate_real_source(_valid_source(dtype), host_entry, kernel_entry, dtype)
    assert result["gm_to_ub_copy_count"] == 19
    assert result["ub_to_gm_copy_count"] == 3
    assert result["runtime_extent_count"] == len(SYMBOLIC_EXTENTS)
    assert result["flat_global_tensor_bindings"] == FLAT_BUFFER_HANDLES
    assert result["cast_rint_count"] == (0 if dtype == "float32" else 3)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_retained_generated_source_is_accepted(dtype: str) -> None:
    retained_root_value = os.environ.get("TILELANG_FA_BWD_RETAINED_SOURCE_DIR")
    if retained_root_value is None:
        pytest.skip("set TILELANG_FA_BWD_RETAINED_SOURCE_DIR for retained-source gate")
    retained_root = Path(retained_root_value)
    source = (retained_root / f"fa_bwd_owned_tile_{dtype}.cpp").read_text(
        encoding="utf-8"
    )
    host_entry, kernel_entry = _entry_names(dtype)
    result = validate_real_source(source, host_entry, kernel_entry, dtype)
    assert result["gm_to_ub_copy_count"] == 19
    assert result["ub_to_gm_copy_count"] == 3


def test_injected_flat_global_read_and_write_are_rejected() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype) + (
        "q_flat.GetValue(0);\n"
        "dq_flat.SetValue(0, static_cast<half>(0));\n"
    )
    with pytest.raises(AssertionError, match="forbidden flat GlobalTensor scalar access"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_removed_output_copy_is_rejected() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    source = _valid_source(dtype)
    known_bad = "\n".join(
        line
        for line in source.splitlines()
        if not ("copy_ub_to_gm" in line and "dq_flat" in line)
    )
    with pytest.raises(AssertionError, match="expected exactly 3 UB-to-GM copies"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_obsolete_parent_scalar_pattern_is_rejected() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype) + (
        "q.GetValue(0);\n"
        "k.GetValue(0);\n"
        "dq.SetValue(0, static_cast<half>(0));\n"
    )
    with pytest.raises(AssertionError, match="obsolete parent global scalar pattern"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda source: source + "\n void main_kernel();\n", "generic device entry"),
        (lambda source: source + "\nABI_ONLY\n", "ABI sentinel"),
        (
            lambda source: source.replace(
                "scratch.SetValue(0, 0.000000e+00f);", "", 1
            ),
            "accumulators are not reset",
        ),
        (lambda source: source + "\nfloat dq_acc;\n", "lifted out of its scope"),
    ],
)
def test_existing_negative_contracts_remain_enforced(mutation, message: str) -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    with pytest.raises(AssertionError, match=message):
        validate_real_source(mutation(_valid_source(dtype)), host_entry, kernel_entry, dtype)


def test_wrong_dtype_cast_contract_is_rejected() -> None:
    dtype = "bfloat16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype).replace(
        "AscendC::RoundMode::CAST_RINT", "AscendC::RoundMode::CAST_NONE", 1
    )
    with pytest.raises(AssertionError, match="CAST_RINT output conversions"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_local_ub_scalar_access_remains_allowed() -> None:
    dtype = "float32"
    host_entry, kernel_entry = _entry_names(dtype)
    source = _valid_source(dtype)
    assert "scratch.SetValue(1, scratch.GetValue(1));" in source
    validate_real_source(source, host_entry, kernel_entry, dtype)
