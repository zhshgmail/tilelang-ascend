"""Focused device-free tests for the owned-tile generated-source validator."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from poc.run_fa_bwd_real_lowering import (
    CPP_DATA_TYPES,
    FA_BWD_BISHENG_COMPILE_FLAGS,
    FA_BWD_PASS_CONFIGS,
    FLAT_BUFFER_HANDLES,
    INPUT_COPY_CONTRACT,
    OUTPUT_FLAT_BUFFERS,
    SYMBOLIC_EXTENTS,
    _strip_cpp_comments_and_literals,
    validate_real_source,
)


def _entry_names(dtype: str) -> tuple[str, str]:
    return f"host_fa_bwd_{dtype}", f"kernel_fa_bwd_{dtype}"


def test_generator_enables_explicit_sync_and_disables_bisheng_auto_sync() -> None:
    assert FA_BWD_PASS_CONFIGS == {
        "tl.disable_safe_memory_legalize": True,
        "tl.ascend_auto_sync": False,
        "tl.ascend_auto_sync_vs": True,
        "tl.ascend_memory_planning": True,
        "tl.ascend_auto_cv_combine": True,
    }
    assert FA_BWD_BISHENG_COMPILE_FLAGS == ("-O3", "--cce-auto-sync=off")


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
    for buffer_name, (destination_name, copy_count) in INPUT_COPY_CONTRACT.items():
        buffer_type = "float" if buffer_name in {"max_flat", "sum_flat"} else data_type
        width = 8 if buffer_name in {"max_flat", "sum_flat"} else 32
        for _ in range(copy_count):
            lines.append(
                f"tl::ascend::copy_gm_to_ub<{buffer_type}, {width}>("
                f"{destination_name}[0], {buffer_name}[0], 32, 1, {width}, 0);"
            )
    lines.extend(
        [
            "AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);",
            "AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);",
            "scratch.SetValue(0, 0.000000e+00f);",
            "scratch.SetValue(0, 0.000000e+00f);",
            "scratch.SetValue(0, 0.000000e+00f);",
            "scratch.SetValue(1, scratch.GetValue(1));",
            "AscendC::Exp(exp_out[0], exp_in[0], 32);",
            "AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);",
            "AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);",
        ]
    )
    output_source = "acc_tile" if dtype == "float32" else "out_tile"
    for buffer_name in OUTPUT_FLAT_BUFFERS:
        if dtype != "float32":
            lines.append(
                "AscendC::Cast(out_tile[0], acc_tile[0], "
                "AscendC::RoundMode::CAST_RINT, 32);"
            )
            lines.extend(
                [
                    "AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);",
                    "AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);",
                ]
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
    assert result["flat_input_copy_multiplicities"] == {
        name: count for name, (_, count) in INPUT_COPY_CONTRACT.items()
    }
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


_ATTEMPT2_VARIANTS = {
    "float16": ("fa_bwd_fp16.cpp", "call_fa_bwd_fp16", "fa_bwd_fp16_kernel"),
    "bfloat16": ("fa_bwd_bf16.cpp", "call_fa_bwd_bf16", "fa_bwd_bf16_kernel"),
    "float32": ("fa_bwd_fp32.cpp", "call_fa_bwd_fp32", "fa_bwd_fp32_kernel"),
}


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_exact_attempt2_source_is_accepted_with_real_plan_entries(dtype: str) -> None:
    retained_root_value = os.environ.get("TILELANG_FA_BWD_ATTEMPT2_SOURCE_DIR")
    if retained_root_value is None:
        pytest.skip("set TILELANG_FA_BWD_ATTEMPT2_SOURCE_DIR for exact attempt2 gate")
    filename, host_entry, kernel_entry = _ATTEMPT2_VARIANTS[dtype]
    source = (Path(retained_root_value) / filename).read_text(encoding="utf-8")
    result = validate_real_source(source, host_entry, kernel_entry, dtype)
    assert result["gm_to_ub_copy_count"] == 19
    assert result["ub_to_gm_copy_count"] == 3
    assert result["flat_input_copy_multiplicities"] == {
        name: count for name, (_, count) in INPUT_COPY_CONTRACT.items()
    }


def test_injected_flat_global_read_and_write_are_rejected() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype) + (
        "q_flat.GetValue(0);\n"
        "dq_flat.SetValue(0, static_cast<half>(0));\n"
    )
    with pytest.raises(AssertionError, match="forbidden flat GlobalTensor scalar access"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        ("q_flat[0]", "q_flat[(0) + 1]", "invalid GM-to-UB source offset"),
        ("dq_flat[0]", "dq_flat[(0) + 1]", "invalid UB-to-GM target offset"),
        (
            "q_ub[0], q_flat[0], 32, 1, 32, 0",
            "q_ub[0], q_flat[0], 32, 1, 0, 0",
            "invalid GM-to-UB copy extent",
        ),
        (
            "q_ub[0], q_flat[0], 32, 1, 32, 0",
            "q_ub[0], q_flat[0], 32, 1, 33, 0",
            "invalid GM-to-UB copy extent",
        ),
        (
            "dq_flat[0], out_tile[0], 32, 1, 32",
            "dq_flat[0], out_tile[0], 32, 1, 0",
            "invalid UB-to-GM copy extent",
        ),
    ],
)
def test_copy_base_and_extent_mutations_are_rejected(
    needle: str, replacement: str, message: str
) -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    source = _valid_source(dtype)
    assert needle in source
    known_bad = source.replace(needle, replacement, 1)
    with pytest.raises(AssertionError, match=message):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_injected_dcci_is_rejected() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype) + (
        "AscendC::DataCacheCleanAndInvalid<half, "
        "AscendC::CacheLine::SINGLE_CACHE_LINE, "
        "AscendC::DcciDst::CACHELINE_OUT>(dq_flat[0]);\n"
    )
    with pytest.raises(AssertionError, match="forbidden DCCI"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_scalar_access_on_rogue_declared_global_tensor_is_rejected() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype) + (
        "AscendC::GlobalTensor<half> rogue_global;\n"
        "rogue_global.GetValue(0);\n"
    )
    with pytest.raises(AssertionError, match="declared GlobalTensor scalar access"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_missing_required_copy_dependency_edges_are_rejected() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = "\n".join(
        line
        for line in _valid_source(dtype).splitlines()
        if "SetFlag<" not in line and "WaitFlag<" not in line
    )
    with pytest.raises(AssertionError, match="dependency"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_unpaired_copy_dependency_event_is_rejected() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype).replace(
        "AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);", "", 1
    )
    with pytest.raises(AssertionError, match="unpaired copy dependency events"):
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


def test_extra_wrong_q_rebind_is_rejected() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype).replace(
        "q_flat.SetGlobalBuffer((__gm__ half*)q_handle);",
        "q_flat.SetGlobalBuffer((__gm__ half*)q_handle);\n"
        "q_flat.SetGlobalBuffer((__gm__ half*)k_handle);",
        1,
    )
    with pytest.raises(AssertionError, match="invalid flat GlobalTensor binding"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_q_copy_redistribution_is_rejected_with_same_total_and_participation() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype).replace(
        "q_ub[0], q_flat[0], 32, 1, 32, 0);",
        "k_ub[0], k_flat[0], 32, 1, 32, 0);",
        1,
    )
    assert known_bad.count("copy_gm_to_ub") == 19
    assert "q_flat[0]" in known_bad
    with pytest.raises(AssertionError, match="wrong GM-to-UB source multiplicities"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_real_rint_change_cannot_be_repaired_by_comment_token() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype).replace(
        "AscendC::RoundMode::CAST_RINT",
        "AscendC::RoundMode::CAST_NONE",
        1,
    ) + "// AscendC::RoundMode::CAST_RINT\n"
    with pytest.raises(AssertionError, match="invalid output cast arguments"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


@pytest.mark.parametrize(
    "spoof",
    [
        "/* AscendC::Exp */",
        'const char *spoof = "AscendC::Exp";',
        'const char *spoof = "escaped \\\" AscendC::Exp";',
        'const char *spoof = R"tag(AscendC::Exp)tag";',
    ],
)
def test_comment_or_string_cannot_supply_required_exp_token(spoof: str) -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype).replace(
        "AscendC::Exp(exp_out[0], exp_in[0], 32);", spoof, 1
    )
    with pytest.raises(AssertionError, match="missing real-lowering tokens"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_comments_cannot_supply_a_missing_input_copy() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    source = _valid_source(dtype)
    copy = (
        "tl::ascend::copy_gm_to_ub<half, 32>("
        "q_ub[0], q_flat[0], 32, 1, 32, 0);"
    )
    known_bad = source.replace(copy, f"/* {copy} */", 1)
    with pytest.raises(AssertionError, match="expected exactly 19 GM-to-UB copies"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_wrong_input_copy_dtype_is_rejected() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype).replace(
        "copy_gm_to_ub<half, 32>(q_ub[0], q_flat[0]",
        "copy_gm_to_ub<float, 32>(q_ub[0], q_flat[0]",
        1,
    )
    with pytest.raises(AssertionError, match="wrong GM-to-UB dtype/width"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_wrong_input_copy_destination_is_rejected() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype).replace(
        "q_ub[0], q_flat[0]", "k_ub[0], q_flat[0]", 1
    )
    with pytest.raises(AssertionError, match="wrong GM-to-UB destination"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_wrong_output_cast_width_is_rejected() -> None:
    dtype = "bfloat16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype).replace(
        "AscendC::RoundMode::CAST_RINT, 32);",
        "AscendC::RoundMode::CAST_RINT, 31);",
        1,
    )
    with pytest.raises(AssertionError, match="invalid output cast arguments"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_commented_wrong_rebind_is_not_executable() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    source = _valid_source(dtype) + (
        "/* q_flat.SetGlobalBuffer((__gm__ half*)k_handle); */\n"
    )
    validate_real_source(source, host_entry, kernel_entry, dtype)


@pytest.mark.parametrize(
    "replacement",
    [
        "q_flat.SetGlobalBuffer((__gm__ float*)q_handle);",
        "q_flat.SetGlobalBuffer((__gm__ half*)k_handle);",
    ],
)
def test_wrong_flat_binding_dtype_or_handle_is_rejected(replacement: str) -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype).replace(
        "q_flat.SetGlobalBuffer((__gm__ half*)q_handle);", replacement, 1
    )
    with pytest.raises(AssertionError, match="wrong dtype/cast/handle binding"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_unapproved_flat_alias_cannot_supply_an_input_copy() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype).replace("q_flat[0]", "rogue_flat[0]", 1)
    with pytest.raises(AssertionError, match="not an approved flat input"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_fp16_output_copy_cannot_bypass_cast_tile() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype).replace(
        "dq_flat[0], out_tile[0]", "dq_flat[0], acc_tile[0]", 1
    )
    with pytest.raises(AssertionError, match="wrong UB-to-GM source"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_lexer_blanks_chars_strings_comments_and_escaped_delimiters() -> None:
    source = r'''kept();
char slash = '/'; char quote = '\''; char backslash = '\\';
const char *text = "/* fake */ \\\" still string";
// q_flat.SetGlobalBuffer((__gm__ half*)k_handle);
/* tl::ascend::copy_gm_to_ub<half, 32>(q_ub[0], q_flat[0], 32, 1, 32, 0); */
tail();'''
    executable = _strip_cpp_comments_and_literals(source)
    assert "kept();" in executable
    assert "tail();" in executable
    assert "SetGlobalBuffer" not in executable
    assert "copy_gm_to_ub" not in executable
    assert executable.count("\n") == source.count("\n")


@pytest.mark.parametrize("suffix", ["/* unterminated", '"unterminated', "'x"])
def test_unterminated_lexical_construct_fails_closed(suffix: str) -> None:
    with pytest.raises(AssertionError, match="unterminated C\\+\\+"):
        _strip_cpp_comments_and_literals("kept(); " + suffix)


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


def test_a5_source_rejects_illegal_pipe_all_barrier() -> None:
    dtype = "float16"
    host_entry, kernel_entry = _entry_names(dtype)
    known_bad = _valid_source(dtype).replace(
        "scratch.SetValue(0, 0.000000e+00f);",
        "AscendC::PipeBarrier<PIPE_ALL>();\n"
        "scratch.SetValue(0, 0.000000e+00f);",
        1,
    )
    with pytest.raises(AssertionError, match="illegal A5 PIPE_ALL"):
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
    with pytest.raises(AssertionError, match="invalid output cast arguments"):
        validate_real_source(known_bad, host_entry, kernel_entry, dtype)


def test_local_ub_scalar_access_remains_allowed() -> None:
    dtype = "float32"
    host_entry, kernel_entry = _entry_names(dtype)
    source = _valid_source(dtype)
    assert "scratch.SetValue(1, scratch.GetValue(1));" in source
    validate_real_source(source, host_entry, kernel_entry, dtype)
