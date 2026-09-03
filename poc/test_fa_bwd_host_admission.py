"""Device-free tests for the generated FA-Bwd host admission contract."""

from __future__ import annotations

import ctypes
import dataclasses
import importlib.util
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
CASE_CSV = REPO / "poc/inputs/op29_fa_bwd_fixed50_shapes.csv"
POINTER_NAMES = (
    "q",
    "k",
    "v",
    "dy",
    "softmax_max",
    "softmax_sum",
    "attention",
    "dq",
    "dk",
    "dv",
)


def _load_dispatch_module():
    path = REPO / "tilelang/jit/adapter/ascendc_dispatch.py"
    spec = importlib.util.spec_from_file_location("ascendc_dispatch_host_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _contiguous_strides(case) -> list[int]:
    shapes = (
        (case.B, case.Sq, case.Hq, case.D),
        (case.B, case.Sk, case.Hk, case.D),
        (case.B, case.Sk, case.Hk, case.D),
        (case.B, case.Sq, case.Hq, case.D),
        (case.B, case.Hq, case.Sq, 8),
        (case.B, case.Hq, case.Sq, 8),
        (case.B, case.Sq, case.Hq, case.D),
        (case.B, case.Sq, case.Hq, case.D),
        (case.B, case.Sk, case.Hk, case.D),
        (case.B, case.Sk, case.Hk, case.D),
    )
    result = []
    for shape in shapes:
        stride = 1
        tensor_strides = []
        for extent in reversed(shape):
            tensor_strides.append(stride)
            stride *= extent
        result.extend(reversed(tensor_strides))
    return result


def _mock_wrappers(module, plan) -> str:
    signature = ", ".join(module._wrapper_argument_declarations())
    definitions = []
    for variant in plan.variants:
        definitions.append(
            f'extern "C" void {variant.host_entry}({signature}) '
            f"{{ g_last_key = {variant.dispatch_key}; }}"
        )
    return (
        '#include "fa_bwd_dispatch.hpp"\n'
        'extern "C" { int g_last_key = -1; }\n'
        + "\n".join(definitions)
        + '\nextern "C" int fa_bwd_last_key() { return g_last_key; }\n'
    )


@pytest.fixture(scope="module")
def host_dispatcher(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler is unavailable")
    module = _load_dispatch_module()
    cases = module.load_fa_bwd_cases(CASE_CSV)
    root = tmp_path_factory.mktemp("fa_bwd_host_admission")
    generated = root / "generated"
    plan = module.write_dispatch_bundle(cases, generated)
    header = (generated / "host/fa_bwd_dispatch.hpp").read_text(encoding="utf-8")
    source = (generated / "host/fa_bwd_dispatch.cpp").read_text(encoding="utf-8")
    assert tuple(module._POINTER_ARGS) == POINTER_NAMES
    assert "const int64_t* tensor_strides" in header
    assert "int32_t tensor_stride_count" in header
    assert "TILELANG_FA_BWD_STRIDE_COUNT = 40" in header
    assert "validate_contiguous_strides" in source
    assert "overlaps(" in source
    mock = root / "mock_wrappers.cpp"
    mock.write_text(_mock_wrappers(module, plan), encoding="utf-8")
    library_path = root / "libfa_bwd_host_test.so"
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wno-unused-parameter",
        "-fPIC",
        "--shared",
        f"-I{generated / 'host'}",
        str(generated / "host/fa_bwd_dispatch.cpp"),
        str(mock),
        "-o",
        str(library_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    library = ctypes.CDLL(str(library_path))
    call = library.tilelang_fa_bwd_call
    call.argtypes = [
        *([ctypes.c_void_p] * 10),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32,
        *([ctypes.c_int64] * 6),
        *([ctypes.c_int32] * 3),
        *([ctypes.c_float] * 2),
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    call.restype = ctypes.c_int
    library.fa_bwd_last_key.argtypes = []
    library.fa_bwd_last_key.restype = ctypes.c_int
    return module, cases, call, library.fa_bwd_last_key


def _invoke(
    call,
    module,
    case,
    *,
    pointers=None,
    strides=None,
    stride_count=40,
    null_stride_pointer=False,
    **overrides,
):
    if pointers is None:
        pointers = [0x1_0000_0000 + index * 0x1000_0000 for index in range(10)]
    if strides is None:
        strides = _contiguous_strides(case)
    stride_array = None
    if not null_stride_pointer:
        stride_array = (ctypes.c_int64 * 40)(*strides)
    values = {
        "B": case.B,
        "Sq": case.Sq,
        "Sk": case.Sk,
        "Hq": case.Hq,
        "Hk": case.Hk,
        "D": case.D,
        "causal": int(case.causal),
        "window_left": case.window_left,
        "window_right": case.window_right,
        "softcap": case.softcap,
        "scale": 1.0,
        "dtype_code": module.DTYPE_CODES[case.dtype],
    }
    values.update(overrides)
    return call(
        *[ctypes.c_void_p(pointer) for pointer in pointers],
        stride_array,
        stride_count,
        values["B"],
        values["Sq"],
        values["Sk"],
        values["Hq"],
        values["Hk"],
        values["D"],
        values["causal"],
        values["window_left"],
        values["window_right"],
        values["softcap"],
        values["scale"],
        values["dtype_code"],
        None,
    )


def test_generated_dispatcher_accepts_all_fixed50_cases(host_dispatcher) -> None:
    module, cases, call, last_key = host_dispatcher
    assert len(cases) == 50
    for case in cases:
        assert _invoke(call, module, case) == 0
        assert last_key() == module.DTYPE_CODES[case.dtype]


@pytest.mark.parametrize("pointer", [None, 0x1_0000_0001])
def test_null_or_unaligned_pointer_is_rejected(host_dispatcher, pointer) -> None:
    module, cases, call, _ = host_dispatcher
    pointers = [0x1_0000_0000 + index * 0x1000_0000 for index in range(10)]
    pointers[0] = pointer
    assert _invoke(call, module, cases[0], pointers=pointers) == -5


def test_noncontiguous_stride_is_rejected(host_dispatcher) -> None:
    module, cases, call, _ = host_dispatcher
    strides = _contiguous_strides(cases[0])
    strides[0] += 1
    assert _invoke(call, module, cases[0], strides=strides) == -6


def test_wrong_stride_metadata_count_is_rejected(host_dispatcher) -> None:
    module, cases, call, _ = host_dispatcher
    assert _invoke(call, module, cases[0], stride_count=39) == -6


def test_null_stride_metadata_pointer_is_rejected(host_dispatcher) -> None:
    module, cases, call, _ = host_dispatcher
    assert _invoke(call, module, cases[0], null_stride_pointer=True) == -5


@pytest.mark.parametrize(
    "overrides",
    [
        {"B": 0},
        {"Hq": 3, "Hk": 2},
        {"D": 17},
        {"causal": 2},
        {"window_left": -2},
        {"window_right": -2},
        {"softcap": math.nan},
        {"scale": 0.0},
        {"scale": math.inf},
    ],
)
def test_illegal_runtime_symbolic_domain_is_rejected(
    host_dispatcher, overrides
) -> None:
    module, cases, call, _ = host_dispatcher
    expected = -2 if overrides == {"B": 0} else -3
    assert _invoke(call, module, cases[0], **overrides) == expected


def test_unsupported_dtype_is_rejected_before_layout_use(host_dispatcher) -> None:
    module, cases, call, _ = host_dispatcher
    assert _invoke(call, module, cases[0], dtype_code=99) == -4


def test_runtime_size_overflow_is_rejected(host_dispatcher) -> None:
    module, cases, call, _ = host_dispatcher
    assert (
        _invoke(
            call,
            module,
            cases[0],
            B=(1 << 62),
            Sq=8,
            Sk=8,
            Hq=2,
            Hk=1,
            D=16,
        )
        == -7
    )


def test_int32_generated_index_domain_overflow_is_rejected(host_dispatcher) -> None:
    module, cases, call, _ = host_dispatcher
    assert (
        _invoke(
            call,
            module,
            cases[0],
            B=(1 << 28),
            Sq=1,
            Sk=1,
            Hq=1,
            Hk=1,
            D=8,
        )
        == -7
    )


def test_pointer_span_address_overflow_is_rejected(host_dispatcher) -> None:
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        pytest.skip("64-bit pointer-range control")
    module, cases, call, _ = host_dispatcher
    pointers = [0x1_0000_0000 + index * 0x1000_0000 for index in range(10)]
    pointers[0] = (1 << 64) - 32
    assert _invoke(call, module, cases[0], pointers=pointers) == -7


@pytest.mark.parametrize("partial", [False, True])
def test_output_input_alias_or_overlap_is_rejected(
    host_dispatcher, partial: bool
) -> None:
    module, cases, call, _ = host_dispatcher
    pointers = [0x1_0000_0000 + index * 0x1000_0000 for index in range(10)]
    pointers[7] = pointers[0] + (32 if partial else 0)
    assert _invoke(call, module, cases[0], pointers=pointers) == -8


def test_output_output_overlap_is_rejected(host_dispatcher) -> None:
    module, cases, call, _ = host_dispatcher
    pointers = [0x1_0000_0000 + index * 0x1000_0000 for index in range(10)]
    pointers[8] = pointers[7]
    assert _invoke(call, module, cases[0], pointers=pointers) == -8


def test_dispatch_product_stays_single_host_three_dtype_kernels() -> None:
    module = _load_dispatch_module()
    cases = module.load_fa_bwd_cases(CASE_CSV)
    plan = module.plan_fa_bwd_dispatch(cases)
    assert plan.host_entry == "tilelang_fa_bwd_call"
    assert len(plan.variants) == 3
    assert {variant.dtype for variant in plan.variants} == {
        "float16",
        "bfloat16",
        "float32",
    }
    assert plan.case_count == 50
    assert sorted(case_id for variant in plan.variants for case_id in variant.case_ids) == list(
        range(50)
    )


@pytest.mark.parametrize(
    "replacement",
    [
        {"window_left": -2},
        {"window_right": -2},
        {"softcap": math.nan},
        {"B": 1 << 28, "Sq": 1, "Sk": 1, "Hq": 1, "Hk": 1, "D": 8},
    ],
)
def test_static_planner_rejects_the_same_unsupported_shape_domain(replacement) -> None:
    module = _load_dispatch_module()
    case = module.load_fa_bwd_cases(CASE_CSV)[0]
    with pytest.raises(module.AscendCSymbolicContractError):
        dataclasses.replace(case, **replacement).validate()
