# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""AscendC symbolic-shape variant planning and host-dispatch generation.

This module deliberately has no TVM import so that the shape contract can be
validated before lowering. Numerical kernels are emitted separately by the
TileLang lowering pipeline; this module only owns variant planning and the
single public host dispatcher.

The first supported contract is BSHD FlashAttention backward. Tensor rank is
fixed while B/Sq/Sk/Hq/Hk/D are runtime extents. Dtype remains a compile-time
kernel property, so float16, bfloat16 and float32 require three variants.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Sequence


DTYPE_CODES = {"float16": 0, "bfloat16": 1, "float32": 2}
DTYPE_SUFFIXES = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}
EXPECTED_RANK_SIGNATURE = (4, 4, 4, 4, 4, 4, 4)
SYMBOLIC_EXTENTS = ("B", "Sq", "Sk", "Hq", "Hk", "D")
MAX_GENERATED_INDEX = (1 << 31) - 1


class AscendCSymbolicContractError(ValueError):
    """Raised when a case cannot share the fixed-rank symbolic ABI."""


@dataclass(frozen=True)
class FABwdCase:
    case_id: int
    B: int
    Sq: int
    Sk: int
    Hq: int
    Hk: int
    D: int
    dtype: str
    causal: bool
    window_left: int
    window_right: int
    softcap: float
    rank_signature: tuple[int, ...]

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> FABwdCase:
        rank_signature = tuple(int(x) for x in row["rank_signature"].split("x"))
        return cls(
            case_id=int(row["case_index"]),
            B=int(row["B"]),
            Sq=int(row["Sq"]),
            Sk=int(row["Sk"]),
            Hq=int(row["Hq"]),
            Hk=int(row["Hk"]),
            D=int(row["D"]),
            dtype=row["dtype"],
            causal=row["causal"].strip().lower() == "true",
            window_left=int(row["window_left"]),
            window_right=int(row["window_right"]),
            softcap=float(row["softcap"]),
            rank_signature=rank_signature,
        )

    def validate(self) -> None:
        if self.rank_signature != EXPECTED_RANK_SIGNATURE:
            raise AscendCSymbolicContractError(
                f"case {self.case_id}: rank signature {self.rank_signature} does not match {EXPECTED_RANK_SIGNATURE}"
            )
        if self.dtype not in DTYPE_CODES:
            raise AscendCSymbolicContractError(
                f"case {self.case_id}: unsupported dtype {self.dtype!r}"
            )
        extents = (self.B, self.Sq, self.Sk, self.Hq, self.Hk, self.D)
        if any(value <= 0 for value in extents):
            raise AscendCSymbolicContractError(
                f"case {self.case_id}: non-positive runtime extent {extents}"
            )
        if self.Hq % self.Hk != 0:
            raise AscendCSymbolicContractError(
                f"case {self.case_id}: Hq={self.Hq} is not divisible by Hk={self.Hk}"
            )
        # D is symbolic, while the current A5 vector/Cube contracts still
        # require an 8-element granularity. This is a runtime guard, not a
        # reason to clone the kernel per D value.
        if self.D % 8 != 0:
            raise AscendCSymbolicContractError(
                f"case {self.case_id}: D={self.D} violates runtime D%8 guard"
            )
        if self.window_left < -1 or self.window_right < -1:
            raise AscendCSymbolicContractError(
                f"case {self.case_id}: window bounds must be -1 or non-negative"
            )
        if not math.isfinite(self.softcap) or self.softcap < 0.0:
            raise AscendCSymbolicContractError(
                f"case {self.case_id}: softcap must be finite and non-negative"
            )
        tensor_elements = (
            self.B * self.Sq * self.Hq * self.D,
            self.B * self.Sk * self.Hk * self.D,
            self.B * self.Hq * self.Sq * 8,
        )
        if any(elements > MAX_GENERATED_INDEX for elements in tensor_elements):
            raise AscendCSymbolicContractError(
                f"case {self.case_id}: logical tensor size exceeds generated int32 "
                f"index domain: {tensor_elements}"
            )


@dataclass(frozen=True)
class AscendCVariant:
    dtype: str
    dispatch_key: int
    host_entry: str
    kernel_symbol: str
    case_ids: tuple[int, ...]


@dataclass(frozen=True)
class AscendCDispatchPlan:
    operator: str
    host_entry: str
    symbolic_extents: tuple[str, ...]
    fixed_rank_signature: tuple[int, ...]
    variants: tuple[AscendCVariant, ...]
    case_count: int
    naive_static_kernel_count: int
    a3_factory_kernel_count: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "operator": self.operator,
            "host_entry": self.host_entry,
            "symbolic_extents": list(self.symbolic_extents),
            "fixed_rank_signature": list(self.fixed_rank_signature),
            "variants": [asdict(v) for v in self.variants],
            "case_count": self.case_count,
            "naive_static_kernel_count": self.naive_static_kernel_count,
            "a3_factory_kernel_count": self.a3_factory_kernel_count,
            "poc_kernel_count": len(self.variants),
            "device_source_authority": "TILELANG_LOWERING_REQUIRED",
        }


def load_fa_bwd_cases(path: str | os.PathLike[str]) -> list[FABwdCase]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        cases = [FABwdCase.from_csv_row(row) for row in csv.DictReader(handle)]
    if not cases:
        raise AscendCSymbolicContractError("case set is empty")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise AscendCSymbolicContractError("duplicate case ids")
    for case in cases:
        case.validate()
    return cases


def plan_fa_bwd_dispatch(cases: Sequence[FABwdCase]) -> AscendCDispatchPlan:
    if not cases:
        raise AscendCSymbolicContractError("case set is empty")
    for case in cases:
        case.validate()

    variants = []
    for dtype, dispatch_key in sorted(DTYPE_CODES.items(), key=lambda item: item[1]):
        case_ids = tuple(case.case_id for case in cases if case.dtype == dtype)
        if not case_ids:
            continue
        suffix = DTYPE_SUFFIXES[dtype]
        variants.append(
            AscendCVariant(
                dtype=dtype,
                dispatch_key=dispatch_key,
                host_entry=f"call_fa_bwd_{suffix}",
                kernel_symbol=f"fa_bwd_{suffix}_kernel",
                case_ids=case_ids,
            )
        )

    # Reproduce the current A3 example's Python factory specialization:
    # dtype, D, Hq and Hk are factory args and therefore enter args_repr.
    a3_factory_keys = {(case.dtype, case.D, case.Hq, case.Hk) for case in cases}
    return AscendCDispatchPlan(
        operator="29_FlashAttentionBwd",
        host_entry="tilelang_fa_bwd_call",
        symbolic_extents=SYMBOLIC_EXTENTS,
        fixed_rank_signature=EXPECTED_RANK_SIGNATURE,
        variants=tuple(variants),
        case_count=len(cases),
        naive_static_kernel_count=len(cases),
        a3_factory_kernel_count=len(a3_factory_keys),
    )


_POINTER_ARGS = (
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
_TENSOR_SHAPES = (
    ("B", "Sq", "Hq", "D"),
    ("B", "Sk", "Hk", "D"),
    ("B", "Sk", "Hk", "D"),
    ("B", "Sq", "Hq", "D"),
    ("B", "Hq", "Sq", "8"),
    ("B", "Hq", "Sq", "8"),
    ("B", "Sq", "Hq", "D"),
    ("B", "Sq", "Hq", "D"),
    ("B", "Sk", "Hk", "D"),
    ("B", "Sk", "Hk", "D"),
)
_RUNTIME_ARGS = (
    ("int64_t", "B"),
    ("int64_t", "Sq"),
    ("int64_t", "Sk"),
    ("int64_t", "Hq"),
    ("int64_t", "Hk"),
    ("int64_t", "D"),
    ("int32_t", "causal"),
    ("int32_t", "window_left"),
    ("int32_t", "window_right"),
    ("float", "softcap"),
    ("float", "scale"),
)


def _argument_declarations(include_dtype: bool) -> list[str]:
    declarations = [f"uint8_t* {name}" for name in _POINTER_ARGS]
    declarations.append("const int64_t* tensor_strides")
    declarations.append("int32_t tensor_stride_count")
    declarations.extend(f"{ctype} {name}" for ctype, name in _RUNTIME_ARGS)
    if include_dtype:
        declarations.append("int32_t dtype_code")
    declarations.append("aclrtStream stream")
    return declarations


def _argument_names() -> list[str]:
    names = list(_POINTER_ARGS)
    names.extend(name for _, name in _RUNTIME_ARGS)
    names.append("stream")
    return names


def _wrapper_argument_declarations() -> list[str]:
    """Match the exact argument order emitted by AscendC host codegen."""

    declarations = [f"uint8_t* {name}" for name in _POINTER_ARGS]
    declarations.extend(
        [
            "int32_t causal",
            "int32_t window_left",
            "int32_t window_right",
            "float softcap",
            "float scale",
            "int64_t B",
            "int64_t Sq",
            "int64_t Hq",
            "int64_t D",
            "int64_t Sk",
            "int64_t Hk",
            "aclrtStream stream",
        ]
    )
    return declarations


def _wrapper_argument_names() -> list[str]:
    return [
        *_POINTER_ARGS,
        "causal",
        "window_left",
        "window_right",
        "softcap",
        "scale",
        "B",
        "Sq",
        "Hq",
        "D",
        "Sk",
        "Hk",
        "stream",
    ]


def render_dispatch_header(plan: AscendCDispatchPlan) -> str:
    del plan
    signature = ",\n    ".join(_argument_declarations(include_dtype=True))
    return f"""#pragma once
#include <cstdint>

using aclrtStream = void*;

enum TileLangFABwdStatus : int32_t {{
  TILELANG_FA_BWD_OK = 0,
  TILELANG_FA_BWD_INVALID_EXTENT = -2,
  TILELANG_FA_BWD_INVALID_SYMBOLIC_DOMAIN = -3,
  TILELANG_FA_BWD_UNSUPPORTED_DTYPE = -4,
  TILELANG_FA_BWD_INVALID_POINTER = -5,
  TILELANG_FA_BWD_NONCONTIGUOUS = -6,
  TILELANG_FA_BWD_SIZE_OVERFLOW = -7,
  TILELANG_FA_BWD_ALIAS_OVERLAP = -8,
}};

// tensor_strides contains 10 row-major element-stride vectors in this order:
// q, k, v, dy, softmax_max, softmax_sum, attention, dq, dk, dv.
// Each tensor has fixed rank four; therefore callers provide exactly 40 values.
enum : int32_t {{
  TILELANG_FA_BWD_TENSOR_COUNT = 10,
  TILELANG_FA_BWD_TENSOR_RANK = 4,
  TILELANG_FA_BWD_STRIDE_COUNT = 40,
}};

extern "C" int tilelang_fa_bwd_call(
    {signature});
"""


def _render_tensor_shapes() -> str:
    return ",\n".join(
        "    {" + ", ".join(shape) + "}" for shape in _TENSOR_SHAPES
    )


def _render_host_admission_support() -> str:
    return """namespace {

constexpr std::uintptr_t kRequiredAlignment = 32;
constexpr std::size_t kFirstOutput = 7;

struct ByteSpan {
  std::uintptr_t begin;
  std::uintptr_t end;
};

bool checked_mul(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t* result) {
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

int validate_contiguous_strides(
    const int64_t* strides,
    const int64_t* shape,
    std::uint64_t element_bytes,
    uint8_t* pointer,
    ByteSpan* span) {
  std::uint64_t elements = 1;
  for (std::size_t axis = 0; axis < TILELANG_FA_BWD_TENSOR_RANK; ++axis) {
    if (!checked_mul(elements, static_cast<std::uint64_t>(shape[axis]), &elements)) {
      return TILELANG_FA_BWD_SIZE_OVERFLOW;
    }
  }
  // The generated scalar loops and flattened lane indices are int32 today.
  // Admit only domains those generated indices can represent exactly.
  if (elements > static_cast<std::uint64_t>(
                     std::numeric_limits<int32_t>::max())) {
    return TILELANG_FA_BWD_SIZE_OVERFLOW;
  }
  std::uint64_t expected_stride = 1;
  for (std::size_t axis = TILELANG_FA_BWD_TENSOR_RANK; axis-- > 0;) {
    if (expected_stride > static_cast<std::uint64_t>(
                              std::numeric_limits<int64_t>::max())) {
      return TILELANG_FA_BWD_SIZE_OVERFLOW;
    }
    if (strides[axis] != static_cast<int64_t>(expected_stride)) {
      return TILELANG_FA_BWD_NONCONTIGUOUS;
    }
    if (!checked_mul(
            expected_stride, static_cast<std::uint64_t>(shape[axis]),
            &expected_stride)) {
      return TILELANG_FA_BWD_SIZE_OVERFLOW;
    }
  }
  std::uint64_t bytes = 0;
  if (!checked_mul(elements, element_bytes, &bytes)) {
    return TILELANG_FA_BWD_SIZE_OVERFLOW;
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    return TILELANG_FA_BWD_SIZE_OVERFLOW;
  }
  span->begin = begin;
  span->end = begin + static_cast<std::uintptr_t>(bytes);
  return TILELANG_FA_BWD_OK;
}

bool overlaps(const ByteSpan& lhs, const ByteSpan& rhs) {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

}  // namespace"""


def _render_host_admission_body() -> str:
    pointer_names = ", ".join(_POINTER_ARGS)
    shapes = _render_tensor_shapes()
    return f"""  if (B <= 0 || Sq <= 0 || Sk <= 0 || Hq <= 0 || Hk <= 0 || D <= 0) {{
    return TILELANG_FA_BWD_INVALID_EXTENT;
  }}
  if ((Hq % Hk) != 0 || (D % 8) != 0 || (causal != 0 && causal != 1) ||
      window_left < -1 || window_right < -1 || !std::isfinite(softcap) ||
      softcap < 0.0f || !std::isfinite(scale) || scale <= 0.0f) {{
    return TILELANG_FA_BWD_INVALID_SYMBOLIC_DOMAIN;
  }}
  if (dtype_code < 0 || dtype_code > 2) {{
    return TILELANG_FA_BWD_UNSUPPORTED_DTYPE;
  }}
  if (tensor_strides == nullptr) {{
    return TILELANG_FA_BWD_INVALID_POINTER;
  }}
  if (tensor_stride_count != TILELANG_FA_BWD_STRIDE_COUNT) {{
    return TILELANG_FA_BWD_NONCONTIGUOUS;
  }}

  uint8_t* tensors[TILELANG_FA_BWD_TENSOR_COUNT] = {{{pointer_names}}};
  for (uint8_t* pointer : tensors) {{
    if (pointer == nullptr ||
        (reinterpret_cast<std::uintptr_t>(pointer) & (kRequiredAlignment - 1)) != 0) {{
      return TILELANG_FA_BWD_INVALID_POINTER;
    }}
  }}

  const int64_t shapes[TILELANG_FA_BWD_TENSOR_COUNT]
                      [TILELANG_FA_BWD_TENSOR_RANK] = {{
{shapes}
  }};
  const std::uint64_t value_bytes = dtype_code == 2 ? 4 : 2;
  const std::uint64_t element_bytes[TILELANG_FA_BWD_TENSOR_COUNT] = {{
      value_bytes, value_bytes, value_bytes, value_bytes, 4,
      4, value_bytes, value_bytes, value_bytes, value_bytes}};
  ByteSpan spans[TILELANG_FA_BWD_TENSOR_COUNT] = {{}};
  for (std::size_t tensor = 0; tensor < TILELANG_FA_BWD_TENSOR_COUNT; ++tensor) {{
    const int status = validate_contiguous_strides(
        tensor_strides + tensor * TILELANG_FA_BWD_TENSOR_RANK,
        shapes[tensor], element_bytes[tensor], tensors[tensor], &spans[tensor]);
    if (status != TILELANG_FA_BWD_OK) return status;
  }}
  for (std::size_t output = kFirstOutput;
       output < TILELANG_FA_BWD_TENSOR_COUNT; ++output) {{
    for (std::size_t other = 0; other < output; ++other) {{
      if (overlaps(spans[output], spans[other])) {{
        return TILELANG_FA_BWD_ALIAS_OVERLAP;
      }}
    }}
  }}"""


def render_dispatch_source(plan: AscendCDispatchPlan) -> str:
    wrapper_signature = ", ".join(_wrapper_argument_declarations())
    declarations = "\n".join(
        f'extern "C" void {variant.host_entry}({wrapper_signature});'
        for variant in plan.variants
    )
    call_args = ", ".join(_wrapper_argument_names())
    cases = "\n".join(
        f"    case {variant.dispatch_key}: {variant.host_entry}({call_args}); "
        "return TILELANG_FA_BWD_OK;"
        for variant in plan.variants
    )
    signature = ", ".join(_argument_declarations(include_dtype=True))
    return f"""#include "fa_bwd_dispatch.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

{declarations}

{_render_host_admission_support()}

extern "C" int tilelang_fa_bwd_call(
    {signature}) {{
{_render_host_admission_body()}

  switch (dtype_code) {{
{cases}
    default: return TILELANG_FA_BWD_UNSUPPORTED_DTYPE;
  }}
}}
"""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_dispatch_bundle(
    cases: Sequence[FABwdCase], output_dir: str | os.PathLike[str]
) -> AscendCDispatchPlan:
    plan = plan_fa_bwd_dispatch(cases)
    output = Path(output_dir)
    _write_text(output / "host" / "fa_bwd_dispatch.hpp", render_dispatch_header(plan))
    _write_text(output / "host" / "fa_bwd_dispatch.cpp", render_dispatch_source(plan))
    _write_text(
        output / "variant_plan.json",
        json.dumps(plan.to_json_dict(), indent=2, sort_keys=True) + "\n",
    )
    return plan
