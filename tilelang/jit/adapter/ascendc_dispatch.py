# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""AscendC symbolic-shape variant planning and host-dispatch generation.

This module deliberately has no TVM import so that the shape contract can be
validated before lowering. The actual kernels are still emitted by TileLang;
the generated device sources are ABI sentinels, not numerical implementations.

The first supported contract is BSHD FlashAttention backward. Tensor rank is
fixed while B/Sq/Sk/Hq/Hk/D are runtime extents. Dtype remains a compile-time
kernel property, so float16, bfloat16 and float32 require three variants.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Sequence


DTYPE_CODES = {"float16": 0, "bfloat16": 1, "float32": 2}
DTYPE_SUFFIXES = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}
EXPECTED_RANK_SIGNATURE = (4, 4, 4, 4, 4, 4, 4)
SYMBOLIC_EXTENTS = ("B", "Sq", "Sk", "Hq", "Hk", "D")


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
            raise AscendCSymbolicContractError(f"case {self.case_id}: unsupported dtype {self.dtype!r}")
        extents = (self.B, self.Sq, self.Sk, self.Hq, self.Hk, self.D)
        if any(value <= 0 for value in extents):
            raise AscendCSymbolicContractError(f"case {self.case_id}: non-positive runtime extent {extents}")
        if self.Hq % self.Hk != 0:
            raise AscendCSymbolicContractError(f"case {self.case_id}: Hq={self.Hq} is not divisible by Hk={self.Hk}")
        # D is symbolic, while the current A5 vector/Cube contracts still
        # require an 8-element granularity. This is a runtime guard, not a
        # reason to clone the kernel per D value.
        if self.D % 8 != 0:
            raise AscendCSymbolicContractError(f"case {self.case_id}: D={self.D} violates runtime D%8 guard")


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
            "device_source_authority": "ABI_ONLY_NON_NUMERICAL",
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


_POINTER_ARGS = ("q", "k", "v", "dy", "attention", "dq", "dk", "dv")
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


def render_dispatch_header(plan: AscendCDispatchPlan) -> str:
    del plan
    signature = ",\n    ".join(_argument_declarations(include_dtype=True))
    return f"""#pragma once
#include <cstdint>

using aclrtStream = void*;

extern "C" int tilelang_fa_bwd_call(
    {signature});
"""


def render_dispatch_source(plan: AscendCDispatchPlan) -> str:
    wrapper_signature = ", ".join(_argument_declarations(include_dtype=False))
    declarations = "\n".join(f'extern "C" void {variant.host_entry}({wrapper_signature});' for variant in plan.variants)
    call_args = ", ".join(_argument_names())
    cases = "\n".join(f"    case {variant.dispatch_key}: {variant.host_entry}({call_args}); return 0;" for variant in plan.variants)
    return f"""#include "fa_bwd_dispatch.hpp"

{declarations}

extern "C" int tilelang_fa_bwd_call(
    {", ".join(_argument_declarations(include_dtype=True))}) {{
  if (B <= 0 || Sq <= 0 || Sk <= 0 || Hq <= 0 || Hk <= 0 || D <= 0) return -2;
  if ((Hq % Hk) != 0 || (D % 8) != 0) return -3;
  switch (dtype_code) {{
{cases}
    default: return -4;
  }}
}}
"""


def render_device_abi_source(variant: AscendCVariant) -> str:
    # This source is intentionally an ABI sentinel. Numerical FA backward
    # bodies must come from TileLang lowering before a product claim is valid.
    return f"""#include "kernel_operator.h"

extern "C" __global__ __aicore__ void {variant.kernel_symbol}(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR dy, GM_ADDR attention,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
    int64_t B, int64_t Sq, int64_t Sk, int64_t Hq, int64_t Hk, int64_t D,
    int32_t causal, int32_t window_left, int32_t window_right,
    float softcap, float scale, uint64_t fftsAddr) {{
  // ABI-only discriminator: all extents are runtime scalars. This kernel is
  // deliberately non-numerical and must fail any precision known-bad gate.
  if (B <= 0 || Sq <= 0 || Sk <= 0 || Hq <= 0 || Hk <= 0 || D <= 0) return;
}}
"""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_dispatch_bundle(cases: Sequence[FABwdCase], output_dir: str | os.PathLike[str]) -> AscendCDispatchPlan:
    plan = plan_fa_bwd_dispatch(cases)
    output = Path(output_dir)
    _write_text(output / "host" / "fa_bwd_dispatch.hpp", render_dispatch_header(plan))
    _write_text(output / "host" / "fa_bwd_dispatch.cpp", render_dispatch_source(plan))
    for variant in plan.variants:
        _write_text(
            output / "kernel" / f"{variant.kernel_symbol}.cpp",
            render_device_abi_source(variant),
        )
    _write_text(
        output / "variant_plan.json",
        json.dumps(plan.to_json_dict(), indent=2, sort_keys=True) + "\n",
    )
    return plan
