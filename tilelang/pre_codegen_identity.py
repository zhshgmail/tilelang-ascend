# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Diagnostic identity for the final Ascend TIR consumed by code generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Callable, Mapping, Optional


SIMULATOR_EVIDENCE_AUTHORITY = "SIMULATOR_DIAGNOSTIC"


@dataclass(frozen=True)
class FinalTirIdentity:
    """Identity of one final, optimized, pre-device-codegen TIR module.

    This record is evidence only. It deliberately contains no pass/fail field
    and must not be promoted to precision, performance, or product authority.
    """

    schema_version: str
    authority: str
    final_tir_sha256: str
    serialization: str
    platform: str
    target_model: str
    target_mcpu: str
    target_repr: str

    def to_dict(self) -> Mapping[str, str]:
        """Return a stable JSON-compatible representation."""
        return asdict(self)


def capture_final_tir_identity(
    optimized_mod: Any,
    *,
    target: Any,
    platform: str,
    serializer: Optional[Callable[[Any], str]] = None,
) -> FinalTirIdentity:
    """Hash the exact IRModule handed to native code generation or simulation."""
    if serializer is None:
        from tilelang import tvm

        serializer = tvm.ir.save_json
        serialization = "tvm.ir.save_json:utf-8"
    else:
        serialization = "caller-supplied:str:utf-8"

    serialized = serializer(optimized_mod)
    if not isinstance(serialized, str):
        raise TypeError("final TIR serializer must return str")
    payload = serialized.encode("utf-8")
    return FinalTirIdentity(
        schema_version="tilelang.final-tir-identity.v1",
        authority=SIMULATOR_EVIDENCE_AUTHORITY,
        final_tir_sha256=sha256(payload).hexdigest(),
        serialization=serialization,
        platform=str(platform).strip().upper(),
        target_model=str(getattr(target, "model", "") or ""),
        target_mcpu=str(getattr(target, "mcpu", "") or ""),
        target_repr=str(target),
    )
