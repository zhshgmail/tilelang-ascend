# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CPU-only tests for the shared final-TIR diagnostic identity."""

from tilelang.pre_codegen_identity import capture_final_tir_identity


class _Target:
    model = "ascendc"
    mcpu = "dav-3510"

    def __str__(self) -> str:
        return "llvm -model=ascendc -mcpu=dav-3510"


def test_final_tir_identity_is_stable_and_has_no_verdict() -> None:
    module = object()
    serializer = lambda value: "same-final-tir" if value is module else "other"

    first = capture_final_tir_identity(
        module,
        target=_Target(),
        platform="A5",
        serializer=serializer,
    )
    second = capture_final_tir_identity(
        module,
        target=_Target(),
        platform="a5",
        serializer=serializer,
    )

    assert first == second
    assert first.authority == "SIMULATOR_DIAGNOSTIC"
    assert first.platform == "A5"
    assert first.target_mcpu == "dav-3510"
    assert len(first.final_tir_sha256) == 64
    assert "verdict" not in first.to_dict()
    assert "pass" not in first.to_dict()


def test_final_tir_identity_changes_with_exact_final_tir_bytes() -> None:
    first = capture_final_tir_identity(
        "module-a",
        target=_Target(),
        platform="A5",
        serializer=lambda value: value,
    )
    second = capture_final_tir_identity(
        "module-b",
        target=_Target(),
        platform="A5",
        serializer=lambda value: value,
    )

    assert first.final_tir_sha256 != second.final_tir_sha256


def test_default_identity_uses_stable_metadata_complete_script() -> None:
    class _ScriptableModule:
        def script(self, *, show_meta: bool) -> str:
            assert show_meta is True
            return "stable-final-tir"

    identity = capture_final_tir_identity(
        _ScriptableModule(), target=_Target(), platform="A5"
    )

    assert identity.schema_version == "tilelang.final-tir-identity.v2"
    assert identity.serialization == "tvm.script(show_meta=True):utf-8"
