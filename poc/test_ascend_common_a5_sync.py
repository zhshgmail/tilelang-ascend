"""DAV3510 compile contracts for the shared Ascend C helper header."""

from __future__ import annotations

from pathlib import Path


COMMON_HEADER = (
    Path(__file__).parents[1] / "src" / "tl_templates" / "ascend" / "common.h"
)


def test_dav3510_common_header_uses_explicit_per_pipe_drain() -> None:
    source = COMMON_HEADER.read_text(encoding="utf-8")

    # CANN 9.2 Bisheng rejects PIPE_ALL for DAV3510 when auto-sync is disabled.
    # Keep the legacy spelling only in the non-3510 preprocessor branch, and
    # route every former shared-header call site through the compatibility
    # helper so generated kernels remain source-product compilable.
    assert "CATLASS_DEVICE void pipe_barrier_all_compat()" in source
    assert source.count("AscendC::PipeBarrier<PIPE_ALL>();") == 1
    assert source.count("pipe_barrier_all_compat();") == 11

    for pipe in ("PIPE_V", "PIPE_MTE2", "PIPE_MTE3", "PIPE_MTE1", "PIPE_M", "PIPE_FIX"):
        assert f"AscendC::PipeBarrier<{pipe}>();" in source
