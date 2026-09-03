"""DAV3510 compile contracts for the shared Ascend C helper header."""

from __future__ import annotations

from pathlib import Path


COMMON_HEADER = (
    Path(__file__).parents[1] / "src" / "tl_templates" / "ascend" / "common.h"
)


def test_dav3510_common_header_uses_directional_hard_events() -> None:
    source = COMMON_HEADER.read_text(encoding="utf-8")

    # CANN 9.2 Bisheng rejects PIPE_ALL for DAV3510 when auto-sync is disabled.
    # PIPE_V is also a no-op on this target.  Tail helpers are modelled as
    # PIPE_V operations, so synchronize their scalar implementation with V_S
    # on entry and S_V on exit.  gemmL1 drains its final FIX producer before M
    # reuses L0C.  FetchEventID avoids colliding with caller-owned literal IDs.
    assert "CATLASS_DEVICE void hard_event_barrier_compat()" in source
    assert "GetTPipePtr()->FetchEventID(event)" in source
    assert "AscendC::SetFlag<event>(event_id);" in source
    assert "AscendC::WaitFlag<event>(event_id);" in source
    assert "AscendC::PipeBarrier<PIPE_ALL>();" not in source
    assert source.count(
        "hard_event_barrier_compat<AscendC::HardEvent::V_S>();"
    ) == 5
    assert source.count(
        "hard_event_barrier_compat<AscendC::HardEvent::S_V>();"
    ) == 5
    assert source.count(
        "hard_event_barrier_compat<AscendC::HardEvent::FIX_M>();"
    ) == 1
