# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Typed Ascend topology profiles with explicitly uncalibrated timing."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from .errors import SimulatorConfigError


@dataclass(frozen=True)
class DeviceProfile:
    """Static topology used by the simulator.

    Timing is deliberately kept separate from topology.  A5 core counts are SKU/runtime
    properties and therefore remain unset in the device-free profile instead of inheriting
    an A2/A3 fallback.
    """

    platform: str
    cube_core_count: Optional[int]
    vector_core_count: Optional[int]
    vector_lanes_per_cube: int
    local_memory_bytes: Mapping[str, int] = field(default_factory=dict)
    capacity_source: str = ""
    requires_runtime_core_counts: bool = False
    cube_pipes: Tuple[str, ...] = ("mte2", "mte1", "m", "fix")
    vector_pipes: Tuple[str, ...] = ("mte2", "v", "mte3")
    calibration: str = "uncalibrated"

    def __post_init__(self) -> None:
        normalized_memory = {
            str(scope).strip().upper(): int(size)
            for scope, size in self.local_memory_bytes.items()
        }
        if any(not scope or size <= 0 for scope, size in normalized_memory.items()):
            raise SimulatorConfigError("local memory capacities must be positive")
        object.__setattr__(self, "local_memory_bytes", MappingProxyType(normalized_memory))


@dataclass(frozen=True)
class TimingProfile:
    """Parameter table for a discrete-event scheduler.

    Empty operation costs are intentional: callers must either provide calibrated values or
    accept ``fallback_cycles``.  This prevents the scaffold from presenting invented hardware
    latencies as measurements.
    """

    platform: str
    operation_cycles: Mapping[str, int] = field(default_factory=dict)
    fallback_cycles: int = 1
    calibration: str = "uncalibrated-unit-cost"

    def __post_init__(self) -> None:
        get_device_profile(self.platform)
        if self.fallback_cycles <= 0:
            raise SimulatorConfigError("fallback_cycles must be positive")
        normalized = {}
        for name, cycles in self.operation_cycles.items():
            if not name:
                raise SimulatorConfigError("operation name must not be empty")
            if cycles <= 0:
                raise SimulatorConfigError(f"operation cycle count must be positive: {name}")
            normalized[str(name)] = int(cycles)
        object.__setattr__(self, "operation_cycles", MappingProxyType(normalized))
        object.__setattr__(self, "platform", normalize_platform(self.platform))

    def estimate_cycles(self, operation: str) -> int:
        """Return a configured cost or the visibly uncalibrated fallback cost."""
        return self.operation_cycles.get(operation, self.fallback_cycles)


A2_A3_LOCAL_MEMORY_BYTES = MappingProxyType({
    "L1": 524032,
    "L0A": 65536,
    "L0B": 65536,
    "L0C": 131072,
    "UB": 196352,
    "BT": 1024,
})

# The compiler-owned values come from src/transform/ascend_memory_planning.cc:
# ASCEND_A5_{SHARED_DYN_MEM,WMMA_MATRIX_A,WMMA_MATRIX_B,
# WMMA_ACCUMULATOR,SHARED_MEM}_SIZE.  DAV3510 BT is 4 KiB in the canonical
# A5Ops hardware profile at
# src/skills/references/hardware/target/ascend950pr.md (source snapshot sha256
# 87b70bf1c9e2fec34f8aa4d8ff6a22705736d484f604dce80943038b9e33edfe).
# Core counts are intentionally absent because they vary by Ascend950 physical
# SKU/vNPU and must be supplied by the active runtime.
A5_DAV3510_LOCAL_MEMORY_BYTES = MappingProxyType({
    "L1": 524288,
    "L0A": 65536,
    "L0B": 65536,
    "L0C": 262144,
    "UB": 253952,
    "BT": 4096,
})

_DEVICE_PROFILES = {
    "A2": DeviceProfile(
        "A2",
        cube_core_count=20,
        vector_core_count=40,
        vector_lanes_per_cube=2,
        local_memory_bytes=A2_A3_LOCAL_MEMORY_BYTES,
        capacity_source="legacy A2/A3 compiler memory-planning constants",
    ),
    "A3": DeviceProfile(
        "A3",
        cube_core_count=20,
        vector_core_count=40,
        vector_lanes_per_cube=2,
        local_memory_bytes=A2_A3_LOCAL_MEMORY_BYTES,
        capacity_source="legacy A2/A3 compiler memory-planning constants",
    ),
    "A5": DeviceProfile(
        "A5",
        cube_core_count=None,
        vector_core_count=None,
        vector_lanes_per_cube=2,
        local_memory_bytes=A5_DAV3510_LOCAL_MEMORY_BYTES,
        capacity_source=(
            "src/transform/ascend_memory_planning.cc DAV3510 constants; "
            "canonical A5Ops ascend950pr.md hardware profile for BT"
        ),
        requires_runtime_core_counts=True,
    ),
}


def normalize_platform(platform: str) -> str:
    """Normalize and validate a simulator platform name."""
    normalized = platform.strip().upper()
    if normalized not in _DEVICE_PROFILES:
        supported = ", ".join(sorted(_DEVICE_PROFILES))
        raise SimulatorConfigError(
            f"unsupported simulator platform {platform!r}; supported platforms: {supported}"
        )
    return normalized


def get_device_profile(platform: str) -> DeviceProfile:
    """Return the immutable device topology for A2, A3, or A5."""
    return _DEVICE_PROFILES[normalize_platform(platform)]


def default_timing_profile(platform: str) -> TimingProfile:
    """Return a unit-cost profile clearly marked as uncalibrated."""
    return TimingProfile(platform=normalize_platform(platform))
