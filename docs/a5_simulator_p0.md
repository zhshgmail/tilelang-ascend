# A5/DAV3510 simulator P0

## Scope and authority

This P0 is a device-free, diagnostic evidence adapter. It lets native AscendC
code generation and simulator analysis bind to the same final optimized TIR. It
does not create a product gate, prove numerical precision, or measure device
performance. Native A5 compile/load, fixed-denominator precision with known-bad
controls, and same-candidate device-time measurement remain authoritative.

This boundary follows the centralized-gate architecture: a backend supplies
typed evidence to the shared consumer; it does not copy product acceptance
logic into the simulator.

## DAV3510 profile sources

The source-controlled A5 memory planner in
`src/transform/ascend_memory_planning.cc` defines the usable capacities consumed
by this profile:

| Scope | Bytes |
|---|---:|
| L1 | 524288 |
| L0A | 65536 |
| L0B | 65536 |
| L0C | 262144 |
| UB | 253952 |

BT is 4096 bytes per the canonical A5Ops hardware profile
`src/skills/references/hardware/target/ascend950pr.md` (source snapshot SHA256
`87b70bf1c9e2fec34f8aa4d8ff6a22705736d484f604dce80943038b9e33edfe`).
Physical core counts vary by Ascend950 SKU and vNPU profile, so the device-free
profile deliberately stores no invented fallback. A future device-backed
adapter must read those counts from the active runtime and record their
provenance.

Both topology and timing are explicitly uncalibrated. Simulated cycles are for
trace structure only and must not be reported as latency or used for autotuner
pruning.

## P0 behavior

- `lower_ascend_ir` is shared by native code generation and simulator analysis.
- The final module is serialized once at the pre-codegen boundary and exposed
  as `SIMULATOR_DIAGNOSTIC` evidence with target, platform, and `mcpu` identity.
- A5 memory allocation uses the DAV3510 capacity table rather than the C220
  table.
- Unknown operations and explicitly unmodeled DAV3510 semantics fail closed.
  They do not inherit a similarly named C220 operation classification.

## Deliberately missing after P0

- symbolic `For`/`If`/`Let` execution;
- scalar `BufferLoad`/`BufferStore` functional semantics;
- bit-accurate BF16 storage, casts, transcendental functions, and reductions;
- DAV3510 BufferID/SSBuffer, SIMT, RegBase, NDDMA, CCU/KFC, and related runtime
  behavior;
- post-codegen AscendC, host ABI, JIT-cache, DSO-load, precision, and calibrated
  performance validation.

Until those slices have differential A5 tests, unsupported operations must stay
fail-closed and the simulator must remain diagnostic-only.
