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
- The final module is rendered with metadata-complete TVMScript at the
  pre-codegen boundary and exposed as stable `SIMULATOR_DIAGNOSTIC` evidence
  with target, platform, and `mcpu` identity. Raw `tvm.ir.save_json` is not
  used for this identity because `Map` node order can vary across processes.
- A5 memory allocation uses the DAV3510 capacity table rather than the C220
  table.
- `build_kernel_program(..., symbol_bindings=...)` can specialize finite
  boolean, integer, and floating scalar symbols before expanding `For`, `If`,
  and `Let` control. Lowered two-argument `auto_{set,wait}_flag` calls preserve
  their combined pipe pair and event id for synchronization scheduling.
- Unknown operations and explicitly unmodeled DAV3510 semantics fail closed.
  They do not inherit a similarly named C220 operation classification.

## Bounded P1 scalar-control behavior

The final-TIR bridge now evaluates the scalar values needed to decide A5
control flow. This is intentionally narrower than functional tensor execution:

- A5 `For`, `IfThenElse`, `LetStmt`, and `AssertStmt` can consume finite scalar
  constants, bound scalar `Var` values, and scalar `BufferLoad` values produced
  by an earlier scalar `BufferStore` in the same modeled core/vector context.
- The expression subset is integer and finite floating arithmetic, integer
  division/modulo, `Min`/`Max`, comparisons, boolean `And`/`Or`/`Not`, `Cast`,
  and branch-lazy `Select`.
- Known scalar stores carry deterministic `scalar_indices` and `scalar_value`
  metadata on their existing `buffer_store` task. A store whose value is not
  modeled remains a static scheduling task; if later control flow consumes it,
  the unknown value propagates and fails closed.
- Read-before-write, an unbound scalar, a non-scalar index, a non-integral loop
  extent, and every expression outside the list above fail closed when needed
  by control flow.
- The behavior is gated to A5. A2/A3 keep the P0 constant-specialization
  boundary, including their existing error for a buffer-dependent condition.

This slice is covered by a device-free differential fixture against direct
Python scalar truth. It does not add an acceptance gate, and does not change
the `SIMULATOR_DIAGNOSTIC` final-TIR identity or its authority.

## Deliberately missing after bounded P1

- parameter-buffer inputs and complete tensor-output functional execution;
- scalar loads whose most recent producer is an unmodeled tensor/vector call;
- bit-accurate BF16 storage, casts, transcendental functions, and reductions;
- DAV3510 BufferID/SSBuffer, SIMT, RegBase, NDDMA, CCU/KFC, and related runtime
  behavior;
- post-codegen AscendC, host ABI, JIT-cache, DSO-load, precision, and calibrated
  performance validation.

Until those slices have differential A5 tests, unsupported operations must stay
fail-closed and the simulator must remain diagnostic-only.
