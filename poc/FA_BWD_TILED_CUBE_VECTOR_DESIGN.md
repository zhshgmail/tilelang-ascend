# Symbolic tiled Cube+Vector FA-Bwd checkpoint

Status: device-free compiler checkpoint.  This document is not NPU precision or
performance evidence and does not change the A5Ops product gate.

## Bound contract

- Route A remains `tilelang.lower(..., target="ascendc", platform="A5")`.
- The public product remains one generated host dispatcher and one device DSO
  per dtype family (`float16`, `bfloat16`, `float32`): three DSOs total.
- `B`, `Sq`, `Sk`, `Hq`, `Hk`, and `D` remain runtime symbols.  The frozen 50
  cases share the same code; there are no case ids, shape tables, or per-shape
  kernel clones in the tiled source.
- BSND tensors, GQA (`Hq % Hk == 0`), causal/window masks, runtime softcap, and
  upstream `softmax_max`/`softmax_sum` stay on the existing ABI.
- Fixed tiles are `BQ=16`, `BK=16`, `D_PAD=128`.  `D<=128` and `D%8==0` are
  host admission facts for the frozen family, while GM/L1 and L0C/GM tail
  copies carry the real runtime extents.

## Kernel ownership and math

The kernel uses 24 physical AIC tasks with persistent symbolic loops.

1. A dQ task owns `(b, hq, q_tile)`, computes Delta with Vector operations,
   streams every K/V tile, and keeps the dQ accumulator in L0C.
2. A dK/dV task owns `(b, hk, k_tile)`, streams all query heads in its GQA
   group and all query tiles, and keeps both accumulators in L0C.

Both paths recompute the same score tile and use five mathematical GEMM roles:
`Q@K^T`, `dY@V^T`, `dS@K`, `dS^T@Q`, and `P^T@dY`.  Vector stages reconstruct
`P` from the upstream softmax statistics, apply causal/window/tail masks and
runtime softcap, and form `dS = P * (dP - Delta) * d(score)/d(qk)`.

This owner split intentionally avoids atomic output updates.  The retained
consumer does not promise zero-filled output buffers, so an atomic design would
silently add to sentinel memory.  Recomputing score is the bounded source-first
tradeoff until first-device precision establishes a safe accumulator strategy.

## Resource estimate

The largest single L0C object is `16*128*4 = 8 KiB`; the three live L0C
accumulators remain below 24 KiB.  Fixed L1/UB tiles are independent of runtime
sequence sizes.  No full score matrix or host workspace is materialized.

## Dispatch coverage

[DISPATCH-COVERAGE] The existing dtype plan still maps all 50 descriptors to
exactly three variants.  `--kernel-path tiled` changes only the lowered body,
not dispatcher ABI, variant keys, or provenance inputs.

[REORDER-COST] The initial tiled checkpoint changes reduction grouping from the
scalar reference and reconstructs P at 16x16 granularity.  First-device
precision must therefore run all 150 outputs and known-bad controls before any
performance claim.

[HOST-METADATA-AUDIT] Shape/stride/alignment/alias admission remains owned by
the existing single dispatcher.  The tiled body introduces no new host scalar,
workspace, per-case table, or output-initialization requirement.

## Checkpoint gates

- Python structural tests: symbolic family, exactly one PrimFunc, five GEMM
  roles, owner scheduling, no case specialization.
- Route-A lowering for all three dtypes and generated-source structural audit.
- A CANN 9.2 / Bisheng 15 DAV3510 compile-only receipt is still required if it
  is not available in the local device-free environment.
- Only after compile admission: fresh NPU full50 precision + known-bad, then
  canonical same-candidate msprof performance.  No result from this document
  satisfies those product gates.

