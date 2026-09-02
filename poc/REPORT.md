# TileLang AscendC symbolic FlashAttention backward compiler PoC

## Scope and authority

- `observed_at`: `2026-09-02T06:13:40Z`
- operator: `29_FlashAttentionBwd`
- mode: isolated compiler author PoC, card-free only
- authority: `AUTHOR_EVIDENCE_ONLY`
- NPU execution: `NOT_RUN_NO_NPU_ADMISSION`
- numerical precision: `NOT_MEASURED`
- performance: `NOT_MEASURED`

This PoC answers one narrower question: can the AscendC backend keep the
rank-4 FlashAttention-backward extents symbolic and generate one host entry
plus a small number of real A5 Ascend C kernels, instead of cloning one kernel
per benchmark case? It does not claim that those kernels are numerically
correct or fast on an A5 device. Those claims require a non-author fixed-50
consumer with a known-bad discriminator.

## Source identities

- official repository: <https://github.com/tile-ai/tilelang-ascend.git>
- official branch/ref: `refs/heads/ascendc_pto`
- official tip at observation: `5e1dbfa15602642938d289ca561279a0c989a234`
- native A5 support PR: <https://github.com/tile-ai/tilelang-ascend/pull/1702>
- PR #1702 state: `OPEN`; head `2b5cc9b705e3142524126d4f2f727498896d596a`; GitHub test-merge ref `bafad01797a0cf8fc31d8f59b10b4f32446ee3e3`
- colleague A3 fork: <https://github.com/wzzll123/tilelang-ascend.git>
- colleague branch/ref: `refs/heads/ascendc_pto`
- colleague tip at observation: `4df68b237f235619a96a8106e21fbcd1830528f4`

Related live follow-up heads found at observation:

| PR | head | relevance |
|---|---|---|
| [#1335](https://github.com/tile-ai/tilelang-ascend/pull/1335) | `d484bdc01b3ac641435d46c86ce6be584dec407f` | symbolic local address/size and memory planning |
| [#1452](https://github.com/tile-ai/tilelang-ascend/pull/1452) | `32fb712acc3bb25236b82986666f62f6cbce688c` | high-rank/compact GM-UB copy flattening |
| [#1693](https://github.com/tile-ai/tilelang-ascend/pull/1693) | `6df22a907b813c239af542f4e032fb3e8a44ee35` | TND shared-prefix FA implementation |
| [#1708](https://github.com/tile-ai/tilelang-ascend/pull/1708) | `d6b6339cad33594fbbe7b779449b65020308ee8a` | A5 AscendC backend roadmap |
| [#1723](https://github.com/tile-ai/tilelang-ascend/pull/1723) | `b8d65d0077cd610e457bfa9b449ff1b4c3bc2953` | multi-tile unaligned K-tail correctness |

The A3 fork adds FA/fusion correctness and completeness repairs, including
copy alignment, CV synchronization, BF16 conversion, scalar flattening, L1/L0
copy handling, and target-core selection. Its unique diff does not introduce a
general symbolic-shape system; it uses the symbolic infrastructure inherited
from upstream. The A5 PoC therefore reuses the upstream symbolic TIR path and
adds the missing compiler-owned dtype variant/host-dispatch layer.

## Fixed-50 input contract

- original benchmark JSON:
  `/home/zheng/workspace/codex02_agent/op29_case36_precision_arm_20260831T2220Z/29_FlashAttentionBwd/29_FlashAttentionBwd.json`
- original JSON SHA256:
  `63b545dd67682935c51910b42c4324c4d375559aff12b44b469c1aaecd97c253`
- build-input CSV SHA256 (original CRLF bytes):
  `8f0b9b816fe534832a1549e03047cb4e10d468c64043bdc4c4339fe2378dd188`
- committed LF-normalized CSV: `poc/inputs/op29_fa_bwd_fixed50_shapes.csv`
- committed CSV SHA256:
  `6cb3fef12299b661ec01923e5216f135aa5be0fbe97abb5aad60f21684efd889`
- denominator: exactly 50 cases

All seven tensors are rank 4 in all 50 cases. Thus the reported "DIM changes"
in this denominator are changes in **extent**, not changes in rank:

- Q/DY/attention: `[B,Sq,Hq,D]`
- K/V: `[B,Sk,Hk,D]`
- softmax max/sum: `[B,Hq,Sq,8]`

Extent/dtype distribution:

- `B`: 1x29, 2x15, 3x4, 4x2
- `Sq`: 22 values, range 3..400
- `Sk`: 25 values, range 5..500
- `Hq`: 12 values: 2,4,6,8,28,32,36,40,48,56,64,72
- `Hk`: 12 values: 1,2,3,4,7,8,9,10,12,14,16,18
- `D`: 16x7, 24x2, 32x3, 40x2, 48x1, 96x5, 128x30
- dtype: FP32x17, FP16x17, BF16x16
- causal: truex26, falsex24

Kernel-count baselines:

- static full case signature: 50 kernels
- A3 example/factory-like `(dtype,D,Hq,Hk)` specialization: 37 kernels
- this PoC: 1 host dispatcher + 3 real TileLang-lowered kernels, keyed only by dtype

## What changed in the compiler PoC

The patch is intentionally limited to the AscendC backend/PoC surfaces:

- `src/target/codegen_ascend.cc`: permits a PrimFunc `ascendc_host_entry`
  attribute to select the generated host wrapper symbol.
- `tilelang/jit/adapter/ascendc_dispatch.py`: validates the fixed-rank/runtime-
  extent contract, groups only by dtype, renders one checked public dispatcher,
  and matches the actual AscendC host-wrapper ABI.
- `poc/fa_bwd_symbolic_lowering.py`: expresses numerical FlashAttention
  backward dataflow in TileLang with runtime `B/Sq/Sk/Hq/Hk/D`, including Q/K/V,
  DY, softmax statistics, attention, DQ/DK/DV, masking, softcap, and exponentials.
- `poc/run_fa_bwd_real_lowering.py`: lowers each dtype variant through TileLang,
  checks the generated source, compiles for native A5, links the dispatcher,
  runs dispatch/negative controls, and writes content-addressed evidence.
- `poc/run_fa_symbolic_dispatch_poc.py`: verifies all 50 host routes and the
  rank/D negative controls without a device.
- `testing/python/language/test_tilelang_ascend_language_alloc_codegen.py`:
  source-level coverage for the host-entry codegen contract.

This is not a handwritten numerical Ascend C kernel. The device `.cpp` files in
the external evidence are the result of `tilelang.lower(...)` over the
`@T.prim_func` in `poc/fa_bwd_symbolic_lowering.py`. The only generated C++
written directly by the Python adapter is the host dispatcher.

## Card-free A5 build result

Build-only container:
`root@141.61.33.141:zheng-codex02-tilelang-fa-symbolic-buildonly-20260902t0510z`.
Docker inspect recorded `devices=[]`; no NPU was requested or used.

Toolchain:
`/data/pri/Ascend/9.1.0.B150/cann-9.1.0/x86_64-linux/ccec_compiler/bin/bisheng`
with `--npu-arch=dav-3510 -DCATLASS_ARCH=3510 -std=c++17 -xasc -O2 --shared`.
The full command vectors are retained verbatim in `poc/RESULT.json`.

Result:

| dtype | generated source SHA256 | A5 ELF SHA256 | build |
|---|---|---|---|
| BF16 | `e4b40e2a053ade21f6559ce56c411fbfb3e34c65cf76223e28c47fbd66da5aa3` | `b4d66e1d52b62ff3a43a28dfd67344d69e39e70886fb102e0bf188c1deab93d5` | rc=0 |
| FP16 | `4261ff994e927602ea0673fc6d181569a8efdf10e898735d2d04ed0bba387112` | `99ef12ac4e384fd1b95a93dabb404d411e5deb12f2441f81b0e650274a9c2f2b` | rc=0 |
| FP32 | `e180007ce9cebad9d9800a2b7f5c63dfb10fe0210e992819cb16053d525f399c` | `f91cf3206ef438d402bda8e18e547383558b1f2536c2b1c48bedd41971f2b419` | rc=0 |

Host dispatcher:

- symbol: `tilelang_fa_bwd_call`
- shared object SHA256:
  `c64c5b822b0b5f51c301b56adcef1aec2466fb46f80dfc9e320bba0cfb918ec9`
- `DT_NEEDED`: exactly the three typed FA libraries above plus normal runtime dependencies
- `RTLD_NOW`: PASS without calling a device wrapper

Generated-source guards passed for every dtype: the runtime extent arguments
are present, Q/K/V/DY and softmax inputs are consumed, DQ/DK/DV are written,
`AscendC::Exp` is present, accumulator state is reset per output, and all
ABI-only sentinel tokens are absent.

Tests:

- host dispatcher over all 50 case rows: PASS
- rank-change negative control: rejected
- `D % 8 != 0` negative control: rejected
- selected compiler tests:
  `3 passed, 16 deselected` (`pytest` rc=0)
- `git diff --cached --check`: PASS before author commit

External generated/build evidence (not committed because it contains ELF and
generated artifacts):

`root@141.61.33.141:/home/zheng/codex02_tilelang_fa_symbolic_build_20260902T0510Z/evidence/real_fa_bwd_symbolic_20260902T0620Z/`

The final external `MANIFEST.sha256` binds every generated source, ELF, host
dispatcher, compiler output, test log, and result receipt.

## What this proves and what it does not

Proved as author evidence:

- extent variation in these 50 rank-4 cases does not inherently require 50
  cloned kernels;
- a compiler-owned variant plan can lower 50 compile-time case descriptions to
  one host ABI and three real A5 Ascend C typed kernels;
- B/Sq/Sk/Hq/Hk/D reach the generated host wrapper and device body as runtime
  values;
- all three variants build successfully for `dav-3510` without an NPU;
- rank variation remains outside this ABI and fails loudly.

Not proved:

- that any one of the 50 cases launches on an A5 device;
- fixed-50 numerical precision, known-bad discrimination, or performance;
- that one generic scalar kernel is sufficiently performant;
- product acceptance or upstream mergeability.

The implemented numerical body is deliberately scalar/correctness-shaped. It
is a compiler-path discriminator, not the final fused Cube+Vector performance
implementation. A later performance implementation may need a small, evidence-
supported multi-key (for example a D/tail or tiling family), but it must not
regress to one kernel per case.

## Independent acceptance gate

The author commit must receive a fresh non-author review on its exact tree.
Then, on a freshly admitted A5 device, an independent consumer must:

1. rebuild or byte-bind the same source/toolchain/host ABI;
2. launch all 50 cases through the one dispatcher without recompilation;
3. prove fixed-50 DQ/DK/DV precision using the bound CPU comparator;
4. run an observable same-session known-bad;
5. verify the dispatch variant count remains three and cache/compiler activity
   does not create shape-specific kernels;
6. measure same-candidate performance against the product reference.

Until those gates pass, the correct status is `AUTHOR_BUILD_PASS / DEVICE_NOT_RUN`.
