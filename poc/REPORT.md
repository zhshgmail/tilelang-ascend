# TileLang AscendC symbolic FlashAttention backward compiler PoC v4

## Bounded verdict

- observed_at_utc: `2026-09-02T17:06:28Z`
- operator: `29_FlashAttentionBwd`
- authority: `AUTHOR_EVIDENCE_ONLY`
- target: A5 AscendC, `dav-3510`, `CATLASS_ARCH=3510`
- source commit/tree: `43a44017deec4698c30631ec125ab044294b8332` / `372f2550fe1ba3a760763c6adbca30cefb76d67a`
- result: one compiler-generated host dispatcher plus three real TileLang-lowered and Bisheng-built typed kernels covers all 50 compile/dispatch rows
- device execution: `NOT_RUN_NO_NPU_ADMISSION`
- fixed50 numerical precision: `NOT_MEASURED`
- performance: `NOT_MEASURED`

This proves the card-free compiler/build/ABI path only. It does not claim that
the kernels are numerically correct or fast on A5 hardware.

## Fresh upstream and colleague source identities

- official repo/ref: <https://github.com/tile-ai/tilelang-ascend.git>
  `refs/heads/ascendc_pto`
- official head: `01e7254dc51685fd77f41e5807c42e6537e17645`
- colleague A3 fork/ref: <https://github.com/wzzll123/tilelang-ascend.git>
  `refs/heads/ascendc_pto`
- colleague A3 head: `b0dcbf42fbc4ce7e28d594b50b3dba8a7b29e093`
- native A5 support [PR #1702](https://github.com/tile-ai/tilelang-ascend/pull/1702): OPEN, head `2b5cc9b705e3142524126d4f2f727498896d596a`

The official branch advanced from the earlier frozen `5e1dbfa1...` through
three commits: persistent-partial single-wave handling, packed-mask cleanup,
and heavy tile-kernel test migration. None touches the PoC's changed compiler
files. The A3 fork advanced once, by documentation only. Related open heads
were refreshed for #1335 symbolic address/size (`d484bdc...`), #1452 high-rank
copy flattening (`32fb712...`), #1693 TND shared-prefix FA (`6df22a9...`),
#1708 roadmap (`d6b6339...`), #1723 K-tail (`b8d65d0...`), and #1724 widened
FP16 reduce-sum (`e238839...`). `poc/SOURCE_OBSERVATION.json` is the machine-
readable receipt; all generated provenance uses its single observation time.

## Fixed50 symbolic-shape contract

The committed case manifest is
`poc/inputs/op29_fa_bwd_fixed50_shapes.csv`, SHA256
`6cb3fef12299b661ec01923e5216f135aa5be0fbe97abb5aad60f21684efd889`.
It contains exactly 50 cases.

All Q/K/V/DY/attention/softmax tensors have rank 4 in every case. Therefore
the observed DIM variation is variation of **extent**, not rank:

- runtime symbolic extents: `B, Sq, Sk, Hq, Hk, D`
- `D`: 16x7, 24x2, 32x3, 40x2, 48x1, 96x5, 128x30
- dtype: FP32x17, FP16x17, BF16x16
- static full-shape baseline: 50 kernels
- A3 factory-like `(dtype,D,Hq,Hk)` baseline: 37 kernels
- this PoC: one host dispatcher plus three dtype-keyed kernels

Rank changes and `D % 8 != 0` remain outside the implemented ABI and are
fail-closed negative controls. Runtime extent variation within rank 4 is not,
by itself, a language restriction requiring case-per-kernel cloning.

## Compiler change

- `src/target/codegen_ascend.cc`: supports compiler-owned, per-PrimFunc
  `ascendc_host_entry` and `ascendc_kernel_entry`; the same unique device entry
  is used in the definition and host launch relocation.
- `tilelang/jit/adapter/ascendc_dispatch.py`: validates the symbolic contract,
  groups only by dtype, and emits the checked host dispatcher.
- `poc/fa_bwd_symbolic_lowering.py`: real TileLang FA-backward DSL with runtime
  extents and Q/K/V/DY/softmax input consumption plus DQ/DK/DV output writes.
- `poc/run_fa_bwd_real_lowering.py`: invokes `tilelang.lower`, copies the real
  compiler output, builds all typed DSOs, links the host DSO, runs symbol and
  dispatch controls, and emits content-addressed provenance.
- `tilelang/jit/adapter/ascendc_provenance.py`: records one observation time,
  source commit/tree/status, top-level and 16 recursive submodules, exact TVM
  patch, loaded compiler libraries, Bisheng identity, inputs, and artifacts.
- `poc/verify_fa_bwd_bundle.py`: independent fail-closed consumer for build
  provenance and manifest closure.

The exact TVM dependency patch is
`poc/patches/tvm_dynamic_slice_unit_step.patch`, SHA256
`99a307ad3fa0ea648f26ad15a84c38588241de96cd11740c8f4210e7d81931a8`,
applied to TVM gitlink `c2921fdaf795b1103d21abc962e83a209c7258d7`.
It keeps explicit unit-step dynamic slices on TVM's BufferRegion path.

## Card-free A5 build

Build-only container:
`root@141.61.33.141:zheng-codex02-tilelang-fa-symbolic-buildonly-20260902t0510z`.
Docker inspection recorded `Devices=[]` and `DeviceRequests=null`.

Compiler/toolchain bindings:

- built with `USE_ASCEND=ON`; `target.build.tilelang_ascend` present
- `libtilelang_module.so` SHA256 `d05fa79cd5110c03bb399e288e17dfaf3a5b3b1225d06b7204fe8caac3274a2b`
- `libtvm.so` SHA256 `d8d84f3c63d40462d99a9199b85d8a1ce587da328800d281962585b7e17cfeec`
- Bisheng B150 SHA256 `8120f5d1fd5cc2df8499f2aa1b301ec7a8126d66bd23226c721de378872a918d`
- build flags include `--npu-arch=dav-3510 -DCATLASS_ARCH=3510 -std=c++17 -xasc -O2 --shared`

| artifact | source SHA256 | DSO SHA256 |
|---|---|---|
| FP16 kernel | `7af413043df3e806c464b079dfbee0d0eb3ad5feb53b4d2d7ce42ea27b480f1a` | `a36dd9071ea4d0d809b5517f8b7658f120298906f059e299cad85a417916daec` |
| BF16 kernel | `85a99bd0d8f98c9639955e7bc46151d1d5d406ebccb7898a66d3d24cf97626ff` | `bf13c97d9b44fbcd6558350033079d8e1d4a2d568c3183b3cbebb9f93850a7c0` |
| FP32 kernel | `b07754fc297b0a49ab0db74006ed199c8f07f289a8045fc5d663e25fe6492f3e` | `70aba7215aacbf8c4f35d1c835b0d133d8779b04a1749fbbd35cbdccf23adc48` |
| host dispatcher | n/a | `c64c5b822b0b5f51c301b56adcef1aec2466fb46f80dfc9e320bba0cfb918ec9` |

The external bundle is:

`root@141.61.33.141:/home/zheng/codex02_tilelang_fa_symbolic_build_20260902T0510Z/provenance_build_fd1be6e5/bundle_43a44017_b150/`

Its `BUILD_PROVENANCE.json`, `RESULT.json`, and `MANIFEST.sha256` hashes are,
respectively, `7672176343389332d081604a7e235e1b8b84ac303aa63c7b3650b8320fe792cb`,
`fc54f51f9471d85f86e1b56c384da28bea6c2f8939fcc84b7705e9326371d52e`,
and `8a7ccbeda3e901a9cbb1033d00ca7aa97b2a606a6712f6a8`.

## Discriminators and tests

- exact provenance regression: `6 passed`
- real lowering/build command: rc 0
- host dispatcher routes all 50 rows: PASS
- rank-change control: REJECTED
- `D % 8 != 0` control: REJECTED
- stale-official-head provenance control: REJECTED
- mutated-artifact provenance control: REJECTED
- standalone bundle consumer: `PROVENANCE_PASS`, rc 0
- `sha256sum -c MANIFEST.sha256`: rc 0; 57 rows over 58 bundle files
- RTLD_NOW host load: PASS without invoking a device wrapper
- unique-device-symbol verdict: `PASS_UNIQUE_SELF_BINDINGS`

The prior `a5754b8b...` design remains a rejected ancestor because all typed
DSOs exported `main_kernel` and BF16/FP32 relocations bound to the FP16 DSO.
In v4, each DSO defines only its typed device entry
(`fa_bwd_fp16_kernel`, `fa_bwd_bf16_kernel`, or `fa_bwd_fp32_kernel`), generic
`main_kernel` is absent, and `LD_DEBUG=bindings` proves each wrapper relocation
binds to its own DSO.

## Independent A5 device gate

After fresh exclusive NPU admission, a non-author consumer must preserve the
source, three kernel DSOs, host DSO, case manifest, comparator, and toolchain
bindings, then:

1. run all 50 cases through the one host dispatcher without recompilation;
2. compare DQ/DK/DV against the bound CPU golden on the fixed denominator;
3. run an observable same-session known-bad;
4. prove no shape-specific kernel files or compiler activity appear;
5. only after precision, measure same-candidate performance against the
   product reference.

Until that gate passes, the only honest status is
`AUTHOR_BUILD_PASS / DEVICE_NOT_RUN / PRECISION_NOT_MEASURED / PERF_NOT_MEASURED`.
