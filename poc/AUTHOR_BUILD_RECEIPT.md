# AUTHOR_BUILD receipt: FA backward AscendC symbol isolation

- observed_at: `2026-09-02T07:45:00Z`
- authority: `AUTHOR_BUILD_EVIDENCE_ONLY`
- rejected ancestor: `a5754b8b941a0df3b1aaa5ee30bdddae0ac3bbe6`
- build container: `root@141.61.33.141:zheng-codex02-tilelang-fa-symbolic-buildonly-20260902t0510z`
- Docker device bindings: `HostConfig.Devices=[]`, `HostConfig.DeviceRequests=null`
- target: `A5 AscendC`, `dav-3510`, `CATLASS_ARCH=3510`
- TVM gitlink: `c2921fdaf795b1103d21abc962e83a209c7258d7`
- required retained dependency patch:
  `poc/patches/tvm_dynamic_slice_unit_step.patch`
- NPU execution: `NOT_RUN_NO_NPU_ADMISSION`
- numerical precision: `NOT_MEASURED`
- performance: `NOT_MEASURED`

## Rejected control

At `a5754b8b`, all three variant DSOs exported `GLOBAL DEFAULT main_kernel`.
The retained `LD_DEBUG=bindings` control returned rc 0 and recorded:

```text
fa_bwd_fp32.so -> fa_bwd_fp16.so: normal symbol `main_kernel'
fa_bwd_bf16.so -> fa_bwd_fp16.so: normal symbol `main_kernel'
fa_bwd_fp16.so -> fa_bwd_fp16.so: normal symbol `main_kernel'
```

This independently reproduces the non-author High: successful load and host
dispatch did not prove that typed wrappers launched typed device entries.

## Repair

The compiler accepts optional PrimFunc attribute `ascendc_kernel_entry` and
uses its value for both the generated `__aicore__` function definition and the
host-wrapper launch relocation. The default remains `<global_symbol>_kernel`,
so existing single-kernel callers retain their ABI. The public dispatcher ABI
`tilelang_fa_bwd_call` and private host wrappers
`call_fa_bwd_fp16/bf16/fp32` are unchanged.

The FA lowering sets the entries to:

- FP16: `fa_bwd_fp16_kernel`
- BF16: `fa_bwd_bf16_kernel`
- FP32: `fa_bwd_fp32_kernel`

## Card-free build and symbol discriminators

All three variants rebuilt with Bisheng B150 using
`--npu-arch=dav-3510 -DCATLASS_ARCH=3510 -std=c++17 -xasc -O2 --shared`.

| dtype | ELF SHA256 | unique symbol | generic `main_kernel` | wrapper relocation | loader binding |
|---|---|---|---|---|---|
| FP16 | `53d0fe95a23c9253686b9662dbfe9a9df00a7f7f76791fe1f60dae5c008b41c5` | GLOBAL DEFAULT | absent | `R_X86_64_GLOB_DAT fa_bwd_fp16_kernel` | FP16 DSO to itself |
| BF16 | `e4af6027021798d08096e6dbbdb955b90f1f87870c58f0aaf0fbdf65de32970f` | GLOBAL DEFAULT | absent | `R_X86_64_GLOB_DAT fa_bwd_bf16_kernel` | BF16 DSO to itself |
| FP32 | `db3b558461a9f95a3e377428c0450c51cd7edaa5917d94f3da1cbf1ebadbe7cf` | GLOBAL DEFAULT | absent | `R_X86_64_GLOB_DAT fa_bwd_fp32_kernel` | FP32 DSO to itself |

Repaired `LD_DEBUG=bindings` returned 0. It contains exactly one self-binding
for each typed kernel entry and no `normal symbol 'main_kernel'` binding.

## Functional controls

- one public host dispatcher plus three real TileLang-lowered kernels
- fixed-50 compile/host dispatch coverage: case IDs 0..49, PASS
- rank-change negative control: REJECTED
- `D % 8 != 0` negative control: REJECTED
- selected compiler regression: `4 passed, 16 deselected`
- CMake compiler rebuild: rc 0
- external evidence manifest: independently checkable with
  `sha256sum -c MANIFEST.sha256`

## Boundary

This author evidence proves compiler-generated source, A5 card-free build,
host ABI preservation, and ELF binding isolation. It does not prove A5 device
execution, fixed-50 DQ/DK/DV precision, known-bad discrimination on device,
or performance. A fresh non-author review must judge the exact repair commit
and its external evidence before acceptance.

The build container's TVM submodule contains exactly the retained dependency
patch above. It is deliberately disclosed and content-addressed; it must not be
mistaken for part of the symbol-isolation repair or silently omitted in a
clean-room replay.
