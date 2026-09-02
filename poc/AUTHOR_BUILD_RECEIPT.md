# AUTHOR_BUILD receipt: FA backward symbolic AscendC v4

- observed_at_utc: `2026-09-02T17:06:28Z`
- authority: `AUTHOR_BUILD_EVIDENCE_ONLY`
- source commit/tree: `43a44017deec4698c30631ec125ab044294b8332` / `372f2550fe1ba3a760763c6adbca30cefb76d67a`
- reviewed compiler ancestor: `1d5a4b096c092badbc73456f113f8b082c849282`
- rejected symbol-interposition ancestor: `a5754b8b941a0df3b1aaa5ee30bdddae0ac3bbe6`
- official current source: `tile-ai/tilelang-ascend@01e7254dc51685fd77f41e5807c42e6537e17645`
- colleague A3 source: `wzzll123/tilelang-ascend@b0dcbf42fbc4ce7e28d594b50b3dba8a7b29e093`
- A5 PR #1702: OPEN at `2b5cc9b705e3142524126d4f2f727498896d596a`
- target: A5 AscendC `dav-3510`, `CATLASS_ARCH=3510`
- NPU: `NOT_RUN_NO_NPU_ADMISSION`
- precision/performance: `NOT_MEASURED / NOT_MEASURED`

## Exact build

- build-only container: `root@141.61.33.141:zheng-codex02-tilelang-fa-symbolic-buildonly-20260902t0510z`
- device binding: `Devices=[]`, `DeviceRequests=null`
- external bundle: `/home/zheng/codex02_tilelang_fa_symbolic_build_20260902T0510Z/provenance_build_fd1be6e5/bundle_43a44017_b150`
- fixed50 input SHA256: `6cb3fef12299b661ec01923e5216f135aa5be0fbe97abb5aad60f21684efd889`
- TVM gitlink: `c2921fdaf795b1103d21abc962e83a209c7258d7`
- TVM patch SHA256: `99a307ad3fa0ea648f26ad15a84c38588241de96cd11740c8f4210e7d81931a8`
- loaded TileLang module SHA256: `d05fa79cd5110c03bb399e288e17dfaf3a5b3b1225d06b7204fe8caac3274a2b`
- loaded TVM SHA256: `d8d84f3c63d40462d99a9199b85d8a1ce587da328800d281962585b7e17cfeec`
- Bisheng B150 SHA256: `8120f5d1fd5cc2df8499f2aa1b301ec7a8126d66bd23226c721de378872a918d`

| output | SHA256 |
|---|---|
| host dispatcher DSO | `c64c5b822b0b5f51c301b56adcef1aec2466fb46f80dfc9e320bba0cfb918ec9` |
| FP16 source | `7af413043df3e806c464b079dfbee0d0eb3ad5feb53b4d2d7ce42ea27b480f1a` |
| FP16 DSO | `a36dd9071ea4d0d809b5517f8b7658f120298906f059e299cad85a417916daec` |
| BF16 source | `85a99bd0d8f98c9639955e7bc46151d1d5d406ebccb7898a66d3d24cf97626ff` |
| BF16 DSO | `bf13c97d9b44fbcd6558350033079d8e1d4a2d568c3183b3cbebb9f93850a7c0` |
| FP32 source | `b07754fc297b0a49ab0db74006ed199c8f07f289a8045fc5d663e25fe6492f3e` |
| FP32 DSO | `70aba7215aacbf8c4f35d1c835b0d133d8779b04a1749fbbd35cbdccf23adc48` |
| BUILD_PROVENANCE.json | `7672176343389332d081604a7e235e1b8b84ac303aa63c7b3650b8320fe792cb` |
| RESULT.json | `fc54f51f9471d85f86e1b56c384da28bea6c2f8939fcc84b7705e9326371d52e` |
| bundle MANIFEST.sha256 | `8a7ccbeda3e901a9cbb1033d00ca7aa97b2af405c8496debfb2a606a6712f6a8` |

## Terminal controls

- compiler provenance pytest: `6 passed`
- `tilelang.lower` plus three Bisheng builds: rc 0
- host routing: 50/50 compile-dispatch rows
- rank-change: REJECTED
- `D % 8 != 0`: REJECTED
- stale official source head: REJECTED
- mutated generated artifact: REJECTED
- RTLD_NOW: PASS without device call
- symbols: `PASS_UNIQUE_SELF_BINDINGS`
- standalone provenance consumer: `PROVENANCE_PASS`, rc 0
- external manifest: 57/57 rows pass, rc 0

## Boundary and next gate

This receipt proves card-free compiler generation, native A5 build, one-host /
three-kernel dispatch, dependency provenance, manifest closure, and typed DSO
self-binding. It is author evidence and does not establish device execution,
numerical precision, performance, or product acceptance.

The next gate is fresh exclusive A5 admission followed by a non-author fixed50
consumer using these exact bytes: one dispatcher, no recompilation, bound CPU
golden for DQ/DK/DV, an observable same-session known-bad, verification that
no shape-specific kernels appear, and only then same-candidate performance.
