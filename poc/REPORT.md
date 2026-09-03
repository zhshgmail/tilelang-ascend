# TileLang e287 A5 symbolic FA backward build claims

## Bounded verdict

- recorded UTC: `2026-09-03T00:39:41Z`
- source commit/tree: `e287677f3e328cd190181edf443f855983af574a` / `33ca51941324468e785c81297619501ebdf06739`
- package: `/work/e287_builds/bundle_e287677f_b150_attempt2_packaged`
- target: A5 AscendC `dav-3510`, `CATLASS_ARCH=3510`
- `A5_BUILD_ABI`: **PASS**, build-only and typed-DSO ABI scope only
- committed claims at attempt1 review: **REJECT**
- device execution / fixed50 precision / performance: **NOT RUN**

This is author evidence only. It does not establish device execution, numerical correctness,
performance, or product acceptance. Fresh non-author exact-head review is mandatory.

## Exact identity

Fresh source observation SHA256 is
`033a023a3b4faf5e2147a7023101759b369e409b02dc25a5c2599c4f28af73b3`, observed
`2026-09-03T00:16:45Z`. The four canonical input hashes are:

- fixed50 JSON: `63b545dd67682935c51910b42c4324c4d375559aff12b44b469c1aaecd97c253`
- fixed50 shapes CSV: `6cb3fef12299b661ec01923e5216f135aa5be0fbe97abb5aad60f21684efd889`
- operator source: `2d714ce7a96c13a632c4548b0c8bd2f3af4b987873488230ae079b74c4b541e7`
- reference model: `71a29698c475bb66844308eb96429fe8300f2fd6bacfb32a3f5c338038e72793`

Compiler and toolchain hashes:

- libtilelang_module.so: `d05fa79cd5110c03bb399e288e17dfaf3a5b3b1225d06b7204fe8caac3274a2b`
- libtvm.so: `d8d84f3c63d40462d99a9199b85d8a1ce587da328800d281962585b7e17cfeec`
- Bisheng B150: `8120f5d1fd5cc2df8499f2aa1b301ec7a8126d66bd23226c721de378872a918d`

Native DSO hashes are host dispatcher `c64c5b822b0b5f51c301b56adcef1aec2466fb46f80dfc9e320bba0cfb918ec9`,
FP16 `fac6d879ab744ef4573f227513bf9c72c1abe7b76d91f12f10b9a648e40dd5b6`,
BF16 `7a3e947aa0718fd2a539f1eb2c3741701696038401bb70a2623635dcfb8ad58f`,
and FP32 `b82ac98ce5ca5d3a90b5075d172e8dc4f9b8011cdce489e96afc80469da35337`.
The build-only container is
`root@141.61.33.141:zheng-codex02-tilelang-fa-symbolic-buildonly-20260902t0510z`;
Docker inspection recorded `Devices=[]`, `DeviceRequests=null`.

## Attempt1 review and packaging correction

Fresh non-author attempt1 review:
`/work/e287_reviews/bundle_e287677f_b150_attempt1_review/REVIEW.md`, SHA256
`d68a7f97f5c89d8f260121dad1e08d60a795e32c97952da2149de4719dd2a162`.
It independently established `A5_BUILD_ABI=PASS`: exact manifest and identity rehash,
isolated typed symbols and self-bindings under RTLD_NOW, plus a retained cross-binding
known-bad that failed. Its overall consumer verdict was REJECT because the committed
BUILD_PROVENANCE claim was stale, attempt1 lacked top-level REPORT and RECEIPT, and its
manifest differed from the committed claim. This was claims packaging failure, not ABI failure.

Attempt2 is an exact copy of immutable attempt1 plus this report and receipt. Core bytes remain:

- BUILD_PROVENANCE.json: `d288a2b950ed29f44176345c3b541b83f0d5a9b6e4a39e2e1ce15bbb31d81bc8`
- RESULT.json: `c57ae8d2d9311fa415e2a7bd94103b0ff8b6821e158ded4ee1542f354689a27f`

MANIFEST.sha256 was rebuilt over every regular package member except itself and checked.
Author logs are under `/work/e287_transfer/claims_author/`. A new non-author must verify the
exact evidence commit and package. Device fixed50 DQ/DK/DV, same-session known-bad,
no-recompile proof, and same-candidate performance remain unrun gates.
