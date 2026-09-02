#!/usr/bin/env python3
"""Fail-closed consumer for a generated FA backward AscendC bundle."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tilelang"
    / "jit"
    / "adapter"
    / "ascendc_provenance.py"
)
_SPEC = importlib.util.spec_from_file_location("ascendc_provenance_consumer", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load provenance verifier: {_MODULE_PATH}")
_PROVENANCE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROVENANCE)
AscendCProvenanceError = _PROVENANCE.AscendCProvenanceError
load_build_identity_policy = _PROVENANCE.load_build_identity_policy
verify_bundle_manifest = _PROVENANCE.verify_bundle_manifest
verify_build_provenance = _PROVENANCE.verify_build_provenance
verify_committed_bundle_claims = _PROVENANCE.verify_committed_bundle_claims


_REQUIRED_INPUT_NAMES = (
    "op29_fixed50_json",
    "op29_reference_model",
    "op29_operator_source",
    "op29_fixed50_shapes_csv",
)
_EXPECTED_TARGET = {
    "backend": "ascendc",
    "platform": "A5",
    "npu_arch": "dav-3510",
    "catlass_arch": "3510",
}
_CLAIM_BINDINGS = {
    "BUILD_PROVENANCE.json": "poc/BUILD_PROVENANCE.author.json",
    "RESULT.json": "poc/RESULT.json",
    "REPORT.md": "poc/REPORT.md",
    "AUTHOR_BUILD_RECEIPT.md": "poc/AUTHOR_BUILD_RECEIPT.md",
    "MANIFEST.sha256": "poc/AUTHOR_BUNDLE_MANIFEST.sha256",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-observation", type=Path, required=True)
    parser.add_argument("--identity-policy", type=Path, required=True)
    parser.add_argument("--identity-policy-sha256", required=True)
    parser.add_argument("--trusted-root", type=Path, required=True)
    parser.add_argument("--trusted-manifest", type=Path, required=True)
    parser.add_argument("--trusted-manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        identity = load_build_identity_policy(
            args.identity_policy, args.identity_policy_sha256
        )
        if identity.get("target") != _EXPECTED_TARGET:
            raise AscendCProvenanceError("identity policy does not bind the A5 target")
        verify_build_provenance(
            args.bundle / "BUILD_PROVENANCE.json",
            args.bundle,
            args.source_observation,
            identity,
            required_input_names=_REQUIRED_INPUT_NAMES,
        )
        verify_bundle_manifest(args.bundle)
        verify_committed_bundle_claims(
            bundle_root=args.bundle,
            trusted_root=args.trusted_root,
            trusted_manifest=args.trusted_manifest,
            expected_trusted_manifest_sha256=args.trusted_manifest_sha256,
            claim_bindings=_CLAIM_BINDINGS,
        )
    except (AscendCProvenanceError, OSError, ValueError, KeyError) as error:
        print(f"PROVENANCE_FAIL: {error}")
        return 2
    print("PROVENANCE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
