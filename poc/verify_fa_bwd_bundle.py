#!/usr/bin/env python3
"""Fail-closed consumer for a generated FA backward AscendC bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from tilelang.jit.adapter.ascendc_provenance import (
    AscendCProvenanceError,
    verify_bundle_manifest,
    verify_build_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-observation", type=Path)
    args = parser.parse_args()
    try:
        verify_build_provenance(
            args.bundle / "BUILD_PROVENANCE.json",
            args.bundle,
            args.source_observation,
        )
        verify_bundle_manifest(args.bundle)
    except (AscendCProvenanceError, OSError, ValueError) as error:
        print(f"PROVENANCE_FAIL: {error}")
        return 2
    print("PROVENANCE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
