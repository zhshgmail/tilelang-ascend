import json
import subprocess
from pathlib import Path

import pytest

from tilelang.jit.adapter import ascendc_provenance
from tilelang.jit.adapter.ascendc_provenance import (
    AscendCProvenanceError,
    sha256,
    validate_source_observation,
    verify_bundle_manifest,
    verify_build_provenance,
)


def _observation() -> dict:
    return {
        "schema_version": 1,
        "observed_at_utc": "2026-09-02T17:06:28Z",
        "official": {
            "remote": "https://github.com/tile-ai/tilelang-ascend.git",
            "ref": "refs/heads/ascendc_pto",
            "head": "1" * 40,
        },
        "colleague_a3": {
            "remote": "https://github.com/wzzll123/tilelang-ascend.git",
            "ref": "refs/heads/ascendc_pto",
            "head": "2" * 40,
        },
        "pull_requests": [{"number": 1702, "head": "3" * 40}],
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    observation = _observation()
    observation_path = tmp_path / "SOURCE_OBSERVATION.json"
    _write_json(observation_path, observation)
    artifact = tmp_path / "kernel.so"
    artifact.write_bytes(b"not-an-elf-but-content-addressed")
    record = {
        "schema_version": 1,
        "device_execution": "NOT_RUN_NO_NPU_ADMISSION",
        "source_observation": {
            "path": observation_path.name,
            "sha256": sha256(observation_path),
            "observed_at_utc": observation["observed_at_utc"],
            "official_head": observation["official"]["head"],
            "colleague_a3_head": observation["colleague_a3"]["head"],
        },
        "dependencies": [],
        "inputs": [],
        "artifacts": [
            {
                "path": artifact.name,
                "sha256": sha256(artifact),
                "bytes": artifact.stat().st_size,
            }
        ],
    }
    provenance = tmp_path / "BUILD_PROVENANCE.json"
    _write_json(provenance, record)
    return provenance, observation_path, artifact


def test_source_observation_has_one_timestamp():
    validate_source_observation(_observation())


def test_source_observation_rejects_nested_mixed_timestamp():
    observation = _observation()
    observation["official"]["observed_at_utc"] = "2026-08-01T00:00:00Z"
    with pytest.raises(AscendCProvenanceError, match="mixed-time"):
        validate_source_observation(observation)


def test_git_porcelain_keeps_leading_index_column(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        cwd=tmp_path,
        check=True,
    )
    tracked.write_text("after\n", encoding="utf-8")
    assert ascendc_provenance._git_lines(tmp_path, "status", "--porcelain=v1") == [
        " M tracked.txt"
    ]


def test_bundle_provenance_positive_and_stale_head_negative(tmp_path):
    provenance, observation, _ = _bundle(tmp_path)
    verify_build_provenance(provenance, tmp_path, observation)
    value = json.loads(provenance.read_text(encoding="utf-8"))
    value["source_observation"]["official_head"] = "0" * 40
    _write_json(provenance, value)
    with pytest.raises(AscendCProvenanceError, match="official source head mismatch"):
        verify_build_provenance(provenance, tmp_path, observation)


def test_bundle_provenance_rejects_artifact_mutation(tmp_path):
    provenance, observation, artifact = _bundle(tmp_path)
    artifact.write_bytes(b"mutated")
    with pytest.raises(AscendCProvenanceError, match="artifact mismatch"):
        verify_build_provenance(provenance, tmp_path, observation)


def test_bundle_manifest_is_closed_and_rejects_unmanifested_file(tmp_path):
    first = tmp_path / "a.txt"
    first.write_text("a\n", encoding="utf-8")
    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text(f"{sha256(first)}  a.txt\n", encoding="utf-8")
    verify_bundle_manifest(tmp_path)
    (tmp_path / "unmanifested.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(AscendCProvenanceError, match="manifest closure mismatch"):
        verify_bundle_manifest(tmp_path)
