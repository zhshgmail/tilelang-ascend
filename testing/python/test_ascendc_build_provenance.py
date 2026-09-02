import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tilelang"
    / "jit"
    / "adapter"
    / "ascendc_provenance.py"
)
SPEC = importlib.util.spec_from_file_location("ascendc_provenance_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ascendc_provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ascendc_provenance)
AscendCProvenanceError = ascendc_provenance.AscendCProvenanceError
build_identity_policy = ascendc_provenance.build_identity_policy
load_build_identity_policy = ascendc_provenance.load_build_identity_policy
sha256 = ascendc_provenance.sha256
validate_source_observation = ascendc_provenance.validate_source_observation
verify_bundle_manifest = ascendc_provenance.verify_bundle_manifest
verify_build_provenance = ascendc_provenance.verify_build_provenance
verify_committed_bundle_claims = ascendc_provenance.verify_committed_bundle_claims


INPUT_NAMES = (
    "op29_fixed50_json",
    "op29_reference_model",
    "op29_operator_source",
    "op29_fixed50_shapes_csv",
)
CLAIM_BINDINGS = {
    "BUILD_PROVENANCE.json": "poc/BUILD_PROVENANCE.author.json",
    "RESULT.json": "poc/RESULT.json",
    "REPORT.md": "poc/REPORT.md",
    "AUTHOR_BUILD_RECEIPT.md": "poc/AUTHOR_BUILD_RECEIPT.md",
    "MANIFEST.sha256": "poc/AUTHOR_BUNDLE_MANIFEST.sha256",
}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_manifest(root: Path, manifest: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == manifest:
            continue
        rows.append(f"{sha256(path)}  {path.relative_to(root)}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict:
    bundle = tmp_path / "bundle"
    trusted = tmp_path / "trusted"
    inputs_dir = bundle / "provenance" / "inputs"
    dependencies_dir = bundle / "provenance" / "dependencies"
    inputs_dir.mkdir(parents=True)
    dependencies_dir.mkdir(parents=True)

    expected_observation = tmp_path / "expected" / "SOURCE_OBSERVATION.json"
    _write_json(expected_observation, _observation())
    packaged_observation = bundle / "provenance" / "SOURCE_OBSERVATION.json"
    _write_json(packaged_observation, _observation())

    input_records = []
    for index, name in enumerate(INPUT_NAMES, 1):
        path = inputs_dir / name
        path.write_bytes(f"canonical-{index}-{name}\n".encode())
        input_records.append(
            {
                "name": name,
                "path": str(path.relative_to(bundle)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )

    patch = dependencies_dir / "tvm.patch"
    patch.write_bytes(b"diff --git a/a b/a\n+required patch\n")
    artifact = bundle / "generated" / "kernel.so"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"content-addressed-kernel")
    record = {
        "schema_version": 1,
        "authority": "AUTHOR_BUILD_PROVENANCE_ONLY",
        "device_execution": "NOT_RUN_NO_NPU_ADMISSION",
        "source_observation": {
            "path": str(packaged_observation.relative_to(bundle)),
            "sha256": sha256(packaged_observation),
            "observed_at_utc": _observation()["observed_at_utc"],
            "official_head": _observation()["official"]["head"],
            "colleague_a3_head": _observation()["colleague_a3"]["head"],
        },
        "source_repo": {
            "path_diagnostic_only": "/build/source",
            "origin": "https://github.com/tile-ai/tilelang-ascend.git",
            "branch": "DETACHED",
            "commit": "4" * 40,
            "tree": "5" * 40,
            "status": [" M 3rdparty/tvm"],
        },
        "loaded_compiler": {
            "libtilelang_module.so": {
                "path": "/build/libtilelang_module.so",
                "sha256": "6" * 64,
                "bytes": 128,
            },
            "libtvm.so": {
                "path": "/build/libtvm.so",
                "sha256": "7" * 64,
                "bytes": 256,
            },
        },
        "toolchain": {
            "path": "/opt/bisheng",
            "sha256": "8" * 64,
            "bytes": 512,
            "version": "Bisheng 15.0.5",
            "version_sha256": hashlib.sha256(b"Bisheng 15.0.5").hexdigest(),
        },
        "target": {
            "backend": "ascendc",
            "platform": "A5",
            "npu_arch": "dav-3510",
            "catlass_arch": "3510",
        },
        "dependencies": [
            {
                "path": "3rdparty/tvm",
                "gitlink": "9" * 40,
                "actual_head": "9" * 40,
                "patch": str(patch.relative_to(bundle)),
                "patch_sha256": sha256(patch),
                "status": [" M python/tvm/tir/buffer.py"],
                "recursive_submodules": [" a" * 20 + " 3rdparty/dlpack"],
            }
        ],
        "inputs": input_records,
        "artifacts": [
            {
                "path": str(artifact.relative_to(bundle)),
                "sha256": sha256(artifact),
                "bytes": artifact.stat().st_size,
            }
        ],
    }
    provenance = bundle / "BUILD_PROVENANCE.json"
    _write_json(provenance, record)
    (bundle / "RESULT.json").write_text('{"status":"AUTHOR_ONLY"}\n', encoding="utf-8")
    (bundle / "REPORT.md").write_text("# Author report\n", encoding="utf-8")
    (bundle / "AUTHOR_BUILD_RECEIPT.md").write_text(
        "# Author receipt\n", encoding="utf-8"
    )
    bundle_manifest = bundle / "MANIFEST.sha256"
    _write_manifest(bundle, bundle_manifest)

    trusted_claims = {
        "poc/BUILD_PROVENANCE.author.json": provenance,
        "poc/RESULT.json": bundle / "RESULT.json",
        "poc/REPORT.md": bundle / "REPORT.md",
        "poc/AUTHOR_BUILD_RECEIPT.md": bundle / "AUTHOR_BUILD_RECEIPT.md",
        "poc/AUTHOR_BUNDLE_MANIFEST.sha256": bundle_manifest,
    }
    for relative, source in trusted_claims.items():
        path = trusted / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
    trusted_manifest = trusted / "poc" / "MANIFEST.sha256"
    _write_manifest(trusted, trusted_manifest)

    policy_path = tmp_path / "BUILD_IDENTITY_POLICY.json"
    _write_json(
        policy_path,
        {"schema_version": 1, "build_identity": build_identity_policy(record)},
    )
    return {
        "bundle": bundle,
        "trusted": trusted,
        "trusted_manifest": trusted_manifest,
        "trusted_manifest_sha256": sha256(trusted_manifest),
        "policy": policy_path,
        "policy_sha256": sha256(policy_path),
        "observation": expected_observation,
        "record": record,
        "provenance": provenance,
    }


def _verify(value: dict) -> None:
    identity = load_build_identity_policy(value["policy"], value["policy_sha256"])
    verify_build_provenance(
        value["provenance"],
        value["bundle"],
        value["observation"],
        identity,
        required_input_names=INPUT_NAMES,
    )
    verify_bundle_manifest(value["bundle"])
    verify_committed_bundle_claims(
        bundle_root=value["bundle"],
        trusted_root=value["trusted"],
        trusted_manifest=value["trusted_manifest"],
        expected_trusted_manifest_sha256=value["trusted_manifest_sha256"],
        claim_bindings=CLAIM_BINDINGS,
    )


def _write_record(value: dict, record: dict) -> None:
    value["record"] = record
    _write_json(value["provenance"], record)


def _refresh_bundle_manifest(value: dict) -> None:
    _write_manifest(value["bundle"], value["bundle"] / "MANIFEST.sha256")


def _refresh_trusted_manifest(value: dict) -> None:
    _write_manifest(value["trusted"], value["trusted_manifest"])


MUTATION_CASES = [
    "review_exact_head_binding_absent",
    "source_commit_mutated",
    "source_tree_mutated",
    "source_repo_missing",
    "loaded_compiler_missing",
    "loaded_compiler_empty",
    "loaded_compiler_one_missing",
    "loaded_compiler_extra",
    "loaded_compiler_digest_mutated",
    "loaded_compiler_path_mutated",
    "toolchain_missing",
    "toolchain_path_mutated",
    "toolchain_digest_mutated",
    "toolchain_version_mutated",
    "target_missing",
    "target_arch_mutated",
    "target_platform_mutated",
    "dependencies_missing",
    "dependencies_empty",
    "dependencies_extra",
    "dependency_recursive_submodules_empty",
    "dependency_status_empty",
    "dependency_patch_empty_self_consistent",
    "dependency_patch_missing",
    "dependency_unlisted_extra_file_self_manifested",
    "inputs_missing",
    "inputs_empty",
    "canonical_json_model_source_binding_absent",
    "input_empty_self_consistent",
    "canonical_fixed50_replaced_self_consistent",
    "inputs_extra",
    "input_unlisted_extra_file_self_manifested",
    "input_file_missing",
    "source_observation_stale_bytes",
    "artifact_bytes_mutated",
    "artifact_and_provenance_self_consistent_rewrite",
    "bundle_extra_unmanifested",
    "bundle_manifest_digest_mutated",
    "bundle_extra_self_manifested",
    "committed_poc_manifest_bytes_mutated",
    "committed_author_bundle_manifest_bytes_mutated",
    "claim_missing_RESULT_json",
    "claim_missing_REPORT_md",
    "claim_missing_AUTHOR_BUILD_RECEIPT_md",
    "claim_missing_AUTHOR_BUNDLE_MANIFEST_sha256",
    "claim_mutated_RESULT_json",
    "claim_mutated_REPORT_md",
    "claim_mutated_AUTHOR_BUILD_RECEIPT_md",
    "stale_external_bundle_source_commit",
]


def _mutate(value: dict, case: str) -> None:
    record = copy.deepcopy(value["record"])
    bundle = value["bundle"]
    if case in {"review_exact_head_binding_absent", "source_commit_mutated"}:
        record["source_repo"]["commit"] = "a" * 40
    elif case == "source_tree_mutated":
        record["source_repo"]["tree"] = "a" * 40
    elif case == "source_repo_missing":
        record.pop("source_repo")
    elif case == "loaded_compiler_missing":
        record.pop("loaded_compiler")
    elif case == "loaded_compiler_empty":
        record["loaded_compiler"] = {}
    elif case == "loaded_compiler_one_missing":
        record["loaded_compiler"].pop("libtvm.so")
    elif case == "loaded_compiler_extra":
        record["loaded_compiler"]["extra.so"] = copy.deepcopy(
            record["loaded_compiler"]["libtvm.so"]
        )
    elif case == "loaded_compiler_digest_mutated":
        record["loaded_compiler"]["libtvm.so"]["sha256"] = "a" * 64
    elif case == "loaded_compiler_path_mutated":
        record["loaded_compiler"]["libtvm.so"]["path"] = "/other/libtvm.so"
    elif case == "toolchain_missing":
        record.pop("toolchain")
    elif case == "toolchain_path_mutated":
        record["toolchain"]["path"] = "/other/bisheng"
    elif case == "toolchain_digest_mutated":
        record["toolchain"]["sha256"] = "a" * 64
    elif case == "toolchain_version_mutated":
        record["toolchain"]["version"] = "other version"
        record["toolchain"]["version_sha256"] = hashlib.sha256(
            b"other version"
        ).hexdigest()
    elif case == "target_missing":
        record.pop("target")
    elif case == "target_arch_mutated":
        record["target"]["npu_arch"] = "dav-2201"
    elif case == "target_platform_mutated":
        record["target"]["platform"] = "A3"
    elif case == "dependencies_missing":
        record.pop("dependencies")
    elif case == "dependencies_empty":
        record["dependencies"] = []
    elif case == "dependencies_extra":
        extra = copy.deepcopy(record["dependencies"][0])
        extra["path"] = "3rdparty/extra"
        record["dependencies"].append(extra)
    elif case == "dependency_recursive_submodules_empty":
        record["dependencies"][0]["recursive_submodules"] = []
    elif case == "dependency_status_empty":
        record["dependencies"][0]["status"] = []
    elif case == "dependency_patch_empty_self_consistent":
        patch = bundle / record["dependencies"][0]["patch"]
        patch.write_bytes(b"")
        record["dependencies"][0]["patch_sha256"] = sha256(patch)
    elif case == "dependency_patch_missing":
        (bundle / record["dependencies"][0]["patch"]).unlink()
    elif case == "dependency_unlisted_extra_file_self_manifested":
        (bundle / "provenance/dependencies/extra.patch").write_bytes(b"extra")
        _refresh_bundle_manifest(value)
        return
    elif case == "inputs_missing":
        record.pop("inputs")
    elif case == "inputs_empty":
        record["inputs"] = []
    elif case == "canonical_json_model_source_binding_absent":
        record["inputs"] = [record["inputs"][-1]]
    elif case == "input_empty_self_consistent":
        item = record["inputs"][0]
        path = bundle / item["path"]
        path.write_bytes(b"")
        item["bytes"] = 0
        item["sha256"] = sha256(path)
    elif case == "canonical_fixed50_replaced_self_consistent":
        item = record["inputs"][0]
        path = bundle / item["path"]
        path.write_bytes(b"replacement canonical input")
        item["bytes"] = path.stat().st_size
        item["sha256"] = sha256(path)
    elif case == "inputs_extra":
        path = bundle / "provenance/inputs/extra"
        path.write_bytes(b"extra")
        record["inputs"].append(
            {
                "name": "extra",
                "path": str(path.relative_to(bundle)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    elif case == "input_unlisted_extra_file_self_manifested":
        (bundle / "provenance/inputs/extra").write_bytes(b"extra")
        _refresh_bundle_manifest(value)
        return
    elif case == "input_file_missing":
        (bundle / record["inputs"][0]["path"]).unlink()
    elif case == "source_observation_stale_bytes":
        (bundle / record["source_observation"]["path"]).write_bytes(b"{}\n")
    elif case == "artifact_bytes_mutated":
        (bundle / record["artifacts"][0]["path"]).write_bytes(b"mutated")
    elif case == "artifact_and_provenance_self_consistent_rewrite":
        item = record["artifacts"][0]
        path = bundle / item["path"]
        path.write_bytes(b"mutated")
        item["bytes"] = path.stat().st_size
        item["sha256"] = sha256(path)
        _write_record(value, record)
        _refresh_bundle_manifest(value)
        return
    elif case == "bundle_extra_unmanifested":
        (bundle / "EXTRA.bin").write_bytes(b"extra")
        return
    elif case == "bundle_manifest_digest_mutated":
        manifest = bundle / "MANIFEST.sha256"
        lines = manifest.read_text(encoding="utf-8").splitlines()
        lines[0] = "0" * 64 + lines[0][64:]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    elif case == "bundle_extra_self_manifested":
        (bundle / "EXTRA.bin").write_bytes(b"extra")
        _refresh_bundle_manifest(value)
        return
    elif case == "committed_poc_manifest_bytes_mutated":
        value["trusted_manifest"].write_text("mutated\n", encoding="utf-8")
        return
    elif case == "committed_author_bundle_manifest_bytes_mutated":
        path = value["trusted"] / "poc/AUTHOR_BUNDLE_MANIFEST.sha256"
        path.write_text("mutated\n", encoding="utf-8")
        _refresh_trusted_manifest(value)
        return
    elif case.startswith("claim_missing_"):
        names = {
            "claim_missing_RESULT_json": "poc/RESULT.json",
            "claim_missing_REPORT_md": "poc/REPORT.md",
            "claim_missing_AUTHOR_BUILD_RECEIPT_md": "poc/AUTHOR_BUILD_RECEIPT.md",
            "claim_missing_AUTHOR_BUNDLE_MANIFEST_sha256": (
                "poc/AUTHOR_BUNDLE_MANIFEST.sha256"
            ),
        }
        (value["trusted"] / names[case]).unlink()
        return
    elif case.startswith("claim_mutated_"):
        names = {
            "claim_mutated_RESULT_json": "poc/RESULT.json",
            "claim_mutated_REPORT_md": "poc/REPORT.md",
            "claim_mutated_AUTHOR_BUILD_RECEIPT_md": "poc/AUTHOR_BUILD_RECEIPT.md",
        }
        (value["trusted"] / names[case]).write_text("mutated\n", encoding="utf-8")
        _refresh_trusted_manifest(value)
        return
    elif case == "stale_external_bundle_source_commit":
        record["source_repo"]["commit"] = "b" * 40
    else:  # pragma: no cover
        raise AssertionError(f"unimplemented mutation case: {case}")
    _write_record(value, record)


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


def test_complete_externally_anchored_bundle_positive(tmp_path):
    _verify(_fixture(tmp_path))


def test_standalone_consumer_accepts_complete_externally_anchored_bundle(tmp_path):
    value = _fixture(tmp_path)
    consumer = Path(__file__).resolve().parents[2] / "poc" / "verify_fa_bwd_bundle.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(consumer),
            "--bundle",
            str(value["bundle"]),
            "--source-observation",
            str(value["observation"]),
            "--identity-policy",
            str(value["policy"]),
            "--identity-policy-sha256",
            value["policy_sha256"],
            "--trusted-root",
            str(value["trusted"]),
            "--trusted-manifest",
            str(value["trusted_manifest"]),
            "--trusted-manifest-sha256",
            value["trusted_manifest_sha256"],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "PROVENANCE_PASS"


def test_standalone_consumer_requires_every_external_anchor(tmp_path):
    consumer = Path(__file__).resolve().parents[2] / "poc" / "verify_fa_bwd_bundle.py"
    completed = subprocess.run(
        [sys.executable, str(consumer), "--bundle", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    for name in (
        "--source-observation",
        "--identity-policy",
        "--identity-policy-sha256",
        "--trusted-root",
        "--trusted-manifest",
        "--trusted-manifest-sha256",
    ):
        assert name in completed.stderr


@pytest.mark.parametrize("case", MUTATION_CASES)
def test_known_bad_mutation_matrix_is_fail_closed(tmp_path, case):
    value = _fixture(tmp_path)
    _verify(value)
    _mutate(value, case)
    with pytest.raises((AscendCProvenanceError, OSError, ValueError, KeyError)):
        _verify(value)
