# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Content-addressed provenance for generated AscendC host+kernel bundles."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class AscendCProvenanceError(RuntimeError):
    """Raised when a generated bundle cannot prove its build identity."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: Sequence[str], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command), cwd=cwd, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise AscendCProvenanceError(
            f"command failed rc={completed.returncode}: {list(command)!r}: "
            f"{completed.stderr.strip()}"
        )
    return completed


def _git(repo: Path, *args: str) -> str:
    return _run(["git", *args], cwd=repo).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_source_observation(value: Mapping[str, Any]) -> None:
    """Reject mixed-time or structurally incomplete public-source receipts."""

    if value.get("schema_version") != 1:
        raise AscendCProvenanceError("source observation schema_version must be 1")
    observed_at = value.get("observed_at_utc")
    if not isinstance(observed_at, str) or not observed_at.endswith("Z"):
        raise AscendCProvenanceError("source observation needs one UTC observed_at_utc")

    def reject_nested_observed_at(node: object, path: str = "") -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                child = f"{path}.{key}" if path else key
                if path and "observed_at" in key:
                    raise AscendCProvenanceError(
                        f"mixed-time source observation field is forbidden: {child}"
                    )
                reject_nested_observed_at(item, child)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                reject_nested_observed_at(item, f"{path}[{index}]")

    reject_nested_observed_at(dict(value))
    for name in ("official", "colleague_a3"):
        source = value.get(name)
        if not isinstance(source, dict):
            raise AscendCProvenanceError(f"source observation missing {name}")
        if not _GIT_SHA_RE.fullmatch(str(source.get("head", ""))):
            raise AscendCProvenanceError(f"source observation has invalid {name}.head")
        if not str(source.get("remote", "")).startswith("https://github.com/"):
            raise AscendCProvenanceError(
                f"source observation has invalid {name}.remote"
            )

    pulls = value.get("pull_requests")
    if not isinstance(pulls, list) or not pulls:
        raise AscendCProvenanceError("source observation has no pull_requests")
    for pull in pulls:
        if not isinstance(pull, dict) or not _GIT_SHA_RE.fullmatch(
            str(pull.get("head", ""))
        ):
            raise AscendCProvenanceError("source observation has invalid PR head")


def load_source_observation(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AscendCProvenanceError("source observation must be a JSON object")
    validate_source_observation(value)
    return value


def _capture_dependency(
    repo: Path, submodule: str, patch_path: Path, provenance_dir: Path
) -> dict[str, Any]:
    entry = _git(repo, "ls-tree", "HEAD", "--", submodule).split()
    if len(entry) < 3 or entry[0] != "160000" or entry[1] != "commit":
        raise AscendCProvenanceError(f"{submodule} is not a gitlink at HEAD")
    gitlink = entry[2]
    dependency = repo / submodule
    actual_head = _git(dependency, "rev-parse", "HEAD")
    if actual_head != gitlink:
        raise AscendCProvenanceError(
            f"{submodule} HEAD {actual_head} does not match gitlink {gitlink}"
        )
    live_patch = _run(["git", "diff", "--binary"], cwd=dependency).stdout.encode()
    expected_patch = patch_path.read_bytes()
    if live_patch != expected_patch:
        raise AscendCProvenanceError(
            f"{submodule} live diff does not match {patch_path.relative_to(repo)}"
        )
    copied_patch = provenance_dir / patch_path.name
    shutil.copyfile(patch_path, copied_patch)
    recursive_status = _run(
        ["git", "submodule", "status", "--recursive"], cwd=dependency
    ).stdout.splitlines()
    unclean_submodules = [line for line in recursive_status if not line.startswith(" ")]
    if unclean_submodules:
        raise AscendCProvenanceError(
            f"{submodule} has unavailable or divergent nested submodules: "
            f"{unclean_submodules}"
        )
    return {
        "path": submodule,
        "gitlink": gitlink,
        "actual_head": actual_head,
        "patch": str(copied_patch.relative_to(provenance_dir.parent)),
        "patch_sha256": sha256(copied_patch),
        "status": _git(
            dependency, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines(),
        "recursive_submodules": recursive_status,
    }


def _capture_loaded_library(basename: str) -> dict[str, Any]:
    paths = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 6 and Path(fields[-1]).name == basename:
            paths.add(str(Path(fields[-1]).resolve()))
    if len(paths) != 1:
        raise AscendCProvenanceError(
            f"expected exactly one loaded {basename}, found {sorted(paths)}"
        )
    path = Path(paths.pop())
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def _capture_tool(executable: str) -> dict[str, Any]:
    resolved = shutil.which(executable)
    if resolved is None:
        raise AscendCProvenanceError(f"tool not found: {executable}")
    path = Path(resolved).resolve()
    version = _run([str(path), "--version"])
    version_text = version.stdout + version.stderr
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "version": version_text.strip(),
        "version_sha256": hashlib.sha256(version_text.encode()).hexdigest(),
    }


def _capture_artifacts(
    bundle_root: Path, paths: Sequence[Path]
) -> list[dict[str, Any]]:
    records = []
    for path in sorted(paths):
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(bundle_root.resolve())
        except ValueError as error:
            raise AscendCProvenanceError(
                f"artifact escapes bundle root: {path}"
            ) from error
        if not resolved.is_file():
            raise AscendCProvenanceError(f"artifact is not a regular file: {path}")
        records.append(
            {
                "path": str(relative),
                "sha256": sha256(resolved),
                "bytes": resolved.stat().st_size,
            }
        )
    return records


def _capture_inputs(
    bundle_root: Path, paths: Mapping[str, Path], provenance_dir: Path
) -> list[dict[str, Any]]:
    inputs_dir = provenance_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for name, source in sorted(paths.items()):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise AscendCProvenanceError(f"invalid provenance input name: {name!r}")
        resolved = source.resolve()
        if not resolved.is_file():
            raise AscendCProvenanceError(f"input is not a regular file: {source}")
        copied = inputs_dir / name
        shutil.copyfile(resolved, copied)
        records.append(
            {
                "name": name,
                "path": str(copied.relative_to(bundle_root)),
                "sha256": sha256(copied),
                "bytes": copied.stat().st_size,
            }
        )
    return records


def capture_build_provenance(
    *,
    repo: Path,
    bundle_root: Path,
    source_observation_path: Path,
    artifact_paths: Sequence[Path],
    input_paths: Mapping[str, Path],
    dependency_patches: Mapping[str, Path],
    toolchain: str,
    target: Mapping[str, str],
) -> tuple[Path, dict[str, Any]]:
    """Capture the exact source, dependency, runtime, tool, and bundle identity."""

    repo = repo.resolve()
    bundle_root = bundle_root.resolve()
    provenance_dir = bundle_root / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=False)

    source_observation = load_source_observation(source_observation_path)
    copied_observation = provenance_dir / "SOURCE_OBSERVATION.json"
    shutil.copyfile(source_observation_path, copied_observation)

    allowed_paths = set(dependency_patches)
    status = _git(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    unexpected = []
    for line in status:
        path = line[3:] if len(line) >= 4 else ""
        if path not in allowed_paths:
            unexpected.append(line)
    if unexpected:
        raise AscendCProvenanceError(f"unexpected dirty source state: {unexpected}")
    if {line[3:] for line in status} != allowed_paths:
        raise AscendCProvenanceError(
            f"expected dirty dependency paths {sorted(allowed_paths)}, got {status}"
        )

    dependencies = [
        _capture_dependency(repo, submodule, patch, provenance_dir)
        for submodule, patch in sorted(dependency_patches.items())
    ]
    record = {
        "schema_version": 1,
        "authority": "AUTHOR_BUILD_PROVENANCE_ONLY",
        "source_observation": {
            "path": str(copied_observation.relative_to(bundle_root)),
            "sha256": sha256(copied_observation),
            "observed_at_utc": source_observation["observed_at_utc"],
            "official_head": source_observation["official"]["head"],
            "colleague_a3_head": source_observation["colleague_a3"]["head"],
        },
        "source_repo": {
            "path_diagnostic_only": str(repo),
            "origin": _git(repo, "remote", "get-url", "origin"),
            "branch": _git(repo, "branch", "--show-current") or "DETACHED",
            "commit": _git(repo, "rev-parse", "HEAD"),
            "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
            "status": status,
        },
        "dependencies": dependencies,
        "loaded_compiler": {
            name: _capture_loaded_library(name)
            for name in ("libtilelang_module.so", "libtvm.so")
        },
        "toolchain": _capture_tool(toolchain),
        "target": dict(target),
        "inputs": _capture_inputs(bundle_root, input_paths, provenance_dir),
        "artifacts": _capture_artifacts(bundle_root, artifact_paths),
        "device_execution": "NOT_RUN_NO_NPU_ADMISSION",
    }
    output = bundle_root / "BUILD_PROVENANCE.json"
    _write_json(output, record)
    verify_build_provenance(output, bundle_root, source_observation_path)
    return output, record


def verify_build_provenance(
    provenance_path: Path,
    bundle_root: Path,
    expected_source_observation: Path | None = None,
) -> None:
    """Verify a packaged provenance sidecar without trusting its author text."""

    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise AscendCProvenanceError("build provenance schema_version must be 1")
    if record.get("device_execution") != "NOT_RUN_NO_NPU_ADMISSION":
        raise AscendCProvenanceError("unexpected device-execution claim")

    source = record.get("source_observation", {})
    source_path = bundle_root / str(source.get("path", ""))
    if not source_path.is_file() or sha256(source_path) != source.get("sha256"):
        raise AscendCProvenanceError("source observation sidecar digest mismatch")
    source_value = load_source_observation(source_path)
    if source_value["observed_at_utc"] != source.get("observed_at_utc"):
        raise AscendCProvenanceError("source observation timestamp mismatch")
    if source_value["official"]["head"] != source.get("official_head"):
        raise AscendCProvenanceError("official source head mismatch")
    if source_value["colleague_a3"]["head"] != source.get("colleague_a3_head"):
        raise AscendCProvenanceError("A3 source head mismatch")
    if (
        expected_source_observation is not None
        and source_path.read_bytes() != expected_source_observation.read_bytes()
    ):
        raise AscendCProvenanceError(
            "packaged source observation differs from bound input"
        )

    for input_record in record.get("inputs", []):
        path = (bundle_root / input_record["path"]).resolve()
        try:
            path.relative_to(bundle_root.resolve())
        except ValueError as error:
            raise AscendCProvenanceError("input path escapes bundle root") from error
        if (
            not path.is_file()
            or path.stat().st_size != input_record["bytes"]
            or sha256(path) != input_record["sha256"]
        ):
            raise AscendCProvenanceError(f"input mismatch: {input_record['path']}")

    for dependency in record.get("dependencies", []):
        patch = bundle_root / dependency["patch"]
        if not patch.is_file() or sha256(patch) != dependency["patch_sha256"]:
            raise AscendCProvenanceError("dependency patch digest mismatch")
        if dependency["gitlink"] != dependency["actual_head"]:
            raise AscendCProvenanceError("dependency gitlink/HEAD mismatch")

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AscendCProvenanceError("build provenance has no artifacts")
    for artifact in artifacts:
        digest = str(artifact.get("sha256", ""))
        if not _SHA256_RE.fullmatch(digest):
            raise AscendCProvenanceError("invalid artifact digest")
        path = (bundle_root / artifact["path"]).resolve()
        try:
            path.relative_to(bundle_root.resolve())
        except ValueError as error:
            raise AscendCProvenanceError("artifact path escapes bundle root") from error
        if (
            not path.is_file()
            or path.stat().st_size != artifact["bytes"]
            or sha256(path) != digest
        ):
            raise AscendCProvenanceError(f"artifact mismatch: {artifact['path']}")


def verify_bundle_manifest(bundle_root: Path) -> None:
    """Verify the manifest covers every regular bundle member exactly once."""

    manifest = bundle_root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise AscendCProvenanceError("bundle manifest is absent")
    recorded: dict[str, str] = {}
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise AscendCProvenanceError(f"malformed manifest row {number}")
        digest, relative = match.groups()
        if relative in recorded:
            raise AscendCProvenanceError(f"duplicate manifest path: {relative}")
        path = (bundle_root / relative).resolve()
        try:
            path.relative_to(bundle_root.resolve())
        except ValueError as error:
            raise AscendCProvenanceError("manifest path escapes bundle root") from error
        if not path.is_file() or path.is_symlink() or sha256(path) != digest:
            raise AscendCProvenanceError(f"manifest member mismatch: {relative}")
        recorded[relative] = digest
    actual = {
        str(path.relative_to(bundle_root))
        for path in bundle_root.rglob("*")
        if path.is_file() and path != manifest
    }
    if set(recorded) != actual:
        raise AscendCProvenanceError(
            "manifest closure mismatch: "
            f"missing={sorted(actual - set(recorded))} "
            f"extra={sorted(set(recorded) - actual)}"
        )


def run_provenance_negative_controls(
    provenance_path: Path, bundle_root: Path, source_observation_path: Path
) -> dict[str, str]:
    """Prove stale source identity and artifact hashes are rejected."""

    original = json.loads(provenance_path.read_text(encoding="utf-8"))
    controls = {}
    for name, mutate in (
        (
            "stale_official_head",
            lambda value: value["source_observation"].__setitem__(
                "official_head", "0" * 40
            ),
        ),
        (
            "artifact_digest",
            lambda value: value["artifacts"][0].__setitem__("sha256", "0" * 64),
        ),
    ):
        mutated = copy.deepcopy(original)
        mutate(mutated)
        path = bundle_root / f"provenance/{name}.known_bad.json"
        _write_json(path, mutated)
        try:
            verify_build_provenance(path, bundle_root, source_observation_path)
        except AscendCProvenanceError:
            controls[name] = "REJECTED"
        else:
            raise AssertionError(f"provenance known-bad did not fire: {name}")
    return controls
