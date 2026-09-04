"""Central-profiler adapter for the exact e28825ac FA-bwd candidate.

This file is copied to ``model_new_ascendc.py`` by
``prepare_fa_bwd_canonical_perf.py``.  It deliberately owns no timing or
verdict logic: the locked central provider profiles ``ModelNew.forward``.
"""

from __future__ import annotations

import hashlib
import json
import os
import ctypes
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu

import fa_bwd_consumer as consumer


ROOT = Path(__file__).resolve().parent
BINDING_PATH = ROOT / "PERF_INPUT_BINDING.json"
BUNDLE = ROOT / "candidate" / "bundle"
MAPPING_DIR = ROOT / "mapping_receipts"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_binding() -> dict:
    binding = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    if binding.get("schema") != "tilelang.fa_bwd_canonical_perf_input/1":
        raise consumer.ContractError("unexpected performance input binding schema")
    if binding.get("source_commit") != "e28825ac9af5264b85a97e8ec0e25f3d238c37a3":
        raise consumer.ContractError("performance input is not source e28825ac")
    expected = binding["candidate_files"]
    for relative, wanted in expected.items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != wanted:
            raise consumer.ContractError(f"candidate hash mismatch: {relative}")
    consumer.verify_sha256_manifest(BUNDLE, BUNDLE / "MANIFEST.sha256")
    return binding


def _mapped_bundle_files(binding: dict) -> list[dict[str, str]]:
    bundle_root = BUNDLE.resolve(strict=True)
    mapped: dict[str, Path] = {}
    for raw in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if len(fields) < 6 or not fields[-1].startswith("/"):
            continue
        path = Path(fields[-1]).resolve(strict=False)
        try:
            relative = path.relative_to(bundle_root).as_posix()
        except ValueError:
            continue
        if path.suffix == ".so":
            mapped[relative] = path
    expected = sorted(binding["mapped_candidate_files"])
    if sorted(mapped) != expected:
        raise consumer.ContractError(
            f"bundle-owned mapped set differs: expected={expected}, observed={sorted(mapped)}"
        )
    rows = []
    for relative in expected:
        path = mapped[relative]
        wanted = binding["candidate_files"][f"candidate/bundle/{relative}"]
        actual = _sha256(path)
        if actual != wanted:
            raise consumer.ContractError(f"mapped candidate hash mismatch: {relative}")
        rows.append({"relative_path": relative, "resolved_path": str(path), "sha256": actual})
    return rows


_BINDING = _load_binding()
_MODE = os.RTLD_NOW | os.RTLD_GLOBAL
_KERNEL_DIR = BUNDLE / "generated" / "kernel"
_KERNEL_HANDLES = tuple(
    ctypes.CDLL(str(_KERNEL_DIR / name), mode=_MODE)
    for name in ("fa_bwd_fp16.cpp.so", "fa_bwd_bf16.cpp.so", "fa_bwd_fp32.cpp.so")
)
_DISPATCHER_HANDLE = ctypes.CDLL(
    str(BUNDLE / "generated" / "libtilelang_fa_bwd_dispatch.so"), mode=_MODE
)
_FUNCTION = consumer.configure_ctypes_function(_DISPATCHER_HANDLE.tilelang_fa_bwd_call)
_RETAINED_HANDLES = (_DISPATCHER_HANDLE, *_KERNEL_HANDLES)
if len(_RETAINED_HANDLES) != 4:
    raise consumer.ContractError("dispatcher plus three typed DSO handles were not retained")
_MAPPED_FILES = _mapped_bundle_files(_BINDING)


def _contract(q, k, causal, window_left, window_right, softcap) -> dict:
    dtype = str(q.dtype).removeprefix("torch.")
    batch, seq_q, heads_q, dim = (int(value) for value in q.shape)
    batch_k, seq_k, heads_k, dim_k = (int(value) for value in k.shape)
    if batch != batch_k or dim != dim_k:
        raise consumer.ContractError("q/k runtime shape mismatch")
    return {
        "B": batch,
        "Sq": seq_q,
        "Sk": seq_k,
        "Hq": heads_q,
        "Hk": heads_k,
        "D": dim,
        "dtype": dtype,
        "causal": bool(causal),
        "window_left": int(window_left),
        "window_right": int(window_right),
        "softcap": float(softcap),
    }


def _case_index(contract: dict) -> int:
    matches = [
        index
        for index, expected in enumerate(_BINDING["case_contracts"])
        if expected == contract
    ]
    if len(matches) != 1:
        raise consumer.ContractError(
            f"runtime contract is not one unique frozen50 case: matches={matches}"
        )
    return matches[0]


def _record_mapping(case_index: int) -> None:
    MAPPING_DIR.mkdir(exist_ok=True)
    target = MAPPING_DIR / f"case_{case_index:02d}_pid_{os.getpid()}.json"
    payload = {
        "schema": "tilelang.fa_bwd_loaded_mapping/1",
        "case_index": case_index,
        "pid": os.getpid(),
        "source_commit": _BINDING["source_commit"],
        "bundle_manifest_sha256": _BINDING["bundle_manifest_sha256"],
        "mapped_bundle_owned_shared_objects": _MAPPED_FILES,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != encoded:
            raise consumer.ContractError(f"mapping receipt changed in one process: {target}")
        return
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, target)


class ModelNew(nn.Module):
    """One host dispatcher and three dtype kernels; no per-shape clone."""

    def __init__(self):
        super().__init__()
        self._outputs: dict[tuple, tuple] = {}

    def forward(
        self,
        q,
        k,
        v,
        dy,
        softmax_max,
        softmax_sum,
        attention_in,
        causal,
        window_left,
        window_right,
        softcap,
    ):
        contract = _contract(q, k, causal, window_left, window_right, softcap)
        case_index = _case_index(contract)
        group = [
            q,
            k,
            v,
            dy,
            softmax_max,
            softmax_sum,
            attention_in,
            causal,
            window_left,
            window_right,
            softcap,
        ]
        key = (
            tuple(q.shape),
            tuple(k.shape),
            str(q.dtype),
            q.device.type,
            q.device.index,
        )
        outputs = self._outputs.get(key)
        if outputs is None:
            # empty_like does not enqueue a fill kernel.  The e288 kernel has
            # already passed full sentinel-overwrite precision for all outputs.
            outputs = (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))
            self._outputs[key] = outputs
        device_index = int(q.device.index or 0)
        stream = torch_npu.npu.current_stream(device_index)
        consumer.validate_current_stream(stream, device_index)
        return_code = consumer.call_candidate(_FUNCTION, group, outputs, contract, stream)
        if return_code != 0:
            raise consumer.ContractError(
                f"e288 dispatcher returned {return_code} for frozen case {case_index}"
            )
        _record_mapping(case_index)
        return outputs
