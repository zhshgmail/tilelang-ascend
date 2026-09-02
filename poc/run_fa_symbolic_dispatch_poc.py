#!/usr/bin/env python3
"""Run the card-free FA symbolic-dispatch compiler POC."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_backend(repo: Path):
    path = repo / "tilelang/jit/adapter/ascendc_dispatch.py"
    spec = importlib.util.spec_from_file_location("ascendc_dispatch_poc", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def wrapper_definition(module, variant) -> str:
    signature = ", ".join(module._argument_declarations(include_dtype=False))
    return f'extern "C" void {variant.host_entry}({signature}) {{ g_last_key = {variant.dispatch_key}; }}\n'


def driver_source(module, cases) -> str:
    rows = []
    for case in cases:
        dtype_code = module.DTYPE_CODES[case.dtype]
        rows.append(
            "  {"
            f"{case.case_id}, {case.B}, {case.Sq}, {case.Sk}, "
            f"{case.Hq}, {case.Hk}, {case.D}, {dtype_code}, "
            f"{1 if case.causal else 0}, {case.window_left}, "
            f"{case.window_right}, {case.softcap}f"
            "},"
        )
    return f"""#include "fa_bwd_dispatch.hpp"
#include <cstdint>
#include <iostream>

extern int g_last_key;
struct Case {{
  int id;
  int64_t B, Sq, Sk, Hq, Hk, D;
  int dtype, causal, wl, wr;
  float softcap;
}};

int main() {{
  Case cases[] = {{
{chr(10).join(rows)}
  }};
  uint8_t byte = 0;
  for (const auto& c : cases) {{
    g_last_key = -1;
    int rc = tilelang_fa_bwd_call(
        &byte, &byte, &byte, &byte, &byte, &byte, &byte, &byte,
        c.B, c.Sq, c.Sk, c.Hq, c.Hk, c.D,
        c.causal, c.wl, c.wr, c.softcap, 1.0f, c.dtype, nullptr);
    if (rc != 0 || g_last_key != c.dtype) {{
      std::cerr << "case " << c.id << " rc=" << rc
                << " dispatched=" << g_last_key << " expected=" << c.dtype << "\\n";
      return 10;
    }}
  }}
  int bad_d = tilelang_fa_bwd_call(
      &byte, &byte, &byte, &byte, &byte, &byte, &byte, &byte,
      1, 3, 5, 2, 1, 17, 0, -1, 0, 0.0f, 1.0f, 0, nullptr);
  if (bad_d != -3) return 11;
  int bad_dtype = tilelang_fa_bwd_call(
      &byte, &byte, &byte, &byte, &byte, &byte, &byte, &byte,
      1, 3, 5, 2, 1, 16, 0, -1, 0, 0.0f, 1.0f, 99, nullptr);
  if (bad_dtype != -4) return 12;
  std::cout << "dispatch_cases=" << (sizeof(cases) / sizeof(cases[0]))
            << " negative_controls=2 PASS\\n";
  return 0;
}}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    module = load_backend(args.repo)
    cases = module.load_fa_bwd_cases(args.cases)
    plan = module.write_dispatch_bundle(cases, args.output / "generated")

    assert plan.case_count == 50
    assert plan.naive_static_kernel_count == 50
    assert plan.a3_factory_kernel_count == 37
    assert len(plan.variants) == 3
    assert sorted(case_id for v in plan.variants for case_id in v.case_ids) == list(range(50))

    negative = dataclasses.replace(cases[0], rank_signature=(3, 4, 4, 4, 4, 4, 4))
    try:
        negative.validate()
    except module.AscendCSymbolicContractError:
        rank_negative = "REJECTED"
    else:
        raise AssertionError("rank-changing known-bad was accepted")

    host_test = args.output / "host_test"
    host_test.mkdir(parents=True, exist_ok=True)
    mock = host_test / "mock_wrappers.cpp"
    mock.write_text(
        '#include "../generated/host/fa_bwd_dispatch.hpp"\n'
        "int g_last_key = -1;\n" + "".join(wrapper_definition(module, variant) for variant in plan.variants),
        encoding="utf-8",
    )
    driver = host_test / "driver.cpp"
    driver.write_text(driver_source(module, cases), encoding="utf-8")
    executable = host_test / "dispatch_test"
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("host C++ compiler not found")
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wno-unused-parameter",
        f"-I{args.output / 'generated/host'}",
        str(args.output / "generated/host/fa_bwd_dispatch.cpp"),
        str(mock),
        str(driver),
        "-o",
        str(executable),
    ]
    compile_run = subprocess.run(command, capture_output=True, text=True, check=False)
    (host_test / "compile.stdout").write_text(compile_run.stdout, encoding="utf-8")
    (host_test / "compile.stderr").write_text(compile_run.stderr, encoding="utf-8")
    if compile_run.returncode != 0:
        raise RuntimeError(f"host compile failed: rc={compile_run.returncode}")
    dispatch_run = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
    (host_test / "run.stdout").write_text(dispatch_run.stdout, encoding="utf-8")
    (host_test / "run.stderr").write_text(dispatch_run.stderr, encoding="utf-8")
    if dispatch_run.returncode != 0:
        raise RuntimeError(f"host dispatcher failed: rc={dispatch_run.returncode}")

    result = {
        "authority": "AUTHOR_EVIDENCE_ONLY",
        "operator": plan.operator,
        "input_cases": str(args.cases),
        "input_cases_sha256": sha256(args.cases),
        "case_count": plan.case_count,
        "rank_signature": list(plan.fixed_rank_signature),
        "symbolic_extents": list(plan.symbolic_extents),
        "naive_static_kernel_count": plan.naive_static_kernel_count,
        "a3_factory_kernel_count": plan.a3_factory_kernel_count,
        "poc_host_count": 1,
        "poc_kernel_count": len(plan.variants),
        "covered_case_ids": sorted(case_id for variant in plan.variants for case_id in variant.case_ids),
        "unsupported_case_ids": [],
        "rank_change_known_bad": rank_negative,
        "runtime_guard_negative_controls": 2,
        "host_compile_rc": compile_run.returncode,
        "host_dispatch_rc": dispatch_run.returncode,
        "host_dispatch_stdout": dispatch_run.stdout.strip(),
        "a5_device_compile": "NOT_RUN_TOOLCHAIN_MISSING",
        "numerical_precision": "NOT_IMPLEMENTED_NOT_MEASURED",
        "npu_used": False,
    }
    result_path = args.output / "poc_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
