#!/usr/bin/env python3
"""Lower, card-free build, and bind the real symbolic FA backward POC."""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock


SYMBOLIC_EXTENTS = ("B", "Sq", "Hq", "D", "Sk", "Hk")
FLAT_BUFFER_HANDLES = {
    "q_flat": "q_handle",
    "k_flat": "k_handle",
    "v_flat": "v_handle",
    "dy_flat": "dy_handle",
    "max_flat": "softmax_max_handle",
    "sum_flat": "softmax_sum_handle",
    "attention_flat": "attention_handle",
    "dq_flat": "dq_handle",
    "dk_flat": "dk_handle",
    "dv_flat": "dv_handle",
}
INPUT_FLAT_BUFFERS = (
    "q_flat",
    "k_flat",
    "v_flat",
    "dy_flat",
    "max_flat",
    "sum_flat",
    "attention_flat",
)
OUTPUT_FLAT_BUFFERS = ("dq_flat", "dk_flat", "dv_flat")
LEGACY_GLOBAL_SCALAR_TOKENS = (
    "q.GetValue",
    "k.GetValue",
    "v.GetValue",
    "dy.GetValue",
    "softmax_max.GetValue",
    "softmax_sum.GetValue",
    "dq.SetValue",
    "dk.SetValue",
    "dv.SetValue",
)
CPP_DATA_TYPES = {
    "float16": "half",
    "bfloat16": "bfloat16_t",
    "float32": "float",
}
INPUT_COPY_CONTRACT = {
    "q_flat": ("q_ub", 3),
    "k_flat": ("k_ub", 3),
    "v_flat": ("v_ub", 2),
    "dy_flat": ("dy_ub", 3),
    "max_flat": ("max_ub", 3),
    "sum_flat": ("sum_ub", 3),
    "attention_flat": ("attention_ub", 2),
}


_RAW_STRING_PREFIX = re.compile(
    r'(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\v\f\r\n]{0,16})\('
)


def _strip_cpp_comments_and_literals(source: str) -> str:
    """Blank C/C++ comments and literals while preserving source positions."""

    output = list(source)
    state = "code"
    index = 0
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            raw_match = _RAW_STRING_PREFIX.match(source, index)
            if raw_match is not None:
                closing = ")" + raw_match.group("delimiter") + '"'
                end = source.find(closing, raw_match.end())
                if end < 0:
                    raise AssertionError("unterminated C++ raw string literal")
                for position in range(index, end + len(closing)):
                    if source[position] not in "\r\n":
                        output[position] = " "
                index = end + len(closing)
                continue
            if char == "/" and following == "/":
                output[index] = output[index + 1] = " "
                state = "line_comment"
                index += 2
                continue
            if char == "/" and following == "*":
                output[index] = output[index + 1] = " "
                state = "block_comment"
                index += 2
                continue
            if char == '"':
                output[index] = " "
                state = "string"
            elif char == "'":
                output[index] = " "
                state = "character"
            index += 1
            continue
        if state == "line_comment":
            if char in "\r\n":
                if index == 0 or source[index - 1] != "\\":
                    state = "code"
            else:
                output[index] = " "
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and following == "/":
                output[index] = output[index + 1] = " "
                state = "code"
                index += 2
            else:
                if char not in "\r\n":
                    output[index] = " "
                index += 1
            continue
        if state in {"string", "character"}:
            quote = '"' if state == "string" else "'"
            if char == "\\":
                output[index] = " "
                if following:
                    if following not in "\r\n":
                        output[index + 1] = " "
                    index += 2
                else:
                    index += 1
                continue
            if char == quote:
                output[index] = " "
                state = "code"
                index += 1
                continue
            if char in "\r\n":
                raise AssertionError(f"unterminated C++ {state} literal")
            output[index] = " "
            index += 1
            continue
        raise AssertionError(f"unknown C++ lexer state: {state}")
    if state not in {"code", "line_comment"}:
        raise AssertionError(f"unterminated C++ lexical construct: {state}")
    return "".join(output)


def _split_top_level_arguments(arguments: str) -> list[str]:
    """Split one already stripped call argument list at top-level commas."""

    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    result = []
    start = 0
    for index, char in enumerate(arguments):
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                raise AssertionError("unbalanced delimiters in generated call")
        elif char == "," and not stack:
            result.append(arguments[start:index].strip())
            start = index + 1
    if stack:
        raise AssertionError("unbalanced delimiters in generated call")
    result.append(arguments[start:].strip())
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_real_source(
    source: str, host_entry: str, kernel_entry: str, dtype: str
) -> dict[str, object]:
    if dtype not in CPP_DATA_TYPES:
        raise AssertionError(f"unsupported generated source dtype: {dtype}")
    data_type = CPP_DATA_TYPES[dtype]
    executable = _strip_cpp_comments_and_literals(source)
    required_checks = {
        "host_entry": re.compile(rf"\bextern\s+void\s+{re.escape(host_entry)}\b"),
        "kernel_entry": re.compile(
            rf"\bextern\s+__global__\s+__aicore__\s+void\s+"
            rf"{re.escape(kernel_entry)}\b"
        ),
        "kernel_launch": re.compile(rf"\b{re.escape(kernel_entry)}\s*<<<"),
        "AscendC::Exp": re.compile(r"\bAscendC::Exp\b"),
        **{
            f"int64_t {name}": re.compile(rf"\bint64_t\s+{re.escape(name)}\b")
            for name in SYMBOLIC_EXTENTS
        },
    }
    missing = [name for name, pattern in required_checks.items() if not pattern.search(executable)]
    if missing:
        raise AssertionError(
            f"{dtype}: generated source missing real-lowering tokens: {missing}"
        )
    forbidden = ["ABI-only", "ABI_ONLY", "non-numerical"]
    present_forbidden = [token for token in forbidden if token in executable]
    if present_forbidden:
        raise AssertionError(
            f"{dtype}: generated source contains ABI sentinel tokens: {present_forbidden}"
        )
    generic_device_entry_tokens = [" void main_kernel(", " main_kernel<<<"]
    present_generic_device_entries = [
        token for token in generic_device_entry_tokens if token in executable
    ]
    if present_generic_device_entries:
        raise AssertionError(
            f"{dtype}: generated source retained preemptible generic device entry: "
            f"{present_generic_device_entries}"
        )

    bound_aliases = re.findall(
        r"\b([A-Za-z_]\w*_flat)\.SetGlobalBuffer\s*\(", executable
    )
    unknown_bound_aliases = sorted(set(bound_aliases) - set(FLAT_BUFFER_HANDLES))
    if unknown_bound_aliases:
        raise AssertionError(
            f"{dtype}: unexpected flat GlobalTensor bindings: {unknown_bound_aliases}"
        )
    flat_bindings = {}
    for buffer_name, handle_name in FLAT_BUFFER_HANDLES.items():
        buffer_type = "float" if buffer_name in {"max_flat", "sum_flat"} else data_type
        declaration = re.compile(
            rf"\bAscendC::GlobalTensor<\s*([^>]+?)\s*>\s+"
            rf"{re.escape(buffer_name)}\s*;"
        )
        binding_calls = re.compile(
            rf"\b{re.escape(buffer_name)}\.SetGlobalBuffer\s*\((.*?)\)\s*;",
            re.DOTALL,
        )
        declarations = declaration.findall(executable)
        bindings = binding_calls.findall(executable)
        if len(declarations) != 1 or len(bindings) != 1:
            raise AssertionError(
                f"{dtype}: invalid flat GlobalTensor binding for {buffer_name}: "
                f"declarations={len(declarations)}, bindings={len(bindings)}, "
                f"expected_handle={handle_name}"
            )
        declared_type = re.sub(r"\s+", "", declarations[0])
        expected_binding = re.compile(
            rf"\(\s*__gm__\s+{re.escape(buffer_type)}\s*\*\s*\)\s*"
            rf"{re.escape(handle_name)}\s*"
        )
        if declared_type != buffer_type or not expected_binding.fullmatch(bindings[0]):
            raise AssertionError(
                f"{dtype}: wrong dtype/cast/handle binding for {buffer_name}: "
                f"declaration={declarations[0]!r}, binding={bindings[0]!r}"
            )
        flat_bindings[buffer_name] = handle_name

    legacy_scalar_tokens = [
        token
        for token in LEGACY_GLOBAL_SCALAR_TOKENS
        if re.search(rf"\b{re.escape(token)}", executable)
    ]
    if legacy_scalar_tokens:
        raise AssertionError(
            f"{dtype}: obsolete parent global scalar pattern present: "
            f"{legacy_scalar_tokens}"
        )

    flat_names = "|".join(map(re.escape, FLAT_BUFFER_HANDLES))
    global_scalar_pattern = re.compile(
        rf"\b(?:{flat_names})\.(?:GetValue|SetValue)\b"
    )
    forbidden_global_scalar_accesses = global_scalar_pattern.findall(executable)
    if forbidden_global_scalar_accesses:
        raise AssertionError(
            f"{dtype}: forbidden flat GlobalTensor scalar access present: "
            f"{forbidden_global_scalar_accesses}"
        )

    gm_to_ub_prefix = re.compile(r"\btl::ascend::copy_gm_to_ub\s*<")
    gm_to_ub_pattern = re.compile(
        r"\btl::ascend::copy_gm_to_ub\s*<\s*([^,>\n]+?)\s*,\s*(\d+)\s*>"
        r"\s*\(([^;\n]*)\)\s*;"
    )
    gm_to_ub_matches = list(gm_to_ub_pattern.finditer(executable))
    gm_to_ub_occurrences = len(gm_to_ub_prefix.findall(executable))
    if len(gm_to_ub_matches) != gm_to_ub_occurrences:
        raise AssertionError(f"{dtype}: malformed or multiline GM-to-UB helper call")
    if len(gm_to_ub_matches) != 19:
        raise AssertionError(
            f"{dtype}: expected exactly 19 GM-to-UB copies, got {len(gm_to_ub_matches)}"
        )

    input_copy_counts = {buffer_name: 0 for buffer_name in INPUT_FLAT_BUFFERS}
    for match in gm_to_ub_matches:
        template_type, width, raw_arguments = match.groups()
        arguments = _split_top_level_arguments(raw_arguments)
        if len(arguments) != 6:
            raise AssertionError(
                f"{dtype}: GM-to-UB helper requires six arguments, got {arguments}"
            )
        source_match = re.fullmatch(r"([A-Za-z_]\w*_flat)\s*\[.*\]", arguments[1])
        if source_match is None or source_match.group(1) not in INPUT_COPY_CONTRACT:
            raise AssertionError(
                f"{dtype}: GM-to-UB source is not an approved flat input: {arguments[1]}"
            )
        buffer_name = source_match.group(1)
        destination_name, _ = INPUT_COPY_CONTRACT[buffer_name]
        expected_type = "float" if buffer_name in {"max_flat", "sum_flat"} else data_type
        expected_width = "8" if buffer_name in {"max_flat", "sum_flat"} else "32"
        if re.sub(r"\s+", "", template_type) != expected_type or width != expected_width:
            raise AssertionError(
                f"{dtype}: wrong GM-to-UB dtype/width for {buffer_name}: "
                f"<{template_type}, {width}>"
            )
        if not re.fullmatch(
            rf"{re.escape(destination_name)}\s*\[\s*0\s*\]", arguments[0]
        ):
            raise AssertionError(
                f"{dtype}: wrong GM-to-UB destination for {buffer_name}: {arguments[0]}"
            )
        input_copy_counts[buffer_name] += 1
    expected_input_copy_counts = {
        buffer_name: count
        for buffer_name, (_, count) in INPUT_COPY_CONTRACT.items()
    }
    if input_copy_counts != expected_input_copy_counts:
        raise AssertionError(
            f"{dtype}: wrong GM-to-UB source multiplicities: "
            f"expected={expected_input_copy_counts}, got={input_copy_counts}"
        )

    ub_to_gm_prefix = re.compile(r"\btl::ascend::copy_ub_to_gm\s*<")
    ub_to_gm_pattern = re.compile(
        r"\btl::ascend::copy_ub_to_gm\s*<\s*([^,>\n]+?)\s*,\s*(\d+)\s*>"
        r"\s*\(([^;\n]*)\)\s*;"
    )
    ub_to_gm_matches = list(ub_to_gm_pattern.finditer(executable))
    ub_to_gm_occurrences = len(ub_to_gm_prefix.findall(executable))
    if len(ub_to_gm_matches) != ub_to_gm_occurrences:
        raise AssertionError(f"{dtype}: malformed or multiline UB-to-GM helper call")
    if len(ub_to_gm_matches) != 3:
        raise AssertionError(
            f"{dtype}: expected exactly 3 UB-to-GM copies, got {len(ub_to_gm_matches)}"
        )
    output_copy_sources = "acc_tile" if dtype == "float32" else "out_tile"
    output_copy_counts = {buffer_name: 0 for buffer_name in OUTPUT_FLAT_BUFFERS}
    output_copy_records = []
    for match in ub_to_gm_matches:
        template_type, width, raw_arguments = match.groups()
        arguments = _split_top_level_arguments(raw_arguments)
        if len(arguments) != 5:
            raise AssertionError(
                f"{dtype}: UB-to-GM helper requires five arguments, got {arguments}"
            )
        target_match = re.fullmatch(r"([A-Za-z_]\w*_flat)\s*\[.*\]", arguments[0])
        if target_match is None or target_match.group(1) not in OUTPUT_FLAT_BUFFERS:
            raise AssertionError(
                f"{dtype}: UB-to-GM target is not an approved flat output: {arguments[0]}"
            )
        buffer_name = target_match.group(1)
        if re.sub(r"\s+", "", template_type) != data_type or width != "32":
            raise AssertionError(
                f"{dtype}: wrong UB-to-GM dtype/width for {buffer_name}: "
                f"<{template_type}, {width}>"
            )
        if not re.fullmatch(
            rf"{re.escape(output_copy_sources)}\s*\[\s*0\s*\]", arguments[1]
        ):
            raise AssertionError(
                f"{dtype}: wrong UB-to-GM source for {buffer_name}: {arguments[1]}"
            )
        output_copy_counts[buffer_name] += 1
        output_copy_records.append((match, buffer_name))
    expected_output_copy_counts = {buffer_name: 1 for buffer_name in OUTPUT_FLAT_BUFFERS}
    if output_copy_counts != expected_output_copy_counts:
        raise AssertionError(
            f"{dtype}: wrong UB-to-GM target multiplicities: "
            f"expected={expected_output_copy_counts}, got={output_copy_counts}"
        )

    cast_pattern = re.compile(r"\bAscendC::Cast\s*\((.*?)\)\s*;", re.DOTALL)
    parsed_casts = []
    for match in cast_pattern.finditer(executable):
        arguments = _split_top_level_arguments(match.group(1))
        if arguments and re.fullmatch(r"out_tile\s*\[.*\]", arguments[0]):
            parsed_casts.append((match, arguments))
    output_cast_occurrences = len(
        re.findall(r"\bAscendC::Cast\s*\(\s*out_tile\s*\[", executable)
    )
    if len(parsed_casts) != output_cast_occurrences:
        raise AssertionError(f"{dtype}: malformed output-target AscendC::Cast call")
    expected_cast_count = 0 if dtype == "float32" else 3
    if len(parsed_casts) != expected_cast_count:
        raise AssertionError(
            f"{dtype}: expected {expected_cast_count} structural output casts, "
            f"got {len(parsed_casts)}"
        )
    for _, arguments in parsed_casts:
        normalized = [re.sub(r"\s+", "", argument) for argument in arguments]
        if normalized != [
            "out_tile[0]",
            "acc_tile[0]",
            "AscendC::RoundMode::CAST_RINT",
            "32",
        ]:
            raise AssertionError(
                f"{dtype}: invalid output cast arguments: {arguments}"
            )
    cast_bound_outputs = []
    if dtype != "float32":
        used_casts = set()
        for output_match, buffer_name in output_copy_records:
            adjacent = [
                index
                for index, (cast_match, _) in enumerate(parsed_casts)
                if cast_match.end() <= output_match.start()
                and not executable[cast_match.end() : output_match.start()].strip()
            ]
            if len(adjacent) != 1 or adjacent[0] in used_casts:
                raise AssertionError(
                    f"{dtype}: output cast is not uniquely adjacent to {buffer_name} copy"
                )
            used_casts.add(adjacent[0])
            cast_bound_outputs.append(buffer_name)
        if len(used_casts) != 3:
            raise AssertionError(f"{dtype}: not all output casts are bound to output copies")

    if executable.count("scratch.SetValue(0, 0.000000e+00f)") != 3:
        raise AssertionError(
            f"{dtype}: output accumulators are not reset for all three outputs"
        )
    leaked_scalar_accumulators = [
        token
        for token in ["float dq_acc", "float dk_acc", "float dv_acc", "int masked"]
        if token in executable
    ]
    if leaked_scalar_accumulators:
        raise AssertionError(
            f"{dtype}: loop-local scalar state was lifted out of its scope: "
            f"{leaked_scalar_accumulators}"
        )
    return {
        "required_tokens": list(required_checks),
        "runtime_extent_count": sum(
            bool(re.search(rf"\bint64_t\s+{re.escape(name)}\b", executable))
            for name in SYMBOLIC_EXTENTS
        ),
        "forbidden_tokens_absent": forbidden,
        "generic_device_entry_absent": True,
        "kernel_entry": kernel_entry,
        "flat_global_tensor_bindings": flat_bindings,
        "gm_to_ub_copy_count": len(gm_to_ub_matches),
        "ub_to_gm_copy_count": len(ub_to_gm_matches),
        "flat_input_copy_multiplicities": input_copy_counts,
        "flat_output_copy_multiplicities": output_copy_counts,
        "flat_output_copy_sources": {
            buffer_name: output_copy_sources for buffer_name in OUTPUT_FLAT_BUFFERS
        },
        "forbidden_flat_global_scalar_access_absent": list(FLAT_BUFFER_HANDLES),
        "legacy_global_scalar_pattern_absent": list(LEGACY_GLOBAL_SCALAR_TOKENS),
        "cast_rint_count": len(parsed_casts),
        "cast_bound_outputs": cast_bound_outputs,
        "lexical_comments_and_literals_stripped": True,
        "per_output_accumulator_resets": 3,
        "lifted_scalar_accumulators_absent": True,
    }


def lower_variant(dtype: str, host_entry: str, kernel_entry: str):
    import tilelang
    from tilelang import tvm

    from poc.fa_bwd_symbolic_lowering import make_fa_bwd_scalar

    function = make_fa_bwd_scalar(dtype, host_entry, kernel_entry)
    with tvm.transform.PassContext(config={"tl.disable_safe_memory_legalize": True}):
        return tilelang.lower(function, target="ascendc", platform="A5")


def compile_variant(source: str, output: Path) -> tuple[list[str], Path, Path]:
    from tilelang.jit.adapter.libgen import LibraryGenerator

    commands: list[list[str]] = []
    real_run = subprocess.run

    def traced(command, *args, **kwargs):
        commands.append([str(item) for item in command])
        return real_run(command, *args, **kwargs)

    generator = LibraryGenerator("ascendc", "A5")
    generator.update_lib_code(source)
    with mock.patch("subprocess.run", side_effect=traced):
        generator.compile_lib(timeout=300)
    if len(commands) != 1:
        raise AssertionError(f"expected one Bisheng command, got {len(commands)}")
    temporary_source = Path(generator.get_source_path())
    temporary_library = Path(generator.get_lib_path())
    shutil.copy2(temporary_source, output.with_suffix(".compiler.cpp"))
    shutil.copy2(temporary_library, output)
    return commands[0], output.with_suffix(".compiler.cpp"), output


def link_dispatcher(
    plan, generated: Path, variant_paths: list[Path]
) -> tuple[list[str], Path]:
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("host C++ compiler not found")
    output = generated / "libtilelang_fa_bwd_dispatch.so"
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "--shared",
        f"-I{generated / 'host'}",
        str(generated / "host" / "fa_bwd_dispatch.cpp"),
        f"-L{generated / 'kernel'}",
        "-Wl,--no-as-needed",
        *[f"-l:{path.name}" for path in variant_paths],
        "-Wl,-z,defs",
        "-Wl,-rpath,$ORIGIN/kernel",
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (generated / "host_link.stdout").write_text(completed.stdout, encoding="utf-8")
    (generated / "host_link.stderr").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"host dispatcher link failed: rc={completed.returncode}")
    # Load-only is a host ABI closure check. No wrapper is called and no NPU is used.
    ctypes.CDLL(str(output), mode=os.RTLD_NOW)
    return command, output


def audit_symbol_isolation(
    output_root: Path, plan, variant_paths: list[Path], host_library: Path
) -> dict[str, object]:
    """Prove each host wrapper's relocation binds to its own device entry."""

    from tilelang.jit.adapter import ascendc_dispatch

    audit_dir = output_root / "symbol_audit"
    audit_dir.mkdir(parents=True, exist_ok=False)
    variants = {}
    for variant, library in zip(plan.variants, variant_paths, strict=True):
        suffix = ascendc_dispatch.DTYPE_SUFFIXES[variant.dtype]
        tool_outputs = {}
        for tool_name, command in {
            "nm_dynamic_defined": ["nm", "-D", "--defined-only", str(library)],
            "readelf_symbols": ["readelf", "-Ws", str(library)],
            "readelf_relocations": ["readelf", "-rW", str(library)],
        }.items():
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            (audit_dir / f"{suffix}.{tool_name}.stdout").write_text(
                completed.stdout, encoding="utf-8"
            )
            (audit_dir / f"{suffix}.{tool_name}.stderr").write_text(
                completed.stderr, encoding="utf-8"
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{variant.dtype}: {tool_name} failed rc={completed.returncode}"
                )
            tool_outputs[tool_name] = completed.stdout

        nm_symbols = {
            line.split()[-1]
            for line in tool_outputs["nm_dynamic_defined"].splitlines()
            if line.split()
        }
        if variant.kernel_symbol not in nm_symbols:
            raise AssertionError(
                f"{variant.dtype}: {variant.kernel_symbol} absent from dynamic symbols"
            )
        if "main_kernel" in nm_symbols:
            raise AssertionError(
                f"{variant.dtype}: preemptible generic main_kernel still exported"
            )
        if variant.kernel_symbol not in tool_outputs["readelf_symbols"]:
            raise AssertionError(
                f"{variant.dtype}: readelf does not define {variant.kernel_symbol}"
            )
        if variant.kernel_symbol not in tool_outputs["readelf_relocations"]:
            raise AssertionError(
                f"{variant.dtype}: wrapper relocation does not reference {variant.kernel_symbol}"
            )
        variants[variant.dtype] = {
            "library": str(library.relative_to(output_root)),
            "kernel_symbol": variant.kernel_symbol,
            "host_entry": variant.host_entry,
            "dynamic_symbol_defined": True,
            "generic_main_kernel_absent": True,
            "self_symbol_relocation_present": True,
        }

    loader_script = (
        f"import ctypes, os; ctypes.CDLL({str(host_library)!r}, mode=os.RTLD_NOW)"
    )
    loader_env = os.environ.copy()
    loader_env["LD_DEBUG"] = "bindings,libs"
    loader = subprocess.run(
        [sys.executable, "-c", loader_script],
        capture_output=True,
        text=True,
        check=False,
        env=loader_env,
    )
    (audit_dir / "loader.stdout").write_text(loader.stdout, encoding="utf-8")
    (audit_dir / "loader.stderr").write_text(loader.stderr, encoding="utf-8")
    (audit_dir / "loader.rc").write_text(f"{loader.returncode}\n", encoding="utf-8")
    if loader.returncode != 0:
        raise RuntimeError(f"LD_DEBUG dispatcher load failed rc={loader.returncode}")
    if "normal symbol `main_kernel'" in loader.stderr:
        raise AssertionError("loader still bound the generic main_kernel symbol")

    binding_lines = []
    for variant, library in zip(plan.variants, variant_paths, strict=True):
        matches = [
            line.strip()
            for line in loader.stderr.splitlines()
            if "binding file" in line
            and line.count(library.name) >= 2
            and f"normal symbol `{variant.kernel_symbol}'" in line
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"{variant.dtype}: expected one self-binding for "
                f"{variant.kernel_symbol}, got {matches}"
            )
        binding_lines.extend(matches)
        variants[variant.dtype]["ld_debug_self_binding"] = matches[0]
    (audit_dir / "bindings.summary.txt").write_text(
        "\n".join(binding_lines) + "\n", encoding="utf-8"
    )
    return {
        "verdict": "PASS_UNIQUE_SELF_BINDINGS",
        "loader_returncode": loader.returncode,
        "generic_main_kernel_binding_absent": True,
        "variants": variants,
    }


def run_planner_controls(repo: Path, cases: Path, output: Path) -> dict[str, object]:
    command = [
        sys.executable,
        str(repo / "poc" / "run_fa_symbolic_dispatch_poc.py"),
        "--repo",
        str(repo),
        "--cases",
        str(cases),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (output.parent / "planner_controls.stdout").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output.parent / "planner_controls.stderr").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"planner controls failed: rc={completed.returncode}")
    return {"command": command, "returncode": completed.returncode}


def write_manifest(root: Path) -> Path:
    manifest = root / "MANIFEST.sha256"
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != manifest):
        rows.append(f"{sha256(path)}  {path.relative_to(root)}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    from tilelang.jit.adapter import ascendc_dispatch, ascendc_provenance

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--canonical-json", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--operator-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-observation", type=Path, required=True)
    args = parser.parse_args()

    bound_inputs = {
        "op29_fixed50_json": args.canonical_json,
        "op29_reference_model": args.reference_model,
        "op29_operator_source": args.operator_source,
        "op29_fixed50_shapes_csv": args.cases,
    }
    invalid_inputs = [
        name
        for name, path in bound_inputs.items()
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0
    ]
    if invalid_inputs:
        raise ValueError(f"canonical provenance inputs are unavailable: {invalid_inputs}")

    args.output.mkdir(parents=True, exist_ok=False)
    generated = args.output / "generated"
    cases = ascendc_dispatch.load_fa_bwd_cases(args.cases)
    plan = ascendc_dispatch.write_dispatch_bundle(cases, generated)

    negative_rank = dataclasses.replace(cases[0], rank_signature=(3, 4, 4, 4, 4, 4, 4))
    try:
        negative_rank.validate()
    except ascendc_dispatch.AscendCSymbolicContractError:
        rank_control = "REJECTED"
    else:
        raise AssertionError("rank-changing negative control was accepted")
    negative_d = dataclasses.replace(cases[0], D=17)
    try:
        negative_d.validate()
    except ascendc_dispatch.AscendCSymbolicContractError:
        d_control = "REJECTED"
    else:
        raise AssertionError("D%8 negative control was accepted")

    commands = {}
    sources = {}
    libraries = {}
    guards = {}
    variant_paths = []
    for variant in plan.variants:
        artifact = lower_variant(
            variant.dtype, variant.host_entry, variant.kernel_symbol
        )
        source_path = (
            generated
            / "kernel"
            / f"fa_bwd_{ascendc_dispatch.DTYPE_SUFFIXES[variant.dtype]}.cpp"
        )
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(artifact.kernel_source, encoding="utf-8")
        guards[variant.dtype] = validate_real_source(
            artifact.kernel_source,
            variant.host_entry,
            variant.kernel_symbol,
            variant.dtype,
        )
        library_path = source_path.with_suffix(".so")
        command, compiler_source, library_path = compile_variant(
            artifact.kernel_source, library_path
        )
        commands[variant.dtype] = command
        sources[variant.dtype] = {
            "path": str(source_path.relative_to(args.output)),
            "sha256": sha256(source_path),
            "bytes": source_path.stat().st_size,
            "compiler_copy_sha256": sha256(compiler_source),
        }
        libraries[variant.dtype] = {
            "path": str(library_path.relative_to(args.output)),
            "sha256": sha256(library_path),
            "bytes": library_path.stat().st_size,
        }
        variant_paths.append(library_path)

    host_command, host_library = link_dispatcher(plan, generated, variant_paths)
    symbol_isolation = audit_symbol_isolation(
        args.output, plan, variant_paths, host_library
    )
    planner_controls = run_planner_controls(
        args.repo, args.cases, args.output / "planner_controls"
    )
    generated_artifacts = [path for path in generated.rglob("*") if path.is_file()]
    provenance_path, provenance_record = ascendc_provenance.capture_build_provenance(
        repo=args.repo,
        bundle_root=args.output,
        source_observation_path=args.source_observation,
        artifact_paths=generated_artifacts,
        input_paths=bound_inputs,
        dependency_patches={
            "3rdparty/tvm": args.repo
            / "poc"
            / "patches"
            / "tvm_dynamic_slice_unit_step.patch"
        },
        toolchain="bisheng",
        target={
            "backend": "ascendc",
            "platform": "A5",
            "npu_arch": "dav-3510",
            "catlass_arch": "3510",
        },
    )
    provenance_controls = ascendc_provenance.run_provenance_negative_controls(
        provenance_path, args.output, args.source_observation
    )
    result = {
        "authority": "AUTHOR_EVIDENCE_ONLY",
        "npu_used": False,
        "operator": plan.operator,
        "case_manifest": {
            "path": str(args.cases),
            "sha256": sha256(args.cases),
            "count": len(cases),
            "canonical_inputs": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in sorted(bound_inputs.items())
            },
        },
        "baseline": {
            "naive_static_kernel_count": plan.naive_static_kernel_count,
            "a3_factory_kernel_count": plan.a3_factory_kernel_count,
        },
        "poc": {
            "host_dispatcher_count": 1,
            "real_tilelang_kernel_variants": len(plan.variants),
            "variant_key": "dtype",
            "symbolic_extents": list(plan.symbolic_extents),
            "fixed_rank_signature": list(plan.fixed_rank_signature),
            "compile_dispatch_covered_case_ids": sorted(case.case_id for case in cases),
            "device_executed_case_ids": [],
            "device_supported_case_ids": "UNKNOWN_NOT_RUN",
            "device_unsupported_case_ids": "UNKNOWN_NOT_RUN",
        },
        "negative_controls": {
            "rank_change": rank_control,
            "D_mod_8": d_control,
            "host_dispatch": "PASS",
        },
        "generated_source_guards": guards,
        "generated_sources": sources,
        "variant_libraries": libraries,
        "bisheng_commands": commands,
        "host_dispatcher": {
            "path": str(host_library.relative_to(args.output)),
            "sha256": sha256(host_library),
            "bytes": host_library.stat().st_size,
            "link_command": host_command,
            "rtld_now": "PASS_NO_WRAPPER_CALL",
        },
        "symbol_isolation": symbol_isolation,
        "planner_controls": planner_controls,
        "build_provenance": {
            "path": str(provenance_path.relative_to(args.output)),
            "sha256": sha256(provenance_path),
            "source_commit": provenance_record["source_repo"]["commit"],
            "source_tree": provenance_record["source_repo"]["tree"],
            "source_observed_at_utc": provenance_record["source_observation"][
                "observed_at_utc"
            ],
            "positive_consumer": "PASS",
            "negative_controls": provenance_controls,
        },
        "device_execution": "NOT_RUN_NO_NPU_ADMISSION",
        "numerical_precision": "NOT_MEASURED",
        "performance": "NOT_MEASURED",
    }
    write_json(args.output / "RESULT.json", result)
    manifest = write_manifest(args.output)
    ascendc_provenance.verify_bundle_manifest(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"manifest={manifest} sha256={sha256(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
