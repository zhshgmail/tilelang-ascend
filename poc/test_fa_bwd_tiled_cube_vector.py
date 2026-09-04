"""Device-free gates for the symbolic tiled Cube+Vector FA-Bwd path."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


POC = Path(__file__).parent
SOURCE = POC / "fa_bwd_tiled_symbolic_lowering.py"
DRIVER = POC / "run_fa_bwd_real_lowering.py"
EMITTER = POC / "emit_fa_bwd_tiled_source_checkpoint.py"
FIXED50 = POC / "inputs" / "op29_fa_bwd_fixed50_shapes.csv"


def _source() -> str:
    return SOURCE.read_text(encoding="utf-8")


def test_tiled_source_is_one_symbolic_family_without_case_specialization() -> None:
    source = _source()
    tree = ast.parse(source, filename=str(SOURCE))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "make_fa_bwd_tiled" in functions
    assert source.count("@T.prim_func") == 1
    for symbol in ("B", "Sq", "Sk", "Hq", "Hk", "D"):
        assert f'{symbol} = T.symbolic("{symbol}")' in source
    assert "case_index" not in source
    assert "FIXED50" not in source
    for row in FIXED50.read_text(encoding="utf-8").splitlines()[1:]:
        case_id = row.split(",", 1)[0]
        assert f"case {case_id}" not in source


def test_tiled_source_has_cube_vector_backward_roles_and_owned_outputs() -> None:
    source = _source()
    assert "BQ = 16" in source
    assert "BK = 16" in source
    assert "D_PAD = 128" in source
    assert source.count("T.gemm_v0(") >= 5
    for role in ("QK", "DP", "DQ", "DK", "DV"):
        assert f'GEMM_ROLE_{role} = "' in source
    assert "dq_task_count" in source
    assert "dkv_task_count" in source
    assert "T.ceildiv(dq_task_count, CORE_NUM)" in source
    assert "T.ceildiv(dkv_task_count, CORE_NUM)" in source
    assert "T.tile.exp(" in source
    assert "T.reduce_sum(" in source
    assert "T.tile.select(" in source
    assert "T.tile.atomic_add" not in source


def test_driver_selects_tiled_route_a_without_changing_dtype_variant_count() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    assert 'choices=("scalar", "tiled")' in source
    assert "make_fa_bwd_tiled" in source
    assert 'target="ascendc", platform="A5"' in source
    assert '"real_tilelang_kernel_variants": len(plan.variants)' in source


def test_checkpoint_emitter_is_device_free_and_fail_closed() -> None:
    source = EMITTER.read_text(encoding="utf-8")
    assert '"authority": "DEVICE_FREE_ROUTE_A_IR_AND_SOURCE_ONLY"' in source
    assert '"authority": "DEVICE_FREE_ROUTE_A_IR_BOUNDARY_ONLY"' in source
    assert '"npu_used": False' in source
    assert '"bisheng_invoked": False' in source
    assert 'kernel_path="tiled"' in source
    assert "len(cases) != 50 or len(plan.variants) != 3" in source
    assert '"generated_source_admitted": False' in source
    assert 'return "ASCEND_COMBINE_CV_SYNC_POINT_MISMATCH"' in source
    assert "return 2" in source
    for gate in (
        "CANN_9_2_BISHENG_15_DAV3510_COMPILE_ONLY",
        "FRESH_NPU_FIXED50_PRECISION_AND_KNOWN_BAD",
        "CANONICAL_SAME_CANDIDATE_MSPROF_GE_1X",
    ):
        assert gate in source


def test_checkpoint_classifies_the_observed_combinecv_boundary() -> None:
    from poc.emit_fa_bwd_tiled_source_checkpoint import classify_lowering_failure

    error = (
        "TVMError: Mismatch in sync points between cube and vec for "
        "workspace workspace_13: cube has 1, vec has 8\n"
        "src/transform/ascend_combinecv.cc:375"
    )
    assert (
        classify_lowering_failure(error)
        == "ASCEND_COMBINE_CV_SYNC_POINT_MISMATCH"
    )
    assert (
        classify_lowering_failure("some other compiler failure")
        == "UNCLASSIFIED_ROUTE_A_LOWERING_FAILURE"
    )


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_tiled_primfunc_builds_for_each_dtype(dtype: str) -> None:
    # Importing the source-tree package installs its vendored TVM path.  A
    # direct ``import tvm`` can otherwise skip a real native TileLang build.
    pytest.importorskip(
        "tilelang", reason="native TileLang/TVM build is optional locally"
    )
    pytest.importorskip("tvm", reason="native TileLang/TVM build is optional locally")
    from poc.fa_bwd_tiled_symbolic_lowering import make_fa_bwd_tiled

    function = make_fa_bwd_tiled(
        dtype,
        f"call_fa_bwd_{dtype}",
        f"fa_bwd_{dtype}_kernel",
    )
    text = str(function)
    assert "ascendc_host_entry" in text
    assert "ascendc_kernel_entry" in text
    assert "T.ascend_gemm_v0" in text
    assert all(name in text for name in ("B", "Sq", "Sk", "Hq", "Hk", "D"))
