# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CPU-only tests for final-TIR operation classification."""

import pytest

from tilelang.simulator import Lane, Pipe, UnsupportedSimOpError, classify_operation


@pytest.mark.parametrize(
    ("operation", "lane", "expected_lane", "expected_pipe"),
    [
        ("tl.ascend_gemm_v0", Lane.CUBE, Lane.CUBE, Pipe.MATRIX),
        ("tl.ascend_im2col", Lane.CUBE, Lane.CUBE, Pipe.MTE1),
        ("tl::ascend::copy_gm_to_l1", Lane.CUBE, Lane.CUBE, Pipe.MTE2),
        ("tl::ascend::copy_l1_to_l0", Lane.CUBE, Lane.CUBE, Pipe.MTE1),
        ("tl::ascend::copy_l0c_to_gm", Lane.CUBE, Lane.CUBE, Pipe.FIX),
        ("tl::ascend::copy_ub_to_gm", Lane.VECTOR_1, Lane.VECTOR_1, Pipe.MTE3),
        ("tl.ascend_add", Lane.VECTOR_0, Lane.VECTOR_0, Pipe.VECTOR),
        ("tl.ascend_sort", Lane.VECTOR_1, Lane.VECTOR_1, Pipe.VECTOR),
        ("tl.ascend_set_flag", Lane.CUBE, Lane.CUBE, Pipe.SCALAR),
    ],
)
def test_known_operations_map_to_c220_resources(
    operation: str, lane: Lane, expected_lane: Lane, expected_pipe: Pipe
) -> None:
    actual_lane, actual_pipe, _ = classify_operation(operation, lane)

    assert (actual_lane, actual_pipe) == (expected_lane, expected_pipe)


def test_unknown_operation_fails_closed() -> None:
    with pytest.raises(UnsupportedSimOpError, match="unsupported lowered"):
        classify_operation("tl.ascend_future_magic", Lane.VECTOR_0)


def test_unknown_a5_operation_reports_platform_and_fails_closed() -> None:
    with pytest.raises(
        UnsupportedSimOpError,
        match=r"unsupported lowered.*platform=A5",
    ):
        classify_operation(
            "tl.ascend_future_magic", Lane.VECTOR_0, platform="A5"
        )


@pytest.mark.parametrize(
    "operation",
    [
        "tl.ascend_simt_launch",
        "tl.ascend_regbase_vector",
        "tl.ascend_nddma_copy",
        "tl.ascend_ssbuffer_signal",
        "tl.ascend_buffer_id_wait",
    ],
)
def test_unmodeled_a5_semantics_do_not_inherit_c220_classification(
    operation: str,
) -> None:
    with pytest.raises(UnsupportedSimOpError, match="unsupported A5 simulator semantic"):
        classify_operation(operation, Lane.VECTOR_0, platform="A5")


def test_shmem_operation_is_permanently_unsupported() -> None:
    with pytest.raises(UnsupportedSimOpError, match="intentionally unsupported"):
        classify_operation("tl.ascend_shmem_put_nbi", Lane.VECTOR_0)
