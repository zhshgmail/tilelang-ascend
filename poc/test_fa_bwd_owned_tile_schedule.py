"""Device-free oracle for the FA-Bwd owned-tile schedule.

This model is intentionally independent of TileLang and can run where TVM and
CANN are unavailable.  It checks the task-to-output mapping, byte-level cache
line ownership, tail coverage, logical-row crossings, and the restored legacy
striped mapping used as a known-bad discriminator.
"""

from __future__ import annotations

import ast
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest


TASKS = 48
TILE_ELEMS = 32
CACHE_LINE_BYTES = 32


@dataclass(frozen=True)
class Tile:
    task: int
    tile_id: int
    start: int
    stop: int


def owned_tiles(count: int, tasks: int = TASKS) -> list[Tile]:
    """Mirror ``tile_id = outer * tasks + task_id`` in execution order."""

    if count < 0:
        raise ValueError("count must be non-negative")
    tile_count = (count + TILE_ELEMS - 1) // TILE_ELEMS
    rounds = (tile_count + tasks - 1) // tasks
    result = []
    for outer in range(rounds):
        for task in range(tasks):
            tile_id = outer * tasks + task
            if tile_id < tile_count:
                start = tile_id * TILE_ELEMS
                result.append(Tile(task, tile_id, start, min(start + TILE_ELEMS, count)))
    return result


def element_owners_from_tiles(count: int, tiles: list[Tile]) -> list[int | None]:
    owners: list[int | None] = [None] * count
    for tile in tiles:
        for index in range(tile.start, tile.stop):
            assert owners[index] is None, f"element {index} has multiple writers"
            owners[index] = tile.task
    return owners


def legacy_element_owners(count: int, tasks: int = TASKS) -> list[int]:
    """Restored old ``linear = outer * tasks + task_id`` mapping."""

    return [index % tasks for index in range(count)]


def cache_line_writers(owners: list[int | None], element_bytes: int) -> dict[int, set[int]]:
    writers: dict[int, set[int]] = defaultdict(set)
    for index, task in enumerate(owners):
        assert task is not None
        first_byte = index * element_bytes
        last_byte = first_byte + element_bytes - 1
        for line in range(
            first_byte // CACHE_LINE_BYTES,
            last_byte // CACHE_LINE_BYTES + 1,
        ):
            writers[line].add(task)
    return dict(writers)


@pytest.mark.parametrize("count", [0, 1, 7, 8, 15, 16, 31, 32, 33, 95, 1537])
def test_owned_tiles_cover_once_without_overlap(count: int) -> None:
    tiles = owned_tiles(count)
    owners = element_owners_from_tiles(count, tiles)
    assert all(owner is not None for owner in owners)
    assert [tile.tile_id for tile in tiles] == list(range(len(tiles)))
    assert all(tile.start % TILE_ELEMS == 0 for tile in tiles)
    assert all(0 < tile.stop - tile.start <= TILE_ELEMS for tile in tiles)
    if tiles:
        assert tiles[-1].stop == count


@pytest.mark.parametrize("element_bytes", [1, 2, 4, 8])
@pytest.mark.parametrize("count", [1, 31, 32, 33, 95, 1537])
def test_owned_tiles_are_cache_line_exclusive(count: int, element_bytes: int) -> None:
    owners = element_owners_from_tiles(count, owned_tiles(count))
    writers = cache_line_writers(owners, element_bytes)
    assert all(len(tasks) == 1 for tasks in writers.values())
    for tile in owned_tiles(count):
        assert (tile.start * element_bytes) % CACHE_LINE_BYTES == 0


@pytest.mark.parametrize("d", [3, 7, 17, 31])
def test_flat_tile_model_includes_logical_row_crossing(d: int) -> None:
    count = d * 5
    tiles = owned_tiles(count)
    crossing = [tile for tile in tiles if tile.start // d != (tile.stop - 1) // d]
    assert crossing, f"test shape D={d} did not exercise a row-crossing tile"
    assert all(tile.stop <= count for tile in crossing)


@pytest.mark.parametrize("element_bytes", [2, 4])
def test_restored_legacy_mapping_is_known_bad(element_bytes: int) -> None:
    owners = legacy_element_owners(TASKS * 2)
    writers = cache_line_writers(owners, element_bytes)
    conflicting = {line: tasks for line, tasks in writers.items() if len(tasks) > 1}
    assert conflicting, "known-bad old striped mapping unexpectedly became exclusive"
    assert max(len(tasks) for tasks in conflicting.values()) >= CACHE_LINE_BYTES // element_bytes


def test_source_uses_flat_copy_views_and_no_direct_global_scalar_access() -> None:
    source = (Path(__file__).parent / "fa_bwd_symbolic_lowering.py").read_text(
        encoding="utf-8"
    )
    assert "TILE_ELEMS = 32" in source
    assert source.count("tile_start = tile_id * TILE_ELEMS") == 3
    for name in ("q", "k", "v", "dy", "softmax_max", "softmax_sum", "attention"):
        assert re.search(rf"_flat_storage_view\({name},", source)
        assert not re.search(rf"\b{name}\s*\[", source)
    for name in ("dq", "dk", "dv"):
        assert re.search(rf"_flat_storage_view\({name},", source)
        assert not re.search(rf"\b{name}\s*\[", source)
        assert re.search(rf"T\.copy\([\s\S]*?{name}_flat\[", source)
    for field in ("data=src.data", "scope=src.scope()"):
        assert field in source
    assert 'scope="shared.ub"' in source
    assert "T.copy(" in source
    assert "GetValue" not in source
    assert "SetValue" not in source
    assert "DataCacheCleanAndInvalid" not in source


def test_output_dtype_conversion_contract_is_explicit() -> None:
    source = (Path(__file__).parent / "fa_bwd_symbolic_lowering.py").read_text(
        encoding="utf-8"
    )
    assert source.count('if dtype == "float32":') == 3
    assert source.count('out_tile, acc_tile, "CAST_RINT", TILE_ELEMS') == 3


def test_stats_copies_have_exact_two_argument_api() -> None:
    path = Path(__file__).parent / "fa_bwd_symbolic_lowering.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    stats_calls: dict[str, list[ast.Call]] = {"max_flat": [], "sum_flat": []}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "T"
            and node.func.attr == "copy"
            and node.args
            and isinstance(node.args[0], ast.Subscript)
            and isinstance(node.args[0].value, ast.Name)
        ):
            continue
        source_name = node.args[0].value.id
        if source_name in stats_calls:
            stats_calls[source_name].append(node)

    assert {name: len(calls) for name, calls in stats_calls.items()} == {
        "max_flat": 3,
        "sum_flat": 3,
    }
    for calls in stats_calls.values():
        for call in calls:
            assert len(call.args) == 2
            assert not call.keywords


def _values(count: int, phase: float) -> list[float]:
    return [math.sin(index * 0.37 + phase) * 0.25 for index in range(count)]


def _row_direct(values: list[float], base: int, width: int) -> list[float]:
    return values[base : base + width]


def _row_staged(values: list[float], base: int, width: int) -> list[float]:
    """CPU model of fixed-32 UB chunks with a zero-filled final tail."""

    result = []
    for chunk_start in range(0, width, TILE_ELEMS):
        valid = min(TILE_ELEMS, width - chunk_start)
        ub = [0.0] * TILE_ELEMS
        ub[:valid] = values[base + chunk_start : base + chunk_start + valid]
        result.extend(ub[:valid])
    return result


def _cpu_fa_bwd(
    row_reader,
    *,
    causal: int,
    window_left: int,
    window_right: int,
    softcap: float,
) -> tuple[list[float], list[float], list[float]]:
    """Independent small-shape CPU implementation of the three equations."""

    batch, sq_len, sk_len, hq_count, hk_count, d_size = 1, 3, 4, 2, 1, 7
    q_count = batch * sq_len * hq_count * d_size
    kv_count = batch * sk_len * hk_count * d_size
    q = _values(q_count, 0.1)
    k = _values(kv_count, 0.2)
    v = _values(kv_count, 0.3)
    dy = _values(q_count, 0.4)
    attention = _values(q_count, 0.5)
    scale = 0.7
    group = hq_count // hk_count
    dq = [0.0] * q_count
    dk = [0.0] * kv_count
    dv = [0.0] * kv_count

    for b_i in range(batch):
        for hq_i in range(hq_count):
            hk_i = hq_i // group
            for sq_i in range(sq_len):
                q_base = ((b_i * sq_len + sq_i) * hq_count + hq_i) * d_size
                q_row = row_reader(q, q_base, d_size)
                dy_row = row_reader(dy, q_base, d_size)
                attention_row = row_reader(attention, q_base, d_size)
                softmax_max = 0.15 + 0.01 * (hq_i + sq_i)
                softmax_sum = 1.3 + 0.03 * (hq_i + sq_i)
                for sk_i in range(sk_len):
                    relative = sk_i - (sq_i + sk_len - sq_len)
                    masked = (
                        (causal != 0 and relative > 0)
                        or (window_left >= 0 and relative < -window_left)
                        or (window_right >= 0 and relative > window_right)
                    )
                    if masked:
                        continue
                    kv_base = ((b_i * sk_len + sk_i) * hk_count + hk_i) * d_size
                    k_row = row_reader(k, kv_base, d_size)
                    v_row = row_reader(v, kv_base, d_size)
                    score_before = 0.0
                    dp = 0.0
                    drow = 0.0
                    for rd_i in range(d_size):
                        score_before += q_row[rd_i] * k_row[rd_i]
                        dp += dy_row[rd_i] * v_row[rd_i]
                        drow += dy_row[rd_i] * attention_row[rd_i]
                    score_before *= scale
                    score = score_before
                    softcap_derivative = 1.0
                    if softcap > 0.0:
                        tanh_value = math.tanh(score_before / softcap)
                        score = softcap * tanh_value
                        softcap_derivative = 1.0 - tanh_value * tanh_value
                    probability = math.exp(score - softmax_max) / softmax_sum
                    ds = probability * (dp - drow) * softcap_derivative * scale
                    for d_i in range(d_size):
                        dq[q_base + d_i] += ds * k_row[d_i]
                        dk[kv_base + d_i] += ds * q_row[d_i]
                        dv[kv_base + d_i] += probability * dy_row[d_i]
    return dq, dk, dv


def _publish_owned_tiles(values: list[float]) -> list[float]:
    output = [math.nan] * len(values)
    for tile in owned_tiles(len(values)):
        output[tile.start : tile.stop] = values[tile.start : tile.stop]
    assert not any(math.isnan(value) for value in output)
    return output


@pytest.mark.parametrize(
    "causal,window_left,window_right,softcap",
    [
        (0, -1, -1, 0.0),
        (1, -1, -1, 0.0),
        (0, 1, 1, 1.3),
    ],
)
def test_staged_owned_tile_math_matches_direct_cpu_reference(
    causal: int,
    window_left: int,
    window_right: int,
    softcap: float,
) -> None:
    expected = _cpu_fa_bwd(
        _row_direct,
        causal=causal,
        window_left=window_left,
        window_right=window_right,
        softcap=softcap,
    )
    staged = _cpu_fa_bwd(
        _row_staged,
        causal=causal,
        window_left=window_left,
        window_right=window_right,
        softcap=softcap,
    )
    for expected_output, staged_output in zip(expected, staged, strict=True):
        published = _publish_owned_tiles(staged_output)
        assert published == pytest.approx(expected_output, rel=1e-13, abs=1e-13)
