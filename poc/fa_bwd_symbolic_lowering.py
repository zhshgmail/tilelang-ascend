"""Owned-tile FlashAttention backward DSL for the symbolic A5 POC.

Each vector task owns an aligned contiguous 32-element output tile.  All
global-memory traffic is expressed as ``T.copy`` between a flat view of the
original parameter and explicitly scoped UB staging buffers; scalar arithmetic
only reads or writes LocalTensor storage.

This remains a correctness-shaped compiler probe rather than an optimized
kernel.  In particular, each output lane stages its input chunks independently.
That keeps the original reduction order and makes the production API boundary
explicit, but it is a known runtime-copy inefficiency to address only after the
card-free compiler and numerical gates pass.
"""

from __future__ import annotations

from tilelang import language as T
from tvm import tir


CORE_NUM = 24
VEC_NUM = 2
TASKS = CORE_NUM * VEC_NUM
TILE_ELEMS = 32
STATS_ELEMS = 8


def _flat_storage_view(src, elements):
    """Return a one-dimensional view over the exact same parameter data Var.

    Parent ``4a203f4291`` exports ``T.reshape``, but that helper resolves
    ``T.Buffer`` to the zero-argument language proxy and raises
    ``Buffer() takes no arguments`` during TVMScript parsing.  Keep this
    workaround local to the PoC rather than broadening Revision 2 into a DSL
    fix.  Parameter buffers are contiguous and have zero element offset.
    """

    return tir.decl_buffer(
        (elements,),
        src.dtype,
        name=f"{src.name}_flat",
        data=src.data,
        scope=src.scope(),
    )


def make_fa_bwd_scalar(dtype: str, host_entry: str, kernel_entry: str):
    """Return a symbolic, numerical FlashAttention backward PrimFunc."""

    B = T.symbolic("B")
    Sq = T.symbolic("Sq")
    Sk = T.symbolic("Sk")
    Hq = T.symbolic("Hq")
    Hk = T.symbolic("Hk")
    D = T.symbolic("D")

    @T.prim_func
    def main(
        q: T.Tensor([B, Sq, Hq, D], dtype),
        k: T.Tensor([B, Sk, Hk, D], dtype),
        v: T.Tensor([B, Sk, Hk, D], dtype),
        dy: T.Tensor([B, Sq, Hq, D], dtype),
        softmax_max: T.Tensor([B, Hq, Sq, STATS_ELEMS], "float32"),
        softmax_sum: T.Tensor([B, Hq, Sq, STATS_ELEMS], "float32"),
        attention: T.Tensor([B, Sq, Hq, D], dtype),
        dq: T.Tensor([B, Sq, Hq, D], dtype),
        dk: T.Tensor([B, Sk, Hk, D], dtype),
        dv: T.Tensor([B, Sk, Hk, D], dtype),
        causal: T.int32,
        window_left: T.int32,
        window_right: T.int32,
        softcap: T.float32,
        scale: T.float32,
    ):
        q_elements = B * Sq * Hq * D
        kv_elements = B * Sk * Hk * D
        stats_elements = B * Hq * Sq * STATS_ELEMS
        q_flat = _flat_storage_view(q, q_elements)
        k_flat = _flat_storage_view(k, kv_elements)
        v_flat = _flat_storage_view(v, kv_elements)
        dy_flat = _flat_storage_view(dy, q_elements)
        max_flat = _flat_storage_view(softmax_max, stats_elements)
        sum_flat = _flat_storage_view(softmax_sum, stats_elements)
        attention_flat = _flat_storage_view(attention, q_elements)
        dq_flat = _flat_storage_view(dq, q_elements)
        dk_flat = _flat_storage_view(dk, kv_elements)
        dv_flat = _flat_storage_view(dv, kv_elements)

        with T.Kernel(CORE_NUM, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                task_id = cid * VEC_NUM + vid
                group = Hq // Hk

                q_ub = T.alloc_shared([TILE_ELEMS], dtype, scope="shared.ub")
                k_ub = T.alloc_shared([TILE_ELEMS], dtype, scope="shared.ub")
                v_ub = T.alloc_shared([TILE_ELEMS], dtype, scope="shared.ub")
                dy_ub = T.alloc_shared([TILE_ELEMS], dtype, scope="shared.ub")
                attention_ub = T.alloc_shared(
                    [TILE_ELEMS], dtype, scope="shared.ub"
                )
                q_f32_ub = T.alloc_shared(
                    [TILE_ELEMS], "float32", scope="shared.ub"
                )
                k_f32_ub = T.alloc_shared(
                    [TILE_ELEMS], "float32", scope="shared.ub"
                )
                v_f32_ub = T.alloc_shared(
                    [TILE_ELEMS], "float32", scope="shared.ub"
                )
                dy_f32_ub = T.alloc_shared(
                    [TILE_ELEMS], "float32", scope="shared.ub"
                )
                attention_f32_ub = T.alloc_shared(
                    [TILE_ELEMS], "float32", scope="shared.ub"
                )
                acc_tile = T.alloc_shared(
                    [TILE_ELEMS], "float32", scope="shared.ub"
                )
                out_tile = T.alloc_shared(
                    [TILE_ELEMS], dtype, scope="shared.ub"
                )
                max_ub = T.alloc_shared(
                    [STATS_ELEMS], "float32", scope="shared.ub"
                )
                sum_ub = T.alloc_shared(
                    [STATS_ELEMS], "float32", scope="shared.ub"
                )
                exp_in = T.alloc_shared(
                    [TILE_ELEMS], "float32", scope="shared.ub"
                )
                exp_out = T.alloc_shared(
                    [TILE_ELEMS], "float32", scope="shared.ub"
                )
                scratch = T.alloc_shared(
                    [TILE_ELEMS], "float32", scope="shared.ub"
                )
                mask = T.alloc_shared(
                    [TILE_ELEMS], "int32", scope="shared.ub"
                )

                # DQ: preserve the original Sk -> D reduction order per lane,
                # then publish the whole owned output tile with one copy.
                dq_tiles = T.ceildiv(q_elements, TILE_ELEMS)
                for outer in T.serial(T.ceildiv(dq_tiles, TASKS)):
                    tile_id = outer * TASKS + task_id
                    if tile_id < dq_tiles:
                        tile_start = tile_id * TILE_ELEMS
                        valid = T.min(TILE_ELEMS, q_elements - tile_start)
                        for lane in T.serial(TILE_ELEMS):
                            acc_tile[lane] = 0.0
                            if lane < valid:
                                linear = tile_start + lane
                                d_i = linear % D
                                row_q = linear // D
                                hq_i = row_q % Hq
                                tmp_q = row_q // Hq
                                sq_i = tmp_q % Sq
                                b_i = tmp_q // Sq
                                hk_i = hq_i // group
                                q_base = row_q * D
                                dy_base = row_q * D
                                attention_base = row_q * D
                                stats_base = (
                                    (b_i * Hq + hq_i) * Sq + sq_i
                                ) * STATS_ELEMS
                                T.copy(
                                    max_flat[
                                        stats_base : stats_base + STATS_ELEMS
                                    ],
                                    max_ub,
                                )
                                T.copy(
                                    sum_flat[
                                        stats_base : stats_base + STATS_ELEMS
                                    ],
                                    sum_ub,
                                )
                                scratch[0] = 0.0

                                for sk_i in T.serial(Sk):
                                    relative = sk_i - (sq_i + Sk - Sq)
                                    mask[0] = 0
                                    if causal != 0 and relative > 0:
                                        mask[0] = 1
                                    if window_left >= 0 and relative < -window_left:
                                        mask[0] = 1
                                    if window_right >= 0 and relative > window_right:
                                        mask[0] = 1
                                    if mask[0] == 0:
                                        scratch[1] = 0.0
                                        scratch[2] = 0.0
                                        scratch[3] = 0.0
                                        scratch[8] = 0.0
                                        k_base = (
                                            (b_i * Sk + sk_i) * Hk + hk_i
                                        ) * D
                                        v_base = k_base

                                        for rd_outer in T.serial(
                                            T.ceildiv(D, TILE_ELEMS)
                                        ):
                                            rd_base = rd_outer * TILE_ELEMS
                                            rd_valid = T.min(
                                                TILE_ELEMS, D - rd_base
                                            )
                                            T.copy(
                                                q_flat[
                                                    q_base
                                                    + rd_base : q_base
                                                    + rd_base
                                                    + rd_valid
                                                ],
                                                q_ub[0:rd_valid],
                                            )
                                            T.copy(
                                                k_flat[
                                                    k_base
                                                    + rd_base : k_base
                                                    + rd_base
                                                    + rd_valid
                                                ],
                                                k_ub[0:rd_valid],
                                            )
                                            T.copy(
                                                dy_flat[
                                                    dy_base
                                                    + rd_base : dy_base
                                                    + rd_base
                                                    + rd_valid
                                                ],
                                                dy_ub[0:rd_valid],
                                            )
                                            T.copy(
                                                v_flat[
                                                    v_base
                                                    + rd_base : v_base
                                                    + rd_base
                                                    + rd_valid
                                                ],
                                                v_ub[0:rd_valid],
                                            )
                                            T.copy(
                                                attention_flat[
                                                    attention_base
                                                    + rd_base : attention_base
                                                    + rd_base
                                                    + rd_valid
                                                ],
                                                attention_ub[0:rd_valid],
                                            )
                                            if dtype == "bfloat16":
                                                # The VS-only dependency pass does not
                                                # cover MTE2 -> V.  Bisheng auto-sync is
                                                # disabled for this product, so bind the
                                                # BF16 widening to its input DMA here.
                                                T.set_flag("MTE2", "V", 0)
                                                T.wait_flag("MTE2", "V", 0)
                                                T.tile.cast(
                                                    q_f32_ub,
                                                    q_ub,
                                                    "CAST_NONE",
                                                    TILE_ELEMS,
                                                )
                                                T.tile.cast(
                                                    k_f32_ub,
                                                    k_ub,
                                                    "CAST_NONE",
                                                    TILE_ELEMS,
                                                )
                                                T.tile.cast(
                                                    dy_f32_ub,
                                                    dy_ub,
                                                    "CAST_NONE",
                                                    TILE_ELEMS,
                                                )
                                                T.tile.cast(
                                                    v_f32_ub,
                                                    v_ub,
                                                    "CAST_NONE",
                                                    TILE_ELEMS,
                                                )
                                                T.tile.cast(
                                                    attention_f32_ub,
                                                    attention_ub,
                                                    "CAST_NONE",
                                                    TILE_ELEMS,
                                                )

                                            for rd_lane in T.serial(TILE_ELEMS):
                                                if rd_lane < rd_valid:
                                                    rd_i = rd_base + rd_lane
                                                    if dtype == "bfloat16":
                                                        scratch[1] = (
                                                            scratch[1]
                                                            + q_f32_ub[rd_lane]
                                                            * k_f32_ub[rd_lane]
                                                        )
                                                        scratch[2] = (
                                                            scratch[2]
                                                            + dy_f32_ub[rd_lane]
                                                            * v_f32_ub[rd_lane]
                                                        )
                                                        scratch[3] = (
                                                            scratch[3]
                                                            + dy_f32_ub[rd_lane]
                                                            * attention_f32_ub[rd_lane]
                                                        )
                                                        if rd_i == d_i:
                                                            scratch[8] = k_f32_ub[
                                                                rd_lane
                                                            ]
                                                    else:
                                                        scratch[1] = (
                                                            scratch[1]
                                                            + T.Cast(
                                                                "float32", q_ub[rd_lane]
                                                            )
                                                            * T.Cast(
                                                                "float32", k_ub[rd_lane]
                                                            )
                                                        )
                                                        scratch[2] = (
                                                            scratch[2]
                                                            + T.Cast(
                                                                "float32",
                                                                dy_ub[rd_lane],
                                                            )
                                                            * T.Cast(
                                                                "float32", v_ub[rd_lane]
                                                            )
                                                        )
                                                        scratch[3] = (
                                                            scratch[3]
                                                            + T.Cast(
                                                                "float32",
                                                                dy_ub[rd_lane],
                                                            )
                                                            * T.Cast(
                                                                "float32",
                                                                attention_ub[rd_lane],
                                                            )
                                                        )
                                                        if rd_i == d_i:
                                                            scratch[8] = T.Cast(
                                                                "float32", k_ub[rd_lane]
                                                            )

                                        scratch[1] = scratch[1] * scale
                                        scratch[4] = scratch[1]
                                        scratch[5] = 1.0
                                        if softcap > 0.0:
                                            exp_in[0] = 2.0 * scratch[1] / softcap
                                            T.tile.exp(exp_out, exp_in)
                                            scratch[6] = (
                                                exp_out[0] - 1.0
                                            ) / (exp_out[0] + 1.0)
                                            scratch[4] = softcap * scratch[6]
                                            scratch[5] = (
                                                1.0 - scratch[6] * scratch[6]
                                            )
                                        exp_in[0] = scratch[4] - max_ub[0]
                                        T.tile.exp(exp_out, exp_in)
                                        scratch[7] = (
                                            exp_out[0]
                                            / sum_ub[0]
                                            * (scratch[2] - scratch[3])
                                            * scratch[5]
                                            * scale
                                        )
                                        scratch[0] = (
                                            scratch[0] + scratch[7] * scratch[8]
                                        )
                                acc_tile[lane] = scratch[0]

                        if dtype == "float32":
                            T.set_flag("S", "MTE3", 1)
                            T.wait_flag("S", "MTE3", 1)
                            T.copy(
                                acc_tile[0:valid],
                                dq_flat[tile_start : tile_start + valid],
                            )
                            T.set_flag("MTE3", "S", 1)
                            T.wait_flag("MTE3", "S", 1)
                        else:
                            T.tile.cast(
                                out_tile, acc_tile, "CAST_RINT", TILE_ELEMS
                            )
                            T.set_flag("V", "MTE3", 1)
                            T.wait_flag("V", "MTE3", 1)
                            T.copy(
                                out_tile[0:valid],
                                dq_flat[tile_start : tile_start + valid],
                            )
                            T.set_flag("MTE3", "V", 1)
                            T.wait_flag("MTE3", "V", 1)

                # DK: preserve group -> Sq -> D reduction order per lane.
                dk_tiles = T.ceildiv(kv_elements, TILE_ELEMS)
                for outer in T.serial(T.ceildiv(dk_tiles, TASKS)):
                    tile_id = outer * TASKS + task_id
                    if tile_id < dk_tiles:
                        tile_start = tile_id * TILE_ELEMS
                        valid = T.min(TILE_ELEMS, kv_elements - tile_start)
                        for lane in T.serial(TILE_ELEMS):
                            acc_tile[lane] = 0.0
                            if lane < valid:
                                linear = tile_start + lane
                                d_i = linear % D
                                row_k = linear // D
                                hk_i = row_k % Hk
                                tmp_k = row_k // Hk
                                sk_i = tmp_k % Sk
                                b_i = tmp_k // Sk
                                k_base = row_k * D
                                v_base = row_k * D
                                scratch[0] = 0.0

                                for group_i in T.serial(group):
                                    hq_i = hk_i * group + group_i
                                    for sq_i in T.serial(Sq):
                                        relative = sk_i - (sq_i + Sk - Sq)
                                        mask[0] = 0
                                        if causal != 0 and relative > 0:
                                            mask[0] = 1
                                        if window_left >= 0 and relative < -window_left:
                                            mask[0] = 1
                                        if window_right >= 0 and relative > window_right:
                                            mask[0] = 1
                                        if mask[0] == 0:
                                            q_row = (
                                                (b_i * Sq + sq_i) * Hq + hq_i
                                            )
                                            q_base = q_row * D
                                            dy_base = q_row * D
                                            attention_base = q_row * D
                                            stats_base = (
                                                (b_i * Hq + hq_i) * Sq + sq_i
                                            ) * STATS_ELEMS
                                            T.copy(
                                                max_flat[
                                                    stats_base : stats_base
                                                    + STATS_ELEMS
                                                ],
                                                max_ub,
                                            )
                                            T.copy(
                                                sum_flat[
                                                    stats_base : stats_base
                                                    + STATS_ELEMS
                                                ],
                                                sum_ub,
                                            )
                                            scratch[1] = 0.0
                                            scratch[2] = 0.0
                                            scratch[3] = 0.0
                                            scratch[8] = 0.0

                                            for rd_outer in T.serial(
                                                T.ceildiv(D, TILE_ELEMS)
                                            ):
                                                rd_base = rd_outer * TILE_ELEMS
                                                rd_valid = T.min(
                                                    TILE_ELEMS, D - rd_base
                                                )
                                                T.copy(
                                                    q_flat[
                                                        q_base
                                                        + rd_base : q_base
                                                        + rd_base
                                                        + rd_valid
                                                    ],
                                                    q_ub[0:rd_valid],
                                                )
                                                T.copy(
                                                    k_flat[
                                                        k_base
                                                        + rd_base : k_base
                                                        + rd_base
                                                        + rd_valid
                                                    ],
                                                    k_ub[0:rd_valid],
                                                )
                                                T.copy(
                                                    dy_flat[
                                                        dy_base
                                                        + rd_base : dy_base
                                                        + rd_base
                                                        + rd_valid
                                                    ],
                                                    dy_ub[0:rd_valid],
                                                )
                                                T.copy(
                                                    v_flat[
                                                        v_base
                                                        + rd_base : v_base
                                                        + rd_base
                                                        + rd_valid
                                                    ],
                                                    v_ub[0:rd_valid],
                                                )
                                                T.copy(
                                                    attention_flat[
                                                        attention_base
                                                        + rd_base : attention_base
                                                        + rd_base
                                                        + rd_valid
                                                    ],
                                                    attention_ub[0:rd_valid],
                                                )
                                                if dtype == "bfloat16":
                                                    T.set_flag("MTE2", "V", 2)
                                                    T.wait_flag("MTE2", "V", 2)
                                                    T.tile.cast(
                                                        q_f32_ub,
                                                        q_ub,
                                                        "CAST_NONE",
                                                        TILE_ELEMS,
                                                    )
                                                    T.tile.cast(
                                                        k_f32_ub,
                                                        k_ub,
                                                        "CAST_NONE",
                                                        TILE_ELEMS,
                                                    )
                                                    T.tile.cast(
                                                        dy_f32_ub,
                                                        dy_ub,
                                                        "CAST_NONE",
                                                        TILE_ELEMS,
                                                    )
                                                    T.tile.cast(
                                                        v_f32_ub,
                                                        v_ub,
                                                        "CAST_NONE",
                                                        TILE_ELEMS,
                                                    )
                                                    T.tile.cast(
                                                        attention_f32_ub,
                                                        attention_ub,
                                                        "CAST_NONE",
                                                        TILE_ELEMS,
                                                    )

                                                for rd_lane in T.serial(TILE_ELEMS):
                                                    if rd_lane < rd_valid:
                                                        rd_i = rd_base + rd_lane
                                                        if dtype == "bfloat16":
                                                            scratch[1] = (
                                                                scratch[1]
                                                                + q_f32_ub[rd_lane]
                                                                * k_f32_ub[rd_lane]
                                                            )
                                                            scratch[2] = (
                                                                scratch[2]
                                                                + dy_f32_ub[rd_lane]
                                                                * v_f32_ub[rd_lane]
                                                            )
                                                            scratch[3] = (
                                                                scratch[3]
                                                                + dy_f32_ub[rd_lane]
                                                                * attention_f32_ub[
                                                                    rd_lane
                                                                ]
                                                            )
                                                            if rd_i == d_i:
                                                                scratch[8] = q_f32_ub[
                                                                    rd_lane
                                                                ]
                                                        else:
                                                            scratch[1] = (
                                                                scratch[1]
                                                                + T.Cast(
                                                                    "float32",
                                                                    q_ub[rd_lane],
                                                                )
                                                                * T.Cast(
                                                                    "float32",
                                                                    k_ub[rd_lane],
                                                                )
                                                            )
                                                            scratch[2] = (
                                                                scratch[2]
                                                                + T.Cast(
                                                                    "float32",
                                                                    dy_ub[rd_lane],
                                                                )
                                                                * T.Cast(
                                                                    "float32",
                                                                    v_ub[rd_lane],
                                                                )
                                                            )
                                                            scratch[3] = (
                                                                scratch[3]
                                                                + T.Cast(
                                                                    "float32",
                                                                    dy_ub[rd_lane],
                                                                )
                                                                * T.Cast(
                                                                    "float32",
                                                                    attention_ub[rd_lane],
                                                                )
                                                            )
                                                            if rd_i == d_i:
                                                                scratch[8] = T.Cast(
                                                                    "float32",
                                                                    q_ub[rd_lane],
                                                                )

                                            scratch[1] = scratch[1] * scale
                                            scratch[4] = scratch[1]
                                            scratch[5] = 1.0
                                            if softcap > 0.0:
                                                exp_in[0] = 2.0 * scratch[1] / softcap
                                                T.tile.exp(exp_out, exp_in)
                                                scratch[6] = (
                                                    exp_out[0] - 1.0
                                                ) / (exp_out[0] + 1.0)
                                                scratch[4] = softcap * scratch[6]
                                                scratch[5] = (
                                                    1.0 - scratch[6] * scratch[6]
                                                )
                                            exp_in[0] = scratch[4] - max_ub[0]
                                            T.tile.exp(exp_out, exp_in)
                                            scratch[7] = (
                                                exp_out[0]
                                                / sum_ub[0]
                                                * (scratch[2] - scratch[3])
                                                * scratch[5]
                                                * scale
                                            )
                                            scratch[0] = (
                                                scratch[0] + scratch[7] * scratch[8]
                                            )
                                acc_tile[lane] = scratch[0]

                        if dtype == "float32":
                            T.set_flag("S", "MTE3", 3)
                            T.wait_flag("S", "MTE3", 3)
                            T.copy(
                                acc_tile[0:valid],
                                dk_flat[tile_start : tile_start + valid],
                            )
                            T.set_flag("MTE3", "S", 3)
                            T.wait_flag("MTE3", "S", 3)
                        else:
                            T.tile.cast(
                                out_tile, acc_tile, "CAST_RINT", TILE_ELEMS
                            )
                            T.set_flag("V", "MTE3", 3)
                            T.wait_flag("V", "MTE3", 3)
                            T.copy(
                                out_tile[0:valid],
                                dk_flat[tile_start : tile_start + valid],
                            )
                            T.set_flag("MTE3", "V", 3)
                            T.wait_flag("MTE3", "V", 3)

                # DV: preserve group -> Sq -> D score reduction order per lane.
                dv_tiles = T.ceildiv(kv_elements, TILE_ELEMS)
                for outer in T.serial(T.ceildiv(dv_tiles, TASKS)):
                    tile_id = outer * TASKS + task_id
                    if tile_id < dv_tiles:
                        tile_start = tile_id * TILE_ELEMS
                        valid = T.min(TILE_ELEMS, kv_elements - tile_start)
                        for lane in T.serial(TILE_ELEMS):
                            acc_tile[lane] = 0.0
                            if lane < valid:
                                linear = tile_start + lane
                                d_i = linear % D
                                row_v = linear // D
                                hk_i = row_v % Hk
                                tmp_v = row_v // Hk
                                sk_i = tmp_v % Sk
                                b_i = tmp_v // Sk
                                scratch[0] = 0.0

                                for group_i in T.serial(group):
                                    hq_i = hk_i * group + group_i
                                    for sq_i in T.serial(Sq):
                                        relative = sk_i - (sq_i + Sk - Sq)
                                        mask[0] = 0
                                        if causal != 0 and relative > 0:
                                            mask[0] = 1
                                        if window_left >= 0 and relative < -window_left:
                                            mask[0] = 1
                                        if window_right >= 0 and relative > window_right:
                                            mask[0] = 1
                                        if mask[0] == 0:
                                            q_row = (
                                                (b_i * Sq + sq_i) * Hq + hq_i
                                            )
                                            q_base = q_row * D
                                            dy_base = q_row * D
                                            k_base = (
                                                (b_i * Sk + sk_i) * Hk + hk_i
                                            ) * D
                                            stats_base = (
                                                (b_i * Hq + hq_i) * Sq + sq_i
                                            ) * STATS_ELEMS
                                            T.copy(
                                                max_flat[
                                                    stats_base : stats_base
                                                    + STATS_ELEMS
                                                ],
                                                max_ub,
                                            )
                                            T.copy(
                                                sum_flat[
                                                    stats_base : stats_base
                                                    + STATS_ELEMS
                                                ],
                                                sum_ub,
                                            )
                                            scratch[1] = 0.0
                                            scratch[8] = 0.0

                                            for rd_outer in T.serial(
                                                T.ceildiv(D, TILE_ELEMS)
                                            ):
                                                rd_base = rd_outer * TILE_ELEMS
                                                rd_valid = T.min(
                                                    TILE_ELEMS, D - rd_base
                                                )
                                                T.copy(
                                                    q_flat[
                                                        q_base
                                                        + rd_base : q_base
                                                        + rd_base
                                                        + rd_valid
                                                    ],
                                                    q_ub[0:rd_valid],
                                                )
                                                T.copy(
                                                    k_flat[
                                                        k_base
                                                        + rd_base : k_base
                                                        + rd_base
                                                        + rd_valid
                                                    ],
                                                    k_ub[0:rd_valid],
                                                )
                                                T.copy(
                                                    dy_flat[
                                                        dy_base
                                                        + rd_base : dy_base
                                                        + rd_base
                                                        + rd_valid
                                                    ],
                                                    dy_ub[0:rd_valid],
                                                )
                                                if dtype == "bfloat16":
                                                    T.set_flag("MTE2", "V", 4)
                                                    T.wait_flag("MTE2", "V", 4)
                                                    T.tile.cast(
                                                        q_f32_ub,
                                                        q_ub,
                                                        "CAST_NONE",
                                                        TILE_ELEMS,
                                                    )
                                                    T.tile.cast(
                                                        k_f32_ub,
                                                        k_ub,
                                                        "CAST_NONE",
                                                        TILE_ELEMS,
                                                    )
                                                    T.tile.cast(
                                                        dy_f32_ub,
                                                        dy_ub,
                                                        "CAST_NONE",
                                                        TILE_ELEMS,
                                                    )

                                                for rd_lane in T.serial(TILE_ELEMS):
                                                    if rd_lane < rd_valid:
                                                        rd_i = rd_base + rd_lane
                                                        if dtype == "bfloat16":
                                                            scratch[1] = (
                                                                scratch[1]
                                                                + q_f32_ub[rd_lane]
                                                                * k_f32_ub[rd_lane]
                                                            )
                                                            if rd_i == d_i:
                                                                scratch[8] = dy_f32_ub[
                                                                    rd_lane
                                                                ]
                                                        else:
                                                            scratch[1] = (
                                                                scratch[1]
                                                                + T.Cast(
                                                                    "float32",
                                                                    q_ub[rd_lane],
                                                                )
                                                                * T.Cast(
                                                                    "float32",
                                                                    k_ub[rd_lane],
                                                                )
                                                            )
                                                            if rd_i == d_i:
                                                                scratch[8] = T.Cast(
                                                                    "float32",
                                                                    dy_ub[rd_lane],
                                                                )

                                            scratch[1] = scratch[1] * scale
                                            scratch[4] = scratch[1]
                                            if softcap > 0.0:
                                                exp_in[0] = 2.0 * scratch[1] / softcap
                                                T.tile.exp(exp_out, exp_in)
                                                scratch[4] = softcap * (
                                                    exp_out[0] - 1.0
                                                ) / (exp_out[0] + 1.0)
                                            exp_in[0] = scratch[4] - max_ub[0]
                                            T.tile.exp(exp_out, exp_in)
                                            scratch[7] = exp_out[0] / sum_ub[0]
                                            scratch[0] = (
                                                scratch[0] + scratch[7] * scratch[8]
                                            )
                                acc_tile[lane] = scratch[0]

                        if dtype == "float32":
                            T.set_flag("S", "MTE3", 5)
                            T.wait_flag("S", "MTE3", 5)
                            T.copy(
                                acc_tile[0:valid],
                                dv_flat[tile_start : tile_start + valid],
                            )
                            T.set_flag("MTE3", "S", 5)
                            T.wait_flag("MTE3", "S", 5)
                        else:
                            T.tile.cast(
                                out_tile, acc_tile, "CAST_RINT", TILE_ELEMS
                            )
                            T.set_flag("V", "MTE3", 5)
                            T.wait_flag("V", "MTE3", 5)
                            T.copy(
                                out_tile[0:valid],
                                dv_flat[tile_start : tile_start + valid],
                            )
                            T.set_flag("MTE3", "V", 5)
                            T.wait_flag("MTE3", "V", 5)

    return main.with_attr("ascendc_host_entry", host_entry).with_attr(
        "ascendc_kernel_entry", kernel_entry
    )
