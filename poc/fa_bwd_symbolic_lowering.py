"""Real scalar FlashAttention backward DSL used by the symbolic A5 POC.

This is deliberately a correctness-shaped compiler probe, not an optimized
kernel. The complete backward equations are expressed in TileLang. B, Sq, Sk,
Hq, Hk and D remain TIR variables and the only factory specialization is the
storage dtype. Each output element is owned by one vector task, avoiding
cross-core atomics while preserving the actual FA backward dataflow.
"""

from __future__ import annotations

from tilelang import language as T


CORE_NUM = 24
VEC_NUM = 2
TASKS = CORE_NUM * VEC_NUM


def make_fa_bwd_scalar(dtype: str, host_entry: str):
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
        softmax_max: T.Tensor([B, Hq, Sq, 8], "float32"),
        softmax_sum: T.Tensor([B, Hq, Sq, 8], "float32"),
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
        with T.Kernel(CORE_NUM, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                task_id = cid * VEC_NUM + vid
                group = Hq // Hk

                exp_in = T.alloc_shared([32], "float32")
                exp_out = T.alloc_shared([32], "float32")
                cast_out = T.alloc_shared([32], dtype)
                scratch = T.alloc_shared([32], "float32", scope="shared.ub")
                mask = T.alloc_shared([32], "int32", scope="shared.ub")
                dq_elements = B * Sq * Hq * D
                for outer in T.serial(T.ceildiv(dq_elements, TASKS)):
                    linear = outer * TASKS + task_id
                    if linear < dq_elements:
                        d_i = linear % D
                        tmp_q = linear // D
                        hq_i = tmp_q % Hq
                        tmp_q = tmp_q // Hq
                        sq_i = tmp_q % Sq
                        b_i = tmp_q // Sq
                        hk_i = hq_i // group
                        scratch[0] = 0.0  # output accumulator
                        for sk_i in T.serial(Sk):
                            relative = sk_i - (sq_i + Sk - Sq)
                            mask[0] = 0
                            if causal != 0 and relative > 0:
                                mask[0] = 1
                            if window_left >= 0 and relative < 0 - window_left:
                                mask[0] = 1
                            if window_right >= 0 and relative > window_right:
                                mask[0] = 1
                            if mask[0] == 0:
                                scratch[1] = 0.0  # score_before
                                scratch[2] = 0.0  # dp
                                scratch[3] = 0.0  # drow
                                for rd in T.serial(D):
                                    if dtype == "bfloat16":
                                        cast_out[0] = q[b_i, sq_i, hq_i, rd]
                                        T.tile.cast(exp_out, cast_out, "CAST_NONE", 32)
                                        cast_out[0] = k[b_i, sk_i, hk_i, rd]
                                        T.tile.cast(exp_in, cast_out, "CAST_NONE", 32)
                                        scratch[1] = scratch[1] + exp_out[0] * exp_in[0]
                                        cast_out[0] = dy[b_i, sq_i, hq_i, rd]
                                        T.tile.cast(exp_out, cast_out, "CAST_NONE", 32)
                                        cast_out[0] = v[b_i, sk_i, hk_i, rd]
                                        T.tile.cast(exp_in, cast_out, "CAST_NONE", 32)
                                        scratch[2] = scratch[2] + exp_out[0] * exp_in[0]
                                        cast_out[0] = attention[b_i, sq_i, hq_i, rd]
                                        T.tile.cast(exp_in, cast_out, "CAST_NONE", 32)
                                        scratch[3] = scratch[3] + exp_out[0] * exp_in[0]
                                    else:
                                        scratch[1] = scratch[1] + T.Cast(
                                            "float32", q[b_i, sq_i, hq_i, rd]
                                        ) * T.Cast("float32", k[b_i, sk_i, hk_i, rd])
                                        scratch[2] = scratch[2] + T.Cast(
                                            "float32", dy[b_i, sq_i, hq_i, rd]
                                        ) * T.Cast("float32", v[b_i, sk_i, hk_i, rd])
                                        scratch[3] = scratch[3] + T.Cast(
                                            "float32", dy[b_i, sq_i, hq_i, rd]
                                        ) * T.Cast(
                                            "float32", attention[b_i, sq_i, hq_i, rd]
                                        )
                                scratch[1] = scratch[1] * scale
                                scratch[4] = scratch[1]  # score
                                scratch[5] = 1.0  # softcap derivative
                                if softcap > 0.0:
                                    exp_in[0] = 2.0 * scratch[1] / softcap
                                    T.tile.exp(exp_out, exp_in)
                                    scratch[6] = (exp_out[0] - 1.0) / (exp_out[0] + 1.0)
                                    scratch[4] = softcap * scratch[6]
                                    scratch[5] = 1.0 - scratch[6] * scratch[6]
                                exp_in[0] = scratch[4] - softmax_max[b_i, hq_i, sq_i, 0]
                                T.tile.exp(exp_out, exp_in)
                                scratch[7] = (
                                    exp_out[0]
                                    / softmax_sum[b_i, hq_i, sq_i, 0]
                                    * (scratch[2] - scratch[3])
                                    * scratch[5]
                                    * scale
                                )
                                if dtype == "bfloat16":
                                    cast_out[0] = k[b_i, sk_i, hk_i, d_i]
                                    T.tile.cast(exp_in, cast_out, "CAST_NONE", 32)
                                    scratch[0] = scratch[0] + scratch[7] * exp_in[0]
                                else:
                                    scratch[0] = scratch[0] + scratch[7] * T.Cast(
                                        "float32", k[b_i, sk_i, hk_i, d_i]
                                    )
                        if dtype == "float32":
                            dq[b_i, sq_i, hq_i, d_i] = scratch[0]
                        else:
                            exp_in[0] = scratch[0]
                            T.tile.cast(cast_out, exp_in, "CAST_RINT", 32)
                            dq[b_i, sq_i, hq_i, d_i] = cast_out[0]

                dk_elements = B * Sk * Hk * D
                for outer in T.serial(T.ceildiv(dk_elements, TASKS)):
                    linear = outer * TASKS + task_id
                    if linear < dk_elements:
                        d_i = linear % D
                        tmp_k = linear // D
                        hk_i = tmp_k % Hk
                        tmp_k = tmp_k // Hk
                        sk_i = tmp_k % Sk
                        b_i = tmp_k // Sk
                        scratch[0] = 0.0  # output accumulator
                        for group_i in T.serial(group):
                            hq_i = hk_i * group + group_i
                            for sq_i in T.serial(Sq):
                                relative = sk_i - (sq_i + Sk - Sq)
                                mask[0] = 0
                                if causal != 0 and relative > 0:
                                    mask[0] = 1
                                if window_left >= 0 and relative < 0 - window_left:
                                    mask[0] = 1
                                if window_right >= 0 and relative > window_right:
                                    mask[0] = 1
                                if mask[0] == 0:
                                    scratch[1] = 0.0
                                    scratch[2] = 0.0
                                    scratch[3] = 0.0
                                    for rd in T.serial(D):
                                        if dtype == "bfloat16":
                                            cast_out[0] = q[b_i, sq_i, hq_i, rd]
                                            T.tile.cast(
                                                exp_out, cast_out, "CAST_NONE", 32
                                            )
                                            cast_out[0] = k[b_i, sk_i, hk_i, rd]
                                            T.tile.cast(
                                                exp_in, cast_out, "CAST_NONE", 32
                                            )
                                            scratch[1] = (
                                                scratch[1] + exp_out[0] * exp_in[0]
                                            )
                                            cast_out[0] = dy[b_i, sq_i, hq_i, rd]
                                            T.tile.cast(
                                                exp_out, cast_out, "CAST_NONE", 32
                                            )
                                            cast_out[0] = v[b_i, sk_i, hk_i, rd]
                                            T.tile.cast(
                                                exp_in, cast_out, "CAST_NONE", 32
                                            )
                                            scratch[2] = (
                                                scratch[2] + exp_out[0] * exp_in[0]
                                            )
                                            cast_out[0] = attention[b_i, sq_i, hq_i, rd]
                                            T.tile.cast(
                                                exp_in, cast_out, "CAST_NONE", 32
                                            )
                                            scratch[3] = (
                                                scratch[3] + exp_out[0] * exp_in[0]
                                            )
                                        else:
                                            scratch[1] = scratch[1] + T.Cast(
                                                "float32", q[b_i, sq_i, hq_i, rd]
                                            ) * T.Cast(
                                                "float32", k[b_i, sk_i, hk_i, rd]
                                            )
                                            scratch[2] = scratch[2] + T.Cast(
                                                "float32", dy[b_i, sq_i, hq_i, rd]
                                            ) * T.Cast(
                                                "float32", v[b_i, sk_i, hk_i, rd]
                                            )
                                            scratch[3] = scratch[3] + T.Cast(
                                                "float32", dy[b_i, sq_i, hq_i, rd]
                                            ) * T.Cast(
                                                "float32",
                                                attention[b_i, sq_i, hq_i, rd],
                                            )
                                    scratch[1] = scratch[1] * scale
                                    scratch[4] = scratch[1]
                                    scratch[5] = 1.0
                                    if softcap > 0.0:
                                        exp_in[0] = 2.0 * scratch[1] / softcap
                                        T.tile.exp(exp_out, exp_in)
                                        scratch[6] = (exp_out[0] - 1.0) / (
                                            exp_out[0] + 1.0
                                        )
                                        scratch[4] = softcap * scratch[6]
                                        scratch[5] = 1.0 - scratch[6] * scratch[6]
                                    exp_in[0] = (
                                        scratch[4] - softmax_max[b_i, hq_i, sq_i, 0]
                                    )
                                    T.tile.exp(exp_out, exp_in)
                                    scratch[7] = (
                                        exp_out[0]
                                        / softmax_sum[b_i, hq_i, sq_i, 0]
                                        * (scratch[2] - scratch[3])
                                        * scratch[5]
                                        * scale
                                    )
                                    if dtype == "bfloat16":
                                        cast_out[0] = q[b_i, sq_i, hq_i, d_i]
                                        T.tile.cast(exp_in, cast_out, "CAST_NONE", 32)
                                        scratch[0] = scratch[0] + scratch[7] * exp_in[0]
                                    else:
                                        scratch[0] = scratch[0] + scratch[7] * T.Cast(
                                            "float32", q[b_i, sq_i, hq_i, d_i]
                                        )
                        if dtype == "float32":
                            dk[b_i, sk_i, hk_i, d_i] = scratch[0]
                        else:
                            exp_in[0] = scratch[0]
                            T.tile.cast(cast_out, exp_in, "CAST_RINT", 32)
                            dk[b_i, sk_i, hk_i, d_i] = cast_out[0]

                dv_elements = B * Sk * Hk * D
                for outer in T.serial(T.ceildiv(dv_elements, TASKS)):
                    linear = outer * TASKS + task_id
                    if linear < dv_elements:
                        d_i = linear % D
                        tmp_v = linear // D
                        hk_i = tmp_v % Hk
                        tmp_v = tmp_v // Hk
                        sk_i = tmp_v % Sk
                        b_i = tmp_v // Sk
                        scratch[0] = 0.0  # output accumulator
                        for group_i in T.serial(group):
                            hq_i = hk_i * group + group_i
                            for sq_i in T.serial(Sq):
                                relative = sk_i - (sq_i + Sk - Sq)
                                mask[0] = 0
                                if causal != 0 and relative > 0:
                                    mask[0] = 1
                                if window_left >= 0 and relative < 0 - window_left:
                                    mask[0] = 1
                                if window_right >= 0 and relative > window_right:
                                    mask[0] = 1
                                if mask[0] == 0:
                                    scratch[1] = 0.0
                                    for rd in T.serial(D):
                                        if dtype == "bfloat16":
                                            cast_out[0] = q[b_i, sq_i, hq_i, rd]
                                            T.tile.cast(
                                                exp_out, cast_out, "CAST_NONE", 32
                                            )
                                            cast_out[0] = k[b_i, sk_i, hk_i, rd]
                                            T.tile.cast(
                                                exp_in, cast_out, "CAST_NONE", 32
                                            )
                                            scratch[1] = (
                                                scratch[1] + exp_out[0] * exp_in[0]
                                            )
                                        else:
                                            scratch[1] = scratch[1] + T.Cast(
                                                "float32", q[b_i, sq_i, hq_i, rd]
                                            ) * T.Cast(
                                                "float32", k[b_i, sk_i, hk_i, rd]
                                            )
                                    scratch[1] = scratch[1] * scale
                                    scratch[4] = scratch[1]
                                    if softcap > 0.0:
                                        exp_in[0] = 2.0 * scratch[1] / softcap
                                        T.tile.exp(exp_out, exp_in)
                                        scratch[4] = (
                                            softcap
                                            * (exp_out[0] - 1.0)
                                            / (exp_out[0] + 1.0)
                                        )
                                    exp_in[0] = (
                                        scratch[4] - softmax_max[b_i, hq_i, sq_i, 0]
                                    )
                                    T.tile.exp(exp_out, exp_in)
                                    scratch[7] = (
                                        exp_out[0] / softmax_sum[b_i, hq_i, sq_i, 0]
                                    )
                                    if dtype == "bfloat16":
                                        cast_out[0] = dy[b_i, sq_i, hq_i, d_i]
                                        T.tile.cast(exp_in, cast_out, "CAST_NONE", 32)
                                        scratch[0] = scratch[0] + scratch[7] * exp_in[0]
                                    else:
                                        scratch[0] = scratch[0] + scratch[7] * T.Cast(
                                            "float32", dy[b_i, sq_i, hq_i, d_i]
                                        )
                        if dtype == "float32":
                            dv[b_i, sk_i, hk_i, d_i] = scratch[0]
                        else:
                            exp_in[0] = scratch[0]
                            T.tile.cast(cast_out, exp_in, "CAST_RINT", 32)
                            dv[b_i, sk_i, hk_i, d_i] = cast_out[0]

    return main.with_attr("ascendc_host_entry", host_entry)
