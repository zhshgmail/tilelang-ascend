"""Symbolic tiled Cube+Vector FlashAttention backward for A5 AscendC.

This is the performance-shaped successor to ``make_fa_bwd_scalar``.  It keeps
the same runtime-symbolic BSND ABI while mapping the five matrix products to
Cube and the probability/softcap/mask/Delta stages to Vector.  Output ownership
is split between dQ tiles and dK/dV tiles so correctness never depends on
pre-zeroed output buffers or atomic accumulation.
"""

from tilelang import language as T


CORE_NUM = 24
BQ = 16
BK = 16
D_PAD = 128
STATS_ELEMS = 8

# Stable labels make structural receipts independent of backend spelling.
GEMM_ROLE_QK = "Q @ K^T"
GEMM_ROLE_DP = "dY @ V^T"
GEMM_ROLE_DQ = "dS @ K"
GEMM_ROLE_DK = "dS^T @ Q"
GEMM_ROLE_DV = "P^T @ dY"


TILED_FA_BWD_PASS_CONFIGS = {
    "tl.disable_safe_memory_legalize": True,
    "tl.ascend_auto_cv_combine": True,
    "tl.ascend_auto_cross_core_sync": True,
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_tail_mask": True,
}


def make_fa_bwd_tiled(dtype: str, host_entry: str, kernel_entry: str):
    """Build one runtime-symbolic tiled FA-Bwd PrimFunc for one dtype family."""

    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"unsupported FA-Bwd dtype: {dtype}")

    B = T.symbolic("B")
    Sq = T.symbolic("Sq")
    Sk = T.symbolic("Sk")
    Hq = T.symbolic("Hq")
    Hk = T.symbolic("Hk")
    D = T.symbolic("D")

    @T.macro
    def stage_rows_to_l1(src, stage_ub, dst_l1, b_i, row_start, h_i, row_limit):
        """Zero-pad a BSND tile in UB, then transfer the full tile to L1."""

        T.tile.fill(stage_ub, 0.0)
        for row in T.serial(BQ):
            if row_start + row < row_limit:
                T.copy(
                    src[b_i, row_start + row, h_i, 0:D],
                    stage_ub[row, 0:D],
                )
        T.copy(stage_ub, dst_l1)

    @T.macro
    def load_stats_rows(src, dst_ub, b_i, hq_i, q_start, default_value):
        T.tile.fill(dst_ub, default_value)
        for row in T.serial(BQ):
            if q_start + row < Sq:
                T.copy(
                    src[b_i, hq_i, q_start + row, 0:1],
                    dst_ub[row : row + 1],
                )

    @T.macro
    def load_rows_as_fp32(src, stage_ub, dst_f32_ub, b_i, row_start, h_i, row_limit):
        T.tile.fill(stage_ub, 0.0)
        for row in T.serial(BQ):
            if row_start + row < row_limit:
                T.copy(
                    src[b_i, row_start + row, h_i, 0:D],
                    stage_ub[row, 0:D],
                )
        T.copy(stage_ub, dst_f32_ub)

    @T.macro
    def build_probability_and_ds(
        q,
        k,
        v,
        dy,
        softmax_max,
        softmax_sum,
        attention,
        b_i,
        hq_i,
        hk_i,
        q_start,
        k_start,
        causal_value,
        window_left_value,
        window_right_value,
        softcap_value,
        scale_value,
        q_l1,
        k_l1,
        dy_l1,
        v_l1,
        p_l1,
        ds_l1,
        l0c_score,
        l0c_dp,
        stage_ub,
        attention_f32_ub,
        dy_f32_ub,
        delta_product_ub,
        delta_row_ub,
        delta_2d_ub,
        score_ub,
        probability_ub,
        dp_ub,
        derivative_ub,
        exp_ub,
        numerator_ub,
        denominator_ub,
        one_ub,
        max_row_ub,
        sum_row_ub,
        max_2d_ub,
        sum_2d_ub,
        q_pos_ub,
        k_pos_ub,
        relative_ub,
        mask_ub,
        mask_tmp_ub,
        p_dtype_ub,
        ds_dtype_ub,
    ):
        # Loop-carried tensors are staged through fixed UB tiles so D<128 is
        # explicitly zero padded before Cube consumes the L1 operands.
        stage_rows_to_l1(q, stage_ub, q_l1, b_i, q_start, hq_i, Sq)
        stage_rows_to_l1(k, stage_ub, k_l1, b_i, k_start, hk_i, Sk)
        stage_rows_to_l1(dy, stage_ub, dy_l1, b_i, q_start, hq_i, Sq)
        stage_rows_to_l1(v, stage_ub, v_l1, b_i, k_start, hk_i, Sk)

        # GEMM_ROLE_QK: score = Q @ K^T.
        T.gemm_v0(q_l1, k_l1, l0c_score, transpose_B=True, init=True)
        T.copy(l0c_score, score_ub)
        T.tile.mul(score_ub, score_ub, scale_value)

        load_stats_rows(softmax_max, max_row_ub, b_i, hq_i, q_start, 0.0)
        load_stats_rows(softmax_sum, sum_row_ub, b_i, hq_i, q_start, 1.0)
        T.tile.broadcast(max_2d_ub, max_row_ub, axis=1)
        T.tile.broadcast(sum_2d_ub, sum_row_ub, axis=1)

        # Runtime softcap and its derivative.  Keeping this as tile operations
        # makes the entire 16x16 stage a Vector operation rather than scalar
        # per-output arithmetic.
        T.tile.fill(one_ub, 1.0)
        T.tile.fill(derivative_ub, 1.0)
        if softcap_value > 0.0:
            T.tile.mul(exp_ub, score_ub, 2.0 / softcap_value)
            T.tile.exp(exp_ub, exp_ub)
            T.tile.sub(numerator_ub, exp_ub, one_ub)
            T.tile.add(denominator_ub, exp_ub, 1.0)
            T.tile.div(probability_ub, numerator_ub, denominator_ub)
            T.tile.mul(score_ub, probability_ub, softcap_value)
            T.tile.mul(derivative_ub, probability_ub, probability_ub)
            T.tile.sub(derivative_ub, one_ub, derivative_ub)

        T.tile.sub(probability_ub, score_ub, max_2d_ub)
        T.tile.exp(probability_ub, probability_ub)
        T.tile.div(probability_ub, probability_ub, sum_2d_ub)

        # A packed compare mask covers q/k tails, causal, and both window
        # bounds.  Relative position matches the frozen scalar implementation.
        for qi, ki in T.Parallel(BQ, BK):
            q_pos_ub[qi, ki] = T.Cast("float32", q_start + qi)
            k_pos_ub[qi, ki] = T.Cast("float32", k_start + ki)
            relative_ub[qi, ki] = T.Cast(
                "float32", k_start + ki - (q_start + qi + Sk - Sq)
            )
        T.tile.compare(mask_ub, q_pos_ub, T.Cast("float32", Sq), "LT")
        T.tile.compare(mask_tmp_ub, k_pos_ub, T.Cast("float32", Sk), "LT")
        T.tile.bitwise_and(mask_ub, mask_ub, mask_tmp_ub)
        if causal_value != 0:
            T.tile.compare(mask_tmp_ub, relative_ub, 0.0, "LE")
            T.tile.bitwise_and(mask_ub, mask_ub, mask_tmp_ub)
        if window_left_value >= 0:
            T.tile.compare(
                mask_tmp_ub,
                relative_ub,
                T.Cast("float32", -window_left_value),
                "GE",
            )
            T.tile.bitwise_and(mask_ub, mask_ub, mask_tmp_ub)
        if window_right_value >= 0:
            T.tile.compare(
                mask_tmp_ub,
                relative_ub,
                T.Cast("float32", window_right_value),
                "LE",
            )
            T.tile.bitwise_and(mask_ub, mask_ub, mask_tmp_ub)
        T.tile.select(
            probability_ub,
            mask_ub,
            probability_ub,
            0.0,
            "VSEL_TENSOR_SCALAR_MODE",
        )

        # Delta = sum(attention * dY, D).  UB is explicitly padded before the
        # reduction because PTO GM->UB tail padding is not a correctness API.
        load_rows_as_fp32(
            attention,
            stage_ub,
            attention_f32_ub,
            b_i,
            q_start,
            hq_i,
            Sq,
        )
        load_rows_as_fp32(
            dy, stage_ub, dy_f32_ub, b_i, q_start, hq_i, Sq
        )
        T.tile.mul(delta_product_ub, attention_f32_ub, dy_f32_ub)
        T.reduce_sum(delta_product_ub, delta_row_ub, dim=-1)
        T.tile.broadcast(delta_2d_ub, delta_row_ub, axis=1)

        # GEMM_ROLE_DP: dP = dY @ V^T.
        T.gemm_v0(dy_l1, v_l1, l0c_dp, transpose_B=True, init=True)
        T.copy(l0c_dp, dp_ub)
        T.tile.sub(dp_ub, dp_ub, delta_2d_ub)
        T.tile.mul(dp_ub, dp_ub, probability_ub)
        T.tile.mul(dp_ub, dp_ub, derivative_ub)
        T.tile.mul(dp_ub, dp_ub, scale_value)
        T.tile.select(
            dp_ub,
            mask_ub,
            dp_ub,
            0.0,
            "VSEL_TENSOR_SCALAR_MODE",
        )

        if dtype == "float32":
            T.copy(probability_ub, p_dtype_ub)
            T.copy(dp_ub, ds_dtype_ub)
        else:
            T.tile.cast(p_dtype_ub, probability_ub, "CAST_RINT", BQ * BK)
            T.tile.cast(ds_dtype_ub, dp_ub, "CAST_RINT", BQ * BK)
        T.copy(p_dtype_ub, p_l1)
        T.copy(ds_dtype_ub, ds_l1)

    @T.macro
    def store_owned_rows(src_l0c, out, out_f32_ub, out_dtype_ub, b_i, row_start, h_i, row_limit):
        T.copy(src_l0c, out_f32_ub)
        if dtype == "float32":
            T.copy(out_f32_ub, out_dtype_ub)
        else:
            T.tile.cast(out_dtype_ub, out_f32_ub, "CAST_RINT", BQ * D_PAD)
        for row in T.serial(BQ):
            if row_start + row < row_limit:
                T.copy(
                    out_dtype_ub[row, 0:D],
                    out[b_i, row_start + row, h_i, 0:D],
                )

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
        with T.Kernel(CORE_NUM, threads=1, is_npu=True) as cid:
            # Fixed on-chip shapes; runtime dimensions only control persistent
            # task counts, address arithmetic, valid regions, and masks.
            q_l1 = T.alloc_L1([BQ, D_PAD], dtype)
            k_l1 = T.alloc_L1([BK, D_PAD], dtype)
            dy_l1 = T.alloc_L1([BQ, D_PAD], dtype)
            v_l1 = T.alloc_L1([BK, D_PAD], dtype)
            p_l1 = T.alloc_L1([BQ, BK], dtype)
            ds_l1 = T.alloc_L1([BQ, BK], dtype)

            l0c_score = T.alloc_L0C([BQ, BK], "float32")
            l0c_dp = T.alloc_L0C([BQ, BK], "float32")
            l0c_dq = T.alloc_L0C([BQ, D_PAD], "float32")
            l0c_dk = T.alloc_L0C([BK, D_PAD], "float32")
            l0c_dv = T.alloc_L0C([BK, D_PAD], "float32")

            stage_ub = T.alloc_ub([BQ, D_PAD], dtype)
            attention_f32_ub = T.alloc_ub([BQ, D_PAD], "float32")
            dy_f32_ub = T.alloc_ub([BQ, D_PAD], "float32")
            delta_product_ub = T.alloc_ub([BQ, D_PAD], "float32")
            delta_row_ub = T.alloc_ub([BQ], "float32")
            delta_2d_ub = T.alloc_ub([BQ, BK], "float32")
            score_ub = T.alloc_ub([BQ, BK], "float32")
            probability_ub = T.alloc_ub([BQ, BK], "float32")
            dp_ub = T.alloc_ub([BQ, BK], "float32")
            derivative_ub = T.alloc_ub([BQ, BK], "float32")
            exp_ub = T.alloc_ub([BQ, BK], "float32")
            numerator_ub = T.alloc_ub([BQ, BK], "float32")
            denominator_ub = T.alloc_ub([BQ, BK], "float32")
            one_ub = T.alloc_ub([BQ, BK], "float32")
            max_row_ub = T.alloc_ub([BQ], "float32")
            sum_row_ub = T.alloc_ub([BQ], "float32")
            max_2d_ub = T.alloc_ub([BQ, BK], "float32")
            sum_2d_ub = T.alloc_ub([BQ, BK], "float32")
            q_pos_ub = T.alloc_ub([BQ, BK], "float32")
            k_pos_ub = T.alloc_ub([BQ, BK], "float32")
            relative_ub = T.alloc_ub([BQ, BK], "float32")
            mask_ub = T.alloc_ub([BQ, BK], "float32")
            mask_tmp_ub = T.alloc_ub([BQ, BK], "float32")
            p_dtype_ub = T.alloc_ub([BQ, BK], dtype)
            ds_dtype_ub = T.alloc_ub([BQ, BK], dtype)
            out_f32_ub = T.alloc_ub([BQ, D_PAD], "float32")
            out_dtype_ub = T.alloc_ub([BQ, D_PAD], dtype)

            q_tiles = T.ceildiv(Sq, BQ)
            k_tiles = T.ceildiv(Sk, BK)
            group = Hq // Hk

            # dQ ownership: one task owns the complete Sk reduction for a
            # (batch, query-head, query-tile) output tile.
            dq_task_count = B * Hq * q_tiles
            for task_round in T.serial(T.ceildiv(dq_task_count, CORE_NUM)):
                task_id = task_round * CORE_NUM + cid
                if task_id < dq_task_count:
                    q_tile = task_id % q_tiles
                    hq_i = task_id // q_tiles % Hq
                    b_i = task_id // q_tiles // Hq
                    hk_i = hq_i // group
                    q_start = q_tile * BQ
                    for k_tile in T.serial(k_tiles):
                        k_start = k_tile * BK
                        build_probability_and_ds(
                            q,
                            k,
                            v,
                            dy,
                            softmax_max,
                            softmax_sum,
                            attention,
                            b_i,
                            hq_i,
                            hk_i,
                            q_start,
                            k_start,
                            causal,
                            window_left,
                            window_right,
                            softcap,
                            scale,
                            q_l1,
                            k_l1,
                            dy_l1,
                            v_l1,
                            p_l1,
                            ds_l1,
                            l0c_score,
                            l0c_dp,
                            stage_ub,
                            attention_f32_ub,
                            dy_f32_ub,
                            delta_product_ub,
                            delta_row_ub,
                            delta_2d_ub,
                            score_ub,
                            probability_ub,
                            dp_ub,
                            derivative_ub,
                            exp_ub,
                            numerator_ub,
                            denominator_ub,
                            one_ub,
                            max_row_ub,
                            sum_row_ub,
                            max_2d_ub,
                            sum_2d_ub,
                            q_pos_ub,
                            k_pos_ub,
                            relative_ub,
                            mask_ub,
                            mask_tmp_ub,
                            p_dtype_ub,
                            ds_dtype_ub,
                        )
                        # GEMM_ROLE_DQ: accumulate dQ = dS @ K.
                        T.gemm_v0(
                            ds_l1,
                            k_l1,
                            l0c_dq,
                            init=(k_tile == 0),
                        )
                    store_owned_rows(
                        l0c_dq,
                        dq,
                        out_f32_ub,
                        out_dtype_ub,
                        b_i,
                        q_start,
                        hq_i,
                        Sq,
                    )

            # dK/dV ownership: one task owns all query-head/query-tile
            # contributions for a (batch, kv-head, key-tile) output tile.
            dkv_task_count = B * Hk * k_tiles
            for task_round in T.serial(T.ceildiv(dkv_task_count, CORE_NUM)):
                task_id = task_round * CORE_NUM + cid
                if task_id < dkv_task_count:
                    k_tile = task_id % k_tiles
                    hk_i = task_id // k_tiles % Hk
                    b_i = task_id // k_tiles // Hk
                    k_start = k_tile * BK
                    for group_i in T.serial(group):
                        hq_i = hk_i * group + group_i
                        for q_tile in T.serial(q_tiles):
                            q_start = q_tile * BQ
                            build_probability_and_ds(
                                q,
                                k,
                                v,
                                dy,
                                softmax_max,
                                softmax_sum,
                                attention,
                                b_i,
                                hq_i,
                                hk_i,
                                q_start,
                                k_start,
                                causal,
                                window_left,
                                window_right,
                                softcap,
                                scale,
                                q_l1,
                                k_l1,
                                dy_l1,
                                v_l1,
                                p_l1,
                                ds_l1,
                                l0c_score,
                                l0c_dp,
                                stage_ub,
                                attention_f32_ub,
                                dy_f32_ub,
                                delta_product_ub,
                                delta_row_ub,
                                delta_2d_ub,
                                score_ub,
                                probability_ub,
                                dp_ub,
                                derivative_ub,
                                exp_ub,
                                numerator_ub,
                                denominator_ub,
                                one_ub,
                                max_row_ub,
                                sum_row_ub,
                                max_2d_ub,
                                sum_2d_ub,
                                q_pos_ub,
                                k_pos_ub,
                                relative_ub,
                                mask_ub,
                                mask_tmp_ub,
                                p_dtype_ub,
                                ds_dtype_ub,
                            )
                            first_contribution = T.And(
                                group_i == 0, q_tile == 0
                            )
                            # GEMM_ROLE_DK: dK = dS^T @ Q.
                            T.gemm_v0(
                                ds_l1,
                                q_l1,
                                l0c_dk,
                                transpose_A=True,
                                init=first_contribution,
                            )
                            # GEMM_ROLE_DV: dV = P^T @ dY.
                            T.gemm_v0(
                                p_l1,
                                dy_l1,
                                l0c_dv,
                                transpose_A=True,
                                init=first_contribution,
                            )
                    store_owned_rows(
                        l0c_dk,
                        dk,
                        out_f32_ub,
                        out_dtype_ub,
                        b_i,
                        k_start,
                        hk_i,
                        Sk,
                    )
                    store_owned_rows(
                        l0c_dv,
                        dv,
                        out_f32_ub,
                        out_dtype_ub,
                        b_i,
                        k_start,
                        hk_i,
                        Sk,
                    )

    return main.with_attr("ascendc_host_entry", host_entry).with_attr(
        "ascendc_kernel_entry", kernel_entry
    )
