# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from .lower import lower, lower_ascend_ir, resolve_ascend_target, is_device_call  # noqa: F401
from tilelang.pre_codegen_identity import (  # noqa: F401
    FinalTirIdentity,
    SIMULATOR_EVIDENCE_AUTHORITY,
    capture_final_tir_identity,
)
from .param import KernelParam  # noqa: F401
from .callback import register_cuda_postproc, register_hip_postproc  # noqa: F401
