# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import (
    HAVE_TE_GEMM_SWIZZLED_MASTER_WEIGHT_SCALES,
    HAVE_TE_MXFP8TENSOR,
    fp8_quantize,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(
    not (HAVE_TE_MXFP8TENSOR and HAVE_TE_GEMM_SWIZZLED_MASTER_WEIGHT_SCALES),
    reason="Requires TE quantize_master_weights with gemm_swizzled_scales",
)
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 requires Blackwell",
)
@pytest.mark.parametrize("gemm_swizzled_scales", [False, True])
def test_fp8_quantize_gemm_swizzled_scales(gemm_swizzled_scales):
    """Sharded MXFP8 cast: the scale layout follows the flag and the values do not change."""
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch import MXFP8Quantizer

    Utils.initialize_model_parallel()
    try:
        group = parallel_state.get_data_parallel_group()
        rank, world_size = group.rank(), group.size()
        h, w = 256, 512
        full = torch.randn(h, w, device="cuda", generator=torch.Generator("cuda").manual_seed(7))
        quantizer = MXFP8Quantizer(te.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        weight = quantizer(torch.zeros(h, w, dtype=torch.bfloat16, device="cuda"))
        # Unaligned FSDP-style shard of the flattened weight.
        cuts = [0] + [k * h * w // world_size + 3 for k in range(1, world_size)] + [h * w]
        lo, hi = cuts[rank], cuts[rank + 1]
        fragment = (weight._rowwise_data.view(-1)[lo:hi], weight._columnwise_data.view(-1)[lo:hi])
        fp8_quantize(
            [weight],
            [full.flatten()[lo:hi].clone()],
            [lo],
            group,
            [fragment],
            gemm_swizzled_scales=gemm_swizzled_scales,
        )

        assert weight._with_gemm_swizzled_scales == gemm_swizzled_scales
        expected = quantizer(full.to(torch.bfloat16)).dequantize().flatten()
        torch.testing.assert_close(
            weight.dequantize().flatten()[lo:hi], expected[lo:hi], rtol=0, atol=0
        )
    finally:
        Utils.destroy_model_parallel()
