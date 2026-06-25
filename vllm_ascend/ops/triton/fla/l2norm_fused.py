# SPDX-License-Identifier: Apache-2.0
# Fused l2norm for q and k to reduce kernel launch overhead

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["eps", "M", "NUM_CHUNKS"])
def l2norm_fwd_fused_kernel(
    X1, Y1, X2, Y2,
    eps, M, N: tl.constexpr, MBLOCK: tl.constexpr, NUM_CHUNKS
):
    """Fused l2norm for two tensors (q and k) to reduce kernel launches"""
    base_row = tl.program_id(0) * (NUM_CHUNKS * MBLOCK)
    rindex = tl.arange(0, N)[None, :]

    for chunk in range(NUM_CHUNKS):
        row_idx = base_row + chunk * MBLOCK + tl.arange(0, MBLOCK)[:, None]
        xmask = row_idx < M

        # Process first tensor (q)
        xs1 = tl.load(X1 + (rindex + N * row_idx), mask=xmask, other=0.0).to(tl.float32)
        square1 = xs1 * xs1
        square_sum1 = tl.sum(square1, 1)[:, None]
        rsqrt1 = tl.rsqrt(square_sum1 + eps)
        tl.store(Y1 + (rindex + N * row_idx), xs1 * rsqrt1, xmask)

        # Process second tensor (k)
        xs2 = tl.load(X2 + (rindex + N * row_idx), mask=xmask, other=0.0).to(tl.float32)
        square2 = xs2 * xs2
        square_sum2 = tl.sum(square2, 1)[:, None]
        rsqrt2 = tl.rsqrt(square_sum2 + eps)
        tl.store(Y2 + (rindex + N * row_idx), xs2 * rsqrt2, xmask)


def l2norm_fwd_fused(
    x1: torch.Tensor,
    x2: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused l2norm for two tensors to reduce kernel launch overhead.
    
    Args:
        x1: First input tensor (e.g., query)
        x2: Second input tensor (e.g., key)
        eps: Epsilon for numerical stability
        output_dtype: Output dtype, defaults to input dtype
    
    Returns:
        Tuple of normalized tensors (y1, y2)
    """
    assert x1.shape == x2.shape, "Input tensors must have the same shape"
    
    x1_shape_og = x1.shape
    x1 = x1.reshape(-1, x1.shape[-1])
    x2 = x2.reshape(-1, x2.shape[-1])
    
    if output_dtype is None:
        y1 = torch.empty_like(x1)
        y2 = torch.empty_like(x2)
    else:
        y1 = torch.empty_like(x1, dtype=output_dtype)
        y2 = torch.empty_like(x2, dtype=output_dtype)
    
    assert y1.stride(-1) == 1
    assert y2.stride(-1) == 1
    
    T, D = x1.shape[0], x1.shape[-1]
    
    MAX_FUSED_SIZE = 65536 // x1.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError(f"l2norm_fwd_fused: feature dim {D} exceeds max {BD}")

    MBLOCK = 69
    num_core = get_vectorcore_num()
    main_bs = triton.cdiv(T, num_core)
    num_sub_blocks = triton.cdiv(main_bs, MBLOCK)
    grid = (num_core,)
    
    l2norm_fwd_fused_kernel[grid](
        X1=x1, Y1=y1,
        X2=x2, Y2=y2,
        eps=eps,
        M=T,
        N=D,
        MBLOCK=MBLOCK,
        NUM_CHUNKS=num_sub_blocks,
    )

    return y1.view(x1_shape_og), y2.view(x1_shape_og)
