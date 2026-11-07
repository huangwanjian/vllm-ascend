# SPDX-License-Identifier: Apache-2.0
# MoeRoutingInitFused 单元测试：与 CANN npu_moe_init_routing_v2 的功能等价（四层校验）。
#
# 语义说明：组内（同一 expert 段内）顺序允许与 CANN 不同（counting-sort 稳定序 vs
# CANN Sort 的序），因此不做逐位对拍，而是四层功能等价：
#   1. 自洽性：sorted_hidden[expanded_row_idx[i]] == hidden[i//k]（gather-back）
#   2. 段正确：expert e 的段 = {hidden[i//k] : topk_ids[i] == e} 的多重集合
#      （按 dst 行序的源槽位 expert 序列必须等于段结构）
#   3. 计数：expert_tokens 与 CANN 参考一致（int64、count 模式）
#   4. unpermute 回环：npu_moe_token_unpermute(permuted, row_idx, probs) 的
#      重建输出与参考路径一致（GMM/unpermute 下游真正消费的契约）
#
# 运行环境：装有含本算子的 vllm_ascend（_C 扩展）+ 已安装 cann-ops 包
# （libcust_opapi 含 aclnnMoeRoutingInitFused）+ NPU 设备。
# pytest tests/ut/ops/test_moe_routing_init_fused.py

from __future__ import annotations

import pytest
import torch
import torch_npu

# 导入顺序：先走 ops 包入口（与框架真实路径一致），避免预存循环导入
# device_op -> ops.triton.fla -> ops/__init__ -> fused_moe -> experts_selector -> device_op
import vllm_ascend.ops.fused_moe.fused_moe  # noqa: F401
from vllm_ascend.utils import enable_custom_op

# C 扩展懒注册：enable_custom_op 内 import vllm_ascend_C 触发 TORCH_LIBRARY 注册；
# 真实 vllm 路径由 platform init 调用，此处须显式触发
_HAS_OP = enable_custom_op() and hasattr(torch.ops._C_ascend, "moe_routing_init_fused")

pytestmark = pytest.mark.skipif(not _HAS_OP, reason="moe_routing_init_fused 未注册（需 NPU + 已安装含本算子的算子包）")

E = 64  # 专家数（Xing4.0-29B-A4B routed experts）
K = 3584
K_TOPK = 4


def _ref_cann(hidden, ids, weights):
    sorted_h, row_idx, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
        hidden,
        ids,
        scale=None,
        active_num=hidden.shape[0] * ids.shape[1],
        expert_num=E,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, E],
        quant_mode=-1,
    )
    return sorted_h, row_idx.abs(), expert_tokens.to(torch.int64)


def _fused(hidden, ids, weights):
    from vllm_ascend.device.device_op import DeviceOperator

    return DeviceOperator.moe_routing_init_fused(hidden, ids, weights, expert_num=E)


def _check_case(hidden, ids, weights) -> bool:
    ref_h, ref_idx, ref_counts = _ref_cann(hidden, ids, weights)
    fus_h, fus_idx, fus_counts, fus_w = _fused(hidden, ids, weights)

    ok = True

    # 1. 自洽性（gather-back）
    back = torch.zeros_like(fus_h)
    back[fus_idx.long()] = hidden.repeat_interleave(K_TOPK, dim=0)
    if not torch.equal(back, fus_h):
        ok = False

    # 2. 段归属 + expert_tokens
    if not torch.equal(ref_counts.cpu(), fus_counts.cpu()):
        ok = False
    else:
        seg_of = torch.repeat_interleave(torch.arange(E, dtype=torch.int64), fus_counts.cpu().to(torch.int64)).to("npu")
        # p = argsort(fus_idx) 给出按 dst 行序的源槽位排列，其 expert 序列须等于段结构
        got_seg = ids.reshape(-1).long()[torch.argsort(fus_idx.long())]
        if not torch.equal(got_seg, seg_of):
            ok = False

    # 3. unpermute 回环（下游真实契约；两侧 idx 均已非负）
    un_fused = torch_npu.npu_moe_token_unpermute(permuted_tokens=fus_h, sorted_indices=fus_idx, probs=fus_w)
    un_ref = torch_npu.npu_moe_token_unpermute(permuted_tokens=ref_h, sorted_indices=ref_idx, probs=weights)
    if not torch.allclose(un_fused.float(), un_ref.float(), atol=2e-2, rtol=2e-2):
        ok = False
    return ok


@pytest.mark.parametrize("S", [1, 2, 5, 8, 10, 16])
def test_correctness_matrix(S):
    """均匀随机路由下的四层功能等价（S 上限覆盖 gate 的 activeNum<=64 区间）。"""
    torch.manual_seed(0)
    hidden = torch.randn(S, K, dtype=torch.bfloat16, device="npu")
    ids = torch.randint(0, E, (S, K_TOPK), dtype=torch.int32, device="npu")
    weights = torch.rand(S, K_TOPK, dtype=torch.bfloat16, device="npu")
    assert _check_case(hidden, ids, weights)


@pytest.mark.parametrize("S", [1, 8, 16])
@pytest.mark.parametrize("mode", ["all_one_expert", "heavy_skew"])
def test_skewed_routing(S, mode):
    """倾斜分布：全挤单 expert 与 90% 挤单 expert（counting-sort 边界）。"""
    torch.manual_seed(1)
    hidden = torch.randn(S, K, dtype=torch.bfloat16, device="npu")
    ids = torch.full((S, K_TOPK), 0, dtype=torch.int32, device="npu")
    if mode == "heavy_skew":
        spread = torch.randint(0, E, (S, K_TOPK), dtype=torch.int32, device="npu")
        mask = torch.rand(S, K_TOPK, device="npu") > 0.9
        ids = torch.where(mask, spread, ids)
    weights = torch.rand(S, K_TOPK, dtype=torch.bfloat16, device="npu")
    assert _check_case(hidden, ids, weights)
