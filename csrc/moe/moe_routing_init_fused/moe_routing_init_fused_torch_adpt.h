/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MOE_ROUTING_INIT_FUSED_TORCH_ADPT_H
#define MOE_ROUTING_INIT_FUSED_TORCH_ADPT_H
namespace vllm_ascend {

// MoeRoutingInitFused：decode 小 batch MoE 派发融合
// 替代 npu_moe_init_routing_v2 + torch.abs(expanded_row_idx) + expert_tokens.to(int64) 三个 kernel。
// topk_weights 与 hidden 同 dtype、纯透传（host 侧以输入别名作输出，kernel 零工作）——
// 上游 fused_moe.py 的 .to(x.dtype) cast 保留（阶段 2 再考虑 fp32 截流消掉它）。
// 调用方需先过 gate：EP=1（无 expert_map）、非量化派发、无 EPLB 冗余专家、active_num <= 512。
// 输出契约与 CANN 路径逐字段一致：sorted_hidden 按 expert 稳定排序（count 模式 group_list）、
// expanded_row_idx[i]（i = row*k+slot）= 目标行号（恒非负）。
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> moe_routing_init_fused(
    const at::Tensor& hidden,
    const at::Tensor& topk_ids,
    const at::Tensor& topk_weights,
    int64_t expert_num)
{
    TORCH_CHECK(hidden.dim() == 2, "The hidden should be 2D");
    TORCH_CHECK(topk_ids.dim() == 2, "The topk_ids should be 2D");
    TORCH_CHECK(topk_weights.dim() == 2, "The topk_weights should be 2D");
    TORCH_CHECK(topk_ids.scalar_type() == at::kInt, "The topk_ids should be int32");
    TORCH_CHECK(topk_weights.scalar_type() == hidden.scalar_type(),
                "The topk_weights should have the same dtype as hidden (pass-through)");
    TORCH_CHECK(expert_num > 0, "expert_num should be positive");

    int64_t tokenNum = hidden.size(0);
    int64_t cols = hidden.size(1);
    int64_t k = topk_ids.size(1);
    TORCH_CHECK(topk_ids.size(0) == tokenNum && topk_weights.size(0) == tokenNum &&
                    topk_weights.size(1) == k,
                "topk_ids/topk_weights shape mismatch with hidden");

    at::Tensor sorted_hidden = at::empty({tokenNum * k, cols}, hidden.options());
    at::Tensor expanded_row_idx = at::empty({tokenNum * k}, hidden.options().dtype(at::kInt));
    at::Tensor expert_tokens = at::empty({expert_num}, hidden.options().dtype(at::kLong));
    at::Tensor topk_weights_out = topk_weights; // 纯透传别名

    EXEC_NPU_CMD(aclnnMoeRoutingInitFused, hidden, topk_ids, topk_weights, expert_num, sorted_hidden,
                 expanded_row_idx, expert_tokens, topk_weights_out);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(sorted_hidden, expanded_row_idx,
                                                                      expert_tokens, topk_weights_out);
}

} // namespace vllm_ascend
#endif
