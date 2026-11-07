/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. Do not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_routing_init_fused_proto.h
 * \brief MoeRoutingInitFused: MoE 派发融合算子（decode 小 batch 路径）。
 * \par Inputs:
 * \li hidden: [token_num, cols]，派发前隐层，bf16（Xing4.0 decode 路径专用）。
 * \li topk_ids: [token_num, k] int32，gating topk 的专家序号。
 * \li topk_weights: [token_num, k] bf16（gating 后已转好，本算子纯透传别名）。
 * \par Outputs:
 * \li sorted_hidden: [token_num*k, cols]，按 expert 稳定排序后的隐层（GMM 的 x）。
 * \li expanded_row_idx: [token_num*k] int32，源位置 i(=row*k+slot) → 目标行号（非负，
 *     与 CANN MoeInitRoutingV3 的负值 padding 语义对齐时本算子恒非负——不 drop）。
 * \li expert_tokens: [expert_num] int64，每 expert 行数（count 模式，GMM group_list）。
 * \li topk_weights_out: [token_num, k]，与 hidden 同 dtype（host 侧以输入别名返回，kernel 零工作）。
 * \par Attributes:
 * \li expert_num: required int，专家总数。
 */
#ifndef OPS_OP_PROTO_INC_MOEROUTINGINITFUSED_H_
#define OPS_OP_PROTO_INC_MOEROUTINGINITFUSED_H_

#include "graph/operator_reg.h"

namespace ge {
REG_OP(MoeRoutingInitFused)
    .INPUT(hidden, TensorType({DT_BF16}))
    .INPUT(topk_ids, TensorType({DT_INT32}))
    .INPUT(topk_weights, TensorType({DT_BF16}))
    .OUTPUT(sorted_hidden, TensorType({DT_BF16}))
    .OUTPUT(expanded_row_idx, TensorType({DT_INT32}))
    .OUTPUT(expert_tokens, TensorType({DT_INT64}))
    .OUTPUT(topk_weights_out, TensorType({DT_BF16}))
    .REQUIRED_ATTR(expert_num, Int)
    .OP_END_FACTORY_REG(MoeRoutingInitFused)
} // namespace ge
#endif // OPS_OP_PROTO_INC_MOEROUTINGINITFUSED_H_
