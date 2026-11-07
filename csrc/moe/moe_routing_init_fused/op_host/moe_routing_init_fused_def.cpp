/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. Do not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_routing_init_fused_def.cpp
 * \brief MoeRoutingInitFused: decode 小 batch 专用（Xing4.0 路径，bf16 专用）的 MoE 派发融合算子。
 * 一次 launch 完成 MoeInitRoutingV3(按 expert 稳定排序 + gather) + expanded_row_idx 非负化
 * (消 host 侧 torch.abs) + expert_tokens int64 化 (消 .to(int64))；topk_weights 纯透传别名。
 */

#include "register/op_def_registry.h"

namespace ops {
class MoeRoutingInitFused : public OpDef {
public:
    explicit MoeRoutingInitFused(const char *name) : OpDef(name)
    {
        this->Input("hidden")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("topk_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("topk_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("sorted_hidden")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("expanded_row_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("expert_tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("topk_weights_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("expert_num").Int();
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(MoeRoutingInitFused);
} // namespace ops
