/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. Do not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_routing_init_fused_infershape.cpp
 * \brief
 */
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

#include <string>

#define TO_STRING(x) std::string(#x)

using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t HIDDEN_INDEX = 0;
static constexpr int64_t TOPK_IDS_INDEX = 1;
static constexpr int64_t SORTED_HIDDEN_INDEX = 0;
static constexpr int64_t EXPANDED_ROW_IDX_INDEX = 1;
static constexpr int64_t EXPERT_TOKENS_INDEX = 2;
static constexpr int64_t TOPK_WEIGHTS_OUT_INDEX = 3;
static constexpr int64_t EXPERT_NUM_ATTR_INDEX = 0;

static ge::graphStatus InferShape4MoeRoutingInitFused(gert::InferShapeContext *context)
{
    OP_LOGD(context, "Begin to do MoeRoutingInitFusedInfershape.");

    const gert::Shape *hiddenShape = context->GetInputShape(HIDDEN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, hiddenShape);
    const gert::Shape *topkIdsShape = context->GetInputShape(TOPK_IDS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsShape);
    gert::Shape *sortedHiddenShape = context->GetOutputShape(SORTED_HIDDEN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortedHiddenShape);
    gert::Shape *expandedRowIdxShape = context->GetOutputShape(EXPANDED_ROW_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedRowIdxShape);
    gert::Shape *expertTokensShape = context->GetOutputShape(EXPERT_TOKENS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokensShape);
    gert::Shape *topkWeightsOutShape = context->GetOutputShape(TOPK_WEIGHTS_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsOutShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *expertNumPtr = attrs->GetAttrPointer<int64_t>(EXPERT_NUM_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertNumPtr);
    const int64_t expertNum = *expertNumPtr;
    if (expertNum < 0) {
        OP_LOGE(context, "expert_num must be a non-negative number.");
        return ge::GRAPH_FAILED;
    }

    if (hiddenShape->GetDimNum() != DIM_TWO || topkIdsShape->GetDimNum() != DIM_TWO) {
        OP_LOGE(context, "hidden/topk_ids should be 2D, hidden is %s, topk_ids is %s.",
                TO_STRING(*hiddenShape).c_str(), TO_STRING(*topkIdsShape).c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t tokenNum = hiddenShape->GetDim(0);
    int64_t cols = hiddenShape->GetDim(1);
    int64_t k = topkIdsShape->GetDim(1);
    int64_t activeNum = (tokenNum == NEG_ONE || k == NEG_ONE) ? NEG_ONE : tokenNum * k;

    sortedHiddenShape->SetDimNum(DIM_TWO);
    sortedHiddenShape->SetDim(0U, activeNum);
    sortedHiddenShape->SetDim(1U, cols);

    expandedRowIdxShape->SetDimNum(DIM_ONE);
    expandedRowIdxShape->SetDim(0U, activeNum);

    expertTokensShape->SetDimNum(DIM_ONE);
    expertTokensShape->SetDim(0U, expertNum);

    topkWeightsOutShape->SetDimNum(DIM_TWO);
    topkWeightsOutShape->SetDim(0U, tokenNum);
    topkWeightsOutShape->SetDim(1U, k);

    OP_LOGD(context, "sorted_hidden shape is: %s after infershape.", TO_STRING(*sortedHiddenShape).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeRoutingInitFused(gert::InferDataTypeContext *context)
{
    OP_LOGD(context, "Begin to do MoeRoutingInitFusedInferDataType.");
    auto hiddenDtype = context->GetInputDataType(HIDDEN_INDEX);
    context->SetOutputDataType(SORTED_HIDDEN_INDEX, hiddenDtype);
    context->SetOutputDataType(EXPANDED_ROW_IDX_INDEX, ge::DT_INT32);
    context->SetOutputDataType(EXPERT_TOKENS_INDEX, ge::DT_INT64);
    context->SetOutputDataType(TOPK_WEIGHTS_OUT_INDEX, hiddenDtype);
    OP_LOGD(context, "End to do MoeRoutingInitFusedInferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeRoutingInitFused)
    .InferShape(InferShape4MoeRoutingInitFused)
    .InferDataType(InferDataType4MoeRoutingInitFused);
} // namespace ops
