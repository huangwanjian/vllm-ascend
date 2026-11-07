/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. Do not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_routing_init_fused_tiling.cpp
 * \brief MoeRoutingInitFused tiling：小 batch（activeNum = S*k <= 64）路径，
 * 每个参与核冗余计算 counting-sort 派发映射（结果留 UB），全核按源位置分区 gather。
 */
#include <cmath>
#include "register/op_def_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "platform/platform_info.h"

#include "tiling_base/error_log.h"
#include "moe_routing_init_fused_tiling.h"

#ifndef CEIL_ALIGN
#define CEIL_ALIGN(val, align) ((((val) + (align) - 1) / (align)) * (align))
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

namespace optiling {
const static size_t HIDDEN_INPUT_DIMS = 2;
const static size_t TOPK_INPUT_DIMS = 2;
const static int64_t HIDDEN_INPUT_INDEX = 0;
const static int64_t TOPK_IDS_INPUT_INDEX = 1;
const static int64_t TOPK_WEIGHTS_INPUT_INDEX = 2;
const static int64_t SORTED_HIDDEN_OUTPUT_INDEX = 0;
const static int64_t EXPANDED_ROW_IDX_OUTPUT_INDEX = 1;
const static int64_t EXPERT_TOKENS_OUTPUT_INDEX = 2;
const static int64_t TOPK_WEIGHTS_OUT_OUTPUT_INDEX = 3;
const static int64_t EXPERT_NUM_ATTR_INDEX = 0;
// 保守上限：python 侧另有 gate（S*k <= 512），tiling 再兜一道
const static int64_t MAX_ACTIVE_NUM = 1024;

class MoeRoutingInitFusedTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeRoutingInitFusedTilingBase(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeRoutingInitFusedTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    ge::graphStatus CheckShapeAndAttr();

    const gert::Shape *hiddenShape_ = nullptr;
    const gert::Shape *topkIdsShape_ = nullptr;
    const gert::Shape *topkWeightsShape_ = nullptr;

    int64_t tokenNum_ = 0;
    int64_t k_ = 0;
    int64_t expertNum_ = 0;
    int64_t cols_ = 0;
    int64_t activeNum_ = 0;
    const char *opName_ = "";
    MoeRoutingInitFusedTilingData tilingData_;
};

ge::graphStatus MoeRoutingInitFusedTilingBase::CheckShapeAndAttr()
{
    OP_CHECK_IF(hiddenShape_->GetDimNum() != HIDDEN_INPUT_DIMS,
                OP_LOGE(context_, "The dim number of hidden is: %zu, but should be %zu.",
                        hiddenShape_->GetDimNum(), HIDDEN_INPUT_DIMS),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(topkIdsShape_->GetDimNum() != TOPK_INPUT_DIMS,
                OP_LOGE(context_, "The dim number of topk_ids is: %zu, but should be %zu.",
                        topkIdsShape_->GetDimNum(), TOPK_INPUT_DIMS),
                return ge::GRAPH_FAILED);

    tokenNum_ = hiddenShape_->GetDim(0);
    cols_ = hiddenShape_->GetDim(1);
    OP_CHECK_IF(topkIdsShape_->GetDim(0) != tokenNum_,
                OP_LOGE(context_, "topk_ids dim[0] %ld not equal hidden dim[0] %ld.", topkIdsShape_->GetDim(0),
                        tokenNum_),
                return ge::GRAPH_FAILED);
    k_ = topkIdsShape_->GetDim(1);
    OP_CHECK_IF(k_ <= 0, OP_LOGE(context_, "k is: %ld, but should be greater than 0.", k_),
                return ge::GRAPH_FAILED);
    if (topkWeightsShape_ != nullptr) {
        OP_CHECK_IF(topkWeightsShape_->GetDim(0) != tokenNum_ || topkWeightsShape_->GetDim(1) != k_,
                    OP_LOGE(context_, "topk_weights shape (%ld, %ld) not equal (token_num, k) = (%ld, %ld).",
                            topkWeightsShape_->GetDim(0), topkWeightsShape_->GetDim(1), tokenNum_, k_),
                    return ge::GRAPH_FAILED);
    }

    activeNum_ = tokenNum_ * k_;
    OP_CHECK_IF(activeNum_ > MAX_ACTIVE_NUM,
                OP_LOGE(context_, "active_num is: %ld, but should not be greater than %ld (small-batch path).",
                        activeNum_, MAX_ACTIVE_NUM),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertNum_ <= 0, OP_LOGE(context_, "expert_num is: %ld, but should be greater than 0.", expertNum_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cols_ <= 0, OP_LOGE(context_, "cols is: %ld, but should be greater than 0.", cols_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeRoutingInitFusedTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    auto hiddenShapePtr = context_->GetInputShape(HIDDEN_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hiddenShapePtr);
    hiddenShape_ = &hiddenShapePtr->GetStorageShape();

    auto topkIdsShapePtr = context_->GetInputShape(TOPK_IDS_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, topkIdsShapePtr);
    topkIdsShape_ = &topkIdsShapePtr->GetStorageShape();

    auto topkWeightsShapePtr = context_->GetInputShape(TOPK_WEIGHTS_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, topkWeightsShapePtr);
    topkWeightsShape_ = &topkWeightsShapePtr->GetStorageShape();

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t *expertNumPtr = attrs->GetAttrPointer<int64_t>(EXPERT_NUM_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertNumPtr);
    expertNum_ = *expertNumPtr;

    auto hiddenDesc = context_->GetInputDesc(HIDDEN_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hiddenDesc);
    auto hiddenDtype = hiddenDesc->GetDataType();
    OP_CHECK_IF((hiddenDtype != ge::DataType::DT_BF16),
                OP_LOGE(context_, "hidden dtype %s error, only supports bfloat16 (Xing4.0 decode path).",
                     ge::TypeUtils::DataTypeToSerialString(hiddenDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto sortedHiddenDesc = context_->GetOutputDesc(SORTED_HIDDEN_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sortedHiddenDesc);
    OP_CHECK_IF((sortedHiddenDesc->GetDataType() != hiddenDtype),
                OP_LOGE(context_, "sorted_hidden dtype must be the same with hidden dtype."),
                return ge::GRAPH_FAILED);
    auto weightsOutDesc = context_->GetOutputDesc(TOPK_WEIGHTS_OUT_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightsOutDesc);
    OP_CHECK_IF((weightsOutDesc->GetDataType() != hiddenDtype),
                OP_LOGE(context_, "topk_weights_out dtype must be the same with hidden dtype."),
                return ge::GRAPH_FAILED);
    auto topkWeightsDesc = context_->GetInputDesc(TOPK_WEIGHTS_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, topkWeightsDesc);
    OP_CHECK_IF((topkWeightsDesc->GetDataType() != hiddenDtype),
                OP_LOGE(context_, "topk_weights dtype must be the same with hidden dtype (pass-through)."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeRoutingInitFusedTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    OP_LOGI(context_, "GetPlatformInfo: blockDim = %ld, ubSize = %lu", aicoreParams_.blockDim, aicoreParams_.ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeRoutingInitFusedTilingBase::DoOpTiling()
{
    auto ret = CheckShapeAndAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    tilingData_.set_tokenNum(tokenNum_);
    tilingData_.set_k(k_);
    tilingData_.set_expertNum(expertNum_);
    tilingData_.set_cols(cols_);
    tilingData_.set_activeNum(activeNum_);

    // gather 阶段按 dst 行切核：核数不超过 activeNum（S*k 小时单核也够）
    int64_t gatherCores = std::min<int64_t>(activeNum_, static_cast<int64_t>(aicoreParams_.blockDim));
    int64_t perCoreRows = CEIL_DIV(activeNum_, gatherCores);
    int64_t needCoreNum = CEIL_DIV(activeNum_, perCoreRows);
    int64_t lastCoreRows = activeNum_ % perCoreRows == 0 ? perCoreRows : activeNum_ % perCoreRows;
    tilingData_.set_needCoreNum(needCoreNum);
    tilingData_.set_perCoreRows(perCoreRows);
    tilingData_.set_lastCoreRows(lastCoreRows);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeRoutingInitFusedTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeRoutingInitFusedTilingBase::GetWorkspaceSize()
{
    // kernel 跨核零依赖（每核冗余计算派发映射，结果留 UB），不使用 workspace；
    // 仅保留一个 32B 对齐块，保证入口 workspace 指针非空（入口判空守卫）
    workspaceSize_ = 32;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeRoutingInitFusedTilingBase::PostTiling()
{
    context_->SetBlockDim(tilingData_.get_needCoreNum());
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                              context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeRoutingInitFusedTilingBase::GetTilingKey() const
{
    return 0;
}

void MoeRoutingInitFusedTilingBase::Reset()
{
    opName_ = nullptr;
    return;
}

// 模板 ID 须全局唯一：2000 已被 MoeGatingTopK 占用（同一 liboptiling.so 内冲突 →
// 容器侧 561002 tiling 失败，2026-11-07 实测），本算子用 2001
REGISTER_OPS_TILING_TEMPLATE(MoeRoutingInitFused, MoeRoutingInitFusedTilingBase, 2001);
} // namespace optiling
