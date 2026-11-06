/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_post_tiling.cpp
 * \brief MhcPost tiling 实现（移植自 hermes arch22，核数平台查询、无字面量）
 */

#include <cmath>
#include <algorithm>
#include "mhc_post_tiling.h"
#include "register/op_def_registry.h"
#include "log/log.h"

namespace optiling {

// 对齐常量（元素数）
constexpr uint32_t BF16_FP16_ALIGN_SIZE = 16;  // 16 元素 = 32B（bf16/fp16）
constexpr uint32_t ALIGN_SIZE_512B = 256;      // 256 元素 = 512B（bf16/fp16）

constexpr uint32_t SIZE_OF_16BIT = 2;
constexpr uint32_t SIZE_OF_32BIT = 4;

// 双缓冲配置
constexpr uint32_t DOUBLE_BUFFER_DEPTH = 2;
constexpr uint32_t SINGLE_BUFFER_DEPTH = 1;   // TBuf 单缓冲

// UB 预算：TQue(hOut/x/output) 每流 DB*2B + TBuf(F32) 每流 4B
constexpr uint32_t NUM_DATA_STREAMS = 3;
constexpr uint64_t NON_X_BUFFER_COUNT = 2;    // hOut + output（x 按 n 计）

// 输入索引
constexpr int64_t X_INPUT_INDEX = 0;      // x (B,S,n,D) 或 (T,n,D)
constexpr int64_t H_RES_INPUT_INDEX = 1;  // h_res (…,n,n) FP32
constexpr int64_t H_OUT_INPUT_INDEX = 2;  // h_out (…,D)
constexpr int64_t H_POST_INPUT_INDEX = 3; // h_post (…,n) FP32
constexpr int64_t OUTPUT_INDEX = 0;       // y (…,n,D)

constexpr int64_t N_SIZE_4 = 4;
constexpr int64_t D_MAX_VALUE = 100000;
constexpr int64_t D_ALIGN_SIZE = 128;

static int64_t CeilAlign(int64_t v, int64_t align)
{
    return (v + align - 1) / align * align;
}

static int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

ge::graphStatus MhcPostTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "MhcPost: platform info is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    // 纯 AIV 算子：核数按平台查询（910B=40/910C=48 自适应，禁止字面量）
    aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivCoreNum_ == 0, OP_LOGE(context_, "MhcPost: aiv core num query is 0"),
                return ge::GRAPH_FAILED);
    uint64_t ubSizePlatform = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    ubSize_ = ubSizePlatform;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTiling::CheckParam()
{
    // x: (B,S,n,D) 4D 或 (T,n,D) 3D；n 固定 4；d 128 对齐
    auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
    OP_CHECK_IF(xShapePtr == nullptr, OP_LOGE(context_, "MhcPost: x shape is null"), return ge::GRAPH_FAILED);
    auto xShape = xShapePtr->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != 4 && xShape.GetDimNum() != 3,
                OP_LOGE(context_, "MhcPost: x dim num must be 3 (TND) or 4 (BSND), actual %u", xShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    n_ = xShape.GetDim(xShape.GetDimNum() - 2);
    d_ = xShape.GetDim(xShape.GetDimNum() - 1);
    s_ = (xShape.GetDimNum() == 4) ? xShape.GetDim(0) * xShape.GetDim(1) : xShape.GetDim(0);

    OP_CHECK_IF(n_ != N_SIZE_4, OP_LOGE(context_, "MhcPost: n must be 4, actual %ld", n_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(d_ <= 0 || d_ >= D_MAX_VALUE || d_ % D_ALIGN_SIZE != 0,
                OP_LOGE(context_, "MhcPost: d must be > 0, < 100000 and 128-aligned, actual %ld", d_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(s_ <= 0, OP_LOGE(context_, "MhcPost: total items must be positive, actual %ld", s_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(n_ * d_ > INT32_MAX, OP_LOGE(context_, "MhcPost: n*d overflow, %ld", n_ * d_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTiling::DoOpTiling()
{
    OP_CHECK_IF(CheckParam() == ge::GRAPH_FAILED, OP_LOGE(context_, "MhcPost: param check failed"),
                return ge::GRAPH_FAILED);

    // 核划分：item = (s_total × dOuter)，按核均摊并处理尾核
    uint32_t coreNum = static_cast<uint32_t>(aivCoreNum_);
    uint32_t halfCoreNum = coreNum / 2;
    const uint32_t ubSize = static_cast<uint32_t>(ubSize_);

    // 每 tileD 元素的 UB 字节数：3 流 × (DB×2B + 1×4B)
    uint32_t bytesPerTileD = NUM_DATA_STREAMS * (DOUBLE_BUFFER_DEPTH * SIZE_OF_16BIT + SINGLE_BUFFER_DEPTH * SIZE_OF_32BIT);
    uint32_t maxTileD = ubSize / bytesPerTileD;
    dOuter_ = 1;
    dInner_ = d_;
    dTail_ = d_;
    // 与 hermes 一致的 d 切分策略：item 数不足半核或块超 UB 上限时加倍 dOuter
    while (s_ * dOuter_ <= halfCoreNum || dInner_ >= maxTileD) {
        if (dInner_ <= ALIGN_SIZE_512B) {
            break;
        }
        dOuter_ = dOuter_ * 2;
        dInner_ = d_ / dOuter_;
    }
    dInner_ = CeilAlign(dInner_, static_cast<int64_t>(BF16_FP16_ALIGN_SIZE));
    dOuter_ = CeilDiv(d_, dInner_);
    dTail_ = d_ - (dOuter_ - 1) * dInner_;
    dTailAlign_ = CeilAlign(dTail_, static_cast<int64_t>(BF16_FP16_ALIGN_SIZE));

    int64_t totalCount = s_ * dOuter_;
    usedCoreNum_ = (totalCount < coreNum) ? totalCount : coreNum;
    normalCoreProcessNum_ = CeilDiv(totalCount, usedCoreNum_);
    usedCoreNum_ = CeilDiv(totalCount, normalCoreProcessNum_);
    tailCoreProcessNum_ = totalCount - (usedCoreNum_ - 1) * normalCoreProcessNum_;

    // x 整块常驻 UB 的判据（对应 hermes USE_PERMANENT_X 模板分支）
    uint64_t fullyBytesPerTileD = (static_cast<uint64_t>(n_) + NON_X_BUFFER_COUNT) *
        (DOUBLE_BUFFER_DEPTH * SIZE_OF_16BIT + SIZE_OF_32BIT);
    usePermanentX_ = (fullyBytesPerTileD * static_cast<uint64_t>(dInner_) <= ubSize_) ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTiling::PostTiling()
{
    tilingData_.set_n(n_);
    tilingData_.set_d(d_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_normalCoreProcessNum(normalCoreProcessNum_);
    tilingData_.set_tailCoreProcessNum(tailCoreProcessNum_);
    tilingData_.set_dInner(dInner_);
    tilingData_.set_dOuter(dOuter_);
    tilingData_.set_dTail(dTail_);
    tilingData_.set_dTailAlign(dTailAlign_);
    tilingData_.set_usePermanentX(usePermanentX_);

    context_->SetTilingKey(0);
    context_->SetBlockDim(usedCoreNum_);
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr, OP_LOGE(context_, "MhcPost: workspace ptr is null"),
                return ge::GRAPH_FAILED);
    currentWorkspace[0] = 0; // kernel 不使用 workspace

    // 与 MhcPreClampSinkhorn 相同的写回机制：本地结构体 SaveToBuffer 进 raw tiling data
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForMhcPost(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForMhcPost(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MhcPost", "Tiling context is null"), return ge::GRAPH_FAILED);
    MhcPostTiling mhcPostTiling(context);
    OP_CHECK_IF(mhcPostTiling.GetPlatformInfo() == ge::GRAPH_FAILED,
                OP_LOGE("MhcPost", "GetPlatformInfo failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(mhcPostTiling.DoOpTiling() == ge::GRAPH_FAILED,
                OP_LOGE("MhcPost", "DoOpTiling failed"), return ge::GRAPH_FAILED);
    return mhcPostTiling.PostTiling();
}

IMPL_OP_OPTILING(MhcPost)
    .Tiling(TilingForMhcPost)
    .TilingParse<MhcPostCompileInfo>(TilingPrepareForMhcPost);

} // namespace optiling
