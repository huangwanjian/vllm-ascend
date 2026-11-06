/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mmhc_pre_clamp_sinkhorn_sinkhorn_tiling.cpp
 * \brief MhcPreClampSinkhorn tiling implementation
 */

#include <cstdlib>
#include <sstream>
#include <string>
#include "mhc_pre_clamp_sinkhorn_tiling.h"

using namespace ge;
namespace optiling {
namespace {
constexpr uint64_t WORKSPACE_SIZE = 32;
int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
int64_t DownAlign(int64_t x, int64_t y)
{
    if (y == 0) {
        return x;
    }
    return (x / y) * y;
}
int64_t RoundUp(int64_t x, int64_t y)
{
    return CeilDiv(x, y) * y;
}

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t UB_RESEVED_SIZE = 8192;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint64_t M_L1_MAX_SIZE = 256;
constexpr uint64_t K_MULIT_CORE_SPLIT_BASE_SIZE = 256;
constexpr uint64_t A_L1_SIZE = 128 * 256;
constexpr uint64_t K_L1_MAX_SIZE = 1024;
constexpr int64_t HC_MULT_ATTR_IDX = 0;
constexpr int64_t ITER_TIMES_ATTR_IDX = 1;
constexpr int64_t HC_EPS_ATTR_IDX = 2;
constexpr int64_t NORM_EPS_ATTR_IDX = 3;
constexpr int64_t NEED_BACKWARD_ATTR_IDX = 4;
constexpr int64_t MIN_VAL_ATTR_IDX = 5;
constexpr int64_t MAX_VAL_ATTR_IDX = 6;
constexpr int64_t DEFAULT_ITER_TIMES = 20;
constexpr int64_t BS_SPLIT_THRESHOLD = 128;
constexpr int64_t MAX_BS_PER_LOOP = 32;
constexpr int64_t MAX_REDUCE_SIZE = 256;
// K-split routing threshold. MUST be <= BS_SPLIT_THRESHOLD so that the routed
// k-split kernel runs with bsLoop==1 (no bs-batch-split complication).
constexpr int64_t K_SPLIT_ROUTE_BS_MAX = BS_SPLIT_THRESHOLD;
// Env override for A/B validation: MHC_PRE_KSPLIT_MODE = auto(default) | on | off.
// Read once per tiling call (tiling runs at graph build / retile, not per step).
constexpr int K_SPLIT_MODE_AUTO = 0;
constexpr int K_SPLIT_MODE_ON = 1;
constexpr int K_SPLIT_MODE_OFF = 2;
int GetKSplitModeOverride()
{
    const char *mode = getenv("MHC_PRE_KSPLIT_MODE");
    if (mode == nullptr) {
        return K_SPLIT_MODE_AUTO;
    }
    std::string modeStr(mode);
    if (modeStr == "on" || modeStr == "1") {
        return K_SPLIT_MODE_ON;
    }
    if (modeStr == "off" || modeStr == "0") {
        return K_SPLIT_MODE_OFF;
    }
    return K_SPLIT_MODE_AUTO;
}
// Experiment knob for K-split granularity: overrides the K-block count (kDimNum, i.e. how
// many cube cores split the K axis). 0/unset = default (all spare cube cores ~19).
// Setting it implies forced k-split route under mode=auto, so even cubeBlockDimK==1
// (single-block) variants are reachable for A/B. Inert under MHC_PRE_KSPLIT_MODE=off.
// NOTE: cvLoopKSize stays 1024 (l1b 128KB segment cap), so fewer blocks => more
// sequential k-rounds per core, each re-loading its phi slice into L1.
uint64_t GetKSplitBlocksOverride()
{
    const char *blocks = getenv("MHC_PRE_KSPLIT_BLOCKS");
    if (blocks == nullptr) {
        return 0;
    }
    int64_t v = atoll(blocks);
    if (v <= 0) {
        return 0;
    }
    return static_cast<uint64_t>(v);
}

// ---- key=2 tiny route (decode-only lean kernel) ----
// Route when the flattened token count is small (decode regime, MTP envelope
// max_num_seqs=8 x (1+3) = 32 tokens) and no grad. The tiny kernel launches only
// TINY_K_SPLIT_AIC cube cores (+2x vector) instead of the full 20/40, keeps the
// K-split so the 1.376MB phi weight stream stays spread across cores, and runs a
// single K round per core (no multi-round event protocol).
constexpr int64_t TINY_ROUTE_BS_MAX = 32;
// 8 blocks x 1792 = 14336 exactly; per-core K slice fits the tiny kernel's single
// 256KB L1 slots (B NZ = 32*1792*4 = 229KB, A NZ = align16(M)*1792*4 <= 229KB @ M<=32).
constexpr uint64_t TINY_K_SPLIT_AIC = 8;
// KERNEL_TYPE_MIX_AIC_1_2 fixed 1:2 AIC:AIV ratio (kernel-type property, not a
// platform core-count literal): blockDim(TINY_K_SPLIT_AIC) launches 8 AIC + 16 AIV.
constexpr uint64_t TINY_CV_RATIO = 2;
// tiny kernel 内存预算 gate 常量。L1：kernel 侧 l1a/l1b 尺寸由 tiling 派生
// （align16(mL1Size/hcMix)*cvLoopKSize*4B，见 tiny_core.h Init），gate 按
// "派生总需求 <= 平台查询 L1"判定（平台 L1 查询含保留头：910B3 实测 524032，
// 不可按满容量比较）；L0：kernel 常量 L0_TINY_BUF_SIZE 与 TINY_L0_BUF_BYTES
// 同值，按 64KB 校验。不满足 -> 不路由 tiny，回退 key=1。
// DAV_2201（910B 与 910_93 同架构）恒过。
constexpr uint64_t TINY_L0_BUF_BYTES = 64 * 1024;
// Env override for A/B validation: MHC_PRE_TINYBS = auto(default) | on | off.
constexpr int TINY_MODE_AUTO = 0;
constexpr int TINY_MODE_ON = 1;
constexpr int TINY_MODE_OFF = 2;
int GetTinyBsModeOverride()
{
    const char *mode = getenv("MHC_PRE_TINYBS");
    if (mode == nullptr) {
        return TINY_MODE_AUTO;
    }
    std::string modeStr(mode);
    if (modeStr == "on" || modeStr == "1") {
        return TINY_MODE_ON;
    }
    if (modeStr == "off" || modeStr == "0") {
        return TINY_MODE_OFF;
    }
    return TINY_MODE_AUTO;
}

// Stage2 row blocking: how many 4x4 Sinkhorn matrices one vector pass covers.
// The Sinkhorn body is dominated by per-iteration scalar/issue overhead when
// rowFactor==1 (large d exhausts UB and leaves no room for more rows), so a
// larger block amortizes ~9 tiny vector ops per iteration across rows. The d
// axis is split instead; total ProcessY traffic is unchanged.
// Kernel-side broadcast fixes landed (grouped-coeff paths in base.h); default 8
// verified on 910B3/CANN9.1.0 (msprof kernel avg): prefill S2048 train 637→439us
// (-31%), infer 482→385us (-20%), mid S256 train 187→151us (-19%), decode S4
// infer flat 32.6→32.2us (clamped by rowOfFormerBlock). rf=16 regresses (UB
// budget squeezes dFactor) — keep default at the 4~8 sweet spot.
// Env override for A/B: MHC_PRE_STAGE2_ROW_FACTOR = 1 | 4 | 8 | 16 | 32.
constexpr int64_t STAGE2_ROW_FACTOR_DEFAULT = 8;
int64_t GetStage2RowFactorMin()
{
    const char *rows = getenv("MHC_PRE_STAGE2_ROW_FACTOR");
    if (rows == nullptr) {
        return STAGE2_ROW_FACTOR_DEFAULT;
    }
    int64_t v = atoll(rows);
    return v > 0 ? v : STAGE2_ROW_FACTOR_DEFAULT;
}
}

ge::graphStatus MhcPreClampSinkhornTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<MhcPreClampSinkhornCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
                      return ge::GRAPH_FAILED);
        aivCoreNum_ = compileInfoPtr->coreNum;
        // compileInfo 仅携带 AIV 核数；MIX_AIC_1_2 固定 1:2，按此推导 AIC 数
        // （此前该分支 aicCoreNum_ 恒 0，kDimNum=aicCoreNum_/mDimNum 存在除零隐患）
        aicCoreNum_ = aivCoreNum_ / 2;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        aicCoreNum_ = ascendcPlatform.GetCoreNumAic();
        // 不得无条件覆盖查询值（910B4 时代的 40/20 hack）：910C（Ascend910_9382，
        // 24 Cube/48 Vector）等形态需按真实核数走 tiling——kDimNum（切K）、
        // Part2 行分布（secondUsedCoreNum）、blockDim 全部由此派生。
        // 仅在查询异常（返回 0）时兜底 910B4 形态，防除零与空跑。
        if (aivCoreNum_ == 0 || aicCoreNum_ == 0) {
            OP_LOGW(context_, "platform core num query invalid (aiv=%lu, aic=%lu), fallback to 910B 40/20",
                    static_cast<unsigned long>(aivCoreNum_), static_cast<unsigned long>(aicCoreNum_));
            aivCoreNum_ = 40;
            aicCoreNum_ = 20;
        }

        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        // tiny 路由的内存预算 gate 数据源（910B/910_93 同 DAV_2201：L1 512KB、
        // L0A/L0B 64KB、L0C 128KB）。查询为 0/异常时 gate 失败 -> 不路由 tiny。
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aSize_);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreClampSinkhornTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto hcMultAttr = attrs->GetAttrPointer<int64_t>(HC_MULT_ATTR_IDX);
    hcMult_ = hcMultAttr == nullptr ? 4 : *hcMultAttr;

    auto iterTimesAttr = attrs->GetAttrPointer<int64_t>(ITER_TIMES_ATTR_IDX);
    iterTimes_ = iterTimesAttr == nullptr ? DEFAULT_ITER_TIMES : *iterTimesAttr;

    auto hcEpsAttr = attrs->GetAttrPointer<float>(HC_EPS_ATTR_IDX);
    hcEps_ = hcEpsAttr == nullptr ? 1e-6f : *hcEpsAttr;

    auto normEpsAttr = attrs->GetAttrPointer<float>(NORM_EPS_ATTR_IDX);
    normEps_ = normEpsAttr == nullptr ? 1e-6f : *normEpsAttr;

    auto needBackwardAttr = attrs->GetAttrPointer<bool>(NEED_BACKWARD_ATTR_IDX);
    needGrad_ = needBackwardAttr == nullptr ? true : *needBackwardAttr;

    auto minValAttr = attrs->GetAttrPointer<float>(MIN_VAL_ATTR_IDX);
    clamp_min_ = minValAttr == nullptr ? 0.0f : *minValAttr;

    auto maxValAttr = attrs->GetAttrPointer<float>(MAX_VAL_ATTR_IDX);
    clamp_max_ = maxValAttr == nullptr ? 0.0f : *maxValAttr;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreClampSinkhornTiling::GetShapeAttrsInfoInner()
{
    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    if (xDimNum == 3) {
        bs_ = xShape->GetStorageShape().GetDim(0);
        hcMult_ = xShape->GetStorageShape().GetDim(1);
        d_ = xShape->GetStorageShape().GetDim(2);
    } else if (xDimNum == 4) {
        int64_t b = xShape->GetStorageShape().GetDim(0);
        int64_t s = xShape->GetStorageShape().GetDim(1);
        bs_ = b * s;
        hcMult_ = xShape->GetStorageShape().GetDim(2);
        d_ = xShape->GetStorageShape().GetDim(3);
    }

    auto shapePhi = context_->GetInputShape(1);
    hcMix_ = shapePhi->GetStorageShape().GetDim(0);
    OP_CHECK_IF(shapePhi->GetStorageShape().GetDim(1) != d_ * hcMult_,
                    OP_LOGE(context_->GetNodeName(),
                             "Phi dim 1 should be equal with d_ * hcMult_  %ld, but is %ld", d_ * hcMult_, shapePhi->GetStorageShape().GetDim(1)),
                    return ge::GRAPH_FAILED);

    auto shapeAlpha = context_->GetInputShape(2);
    int64_t scaleFirstDim = shapeAlpha->GetStorageShape().GetDim(0);
    OP_CHECK_IF(scaleFirstDim != 3,
                    OP_LOGE(context_->GetNodeName(),
                             "alpha size should be equal with 3, but is %ld", scaleFirstDim),
                    return ge::GRAPH_FAILED);

    auto shapeBias = context_->GetInputShape(3);
    int64_t baseFirstDim = shapeBias->GetStorageShape().GetDim(0);
    OP_CHECK_IF(baseFirstDim != hcMix_,
                    OP_LOGE(context_->GetNodeName(),
                             "bias size should be equal with mixhc, but is %ld", baseFirstDim),
                    return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS,
                  OP_LOGE(context_->GetNodeName(), "get attr failed."),
                  return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreClampSinkhornTiling::CalcMKSplitCoreMembasePart2Tiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    usedAivCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    rowOfTailBlock_ = bs_ - (usedAivCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t kBlockNum = tilingData_.get_cubeBlockDimK();

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));

    // Exact per-rowFactor UB footprint of Part2, mirroring kernel InitBuffer calls.
    // m-split Part2 works on the already K-reduced hcBeforeNorm, so its mix queues are
    // [rows][...]; k-split Part2 keeps [kcore][rows][...] and is kBlockNum times larger.
    auto FixedUbBytes = [&](int64_t rf) -> int64_t {
        int64_t rowBrcb = RoundUp(rf, 8) * BLOCK_SIZE;
        if (useKSplit_) {
            // mix01 + mix2 + squareSum = 16+32+16 floats per kcore, x2 buffers, x4 bytes.
            return 512 * kBlockNum * rf + 1120 * rf + rowBrcb + 736;
        }
        return 1888 * rf + rowBrcb + 736;
    };

    // Pick the largest feasible row block. Every row is an independent 4x4 Sinkhorn,
    // so a larger block amortizes the ~9 tiny vector ops per iteration; d is split
    // instead and the total ProcessY traffic is unchanged.
    constexpr int64_t MIN_DFACTOR = 64;
    int64_t maxRowFactor = std::min(rowOfFormerBlock_, GetStage2RowFactorMin());
    int64_t chosenRf = 1;
    int64_t chosenDf = std::min<int64_t>(d_, MIN_DFACTOR);
    for (int64_t rf = maxRowFactor; rf >= 1; --rf) {
        int64_t avail = ubSize_ - FixedUbBytes(rf) - UB_RESEVED_SIZE;
        if (avail <= 0) {
            continue;
        }
        // Per d element: x/y bf16 double buffered + fp32 cast x/y = 16+4+16+4 bytes.
        int64_t maxD = DownAlign(avail / (40 * rf), 32);
        if (maxD < 16) {
            continue;
        }
        int64_t df = std::min(d_, maxD);
        if (df < MIN_DFACTOR && d_ >= MIN_DFACTOR) {
            continue;
        }
        chosenRf = rf;
        chosenDf = df;
        break;
    }
    rowFactor_ = chosenRf;
    dFactor_ = chosenDf;
    dLoop_ = CeilDiv(d_, dFactor_);
    tailDFactor_ = d_ % dFactor_ == 0 ? dFactor_ : d_ % dFactor_;
    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;
    
    tilingData_.set_bs(bs_);
    tilingData_.set_hcMix(hcMix_);
    tilingData_.set_hcMult(hcMult_);
    tilingData_.set_d(d_);
    tilingData_.set_hcMultAlign(hcMultAlign_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_stage2RowFactor(rowFactor_);
    tilingData_.set_secondUsedCoreNum(usedAivCoreNums_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_clamp_min(clamp_min_);
    tilingData_.set_clamp_max(clamp_max_);
    tilingData_.set_needGrad(needGrad_ ? 1 : 0);
    return ge::GRAPH_SUCCESS;
}

// 预留功能，切K情况尾核单独计算
ge::graphStatus MhcPreClampSinkhornTiling::CalcBsSplit()
{
    tailBs_ = bs_;
    curBsSplit_ = bs_;
    if (bs_ > BS_SPLIT_THRESHOLD) {
        bsLoop_ = CeilDiv(bs_, BS_SPLIT_THRESHOLD);
        tailBs_ = bs_ % BS_SPLIT_THRESHOLD;
        if (tailBs_ == 0) {
            tailBs_ = BS_SPLIT_THRESHOLD;
        }
        curBsSplit_ = BS_SPLIT_THRESHOLD;
    }
    tilingData_.set_bsSplitThreshold(BS_SPLIT_THRESHOLD);
    tilingData_.set_bsLoop(bsLoop_);
    tilingData_.set_tailBs(tailBs_);
    tilingData_.set_curBsSplit(curBsSplit_);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreClampSinkhornTiling::CalcTailBsTiling()
{
    if (tailBs_ == BS_SPLIT_THRESHOLD || tailBs_ == bs_) {
        tilingData_.set_tailBsRowOfFormerBlock(tilingData_.get_rowOfFormerBlock());
        tilingData_.set_tailBsRowOfTailBlock(tilingData_.get_rowOfTailBlock());
        tilingData_.set_tailBsRowLoopOfFormerBlock(tilingData_.get_rowLoopOfFormerBlock());
        tilingData_.set_tailBsRowLoopOfTailBlock(tilingData_.get_rowLoopOfTailBlock());
        tilingData_.set_tailBsUsedCoreNum(tilingData_.get_secondUsedCoreNum());
        tilingData_.set_tailBsRowFactor(tilingData_.get_stage2RowFactor());
        tilingData_.set_tailBsTailRowFactorOfFormerBlock(tilingData_.get_tailRowFactorOfFormerBlock());
        tilingData_.set_tailBsTailRowFactorOfTailBlock(tilingData_.get_tailRowFactorOfTailBlock());
        tilingData_.set_tailBsML1Size(tilingData_.get_mL1Size());
        tilingData_.set_tailBsKL1Size(tilingData_.get_kL1Size());
        tilingData_.set_tailBsMultCoreSplitMSize(tilingData_.get_multCoreSplitMSize());
        tilingData_.set_tailBsCubeBlockDimM(tilingData_.get_cubeBlockDimM());
        return ge::GRAPH_SUCCESS;
    }
    
    uint64_t tailMDimNum = std::min(aicCoreNum_, static_cast<uint64_t>(CeilDiv(tailBs_, M_L1_MAX_SIZE)));
    uint64_t tailSingleCoreM = RoundUp(CeilDiv(tailBs_, tailMDimNum), AscendC::BLOCK_CUBE);
    
    tailBsCubeBlockDimM_ = static_cast<int64_t>(tailMDimNum);
    tailBsMultCoreSplitMSize_ = static_cast<int64_t>(tailSingleCoreM);
    tailBsML1Size_ = std::min(M_L1_MAX_SIZE, tailSingleCoreM);
    tailBsKL1Size_ = std::min(A_L1_SIZE / tailSingleCoreM, static_cast<uint64_t>(K_L1_MAX_SIZE)) / 128 * 128;
    
    tailBsRowOfFormerBlock_ = CeilDiv(tailBs_, static_cast<int64_t>(aivCoreNum_));
    tailBsUsedCoreNum_ = std::min(CeilDiv(tailBs_, tailBsRowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    tailBsRowOfTailBlock_ = tailBs_ - (tailBsUsedCoreNum_ - 1) * tailBsRowOfFormerBlock_;
    
    int64_t minRowPerCore = 1;
    tailBsRowFactor_ = std::min(tailBsRowOfFormerBlock_, minRowPerCore);
    
    tailBsRowLoopOfFormerBlock_ = CeilDiv(tailBsRowOfFormerBlock_, tailBsRowFactor_);
    tailBsRowLoopOfTailBlock_ = CeilDiv(tailBsRowOfTailBlock_, tailBsRowFactor_);
    tailBsTailRowFactorOfFormerBlock_ = tailBsRowOfFormerBlock_ % tailBsRowFactor_ == 0 ?
                                         tailBsRowFactor_ : tailBsRowOfFormerBlock_ % tailBsRowFactor_;
    tailBsTailRowFactorOfTailBlock_ = tailBsRowOfTailBlock_ % tailBsRowFactor_ == 0 ?
                                       tailBsRowFactor_ : tailBsRowOfTailBlock_ % tailBsRowFactor_;
    
    tilingData_.set_tailBsRowOfFormerBlock(tailBsRowOfFormerBlock_);
    tilingData_.set_tailBsRowOfTailBlock(tailBsRowOfTailBlock_);
    tilingData_.set_tailBsRowLoopOfFormerBlock(tailBsRowLoopOfFormerBlock_);
    tilingData_.set_tailBsRowLoopOfTailBlock(tailBsRowLoopOfTailBlock_);
    tilingData_.set_tailBsUsedCoreNum(tailBsUsedCoreNum_);
    tilingData_.set_tailBsRowFactor(tailBsRowFactor_);
    tilingData_.set_tailBsTailRowFactorOfFormerBlock(tailBsTailRowFactorOfFormerBlock_);
    tilingData_.set_tailBsTailRowFactorOfTailBlock(tailBsTailRowFactorOfTailBlock_);
    tilingData_.set_tailBsML1Size(tailBsML1Size_);
    tilingData_.set_tailBsKL1Size(tailBsKL1Size_);
    tilingData_.set_tailBsMultCoreSplitMSize(tailBsMultCoreSplitMSize_);
    tilingData_.set_tailBsCubeBlockDimM(tailBsCubeBlockDimM_);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreClampSinkhornTiling::CalcOpTiling()
{
    if (CalcBsSplit() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    // key=2 tiny route: decided before any core-count math so the whole tiling chain
    // (mDimNum/kDimNum/splitKSize/Part2 row layout/stage1 fields) derives from the
    // small launch shape. Overrides platform core counts; PostTiling routes to key=2.
    int tinyMode = GetTinyBsModeOverride();
    tinyRoute_ = (tinyMode == TINY_MODE_ON) ? true
               : (tinyMode == TINY_MODE_OFF) ? false
               : (bs_ <= TINY_ROUTE_BS_MAX) && (!needGrad_);
    if (tinyRoute_ && (needGrad_ || bs_ > TINY_ROUTE_BS_MAX)) {
        // safety clamp: tiny kernel is inference-only, bsLoop==1 (bs <= 32 < 128)
        tinyRoute_ = false;
    }
    // ---- tiny 内存预算 gate（平台查询值，不假设芯片规格）----
    // L1 按派生需求比较（与 kernel Init 的 l1a/l1b TBuf 公式同源）：l1Need =
    // (align16(M)+align16(N)) * Kslice * 4B。注意平台 L1 查询含保留头（910B3 实测
    // 524032 = 512KB-256B），不能按"满 2x256KB"比较——否则 2201 全系误判 gate 失败
    // （8ea0e90 曾踩此坑：tiny 被静默关闭）。L0 仍按常量槽 64KB 校验。
    {
        uint64_t tinyBlocks = std::min<uint64_t>(TINY_K_SPLIT_AIC, aicCoreNum_);
        uint64_t tinyKslice = RoundUp(CeilDiv(static_cast<uint64_t>(hcMult_ * d_), tinyBlocks),
                                      K_MULIT_CORE_SPLIT_BASE_SIZE);
        uint64_t aNzRows = RoundUp(static_cast<uint64_t>(bs_), AscendC::BLOCK_CUBE);
        uint64_t bNzRows = RoundUp(static_cast<uint64_t>(hcMix_), AscendC::BLOCK_CUBE);
        uint64_t l1NeedBytes = (aNzRows + bNzRows) * tinyKslice * sizeof(float);
        bool tinyMemOk = (tinyBlocks > 1) && (l1NeedBytes <= l1Size_) &&
                         (l0aSize_ >= TINY_L0_BUF_BYTES) && (l0bSize_ >= TINY_L0_BUF_BYTES) &&
                         (l0cSize_ >= TINY_L0_BUF_BYTES);
        if (tinyRoute_ && !tinyMemOk) {
            if (tinyMode == TINY_MODE_ON) {
                OP_LOGW(context_->GetNodeName(),
                        "MhcPreClampSinkhorn tiny route forced on but memory gate failed "
                        "(l1=%lu need=%lu, l0a=%lu l0b=%lu l0c=%lu), fallback to k-split",
                        static_cast<unsigned long>(l1Size_), static_cast<unsigned long>(l1NeedBytes),
                        static_cast<unsigned long>(l0aSize_), static_cast<unsigned long>(l0bSize_),
                        static_cast<unsigned long>(l0cSize_));
            }
            tinyRoute_ = false;
        }
        if (tinyRoute_) {
            // stash platform counts for the route log before overriding
            platAivCoreNum_ = aivCoreNum_;
            platAicCoreNum_ = aicCoreNum_;
            aicCoreNum_ = tinyBlocks;
            aivCoreNum_ = tinyBlocks * TINY_CV_RATIO;
        }
    }
    uint64_t kSize = hcMult_ * d_;
    tilingData_.set_k(kSize);
    // 切K计算stage1
    uint64_t mDimNum = std::min(aicCoreNum_, static_cast<uint64_t>(CeilDiv(bs_, M_L1_MAX_SIZE)));
    uint64_t singleCoreM = RoundUp(CeilDiv(bs_, mDimNum), AscendC::BLOCK_CUBE);

    uint64_t kDimNum = aicCoreNum_ / mDimNum;
    // Experiment knob: override the K-block count (see GetKSplitBlocksOverride).
    // Inert on the tiny route (tiny fixes its own 8-block layout).
    uint64_t kBlocksOverride = GetKSplitBlocksOverride();
    if (!tinyRoute_ && kBlocksOverride > 0 && kDimNum > 0 && kBlocksOverride < kDimNum) {
        kDimNum = kBlocksOverride;
    }
    uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE);

    tilingData_.set_cubeBlockDimM(mDimNum);
    tilingData_.set_cubeBlockDimK(CeilDiv(kSize, splitKSize));
    tilingData_.set_multCoreSplitMSize(singleCoreM);
    tilingData_.set_mL1Size(std::min(M_L1_MAX_SIZE, singleCoreM));
    tilingData_.set_multCoreSplitKSize(splitKSize);
    tilingData_.set_kL1Size(std::min(A_L1_SIZE / singleCoreM, static_cast<uint64_t>(K_L1_MAX_SIZE)) / 128 * 128);

    if (tinyRoute_) {
        // Tiny kernel runs ONE K round per core (single 256KB L1 slots hold the whole
        // per-core A/B slice; the 1024 default exists because the key=1 kernel's l1b
        // is a 128KB double-buffered slot). cvLoopKSize == kL1Size == splitKSize keeps
        // Part1's outer loop to a single iteration -> no multi-round event protocol.
        tilingData_.set_cvLoopKSize(splitKSize);
        tilingData_.set_kL1Size(splitKSize);
    } else {
        tilingData_.set_cvLoopKSize(1024);
    }

    tilingData_.set_cubeCoreNum(static_cast<int64_t>(aicCoreNum_));
    int64_t lineByteSize = (sizeof(int16_t) + sizeof(int32_t)) * DOUBLE_BUFFER * tilingData_.get_cvLoopKSize();
    int64_t stage1MFactorValue = ubSize_ / lineByteSize;
    tilingData_.set_stage1MFactor(stage1MFactorValue);

    // K-split routing decision:
    //   auto:  k-split (key=1) only for small bs decode (bsLoop==1) + inference + kDimNum>1;
    //          everything else keeps m-split (key=0, current behavior).
    //   env MHC_PRE_KSPLIT_MODE=on|off forces the route for A/B validation
    //   (forced-on is still clamped off for needGrad or bs > BS_SPLIT_THRESHOLD).
    // The tiny route (key=2) has its own env (MHC_PRE_TINYBS) and always takes the
    // k-split workspace layout, so KSPLIT_MODE must not turn its WS branch off.
    if (tinyRoute_) {
        useKSplit_ = true;
        return CalcMKSplitCoreMembasePart2Tiling();
    }
    bool autoKSplit = (bs_ <= K_SPLIT_ROUTE_BS_MAX) && (!needGrad_) &&
                      (tilingData_.get_cubeBlockDimK() > 1);
    int modeOverride = GetKSplitModeOverride();
    // blocks knob (MHC_PRE_KSPLIT_BLOCKS) forces the k-split route under mode=auto so
    // that single/few-block granularity variants are reachable; mode=off still wins.
    bool blocksOverride = GetKSplitBlocksOverride() > 0;
    useKSplit_ = (modeOverride == K_SPLIT_MODE_ON) ? true
               : (modeOverride == K_SPLIT_MODE_OFF) ? false
               : (blocksOverride || autoKSplit);
    if (useKSplit_ && (needGrad_ || bs_ > BS_SPLIT_THRESHOLD)) {
        // safety clamp: k-split kernel path is validated for bsLoop==1 inference only
        useKSplit_ = false;
    }
    return CalcMKSplitCoreMembasePart2Tiling();
}

ge::graphStatus MhcPreClampSinkhornTiling::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    // 预留参数计算（切K）与stage2计算
    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    // 预留参数计算（切K）
    if (CalcTailBsTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    // 目前stage1计算在用分支
    if (CalcStage1Tiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static void PrintTilingData(const gert::TilingContext* context, MhcPreClampSinkhornTilingData& tiling)
{
    OP_LOGD(context->GetNodeName(), ">>>>>>>>>> Start to print MhcPreClampSinkhorn tiling data <<<<<<<<<<");
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: bs is %ld.", tiling.get_bs());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: hcMix is %ld.", tiling.get_hcMix());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: hcMult is %ld.", tiling.get_hcMult());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: d is %ld.", tiling.get_d());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: hcMultAlign is %ld.", tiling.get_hcMultAlign());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: rowOfFormerBlock is %ld.", tiling.get_rowOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: rowOfTailBlock is %ld.", tiling.get_rowOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: rowLoopOfFormerBlock is %ld.", tiling.get_rowLoopOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: rowLoopOfTailBlock is %ld.", tiling.get_rowLoopOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: rowFactor is %ld.", tiling.get_rowFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailRowFactorOfFormerBlock is %ld.", tiling.get_tailRowFactorOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailRowFactorOfTailBlock is %ld.", tiling.get_tailRowFactorOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: dLoop is %ld.", tiling.get_dLoop());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: dFactor is %ld.", tiling.get_dFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailDFactor is %ld.", tiling.get_tailDFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: iterTimes is %ld.", tiling.get_iterTimes());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: hcEps is %f.", tiling.get_hcEps());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: normEps is %f.", tiling.get_normEps());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: kBlockFactor is %ld.", tiling.get_kBlockFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: kFactor is %ld.", tiling.get_kFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailKFactor is %ld.", tiling.get_tailKFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: kLoop is %ld.", tiling.get_kLoop());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: stage2RowFactor is %ld.", tiling.get_stage2RowFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: k is %ld.", tiling.get_k());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: kLoopOfFormerBlock is %ld.", tiling.get_kLoopOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: kLoopOfTailBlock is %ld.", tiling.get_kLoopOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: stage1KFactor is %ld.", tiling.get_stage1KFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: kFactorOfFormerBlock is %ld.", tiling.get_kFactorOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: kL1Size is %ld.", tiling.get_kL1Size());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: mL1Size is %ld.", tiling.get_mL1Size());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: cubeBlockDimM is %ld.", tiling.get_cubeBlockDimM());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: cubeBlockDimK is %ld.", tiling.get_cubeBlockDimK());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: kUbSize is %ld.", tiling.get_kUbSize());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: cvLoopKSize is %ld.", tiling.get_cvLoopKSize());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: multCoreSplitKSize is %ld.", tiling.get_multCoreSplitKSize());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: multCoreSplitMSize is %ld.", tiling.get_multCoreSplitMSize());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailKSizeOfFormerBlock is %ld.", tiling.get_tailKSizeOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailKSizeOfTailBlock is %ld.", tiling.get_tailKSizeOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: mLoopOfFormerBlock is %ld.", tiling.get_mLoopOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: mLoopOfTailBlock is %ld.", tiling.get_mLoopOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: formerMSize is %ld.", tiling.get_formerMSize());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailMSizeOfFormerBlock is %ld.", tiling.get_tailMSizeOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailMSizeOfTailBlock is %ld.", tiling.get_tailMSizeOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: firstUsedCoreNum is %ld.", tiling.get_firstUsedCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: secondUsedCoreNum is %ld.", tiling.get_secondUsedCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: rowInnerFactor is %ld.", tiling.get_rowInnerFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: cubeCoreNum is %ld.", tiling.get_cubeCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: stage1MFactor is %ld.", tiling.get_stage1MFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: bufferPool0Size is %ld.", tiling.get_bufferPool0Size());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: bufferPool1Size is %ld.", tiling.get_bufferPool1Size());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: mUbSize is %ld.", tiling.get_mUbSize());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: needGrad is %ld.", tiling.get_needGrad());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: bsLoop is %ld.", tiling.get_bsLoop());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailBs is %ld.", tiling.get_tailBs());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: bsSplitThreshold is %ld.", tiling.get_bsSplitThreshold());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: curBsSplit is %ld.", tiling.get_curBsSplit());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailBsRowOfFormerBlock is %ld.", tiling.get_tailBsRowOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailBsRowOfTailBlock is %ld.", tiling.get_tailBsRowOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailBsRowLoopOfFormerBlock is %ld.", tiling.get_tailBsRowLoopOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailBsRowLoopOfTailBlock is %ld.", tiling.get_tailBsRowLoopOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailBsUsedCoreNum is %ld.", tiling.get_tailBsUsedCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailBsRowFactor is %ld.", tiling.get_tailBsRowFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailBsTailRowFactorOfFormerBlock is %ld.", tiling.get_tailBsTailRowFactorOfFormerBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailBsTailRowFactorOfTailBlock is %ld.", tiling.get_tailBsTailRowFactorOfTailBlock());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailBsML1Size is %ld.", tiling.get_tailBsML1Size());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailBsKL1Size is %ld.", tiling.get_tailBsKL1Size());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailBsMultCoreSplitMSize is %ld.", tiling.get_tailBsMultCoreSplitMSize());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: tailBsCubeBlockDimM is %ld.", tiling.get_tailBsCubeBlockDimM());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: stage1VecCoreNum is %ld.", tiling.get_stage1VecCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: stage1CubeCoreNum is %ld.", tiling.get_stage1CubeCoreNum());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: stage1BsPerVecCore is %ld.", tiling.get_stage1BsPerVecCore());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: stage1TailBsPerVecCore is %ld.", tiling.get_stage1TailBsPerVecCore());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: stage1BsLoop is %ld.", tiling.get_stage1BsLoop());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: stage1BsFactor is %ld.", tiling.get_stage1BsFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: stage1NcLoop is %ld.", tiling.get_stage1NcLoop());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: stage1NcFactor is %ld.", tiling.get_stage1NcFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: stage1TailNcFactor is %ld.", tiling.get_stage1TailNcFactor());
    OP_LOGD(context->GetNodeName(), "MhcPreClampSinkhorn_tiling: stage1XCastWsSize is %ld.", tiling.get_stage1XCastWsSize());
}

ge::graphStatus MhcPreClampSinkhornTiling::CalcStage1Tiling()
{
    // 1. 计算核心数量
    int64_t vecCoreNum = static_cast<int64_t>(aivCoreNum_);
    int64_t cubeCoreNum = static_cast<int64_t>(aicCoreNum_);
    
    tilingData_.set_stage1VecCoreNum(vecCoreNum);
    tilingData_.set_stage1CubeCoreNum(cubeCoreNum);
    
    // 2. 计算 BS 分核
    int64_t bsPerVecCore = CeilDiv(bs_, vecCoreNum);
    int64_t tailBsPerVecCore = bs_ - (vecCoreNum - 1) * bsPerVecCore;
    if (tailBsPerVecCore <= 0) {
        tailBsPerVecCore = bsPerVecCore;
    }
    
    tilingData_.set_stage1BsPerVecCore(bsPerVecCore);
    tilingData_.set_stage1TailBsPerVecCore(tailBsPerVecCore);
    
    // 3. 计算 BS 循环（每轮最多 32 个 BS）
    int64_t bsLoop = CeilDiv(bsPerVecCore, MAX_BS_PER_LOOP);
    int64_t bsFactor = std::min(bs_, MAX_BS_PER_LOOP);  // L1a: right-size (bs=3->3) to free UB for larger nc chunks
    
    tilingData_.set_stage1BsLoop(bsLoop);
    tilingData_.set_stage1BsFactor(bsFactor);
    
    // 4. ReduceSum 切分 — cast-retiling: ncFactor 2048 ONLY when UB fits (small bs).
    //    cast buffers ~ bsFactor*ncFactor*16 bytes. bs=3 (bsFactor=3): 98KB<192KB -> 2048.
    //    large bs (bsFactor=32): 1MB>>UB -> 256 (safe; cast-retiling is small-shape-only).
    int64_t ncSize = hcMult_ * d_;
    int64_t ncFactor = (bsFactor * 2048 * 16 < ubSize_)
                       ? std::min(ncSize, static_cast<int64_t>(2048))
                       : MAX_REDUCE_SIZE;
    int64_t ncLoop = CeilDiv(ncSize, ncFactor);
    int64_t tailNcFactor = ncSize % ncFactor;
    if (tailNcFactor == 0) {
        tailNcFactor = ncFactor;
    }
    
    tilingData_.set_stage1NcLoop(ncLoop);
    tilingData_.set_stage1NcFactor(ncFactor);
    tilingData_.set_stage1TailNcFactor(tailNcFactor);

    int64_t mm1M = bsFactor * 2;
    int64_t mm1N = hcMult_ * hcMult_ + 2 * hcMult_;
    int64_t mm1K = hcMult_ * d_;

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE("TilingForMhcPreClampSinkhorn", "Tiling platformInfo is null"),
               return ge::GRAPH_FAILED);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfo);

    auto featureDataType = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::MatmulApiTiling mm1Tiling(ascendPlatformInfo);

    mm1Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType, true);
    mm1Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetOrgShape(mm1M, mm1N, mm1K);
    mm1Tiling.SetShape(mm1M, mm1N, mm1K);
    mm1Tiling.SetBias(false);
    mm1Tiling.SetBufferSpace(-1, -1, -1);
    if (mm1Tiling.GetTiling(tilingData_.mm1TilingData) == -1) {
        OP_LOGE(context_, "mm1Tiling.GetTiling failed, M=%ld, N=%ld, K=%ld", mm1M, mm1N, mm1K);
        return ge::GRAPH_FAILED;
    }

    // 5. 计算 Workspace 大小
    // X_cast: cubeCoreNum * maxBsPerCore * n * c * sizeof(float)
    int64_t xCastWsSize = bsFactor * vecCoreNum * 2 * ncSize * sizeof(float);
    workspaceSize_ += xCastWsSize;
    tilingData_.set_stage1XCastWsSize(xCastWsSize);
    
    // 6. 计算额外的 workspace 大小（needGrad=false 时使用）
    if (!needGrad_) {
        // invRms: bs * sizeof(float)
        int64_t invRmsWsSize = bs_ * sizeof(float);
        // hcBeforeNorm: bs * hcMix * sizeof(float)
        int64_t hcMix = hcMult_ * hcMult_ + 2 * hcMult_;
        int64_t hcBeforeNormWsSize = bs_ * hcMix * sizeof(float);
        
        workspaceSize_ += invRmsWsSize + hcBeforeNormWsSize;
    }
    
    // The m-split kernel (tilingKey=0) aggregates exactly one Stage1 partial per row in its
    // Stage2, so it requires cubeBlockDimK==1. Only keep the real K-block count computed by
    // CalcOpTiling when routing to the m+k split kernel (tilingKey=1).
    if (!useKSplit_) {
        tilingData_.set_cubeBlockDimK(1);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreClampSinkhornTiling::GetWorkspaceSize()
{
    // Lever D: 砍掉 128MB padding。m-split(tilingKey 0) 真实占用 ~16-17MB
    // (xCastWsSize ~13.7MB + Stage2 K-reduce scratch ~2.5MB @ m_split_core.h:343-345 + KB 级 invRms/hcBeforeNorm)。
    // 保留 16MB 通用 margin（>> Stage2 K-reduce region），总 ~30MB，host aclrtMalloc 144MB→~30MB。
    // 原: workspaceSize_ += 16 * 1024 * 1024 + 128 * 1024 * 1024;  // 144MB flat padding
    workspaceSize_ += 16 * 1024 * 1024;
    if (useKSplit_) {
        // K-split workspace layout: [xCast per-cube-core | mmOut partial per K-block | squareSum partial]
        // (must stay consistent with MembaseKSplitCorePart1::Init / Part2::Process in op_kernel)
        uint64_t xCastFp32BufSize = tilingData_.get_mL1Size() * RoundUp(tilingData_.get_cvLoopKSize(), 128) *
                                    sizeof(float);
        uint64_t workspaceSize1 = tilingData_.get_cubeCoreNum() * DOUBLE_BUFFER * xCastFp32BufSize;
        uint64_t mmLastAxisSize = RoundUp(tilingData_.get_hcMix(), 128);
        uint64_t workspaceSize2 = RoundUp(tilingData_.get_cubeBlockDimK() * tilingData_.get_bs() *
                                          mmLastAxisSize * sizeof(float), 512);
        uint64_t squareSumSize = RoundUp(tilingData_.get_cubeBlockDimK() *
                                         RoundUp(tilingData_.get_bs(), 16) * 16 * sizeof(float), 512);
        uint64_t kSplitRequired = workspaceSize1 + workspaceSize2 + squareSumSize;
        if (kSplitRequired > workspaceSize_) {
            workspaceSize_ = kSplitRequired;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreClampSinkhornTiling::PostTiling()
{
    // 0: m-split branch (large bs / needGrad / kDimNum==1);
    // 1: m+k split branch (small-bs decode, inference only);
    // 2: tiny branch (small-bs decode, inference only, lean 8-AIC launch)
    uint64_t tilingKey = tinyRoute_ ? 2 : (useKSplit_ ? 1 : 0);
    context_->SetTilingKey(tilingKey);
    if (tinyRoute_) {
        OP_LOGI(context_->GetNodeName(),
                "MhcPreClampSinkhorn route: bs=%ld needGrad=%d cubeBlockDimK=%ld tiny=on "
                "-> tilingKey=2 launch(aiv=%lu aic=%lu) platform(aiv=%lu aic=%lu)",
                bs_, static_cast<int>(needGrad_), tilingData_.get_cubeBlockDimK(),
                static_cast<unsigned long>(aivCoreNum_), static_cast<unsigned long>(aicCoreNum_),
                static_cast<unsigned long>(platAivCoreNum_), static_cast<unsigned long>(platAicCoreNum_));
    } else {
        OP_LOGI(context_->GetNodeName(),
                "MhcPreClampSinkhorn route: bs=%ld needGrad=%d cubeBlockDimK=%ld kDimSplit=%s tiny=off "
                "-> tilingKey=%ld (aiv=%lu aic=%lu)",
                bs_, static_cast<int>(needGrad_), tilingData_.get_cubeBlockDimK(),
                useKSplit_ ? "on" : "off", useKSplit_ ? 1 : 0,
                static_cast<unsigned long>(aivCoreNum_), static_cast<unsigned long>(aicCoreNum_));
    }
    context_->SetBlockDim(aicCoreNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    
    PrintTilingData(context_, tilingData_);
    
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForMhcPreClampSinkhorn(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForMhcPreClampSinkhorn(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MhcPreClampSinkhorn", "Tiling context is null"),
               return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE("TilingForMhcPreClampSinkhorn", "Tiling platformInfo is null"),
               return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    auto socVersion = ascendcPlatform.GetSocVersion();
    // if (socVersion == platform_ascendc::SocVersion::ASCEND950) {
    //     OP_LOGD(context->GetNodeName(), "Using arch35 tiling for ASCEND950");
    //     MhcPreClampSinkhornTilingRegbase mhcPreSinkhornTiling(context);
    //     return mhcPreSinkhornTiling.DoOpTiling();
    // }

    MhcPreClampSinkhornTiling mhcPreSinkhornTiling(context);
    return mhcPreSinkhornTiling.DoOpTiling();
}

IMPL_OP_OPTILING(MhcPreClampSinkhorn)
    .Tiling(TilingForMhcPreClampSinkhorn)
    .TilingParse<MhcPreClampSinkhornCompileInfo>(TilingPrepareForMhcPreClampSinkhorn);

}  // namespace optiling
