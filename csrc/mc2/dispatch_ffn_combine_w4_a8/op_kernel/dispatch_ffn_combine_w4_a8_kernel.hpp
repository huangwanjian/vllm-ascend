/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DISPATCH_FFN_COMBINE_W4A8_KERNEL_HPP
#define DISPATCH_FFN_COMBINE_W4A8_KERNEL_HPP

#include "kernel_operator.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

#include "utils/block_mmad_w4a4.hpp"
#include "utils/copy_gm_to_l1_custom.hpp"
#include "utils/block_epilogue_w4a8post_pertoken_v2.hpp"
#include "utils/block_epilogue_w4a8post_pertoken_swiglu.hpp"
#include "utils/hccl_shmem.hpp"
#include "utils/a2_cross_server.hpp"
#include "utils/const_args.hpp"
#include "utils/layout3d.hpp"

#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2.cpp"
#include "moe_init_routing_quant_v2/moe_v2_fullload_dynamic_quant.h"
#include "unpermute/moe_token_unpermute.h"
#include "utils/get_tensor_addr.hpp"


using namespace AscendC;
// #define SYNC_MODE
// Temporary A2 cross-server diagnostics (disabled in the functional build).
// #define DFFC_XS_STAGE1 1
// #define DFFC_XS_STAGE2 1
// #define DFFC_XS_STAGE4 1
// #define DFFC_XS_STAGE5 1
// #define DFFC_XS_STAGE3 1
// #define DFFC_XS_SELFTEST 1
// #define DFFC_XS_CTX_PROBE 1

namespace Catlass::Gemm::Kernel {

constexpr uint16_t SYNCFLAGC2V = 9;
constexpr uint16_t SYNCFLAGV2C = 10;

template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_>
class DispatchFFNCombineW4A8Kernel {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using ElementScale = uint64_t;
    using LayoutScale = typename layout::VectorLayout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = typename layout::VectorLayout;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue1 = BlockEpilogue1_;
    using BlockEpilogue2 = BlockEpilogue2_;
    using ElementD1 = typename BlockEpilogue1::ElementD;
    using LayoutD1 = typename BlockEpilogue1::LayoutD;
    using ElementD2 = typename BlockEpilogue2::ElementD;
    using LayoutD2 = typename BlockEpilogue2::LayoutD;
    using ElementABefore = std::conditional_t<std::is_same_v<ElementA, AscendC::int4b_t>, int8_t, ElementA>;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        __gm__ ElementABefore *ptrA;
        LayoutA layoutA;
        LayoutA layoutA2;
        __gm__ ElementB *ptrB1;
        LayoutB layoutB1;
        __gm__ float *ptrMAux1;
        __gm__ ElementB *ptrB2;
        LayoutB layoutB2;
        __gm__ float *ptrMAux2;
        __gm__ ElementScale *ptrScale1;
        LayoutScale layoutScale1;
        __gm__ ElementScale *ptrScale2;
        LayoutScale layoutScale2;
        __gm__ ElementD2 *ptrOutput;
        LayoutD1 layoutD1;
        LayoutD2 layoutD2;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrExpertTokenNums;
        int32_t EP;
        int32_t listLen;
        int32_t expertPerRank;
        uint32_t maxOutputSize;
        uint32_t rank;
        uint32_t rankSize;
        int32_t ubMoveNum;
        GM_ADDR symmetricPtr;
        uint64_t kfcSegmentSize;
        //--------------
        GM_ADDR expertIdx;
        GM_ADDR moeInitRoutingQuantV2Scale;
        GM_ADDR moeInitRoutingQuantV2Offset;
        GM_ADDR expandedX;
        GM_ADDR expandedRowIdx;
        GM_ADDR expertTokensCountOrCumsum;
        GM_ADDR expertTokensBeforeCapacity;
        GM_ADDR dynamicQuantScale;
        GM_ADDR probs;
        GM_ADDR ptrXActiveMask;
        int64_t topK;
        uint64_t initRoutingQuantTilingKey;
        uint32_t epilogueCoreNum;
        optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
        float swigluLimit;
        //--------------

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord problemShape_, uint32_t EP_, uint32_t listLen_, 
               uint32_t expertPerRank_, uint32_t maxOutputSize_, uint32_t rank_,
               uint32_t rankSize_, int64_t topK_, uint64_t initRoutingQuantTilingKey_, uint32_t epilogueCoreNum_,
               GM_ADDR ptrA_, LayoutA layoutA_, LayoutA layoutA2_, GM_ADDR ptrB1_,
               LayoutB layoutB1_, GM_ADDR ptrMAux1_, GM_ADDR ptrB2_, LayoutB layoutB2_, GM_ADDR ptrMAux2_,
               GM_ADDR ptrScale1_, LayoutScale layoutScale1_, GM_ADDR ptrScale2_, LayoutScale layoutScale2_,
               GM_ADDR ptrOutput_, LayoutD2 layoutD1_, LayoutD2 layoutD2_, GM_ADDR expertIdx_,
               GM_ADDR moeInitRoutingQuantV2Scale_, GM_ADDR moeInitRoutingQuantV2Offset_,
               GM_ADDR expertTokensBeforeCapacity_, GM_ADDR probs_, GM_ADDR ptrWorkspace_, GM_ADDR gmExpertTokenNums_,
               int32_t ubMoveNum_, GM_ADDR ptrXActiveMask_,
               optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData_, float swigluLimit_,
               GM_ADDR symmetricPtr_ = nullptr, uint64_t kfcSegmentSize_ = 0)
            : problemShape(problemShape_),
              EP(EP_),
              listLen(listLen_),
              expertPerRank(expertPerRank_),
              maxOutputSize(maxOutputSize_),
              rank(rank_),
              rankSize(rankSize_),
              topK(topK_),
              initRoutingQuantTilingKey(initRoutingQuantTilingKey_),
              epilogueCoreNum(epilogueCoreNum_),
              ptrA(reinterpret_cast<__gm__ ElementABefore *>(ptrA_)),
              layoutA(layoutA_),
              layoutA2(layoutA2_),
              ptrB1(reinterpret_cast<__gm__ ElementB *>(ptrB1_)),
              layoutB1(layoutB1_),
              ptrMAux1(reinterpret_cast<__gm__ float *>(ptrMAux1_)),
              ptrB2(reinterpret_cast<__gm__ ElementB *>(ptrB2_)),
              layoutB2(layoutB2_),
              ptrMAux2(reinterpret_cast<__gm__ float *>(ptrMAux2_)),
              ptrScale1(reinterpret_cast<__gm__ ElementScale *>(ptrScale1_)),
              layoutScale1(layoutScale1_),
              ptrScale2(reinterpret_cast<__gm__ ElementScale *>(ptrScale2_)),
              layoutScale2(layoutScale2_),
              ptrOutput(reinterpret_cast<__gm__ ElementD2 *>(ptrOutput_)),
              layoutD1(layoutD1_),
              layoutD2(layoutD2_),
              expertIdx(expertIdx_),
              moeInitRoutingQuantV2Scale(moeInitRoutingQuantV2Scale_),
              moeInitRoutingQuantV2Offset(moeInitRoutingQuantV2Offset_),
              expertTokensBeforeCapacity(expertTokensBeforeCapacity_),
              probs(probs_),
              ptrWorkspace(ptrWorkspace_),
              ptrExpertTokenNums(gmExpertTokenNums_),
              ubMoveNum(ubMoveNum_),
              symmetricPtr(symmetricPtr_),
              kfcSegmentSize(kfcSegmentSize_),
              ptrXActiveMask(ptrXActiveMask_),
              moeInitRoutingQuantV2TilingData(moeInitRoutingQuantV2TilingData_),
              swigluLimit(swigluLimit_)
        {
            moeInitRoutingQuantV2TilingData.vbsComputeParamsOp = moeInitRoutingQuantV2TilingData_.vbsComputeParamsOp;
            moeInitRoutingQuantV2TilingData.vmsMiddleComputeParamsOp =
                moeInitRoutingQuantV2TilingData_.vmsMiddleComputeParamsOp;
            moeInitRoutingQuantV2TilingData.sortOutComputeParamsOp =
                moeInitRoutingQuantV2TilingData_.sortOutComputeParamsOp;
            moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp =
                moeInitRoutingQuantV2TilingData_.srcToDstComputeParamsOp;
            moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp =
                moeInitRoutingQuantV2TilingData_.srcToDstCapacityComputeParamsOp;
            moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp =
                moeInitRoutingQuantV2TilingData_.gatherOutComputeParamsOp;
        }
    };

    // Methods
    CATLASS_DEVICE
    DispatchFFNCombineW4A8Kernel(Params const &params)
    {
        if ASCEND_IS_AIC {
            coreIdx = AscendC::GetBlockIdx();
            coreNum = AscendC::GetBlockNum();
        }

        if ASCEND_IS_AIV {
            coreIdx = get_block_idx() + get_subblockid() * get_block_num();
            coreNum = get_block_num() * get_subblockdim();
        }

        initBuffer(params);
    }

    CATLASS_DEVICE
    ~DispatchFFNCombineW4A8Kernel() {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
#if defined(DFFC_XS_CTX_PROBE) || defined(DFFC_XS_STAGE1) || defined(DFFC_XS_STAGE2) || defined(DFFC_XS_STAGE3)
        // Diagnostics build: the AIV side returns immediately, so the cube
        // side must not enter GMM1/GMM2 (it would wait on AIV cross-core flags
        // that are never set and the kernel would never complete).
        (void)params;
        return;
#endif
#ifdef SYNC_MODE
        AscendC::SyncAll<false>();
#endif
        GMM1(params);
#ifdef SYNC_MODE
        AscendC::SyncAll<false>();
        AscendC::SyncAll<false>();
#endif
        GMM2(params);
#ifdef SYNC_MODE
        AscendC::SyncAll<false>();
#endif
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        DispatchAndCombine(params);
    }

private:
    CATLASS_DEVICE void initBuffer(Params const &params)
    {
#ifndef HCCL_COMM
        shmem.initShmem(params.symmetricPtr, params.rank, params.rankSize);
#else
        if (params.symmetricPtr != nullptr && params.kfcSegmentSize > 0) {
            shmem.initKfc(params.symmetricPtr, static_cast<int32_t>(params.rankSize), params.kfcSegmentSize);
            // A2 cross-server transport: parse kfcContextAddr -> HcclA2CombineOpParam
            // -> aiRMAInfo so cross-server cells can be posted over AI-RDMA.
            crossCtx_.Init(params.symmetricPtr, static_cast<int32_t>(params.rankSize));
        }
#endif
        // AI-RDMA doorbell scratch, kept 4KB below the top of UB (the top ~2KB
        // holds the route caches created by InitRouteCache).
        ubRdmaLocal_ = resource.ubBuf.template GetBufferByByte<uint64_t>(ArchTag::UB_SIZE - 4096);
        ubRdmaLocal_.SetSize(4);
        ubRdmaHead_ = resource.ubBuf.template GetBufferByByte<uint32_t>(ArchTag::UB_SIZE - 4096 + UB_ALIGN);
        ubRdmaHead_.SetSize(8);
        shmem.setCrossServer(&crossCtx_, ubRdmaLocal_, ubRdmaHead_);
#ifdef DFFC_XS_STAGE5
        // Bounded V2 wait + observation dump through the output tensor.
        shmem.setV2Debug(reinterpret_cast<GM_ADDR>(params.ptrOutput), true);
#endif
        workspaceInfo = WorkspaceInfo(params);
        int64_t segmentSize = static_cast<int64_t>(shmem.SegmentSize());
        // Multi-flag KFC context: remote windows are mapped per direction and
        // only cover winSize (the In window). Rebase the whole peermem layout
        // to that window so TPE/flags/sync stay inside the mapped range.
        if (crossCtx_.ready && crossCtx_.winSize > 0 &&
            static_cast<int64_t>(crossCtx_.winSize) < segmentSize) {
            segmentSize = static_cast<int64_t>(crossCtx_.winSize);
            shmem.setSegmentSize(static_cast<size_t>(segmentSize));
        }
        peermemInfo = PeermemInfo(params, segmentSize);

        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspaceInfo.ptrcumsumMM));

        // gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementABefore *>(workspaceInfo.ptrA));
        gmS.SetGlobalBuffer(params.ptrScale1);
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(workspaceInfo.ptrC));

        if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
            gmA1I4.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(workspaceInfo.ptrA1Int4));
            gmA1I4_I8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(workspaceInfo.ptrA1Int4));
            gmA2I4.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(workspaceInfo.ptrA2Int4));
            gmA2I4_I8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(workspaceInfo.ptrA2Int4));
            gmCGMM1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspaceInfo.ptrCGMM1));
            gmCGMM2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspaceInfo.ptrCGMM2));
        }

        // gmPermutedToken.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD1 *>(workspaceInfo.ptrPermutedToken));
        gmS2.SetGlobalBuffer(params.ptrScale2);
        gmC2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(workspaceInfo.ptrC2));

        gmPerTokenScale1.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementPerTokenScale *>(workspaceInfo.ptrPerTokenScale));
        gmPerTokenScale2.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementPerTokenScale *>(workspaceInfo.ptrPerTokenScale2));

        tokenPerExpert.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(shmem() + peermemInfo.offsetPeerTokenPerExpert));

        paddedExpertNumAligned = AlignUp(params.EP * params.expertPerRank + 1, ALIGN_128);
        tokenPerExpertLayout = Layout3D(paddedExpertNumAligned, params.expertPerRank);
        preSumBeforeRank.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspaceInfo.ptrSumBeforeRank));
        // Reuse the (unused) workspace TPE slot for the cross-server D staging
        // base; EP int32s only.
        ptrStageBase_ = crossCtx_.ready
                            ? reinterpret_cast<__gm__ int32_t *>(workspaceInfo.ptrTokenPerExpert)
                            : nullptr;
        gmXActiveMask.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(params.ptrXActiveMask));
    }

    template <typename T>
    CATLASS_DEVICE void CopyGMToGM(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, int32_t elemNum,
                                   int32_t ubMoveNum)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

        using TType = Gemm::GemmType<T, layout::RowMajor>;
        using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
        using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
        CopyGmToUb copyGmToUb;
        CopyUbToGm copyUbToGm;
        constexpr int32_t BufferNum = 2;
        int tmpBufferSize = 32 * 1024 / sizeof(T);  // 32 KB
        AscendC::LocalTensor<T> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<T>(0);
        tmpBuffer1.SetSize(tmpBufferSize);
        int tmpBufferOffset = 96 * 1024;  // half of UB
        AscendC::LocalTensor<T> tmpBuffer2 = resource.ubBuf.template GetBufferByByte<T>(tmpBufferOffset);
        tmpBuffer2.SetSize(tmpBufferSize);

        // [ReduceScatter] 2. Pre Interface Sync
        int pingpongId = 0;
        auto processCount = CeilDiv(elemNum, ubMoveNum);
        for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
            uint32_t curProcessNum = (processIndex == processCount - 1) ? elemNum - ubMoveNum * (processCount - 1) :
                                                                          ubMoveNum;
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
            auto processOffset = processIndex * ubMoveNum;

            auto inputOffset = processOffset;
            auto outputOffset = processOffset;
            // [ReduceScatter] 2. Pre Interface Sync
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            // [ReduceScatter] 3. Start shmem_mte_get_mem_nbi
            copyGmToUb(buf, src[inputOffset], layout::RowMajor{1, curProcessNum}, layout::RowMajor{1, curProcessNum});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            copyUbToGm(dst[outputOffset], buf, layout::RowMajor{1, curProcessNum}, layout::RowMajor{1, curProcessNum});

            // [ReduceScatter] 4. Post Interface Sync
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            pingpongId = (pingpongId + 1) % BufferNum;
        }
        // [ReduceScatter] 4. Post Interface Sync

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }


    CATLASS_DEVICE
    void ApplyXActiveMask(Params const &params) {
        if (params.ptrXActiveMask == nullptr) {
            return;
        }
        int32_t m = params.problemShape.m();
        int32_t topK = params.topK;
        int32_t expertNum = params.expertPerRank * params.EP;
        AscendC::GlobalTensor<int32_t> expertIdxGm;
        expertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.expertIdx));

        int32_t totalElements = m * topK;

        int32_t base = totalElements / coreNum;
        int32_t rem = totalElements % coreNum;

        int32_t startIdx = coreIdx * base + min(coreIdx, rem);
        int32_t endIdx = (coreIdx + 1) * base + min(coreIdx + 1, rem);

        AscendC::LocalTensor<int32_t> tmpExpertIdx = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        int32_t copySize = endIdx - startIdx;

        AscendC::DataCopyPad(tmpExpertIdx[0], expertIdxGm[startIdx], 
                      {1, static_cast<uint16_t>(copySize * sizeof(int32_t)) , 0, 0 }, {} 
        );

        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

        for (int32_t i = 0; i < copySize; ++i) {
            int32_t tokenIdx = (startIdx + i) / topK;
            bool isActive = gmXActiveMask(tokenIdx);
            if (!isActive) {
                tmpExpertIdx.SetValue(i, expertNum);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(expertIdxGm[startIdx], tmpExpertIdx[0], {1, static_cast<uint16_t>(copySize * sizeof(int32_t)), 0, 0, 0});

        AscendC::SyncAll<true>();
    }

    CATLASS_DEVICE void FetchAndPreprocessInt8ToInt4(uint64_t gmOffsetA, AscendC::GlobalTensor<float> dstScale,
                                                     AscendC::GlobalTensor<int8_t> src, int32_t rows,
                                                     int32_t hiddenSize)
    {
        uint32_t tmpBufferOffset = 0;

        AscendC::LocalTensor<int8_t> xTensor0 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
        AscendC::LocalTensor<int8_t> tmpBuffer0 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
        xTensor0.SetSize(hiddenSize);
        tmpBuffer0.SetSize(hiddenSize + ALIGN_512);
        tmpBufferOffset += (hiddenSize + ALIGN_512) * sizeof(int8_t);

        AscendC::LocalTensor<int8_t> xTensor1 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
        AscendC::LocalTensor<int8_t> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
        xTensor1.SetSize(hiddenSize);
        tmpBuffer1.SetSize(hiddenSize + ALIGN_512);
        tmpBufferOffset += (hiddenSize + ALIGN_512) * sizeof(int8_t);

        AscendC::LocalTensor<int4b_t> xHighI4Tensor = resource.ubBuf.template GetBufferByByte<int4b_t>(tmpBufferOffset);
        xHighI4Tensor.SetSize(hiddenSize);
        tmpBufferOffset += hiddenSize / 2;

        AscendC::LocalTensor<int4b_t> xLowI4Tensor = resource.ubBuf.template GetBufferByByte<int4b_t>(tmpBufferOffset);
        xLowI4Tensor.SetSize(hiddenSize);
        tmpBufferOffset += hiddenSize / 2;

        AscendC::LocalTensor<half> xHighHalfTensor = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
        xHighHalfTensor.SetSize(hiddenSize * sizeof(half));
        tmpBufferOffset += hiddenSize * sizeof(half);

        AscendC::LocalTensor<half> xLowHalfTensor = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
        xLowHalfTensor.SetSize(hiddenSize * sizeof(half));
        tmpBufferOffset += hiddenSize * sizeof(half);

        AscendC::LocalTensor<half> xLowHalfTensor2 = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
        xLowHalfTensor2.SetSize(hiddenSize * sizeof(half));
        tmpBufferOffset += hiddenSize * sizeof(half);

        AscendC::LocalTensor<int16_t> xLowI16Tensor = resource.ubBuf.template GetBufferByByte<int16_t>(tmpBufferOffset);
        xLowI16Tensor.SetSize(128 * sizeof(int16_t));

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);

        constexpr int32_t MASK = 128;
        Duplicate(xLowI16Tensor, static_cast<int16_t>(0x0F0F), MASK);
        PipeBarrier<PIPE_V>();
        const size_t LEN_VK = (hiddenSize / 2) / 128;
        const size_t LAST_LEN_VK = (hiddenSize % 256) / 2;
        const half ONE_SIXTEENTH = static_cast<half>(0.0625f);

        uint32_t totalM = rows;

        SetFlag<HardEvent::S_MTE2>(EVENT_ID6);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID6);
        uint32_t curCoreTaskNum, curCoreStartOffset;
        CalculateTaskInfoEachCore(curCoreTaskNum, curCoreStartOffset, totalM);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID7);

        constexpr size_t LEN_128 = 128;

        constexpr int32_t BufferNum = 2;
        uint32_t copyInNum = hiddenSize + ALIGN_512;

        int pingpongId = 0;
        for (uint32_t processIndex = 0; processIndex < rows; ++processIndex) {
            uint64_t relStartAddr = processIndex * hiddenSize;
            uint64_t absStartAddr = gmOffsetA + relStartAddr;
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID6 : EVENT_ID7;
            AscendC::LocalTensor<int8_t> buf = pingpongId == 0 ? tmpBuffer0 : tmpBuffer1;
            AscendC::LocalTensor<int8_t> xTensor = pingpongId == 0 ? xTensor0 : xTensor1;
            AscendC::LocalTensor<float> bufScale = buf[hiddenSize].template ReinterpretCast<float>();
            auto inputOffset = processIndex * copyInNum;
            auto outputOffset = processIndex * hiddenSize;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID);
            AscendC::DataCopy(buf, src[inputOffset], copyInNum);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::DataCopyPad(dstScale[processIndex], bufScale, {1, 4, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID);
            Cast(xHighHalfTensor, xTensor, AscendC::RoundMode::CAST_NONE, hiddenSize);
            PipeBarrier<PIPE_V>();
            Muls(xHighHalfTensor, xHighHalfTensor, ONE_SIXTEENTH, hiddenSize);
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
            Cast(xHighI4Tensor, xHighHalfTensor, AscendC::RoundMode::CAST_FLOOR, hiddenSize);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID6);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID6);
            DataCopy(gmA1I4_I8[absStartAddr], xHighI4Tensor.ReinterpretCast<int8_t>(), hiddenSize / 2);

            SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
            And(xLowHalfTensor.ReinterpretCast<int16_t>(), xTensor.ReinterpretCast<int16_t>(), xLowI16Tensor, LEN_128,
                LEN_VK, {1, 1, 1, 8, 8, 0});
            if (LAST_LEN_VK > 0) {
                And(xLowHalfTensor[LEN_VK * LEN_128].ReinterpretCast<int16_t>(),
                    xTensor[LEN_VK * LEN_128 * TWO].ReinterpretCast<int16_t>(), xLowI16Tensor, LAST_LEN_VK, 1,
                    {1, 1, 1, 8, 8, 0});
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID);
            Cast(xLowHalfTensor2.ReinterpretCast<half>(), xLowHalfTensor.ReinterpretCast<int8_t>(),
                 AscendC::RoundMode::CAST_NONE, hiddenSize);
            PipeBarrier<PIPE_V>();
            const half MINUS_EIGHT = static_cast<half>(-8);
            Adds(xHighHalfTensor, xLowHalfTensor2, MINUS_EIGHT, hiddenSize);
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
            Cast(xLowI4Tensor, xHighHalfTensor.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, hiddenSize);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID7);
            DataCopy(gmA1I4_I8[absStartAddr + hiddenSize / 2], xLowI4Tensor.ReinterpretCast<int8_t>(), hiddenSize / 2);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
            pingpongId = (pingpongId + 1) % BufferNum;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
    }

    CATLASS_DEVICE
    void GetCumsumForMMAIV(AscendC::GlobalTensor<int32_t> &tokenPerExpert, AscendC::GlobalTensor<int32_t> &result,
                           uint32_t expertPerRank, uint32_t rankId, uint32_t EP)
    {
        int32_t expertPerRankAligned = (expertPerRank + 8 - 1) / 8 * 8;
        AscendC::LocalTensor<int32_t> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> tmpResult =
            resource.ubBuf.template GetBufferByByte<int32_t>(EP * expertPerRank * sizeof(int32_t));
#define U16(x) static_cast<uint16_t>(x)

        AscendC::DataCopyPad(tmpBuffer1, tokenPerExpert[rankId * expertPerRank],
                             {U16(EP), U16(expertPerRank * sizeof(int32_t)),
                              U16((paddedExpertNumAligned - expertPerRank) * sizeof(int32_t)), 0},
                             {});

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t i = 1; i < EP; ++i) {
            AscendC::Add(tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[i * expertPerRankAligned],
                         tmpBuffer1[(i - 1) * expertPerRankAligned], expertPerRank);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopyPad(result, tmpBuffer1, {U16(EP), U16((expertPerRank) * sizeof(int32_t)), 0, 0});
    }

    // Cross-node dispatch push: receiver-side row offsets for a destination.
    // Fills dstCum[i * alignedEpr + j] = Σ_{i' <= i} T[i'][dstRank][j], the TPE
    // column owned by dstRank cumsum'd along the source dimension. This is the
    // same quantity dstRank computes as its own cumsumMM, so a sender can place
    // its rows exactly where the destination's conversion loop reads them:
    //   grpBase(g) = Σ_{g'<g} dstCum[(EP-1)*a + g']   (destination group base)
    //   srcPre(g)  = dstCum[(rank-1)*a + g]           (rows from earlier sources)
    // The TPE matrix is all-gathered before the push, so every rank has it.
    // UB must stay valid across the whole push loop: place it outside the
    // [0,32KB) and [96KB,128KB) regions CopyGMToGM uses as ping-pong buffers.
    CATLASS_DEVICE
    void GetColumnCumsumForDst(Params const &params, AscendC::LocalTensor<int32_t> &dstCum, int32_t dstRank)
    {
        const int32_t alignedEpr = (params.expertPerRank + 7) / 8 * 8;
        AscendC::DataCopyPad(
            dstCum, tokenPerExpert[dstRank * params.expertPerRank],
            {static_cast<uint16_t>(params.EP), static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)),
             static_cast<uint16_t>((paddedExpertNumAligned - params.expertPerRank) * sizeof(int32_t)),
             static_cast<uint16_t>((alignedEpr - params.expertPerRank) * sizeof(int32_t))},
            {});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        for (int32_t i = 1; i < params.EP; ++i) {
            AscendC::Add(dstCum[i * alignedEpr], dstCum[i * alignedEpr], dstCum[(i - 1) * alignedEpr],
                         params.expertPerRank);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // The push loop reads these values with scalar GetValue.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void GMM1(Params const &params)
    {
        icache_preload(8);
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);
        float aivFinishGroups = 0.0f;
        __gm__ float *aivFinishPtr = workspaceInfo.ptrSoftFlagBase + params.EP * FLAGSTRIDE;

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;
        uint32_t startCoreIdx = 0;
        uint32_t syncGroupIdx = 0;
        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;

        uint16_t syncgmmIdx = 0;
        AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx /
                                        CROSS_CORE_FLAG_MAX_SET_COUNT);  // Wait for AIV to finish cumsum for matmul

        syncgmmIdx++;
        AscendC::GlobalTensor<ElementB> gmB1;

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);

            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;            } else if (preCurrentmSum + currentM >= params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                currentM = currentM * 2;
            }

            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            gmB1.SetGlobalBuffer(reinterpret_cast<__gm__ AscendC::int4b_t *>(GetTensorAddr<AscendC::int4b_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrB1))));
            gmS.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(GetTensorAddr<int64_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrScale1))));

            if (currentM <= L1TileShape::M) {
                gmB1.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};
            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB1 = params.layoutB1;
            LayoutScale layoutScale = params.layoutScale1;
            // C rows hold N elements (GMM1 output width); the workspace row stride is max(N, K)
            // (see the max(N, n2) C-region sizing in the tiling and workspace carving).
            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n(),
                inGroupProblemShape.n() > inGroupProblemShape.k() ? inGroupProblemShape.n() : inGroupProblemShape.k());
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                for (; syncGroupIdx <= groupIdx; syncGroupIdx++) {
                    AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                    syncgmmIdx++;
                }
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB1.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetS = blockCoord.n() * L1TileShape::N + (params.listLen == 1 ? groupIdx * params.problemShape.n() : 0);
                if (currentM > 0) {
                    if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                            blockMmad(gmS[gmOffsetS], layoutScale, gmA1I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                      gmB1[gmGroupOffsetB + gmOffsetB], layoutB1, gmC[gmGroupOffsetC + gmOffsetC],
                                      layoutC, actualBlockShape, Callback{}, Callback{});
                        } else {
                            blockMmad(gmS[gmOffsetS], layoutScale, gmA1I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                      gmB1[gmGroupOffsetB + gmOffsetB], layoutB1, gmC[gmGroupOffsetC + gmOffsetC],
                                      layoutC, actualBlockShape);
                        }
                    }
                }
            }

            if (IsSyncTask(groupIdx, params.expertPerRank))
            {
                syncLoopIdx++;
                if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                    blockMmad.SynchronizeBlock();
                }
                blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V);
            }

            preCurrentmSum += currentM / 2;
            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() *
                (inGroupProblemShape.n() > inGroupProblemShape.k() ? inGroupProblemShape.n() : inGroupProblemShape.k());
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        for (; syncGroupIdx < params.expertPerRank; syncGroupIdx++) {
            AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmmIdx++;
        }
    }

    CATLASS_DEVICE
    void GMM2(Params const &params)
    {
        icache_preload(8);
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        uint32_t n2 = params.problemShape.k();
        uint32_t k2 = params.problemShape.n() / 2;

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;

        uint32_t startCoreIdx = 0;

        AscendC::PipeBarrier<PIPE_ALL>();

        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM > params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                currentM = currentM * 2;
            }

            AscendC::GlobalTensor<ElementB> gmB2;
            // gmB2.SetGlobalBuffer(params.ptrB2);
            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            gmB2.SetGlobalBuffer(reinterpret_cast<__gm__ AscendC::int4b_t *>(GetTensorAddr<AscendC::int4b_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrB2))));
            gmS2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(GetTensorAddr<int64_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrScale2))));
            if (currentM <= L1TileShape::M) {
                gmB2.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            GemmCoord inGroupProblemShape{currentM, n2, k2};  // M N K

            LayoutA layoutA = params.layoutA2.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB2 = params.layoutB2;
            LayoutScale layoutScale = params.layoutScale2;
            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            if(groupIdx == 0 || IsSyncTask(groupIdx - 1, params.expertPerRank)) {
                AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C);
            }
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                if (loopIdx + coreNum >= coreLoops) {
                    syncLoopIdx = groupIdx;
                }
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};

                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB2.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetS = blockCoord.n() * L1TileShape::N + (params.listLen == 1 ? groupIdx * n2 : 0);
                if (currentM > 0) {
                    if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                            blockMmad(gmS2[gmOffsetS], layoutScale, gmA2I4[gmGroupOffsetA + gmOffsetA], layoutA,
                                      gmB2[gmGroupOffsetB + gmOffsetB], layoutB2, gmC2[gmGroupOffsetC + gmOffsetC],
                                      layoutC, actualBlockShape, Callback{}, Callback{}, syncLoopIdx, 0);
                        } 
                    }
                }
            }
            preCurrentmSum += currentM / 2;
            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
            blockMmad.Finalize(params.expertPerRank - 1, 0);
        }
    }

    CATLASS_DEVICE
    void InitArithProgress(Params const &params)
    {
        AscendC::LocalTensor<float> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<float>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::Duplicate(tmpBuffer1, 0.0f, (params.EP + 1) * FLAGSTRIDE);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::GlobalTensor<float> flagGlobalBase;
        flagGlobalBase.SetGlobalBuffer(workspaceInfo.ptrSoftFlagBase);
        AscendC::DataCopy(flagGlobalBase, tmpBuffer1, (params.EP + 1) * FLAGSTRIDE);
    }

    CATLASS_DEVICE
    void CalculateTaskInfoEachCore(uint32_t &curCoreTaskNum_, uint32_t &curCoreStartOffset_, uint32_t totalM)
    {
        int64_t eachCoreTaskNum = (totalM + coreNum - 1) / coreNum;
        int64_t taskNumPertailCore = eachCoreTaskNum - 1;
        int64_t usedCoreNum = totalM >= coreNum ? coreNum : totalM;
        uint32_t tailCoreIdx = totalM - (eachCoreTaskNum - 1) * usedCoreNum;
        uint32_t curCoreId = GetBlockIdx();
        curCoreTaskNum_ = curCoreId < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
        curCoreStartOffset_ = curCoreId < tailCoreIdx ? eachCoreTaskNum * curCoreId :
                                                        ((eachCoreTaskNum - 1) * curCoreId + tailCoreIdx);
    }

    CATLASS_DEVICE
    void CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(Params const &params,
                                                                        int64_t localTokenPerExpertOffset)
    {
        uint32_t numPerCore = paddedExpertNumAligned;
        AscendC::LocalTensor<int32_t> tmpBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> prevSumBuf = tmpBuffer[numPerCore];

        for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
            if (dstEpIdx == params.rank) {
                continue;
            }
            AscendC::GlobalTensor<int32_t> srcAddress;
            srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(shmem() + localTokenPerExpertOffset));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            using TType = Gemm::GemmType<int32_t, layout::RowMajor>;
            using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
            using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
            CopyGmToUb copyGmToUb;
            CopyUbToGm copyUbToGm;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            copyGmToUb(tmpBuffer, srcAddress[0], layout::RowMajor{1, numPerCore}, layout::RowMajor{1, numPerCore});

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // Same server: IPC direct write; cross server: A2 direct peer
            // writes do not land, so stage the row in the local Out window and
            // post it with AI-RDMA (source LOCAL_OUTPUT -> peer REMOTE_INPUT).
            if (!(crossCtx_.ready && crossCtx_.CrossServer(dstEpIdx))) {
                AscendC::GlobalTensor<int32_t> dstAddress;
                dstAddress.SetGlobalBuffer(
                    reinterpret_cast<__gm__ int32_t *>(shmem(localTokenPerExpertOffset, dstEpIdx)));
                copyUbToGm(dstAddress[0], tmpBuffer, layout::RowMajor{1, numPerCore}, layout::RowMajor{1, numPerCore});
            } else {
                AscendC::GlobalTensor<int32_t> outTpe;
                outTpe.SetGlobalBuffer(
                    reinterpret_cast<__gm__ int32_t *>(crossCtx_.LocalOut() + localTokenPerExpertOffset));
                copyUbToGm(outTpe[0], tmpBuffer, layout::RowMajor{1, numPerCore}, layout::RowMajor{1, numPerCore});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::PipeBarrier<PIPE_ALL>();
                if (!DffcA2::AivRdmaPostSendTry(
                        reinterpret_cast<GM_ADDR>(crossCtx_.LocalOut() + localTokenPerExpertOffset),
                        reinterpret_cast<GM_ADDR>(crossCtx_.In(dstEpIdx) + localTokenPerExpertOffset),
                        static_cast<uint64_t>(dstEpIdx),
                        static_cast<uint64_t>(numPerCore) * sizeof(int32_t), crossCtx_.qp, ubRdmaLocal_,
                        ubRdmaHead_)) {
                    ++xsTpeFails_;
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
            if (dstEpIdx != params.rank) {
                int32_t intPer512 = CACHE_LINE / sizeof(int);
                for (int32_t checkIdx = 0; checkIdx < paddedExpertNumAligned;
                     checkIdx += intPer512) {
                    __gm__ int32_t *sync_check =
                        reinterpret_cast<__gm__ int32_t *>(shmem() + peermemInfo.offsetPeerTokenPerExpert) +
                        tokenPerExpertLayout(dstEpIdx, 0, checkIdx);
                    if (!gm_signal_wait_until_ne_try(sync_check, 0, 3ULL * 1000ULL * 1000ULL * 50ULL)) {
                        ++xsTpePollFails_;
                    }
                }
                AscendC::DataCopy(tmpBuffer, tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Adds(tmpBuffer, tmpBuffer, -0x800000, numPerCore);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], tmpBuffer, numPerCore);
            } else {
                AscendC::DataCopy(tmpBuffer, tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            int32_t prevSum = 0;
            int32_t j = 0;
            for (int32_t i = 0; i < (params.rank + 1) * params.expertPerRank; i++) {
                if (i >= params.rank * params.expertPerRank) {
                    prevSumBuf(j) = prevSum;
                    j++;
                }
                prevSum += tmpBuffer(i);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(
                preSumBeforeRank[dstEpIdx * params.expertPerRank], prevSumBuf,
                AscendC::DataCopyParams{1, static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)), 0, 0});
        }

        AscendC::SyncAll<true>();
    }

    CATLASS_DEVICE
    void ResetTokenPerExpert(int32_t num)
    {
        if (coreIdx != coreNum - 1) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::LocalTensor<int32_t> tmp = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::Duplicate(tmp, 0, num);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(tokenPerExpert, tmp, num);
    }

    CATLASS_DEVICE
    void UpdateAicFlags(const Params &params)
    {
        float flagBase = 1.0f * params.expertPerRank;
        __gm__ float *aicFinishPtr = workspaceInfo.ptrSoftFlagBase + params.EP * FLAGSTRIDE;
        float flag = 0.0f;
        float lastflag = -1.0f;
        AscendC::LocalTensor<float> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<float>(0);
        __gm__ float *flagPtr = workspaceInfo.ptrSoftFlagBase;
        AscendC::GlobalTensor<float> flagGM;
        flagGM.SetGlobalBuffer(flagPtr);
        int32_t flagBufferSize = max(4, params.EP) * FLAGSTRIDE;
        AscendC::LocalTensor<float> dstValueBuffer = resource.ubBuf.template GetBufferByByte<float>(flagBufferSize);
        AscendC::LocalTensor<float> sharedTmpBuffer =
            resource.ubBuf.template GetBufferByByte<float>((flagBufferSize + 64));
        uint64_t mask[1] = {0};
        uint32_t repeatNum = (flagBufferSize / (4 * FLAGSTRIDE));
        for (int32_t i = 0; i < 4; i++) {
            if (i < params.EP) {
                mask[0] |= 1ull * (1ull << (i * 16));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        while (flag < flagBase) {
            flag = flagBase;
            AscendC::DataCopy(tmpBuffer1, flagGM, params.EP * FLAGSTRIDE);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::ReduceMin<float>(dstValueBuffer, tmpBuffer1, sharedTmpBuffer, mask, repeatNum, 8, false);

            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            flag = min(flag, dstValueBuffer.GetValue(0));

            if (flag > lastflag) {
                *aicFinishPtr = flag;
                gm_dcci(aicFinishPtr);
                lastflag = flag;
            }
        }
    }

    CATLASS_DEVICE
    bool IsSyncTask(int32_t task_id, int32_t n_tasks) {
        int32_t offset = n_tasks - task_id;
        if (offset <= 0 || (offset & (offset - 1)) != 0) {
            return false;
        }else
        {
            return true;
        }
    }

    // Temporary transport self-test (DFFC_XS_SELFTEST): bounded probes that
    // decide which cross-server write mechanism works in this environment.
    //   out[0]=rank out[1]=ready out[2]=direct-write landed
    //   out[3]=ai-rdma landed out[4]=qp present
    CATLASS_DEVICE
    int32_t WaitFlagValue(GM_ADDR flagAddr, AscendC::LocalTensor<int32_t> &ub, uint64_t timeoutCycles)
    {
        AscendC::GlobalTensor<int32_t> g;
        g.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flagAddr));
        const int64_t start = AscendC::GetSystemCycle();
        for (uint32_t poll = 0;; ++poll) {
            AscendC::DataCopy(ub, g, 8);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            const int32_t v = ub.GetValue(0);
            if (v != 0) {
                return v;
            }
            if (((poll & 1023U) == 1023U) &&
                static_cast<uint64_t>(AscendC::GetSystemCycle() - start) >= timeoutCycles) {
                return 0;
            }
        }
    }

    CATLASS_DEVICE
    bool TryWaitMte2(GM_ADDR flagAddr, int32_t expect, AscendC::LocalTensor<int32_t> &ub,
                     uint64_t timeoutCycles)
    {
        AscendC::GlobalTensor<int32_t> g;
        g.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flagAddr));
        const int64_t start = AscendC::GetSystemCycle();
        for (uint32_t poll = 0;; ++poll) {
            AscendC::DataCopy(ub, g, 8);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            if (ub.GetValue(0) == expect) {
                return true;
            }
            if (((poll & 1023U) == 1023U) &&
                static_cast<uint64_t>(AscendC::GetSystemCycle() - start) >= timeoutCycles) {
                return false;
            }
        }
    }

    CATLASS_DEVICE
    void CrossServerSelfTest(Params const &params, __gm__ int32_t *out)
    {
        out[0] = params.rank;
        out[1] = static_cast<int32_t>(crossCtx_.ready);
        out[2] = 0;
        out[3] = 0;
        out[4] = (crossCtx_.qp == nullptr) ? 0 : 1;
        if (!crossCtx_.ready || params.EP < 16) {
            return;
        }
        constexpr uint64_t kTo = 3ULL * 1000ULL * 1000ULL * 50ULL;
        const int64_t off1 = static_cast<int64_t>(crossCtx_.winSize) - 4096;
        const int64_t off2 = static_cast<int64_t>(crossCtx_.winSize) - 8192;
        int32_t peer = params.rank + 8;
        if (peer >= params.EP) {
            peer -= params.EP;
        }
        AscendC::LocalTensor<int32_t> ub =
            resource.ubBuf.template GetBufferByByte<int32_t>(ArchTag::UB_SIZE - 4096 + 64);
        ub.SetSize(8);

        // Phase 1: plain MTE3 write into the peer's mapped In window.
        const int32_t v1 = 0x11110000 | params.rank;
        __gm__ int32_t *p1 = reinterpret_cast<__gm__ int32_t *>(crossCtx_.In(peer) + off1);
        for (int32_t i = 0; i < 8; ++i) {
            p1[i] = v1;
        }
        AscendC::GlobalTensor<int32_t> g1;
        g1.SetGlobalBuffer(p1);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(g1);
        AscendC::PipeBarrier<PIPE_ALL>();
        out[2] = TryWaitMte2(crossCtx_.LocalIn() + off1, 0x11110000 | peer, ub, kTo) ? 1 : 0;

        // Phase 2: AI-RDMA to the peer In window at a LOW offset (4KB).
        // Phase 3: same, at a HIGH offset (winSize - 8KB).
        const int64_t offLow = 4096;
        const int64_t offHigh = static_cast<int64_t>(crossCtx_.winSize) - 8192;
        for (int32_t phase = 2; phase <= 3; ++phase) {
            const int64_t off = (phase == 2) ? offLow : offHigh;
            const int32_t val = ((phase == 2) ? 0x22220000 : 0x33330000) | params.rank;
            __gm__ int32_t *pp = reinterpret_cast<__gm__ int32_t *>(crossCtx_.LocalOut() + off);
            for (int32_t i = 0; i < 8; ++i) {
                pp[i] = val;
            }
            AscendC::GlobalTensor<int32_t> gp;
            gp.SetGlobalBuffer(pp);
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(gp);
            AscendC::PipeBarrier<PIPE_ALL>();
            if (!DffcA2::AivRdmaPostSendTry(reinterpret_cast<GM_ADDR>(pp),
                                            reinterpret_cast<GM_ADDR>(crossCtx_.In(peer) + off),
                                            static_cast<uint64_t>(peer), 32, crossCtx_.qp, ubRdmaLocal_,
                                            ubRdmaHead_)) {
                out[phase] = 2; // post failed
                continue;
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            out[phase] = TryWaitMte2(crossCtx_.LocalIn() + off, val & 0xffff0000 | peer, ub, kTo) ? 1 : 0;
        }
    }

    CATLASS_DEVICE
    void DispatchAndCombine(Params const &params)
    {
        icache_preload(8);
#ifdef DFFC_XS_CTX_PROBE
        // Diagnostics: parse the KFC context, run the cross-server RDMA
        // self-test, publish the result to expert_token_nums, then return.
        if (coreIdx == 0) {
            __gm__ int32_t *p = reinterpret_cast<__gm__ int32_t *>(params.ptrExpertTokenNums);
            CrossServerSelfTest(params, p);
            p[5] = crossCtx_.multiFlag;
            p[6] = static_cast<int32_t>(crossCtx_.winSize & 0xffffffffULL);
            p[7] = (crossCtx_.qp == nullptr) ? 0 : 1;
            AscendC::GlobalTensor<int32_t> etnGm;
            etnGm.SetGlobalBuffer(p);
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(etnGm);
            // Raw dump of the context struct into the output tensor (int32
            // view): [0]=sizeof(HcclA2CombineOpParam), [1]=offsetof(multiFlag),
            // [2..257]=first 256 words of HcclA2CombineOpParam,
            // [258..265]=first 8 words of Mc2MoeContext.
            __gm__ int32_t *dump = reinterpret_cast<__gm__ int32_t *>(params.ptrOutput);
            dump[0] = static_cast<int32_t>(sizeof(HcclA2CombineOpParam));
            dump[1] = static_cast<int32_t>(__builtin_offsetof(HcclA2CombineOpParam, multiFlag));
            if (crossCtx_.win != nullptr) {
                __gm__ int32_t *w = reinterpret_cast<__gm__ int32_t *>(crossCtx_.win);
                // words 6..13: winSize, windowsIn[0..3]
                for (int32_t i = 0; i < 8; ++i) {
                    dump[78 + i] = w[6 + i];
                }
                // words 2220..2227: data ptr, dataSize, sizeOfAiRMAInfo, aiRMAInfo
                for (int32_t i = 0; i < 8; ++i) {
                    dump[10 + i] = w[2220 + i];
                }
                __gm__ uint8_t *dBase = reinterpret_cast<__gm__ uint8_t *>(crossCtx_.win->data);
                if (dBase != nullptr) {
                    __gm__ int32_t *d0 = reinterpret_cast<__gm__ int32_t *>(dBase);
                    __gm__ int32_t *d120 = reinterpret_cast<__gm__ int32_t *>(dBase + 8 * 120);
                    __gm__ int32_t *d128 = reinterpret_cast<__gm__ int32_t *>(dBase + 8 * 128);
                    for (int32_t i = 0; i < 30; ++i) {
                        dump[18 + i] = d0[i];   // entry rank 0
                        dump[48 + i] = d120[i]; // entry rank 8 if stride 120
                        dump[78 + 8 + i] = d128[i]; // entry rank 8 if stride 128
                    }
                }
            }
            __gm__ int32_t *mctx = reinterpret_cast<__gm__ int32_t *>(crossCtx_.mc2Ctx);
            for (int32_t i = 0; i < 8; ++i) {
                dump[258 + i] = mctx[i];
            }
            AscendC::GlobalTensor<int32_t> dumpGm;
            dumpGm.SetGlobalBuffer(dump);
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(dumpGm);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::SyncAll<true>();
        return;
#endif
        int64_t localTokenPerExpertOffset =
            peermemInfo.offsetPeerTokenPerExpert + tokenPerExpertLayout(params.rank, 0, 0) * sizeof(int32_t);
        GM_ADDR localTokenPerExpert =
            shmem() + localTokenPerExpertOffset;  // Place the entire communication matrix in peermem
        uint32_t expandedRowIdxOffset = AlignUp(params.problemShape.m(), 256) * params.topK * sizeof(int32_t);

        ApplyXActiveMask(params);

        //---initRouting------
        const bool crossNode = crossCtx_.ready && params.EP > 8;
        // Cross-node: outbound tokens are staged in the local Out window so
        // they can be pushed with AI-RDMA (source must be LOCAL_OUTPUT).
        GM_ADDR expandedXBase = crossNode ? (crossCtx_.LocalOut() + peermemInfo.offsetA)
                                          : (shmem() + peermemInfo.offsetA);
        GM_ADDR dynScaleBase = crossNode ? (crossCtx_.LocalOut() + peermemInfo.offsetPeerPerTokenScale)
                                         : (shmem() + peermemInfo.offsetPeerPerTokenScale);
        moe_init_routing_quant_v2<ElementD2>(
            reinterpret_cast<GM_ADDR>(params.ptrA), params.expertIdx, params.moeInitRoutingQuantV2Scale,
            params.moeInitRoutingQuantV2Offset, expandedXBase, workspaceInfo.expandedRowIdx,
            localTokenPerExpert, params.expertTokensBeforeCapacity, dynScaleBase,
            params.ptrWorkspace + expandedRowIdxOffset, &params.moeInitRoutingQuantV2TilingData,
            params.initRoutingQuantTilingKey);
        
        AscendC::SyncAll<true>();

        if (crossNode) {
            // Clear the per-cell dispatch flags in this rank's In window before
            // anything can write them this call. The flags live in persistent
            // window memory and the receive side polls for "non-zero", so a
            // leftover 1/2 from a previous call could make a later call skip a
            // cell that actually carries data. The sweep must happen before
            // this rank's TPE row is sent: every sender's push pass waits for
            // that row, so no flag write can race the sweep.
            if (coreIdx == 0) {
                const int64_t sweepOff = static_cast<int64_t>(shmem.SegmentSize()) - 3 * MB_SIZE;
                const int32_t sweepInts = static_cast<int32_t>(params.EP) * params.expertPerRank * 16;
                AscendC::LocalTensor<int32_t> zUb = resource.ubBuf.template GetBufferByByte<int32_t>(64 * 1024);
                zUb.SetSize(static_cast<uint32_t>(sweepInts));
                AscendC::Duplicate(zUb, 0, sweepInts);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::GlobalTensor<int32_t> zGm;
                zGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(crossCtx_.LocalIn() + sweepOff));
                AscendC::DataCopy(zGm, zUb, static_cast<uint16_t>(sweepInts));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            AscendC::SyncAll<true>();
        }

        CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(params, localTokenPerExpertOffset);
#ifdef DFFC_XS_STAGE1
        if (coreIdx == 0) {
            __gm__ int32_t *p = reinterpret_cast<__gm__ int32_t *>(params.ptrExpertTokenNums);
            p[0] = params.rank;
            p[1] = 11;
            AscendC::GlobalTensor<int32_t> stGm;
            stGm.SetGlobalBuffer(p);
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(stGm);
        }
        AscendC::SyncAll<true>();
        return;
#endif
#ifdef DFFC_XS_SELFTEST
        if (coreIdx == 0) {
            AscendC::printf("[xs] rank=%d tpe-done\n", params.rank);
        }
#endif

        if (coreIdx == 0) {
            GetCumsumForMMAIV(tokenPerExpert, cumsumMM, params.expertPerRank, params.rank, params.EP);
            if (crossNode) {
                // Per-destination staging base for the cross-server combine D
                // mirror. Destination d owns L_d = sum_g TPE(d, me, g) rows and
                // the mirrors are packed back to back, because the rows of the
                // destinations' own D regions all start at offset 0.
                AscendC::LocalTensor<int32_t> baseUb = resource.ubBuf.template GetBufferByByte<int32_t>(
                    (params.EP * params.expertPerRank + 2048) * static_cast<int32_t>(sizeof(int32_t)));
                baseUb.SetSize(static_cast<uint32_t>(params.EP));
                int32_t acc = 0;
                for (int32_t d = 0; d < static_cast<int32_t>(params.EP); ++d) {
                    baseUb.SetValue(d, acc);
                    for (int32_t g = 0; g < static_cast<int32_t>(params.expertPerRank); ++g) {
                        acc += tokenPerExpert(tokenPerExpertLayout(d, params.rank, g));
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::GlobalTensor<int32_t> baseGm;
                baseGm.SetGlobalBuffer(ptrStageBase_);
                AscendC::DataCopyPad(
                    baseGm, baseUb,
                    AscendC::DataCopyParams{1, static_cast<uint16_t>(params.EP * sizeof(int32_t)), 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                for (int32_t off = 0; off < static_cast<int32_t>(params.EP); off += 16) {
                    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                                      AscendC::DcciDst::CACHELINE_OUT>(baseGm[off]);
                }
            }
        }

        uint32_t curGroupOffset = 0;
        int32_t prevSumBeforeRank = 0;
        int32_t prevSum = 0;
        if (coreIdx < params.EP) {
            prevSum = preSumBeforeRank(coreIdx * params.expertPerRank);
        }
        AscendC::SyncAll<true>();

        AscendC::GlobalTensor<int32_t> ExpertTokenNums;
        ExpertTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.ptrExpertTokenNums));
        if (coreIdx == 0) {
            CopyGMToGM(ExpertTokenNums, cumsumMM[(params.EP - 1) * params.expertPerRank], params.expertPerRank,
                       params.ubMoveNum);
        }
        uint16_t syncgmm1Idx = 0;
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
        syncgmm1Idx++;

        // ---- Dispatch push (cross-server group only) ----
        // Each core owns one destination rank. For every expert group it pushes
        // this rank's rows into that destination's In window at the
        // destination-side rowStart (derived from the all-gathered cumsum) and
        // then posts a 32B flag on the same QP. Self cells are copied locally.
        int32_t xsPushFails = 0;
        int32_t xsFlagFails = 0;
        int32_t xsPushFirst = 0;
        int32_t xsPushSrcRow = 0;
        int32_t xsPushDstRow = 0;
        int64_t dispatchFlagOff = 0;
        int64_t dispatchStageOff = 0;
        if (crossNode) {
            const int64_t slab = static_cast<int64_t>(params.problemShape.k()) + ALIGN_512;
            dispatchFlagOff = static_cast<int64_t>(shmem.SegmentSize()) - 3 * MB_SIZE;
            dispatchStageOff = dispatchFlagOff +
                               static_cast<int64_t>(params.EP) * params.expertPerRank * 64;
            // Cross-server V2 status slots (+64 state==1, +96 state==0).
            shmem.setV2Stage(crossCtx_.LocalOut() + dispatchStageOff + 64);
            if (coreIdx == 0) {
                // Stage with an MTE3 DMA, not scalar stores: the AI-RDMA engine
                // reads this region from HBM, and a scalar store only becomes
                // visible after a cache flush whose granularity is one 64B line
                // (the V2 slots sit on the second line, so they would stay in
                // core 0's cache and the peer would receive zeros).
                AscendC::LocalTensor<int32_t> stgUb = resource.ubBuf.template GetBufferByByte<int32_t>(
                    (params.EP * params.expertPerRank + 2048) * static_cast<int32_t>(sizeof(int32_t)));
                stgUb.SetSize(32);
                for (int32_t i = 0; i < 8; ++i) {
                    stgUb.SetValue(i, (i == 0) ? 1 : 0);              // data marker
                    stgUb.SetValue(8 + i, (i == 0) ? 2 : 0);          // skip marker
                    stgUb.SetValue(16 + i, (i == 0) ? 0x3F800000 : 0);// V2 state 1
                    stgUb.SetValue(24 + i, 0);                        // V2 state 0
                }
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::GlobalTensor<int32_t> stgGm;
                stgGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(crossCtx_.LocalOut() + dispatchStageOff));
                AscendC::DataCopyPad(stgGm, stgUb, AscendC::DataCopyParams{1, 128, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            AscendC::SyncAll<true>();
            // Offset of this rank's first cell for dst==coreIdx inside its own
            // outbound buffer: prefix over (dst' < dst, all groups) in row "me".
            uint32_t srcRow = 0;
            // Receiver-side offsets for the destination this core owns. The
            // destination's conversion loop places source-s rows at
            //   Σ_{g'<g} Σ_i T[i][dst][g']  +  Σ_{s'<s} T[s'][dst][g]
            // (its own group bases and source prefixes). The sender must use
            // the SAME layout, derived from the all-gathered TPE column of dst.
            // (The earlier version used the sender's own cumsumMM — transposed
            // semantics: it tracked "rows dst sends me" instead of "rows I send
            // dst", so only the self cell ever landed correctly.)
            const int32_t xsAlignedEpr = (params.expertPerRank + 7) / 8 * 8;
            int64_t dstBase = 0;   // Σ_{g'<g} Σ_i T[i][dst][g']
            AscendC::LocalTensor<int32_t> dstCumUb =
                resource.ubBuf.template GetBufferByByte<int32_t>(64 * 1024);
            dstCumUb.SetSize(static_cast<uint32_t>(params.EP * xsAlignedEpr));
            if (coreIdx < params.EP) {
                for (int32_t d = 0; d < static_cast<int32_t>(coreIdx); ++d) {
                    for (int32_t gg = 0; gg < params.expertPerRank; ++gg) {
                        srcRow += static_cast<uint32_t>(
                            tokenPerExpert(tokenPerExpertLayout(params.rank, d, gg)));
                    }
                }
                GetColumnCumsumForDst(params, dstCumUb, static_cast<int32_t>(coreIdx));
            }
            for (int32_t g = 0; g < params.expertPerRank; ++g) {
                if (coreIdx < params.EP) {
                    const int32_t dst = coreIdx;
                    // Rows this rank sends to dst's group g: T[me][dst][g].
                    const uint32_t rowsAll =
                        static_cast<uint32_t>(tokenPerExpert(tokenPerExpertLayout(params.rank, dst, g)));
                    const int32_t srcPre = (params.rank == 0)
                                               ? 0
                                               : dstCumUb.GetValue((params.rank - 1) * xsAlignedEpr + g);
                    const uint32_t dstRowStart = static_cast<uint32_t>(dstBase + static_cast<int64_t>(srcPre));
                    uint32_t rows = rowsAll;
                    bool postData = false;
                    if (rows > 0 && dstRowStart < params.maxOutputSize) {
                        if (dstRowStart + rows > params.maxOutputSize) {
                            rows = params.maxOutputSize - dstRowStart;
                        }
                        postData = (rows > 0);
                    }
                    if (true) {
                        if (postData) {
                            GM_ADDR srcPtr = crossCtx_.LocalOut() + peermemInfo.offsetA +
                                             static_cast<int64_t>(srcRow) * slab;
                            GM_ADDR dstPtr = crossCtx_.In(dst) + peermemInfo.offsetA +
                                             static_cast<int64_t>(dstRowStart) * slab;
                            const uint64_t bytes = static_cast<uint64_t>(rows) * static_cast<uint64_t>(slab);
                            if (dst == params.rank || !crossCtx_.CrossServer(dst)) {
                                // Self or same-server peer: IPC direct copy.
                                AscendC::GlobalTensor<int8_t> dG;
                                AscendC::GlobalTensor<int8_t> sG;
                                GM_ADDR peerDst = (dst == params.rank)
                                                      ? dstPtr
                                                      : (shmem(peermemInfo.offsetA +
                                                                   static_cast<int64_t>(dstRowStart) * slab, dst));
                                dG.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(peerDst));
                                sG.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(srcPtr));
                                CopyGMToGM<int8_t>(dG, sG, static_cast<int32_t>(bytes), params.ubMoveNum);
                                if (dst != params.rank) {
                                    __gm__ int32_t *pf = reinterpret_cast<__gm__ int32_t *>(
                                        shmem(dispatchFlagOff +
                                                  static_cast<int64_t>(params.rank * params.expertPerRank + g) * 64,
                                              dst));
                                    gm_dcci(reinterpret_cast<__gm__ uint8_t *>(pf));
                                    *pf = 1;
                                    gm_dcci(reinterpret_cast<__gm__ uint8_t *>(pf));
                                }
                            } else {
                                __gm__ int32_t *dbg = reinterpret_cast<__gm__ int32_t *>(params.ptrOutput);
                                if (g == 0) {
                                    dbg[160 + coreIdx * 4 + 0] = static_cast<int32_t>(srcRow);
                                    dbg[160 + coreIdx * 4 + 1] = static_cast<int32_t>(dstRowStart);
                                    dbg[160 + coreIdx * 4 + 2] = static_cast<int32_t>(bytes);
                                    AscendC::GlobalTensor<int32_t> dg0;
                                    dg0.SetGlobalBuffer(dbg + 160 + coreIdx * 4);
                                    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                                                      AscendC::DcciDst::CACHELINE_OUT>(dg0);
                                }
                                const bool postOk = DffcA2::AivRdmaPostSendTry(srcPtr, dstPtr,
                                                                              static_cast<uint64_t>(dst), bytes,
                                                                              crossCtx_.qp, ubRdmaLocal_, ubRdmaHead_);
                                if (g == 0) {
                                    dbg[160 + coreIdx * 4 + 3] = postOk ? 1 : 2;
                                    AscendC::GlobalTensor<int32_t> dg1;
                                    dg1.SetGlobalBuffer(dbg + 160 + coreIdx * 4);
                                    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                                                      AscendC::DcciDst::CACHELINE_OUT>(dg1);
                                }
                                if (!postOk) {
                                    ++xsPushFails;
                                    if (xsPushFirst == 0) {
                                        xsPushFirst = dst * params.expertPerRank + g + 1;
                                        xsPushSrcRow = static_cast<int32_t>(srcRow);
                                        xsPushDstRow = static_cast<int32_t>(dstRowStart);
                                    }
                                }
                            }
                        }
                        // Every cell posts a flag: 1 = data pushed, 2 = skipped.
                            if (dst != params.rank) {
                                const int64_t markerOff = dispatchStageOff + (postData ? 0 : 32);
                                const int64_t slotOff = dispatchFlagOff +
                                    static_cast<int64_t>(params.rank * params.expertPerRank + g) * 64;
                                if (crossCtx_.CrossServer(dst)) {
                                    if (!DffcA2::AivRdmaPostSendTry(
                                            crossCtx_.LocalOut() + markerOff,
                                            crossCtx_.In(dst) + slotOff,
                                            static_cast<uint64_t>(dst), 32, crossCtx_.qp, ubRdmaLocal_,
                                            ubRdmaHead_)) {
                                        ++xsFlagFails;
                                    }
                                } else {
                                    __gm__ int32_t *pf2 = reinterpret_cast<__gm__ int32_t *>(
                                        shmem(slotOff, dst));
                                    gm_dcci(reinterpret_cast<__gm__ uint8_t *>(pf2));
                                    *pf2 = postData ? 1 : 2;
                                    gm_dcci(reinterpret_cast<__gm__ uint8_t *>(pf2));
                                }
                            }
                    }
                    // Advance along dst's group bases (its own totals) and this
                    // rank's sorted outbound layout.
                    dstBase += static_cast<int64_t>(dstCumUb.GetValue(
                        (params.EP - 1) * xsAlignedEpr + g));
                    srcRow += rowsAll;
                }
            }
            AscendC::SyncAll<true>();
            if (coreIdx == 0) {
                __gm__ int32_t *p0 = reinterpret_cast<__gm__ int32_t *>(params.ptrExpertTokenNums);
                for (int32_t i = 2; i < 8; ++i) {
                    p0[i] = 0;
                }
                AscendC::GlobalTensor<int32_t> zGm;
                zGm.SetGlobalBuffer(p0);
                AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                                  AscendC::DcciDst::CACHELINE_OUT>(zGm);
            }
            AscendC::SyncAll<true>();
        }
#ifdef DFFC_XS_STAGE3
        {
            __gm__ int32_t *p3 = reinterpret_cast<__gm__ int32_t *>(params.ptrExpertTokenNums);
            p3[0] = params.rank;
            p3[1] = 13;
            p3[2] = xsPushFails;
            p3[3] = xsFlagFails;
            p3[4] = xsPushFirst;
            p3[5] = xsPushSrcRow;
            p3[6] = xsPushDstRow;
            p3[7] = xsTpeFails_ * 100 + xsTpePollFails_;
            AscendC::GlobalTensor<int32_t> p3Gm;
            p3Gm.SetGlobalBuffer(p3);
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(p3Gm);
        }
        AscendC::SyncAll<true>();
        return;
#endif

        uint32_t prevGroupSum1 = 0;
        int32_t xsWaitFails = 0;
        int32_t xsFirstFail = 0;
        nSyncSwiglu = 0;
        dequantSum[0] = 0;
        uint32_t dequantSumTemp = 0;
        icache_preload(8);
        for (int32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            // The ith core reads data from the ith rank's peermem
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            for (int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
                uint32_t rowStart =
                    (dstEpIdx == 0 ? 0 : cumsumMM((dstEpIdx - 1) * params.expertPerRank + groupIdx)) + prevGroupSum1;
                if (rowStart < params.maxOutputSize) {
                    uint32_t rows = tokenPerExpert(tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
                    if (rowStart + rows > params.maxOutputSize) {
                        rows = params.maxOutputSize - rowStart;
                    }
                    MatrixCoord offsetA{rowStart, 0};
                    int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
                    if (crossNode) {
                        if (rows == 0 && dstEpIdx == params.rank) {
                            continue;
                        }
                        GM_ADDR srcBase;
                        if (dstEpIdx == params.rank) {
                            // Self cell was copied Out -> In by the push pass.
                            srcBase = crossCtx_.LocalIn() + peermemInfo.offsetA;
                        } else {
                            AscendC::LocalTensor<int32_t> flagUb =
                                resource.ubBuf.template GetBufferByByte<int32_t>(ArchTag::UB_SIZE - 4096 + 64);
                            flagUb.SetSize(8);
                            const int64_t slot = static_cast<int64_t>(dstEpIdx * params.expertPerRank + groupIdx) * 64;
                            GM_ADDR flagAddr = crossCtx_.LocalIn() + dispatchFlagOff + slot;
                            const int32_t fv = WaitFlagValue(flagAddr, flagUb, 1ULL * 1000ULL * 1000ULL * 50ULL);
                            // Reset before acting, including the skip path:
                            // the flag lives in persistent window memory and
                            // the next call polls for "non-zero", so a leftover
                            // 2 could make a later call skip a cell that
                            // actually carries data.
                            {
                                __gm__ int32_t *fp = reinterpret_cast<__gm__ int32_t *>(flagAddr);
                                gm_dcci(reinterpret_cast<__gm__ uint8_t *>(fp));
                                *fp = 0;
                                gm_dcci(reinterpret_cast<__gm__ uint8_t *>(fp));
                            }
                            if (fv == 2) {
                                continue; // sender skipped this cell
                            }
                            if (fv != 1) {
                                // 1s-class timeout: the sender's flag never
                                // landed. Converting zeros would silently
                                // corrupt every output row of this cell —
                                // fail loudly instead.
                                ++xsWaitFails;
                                if (xsFirstFail == 0) {
                                    xsFirstFail = dstEpIdx * params.expertPerRank + groupIdx + 1;
                                }
                                AscendC::Trap();
                            }
                            srcBase = crossCtx_.LocalIn() + peermemInfo.offsetA;
                        }
                        AscendC::GlobalTensor<ElementABefore> gmSrcA;
                        gmSrcA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementABefore *>(srcBase));
                        int64_t srcOffset = static_cast<int64_t>(rowStart) * (params.problemShape.k() + ALIGN_512);
                        FetchAndPreprocessInt8ToInt4(static_cast<uint64_t>(gmOffsetA), gmPerTokenScale1[rowStart],
                                                     gmSrcA[srcOffset], rows, params.problemShape.k());
                    } else {
                        uint32_t rowSrc = prevSum;
                        prevSum += rows;
                        GM_ADDR otherRankPtr = shmem(0, dstEpIdx);
                        AscendC::GlobalTensor<ElementABefore> gmRemoteA;
                        gmRemoteA.SetGlobalBuffer(
                            reinterpret_cast<__gm__ ElementABefore *>(otherRankPtr + peermemInfo.offsetA));
                        int64_t gmOffsetPeer = rowSrc * (params.problemShape.k() + ALIGN_512);
                        FetchAndPreprocessInt8ToInt4(static_cast<uint64_t>(gmOffsetA), gmPerTokenScale1[rowStart],
                                                     gmRemoteA[gmOffsetPeer], rows, params.problemShape.k());
                    }
                }
            }
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmm1Idx++;

            prevGroupSum1 += currentM;

            if(dequantSumTemp + currentM <=params.maxOutputSize)
            {
                dequantSumTemp += currentM;   
            } else if (dequantSumTemp < params.maxOutputSize)
            {
                dequantSumTemp = params.maxOutputSize;
            }

            if(IsSyncTask(groupIdx, params.expertPerRank))
            {
                nSyncSwiglu++;
                dequantSum[nSyncSwiglu] = dequantSumTemp;
            }
        }
#ifdef SYNC_MODE
        AscendC::SyncAll<false>();
#endif

#ifdef DFFC_XS_STAGE2
        {
            __gm__ int32_t *pf = reinterpret_cast<__gm__ int32_t *>(params.ptrExpertTokenNums);
            pf[2] = xsPushFails;
            pf[3] = xsFlagFails;
            pf[4] = xsPushFirst;
            pf[5] = xsPushSrcRow;
            pf[6] = xsPushDstRow;
            pf[7] = xsTpeFails_ * 100 + xsTpePollFails_;
            AscendC::GlobalTensor<int32_t> fGm2;
            fGm2.SetGlobalBuffer(pf);
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(fGm2);
        }
        if (coreIdx == 0) {
            __gm__ int32_t *p = reinterpret_cast<__gm__ int32_t *>(params.ptrExpertTokenNums);
            p[0] = params.rank;
            p[1] = 12;
            AscendC::GlobalTensor<int32_t> st2Gm;
            st2Gm.SetGlobalBuffer(p);
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(st2Gm);
        }
        AscendC::SyncAll<true>();
        return;
#endif

        uint32_t n2 = params.problemShape.k();

        typename BlockEpilogue2::Params epilogueParams{
            static_cast<int32_t>(params.EP),
            static_cast<int32_t>(params.expertPerRank),
            static_cast<int32_t>(params.rank),
            reinterpret_cast<__gm__ int32_t *>(shmem() + peermemInfo.offsetPeerTokenPerExpert),
            params.layoutD2,
            static_cast<int32_t>(n2),
            static_cast<int32_t>(L1TileShape::N),
            shmem,
            static_cast<int32_t>(peermemInfo.offsetD),
            crossNode ? ptrStageBase_ : nullptr};

        uint32_t n = params.problemShape.n();
        // Row stride of the shared C workspace: GMM1 writes N cols/row, GMM2 writes n2=K
        // cols/row into the same max(N, n2)-wide region (see workspace carving above).
        uint32_t cRowStride = params.problemShape.n() > params.problemShape.k()
                                  ? params.problemShape.n()
                                  : params.problemShape.k();
        BlockEpilogue2 blockEpilogue2(resource, epilogueParams);
        BlockEpilogue1 blockEpilogue1(resource, n);
#ifdef SYNC_MODE
        AscendC::SyncAll<false>();
#endif
        for(int32_t syncIdx = 0; syncIdx < nSyncSwiglu; syncIdx++)
        {
            AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V);
            AscendC::SyncAll<true>();
            uint32_t curRowNum = dequantSum[syncIdx+1] - dequantSum[syncIdx];
            if(curRowNum > 0)
            {
                uint32_t rowStartThisCore = dequantSum[syncIdx];
                MatrixCoord offsetC{rowStartThisCore, 0};
                MatrixCoord shapeC{curRowNum, params.problemShape.n()};
                LayoutC layoutC{curRowNum, params.problemShape.n(), cRowStride};
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
                if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                    blockEpilogue1(gmC[gmOffsetC * 2], shapeC, gmPerTokenScale1[rowStartThisCore], params.ptrMAux1,
                                    gmA2I4_I8[gmOffsetD], cumsumMM, rowStartThisCore, gmPerTokenScale2[rowStartThisCore],
                                    params.expertPerRank, params.EP, gmCGMM1[gmOffsetC], params.rank, params.listLen, resource,
                                    params.epilogueCoreNum, params.swigluLimit, cRowStride);
                }
            }
            AscendC::SyncAll<true>();
            // Synchronization signal: SwiGLU notifies GMM2
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C);
        }
        blockEpilogue1.Finalize();
#ifdef SYNC_MODE
        AscendC::SyncAll<false>();
        AscendC::SyncAll<false>();
#endif

        CombineV2(params, blockEpilogue2);
        AscendC::SyncAll<true>();
#ifdef DFFC_XS_STAGE4
        XsStage4Mark(params, 13);   // reached: combine epilogue done
#endif
        PushDCrossServer(params);
        AscendC::SyncAll<true>();
#ifdef DFFC_XS_STAGE4
        XsStage4Mark(params, 14);   // reached: D push posted
        XsStage4Dump(params);
        AscendC::SyncAll<true>();
        return;
#endif
        ResetTokenPerExpert(params.EP * paddedExpertNumAligned);
        shmem.InitStatusTargetSum();
        if (get_subblockid() == 0) {
            AscendC::LocalTensor<int32_t> ctrBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
            shmem.CrossRankSyncV2Set(ctrBuffer);
#ifdef DFFC_XS_SELFTEST
            if (coreIdx == 0) {
                AscendC::printf("[xs] rank=%d v2set-done\n", params.rank);
            }
#endif
        } else {
            uint32_t uboffset = 0;
            uint32_t aicCoreNum = coreNum / 2;
            uint32_t aicCoreIdx = get_block_idx();
            uint32_t sendRankNum_ = params.EP / aicCoreNum;
            uint32_t remainderRankNum = params.EP % aicCoreNum;
            if (aicCoreIdx < remainderRankNum) {
                sendRankNum_++;
            }
            AscendC::LocalTensor<float> statusTensor = resource.ubBuf.template GetBufferByByte<float>(uboffset);
            uboffset += sendRankNum_ * UB_ALIGN;
            AscendC::LocalTensor<float> gatherMaskOutTensor = resource.ubBuf.template GetBufferByByte<float>(uboffset);
            uboffset += AlignUp(params.EP * sizeof(float), 32);
            AscendC::LocalTensor<uint32_t> gatherTmpTensor =
                resource.ubBuf.template GetBufferByByte<uint32_t>(uboffset);
            uboffset += AlignUp(sizeof(uint32_t), 32);
            AscendC::LocalTensor<float> statusSumOutTensor = resource.ubBuf.template GetBufferByByte<float>(uboffset);
            uboffset += AlignUp(sizeof(float), 32);
            shmem.CrossRankSyncV2Wait(statusTensor, gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
            MoeTokenUnpermuteTilingData tilingData;
            MoeTokenUnpermuteTiling(params.problemShape.m() * params.topK, n2, params.topK, tilingData, coreNum / 2);
            KernelMoeTokenUnpermute<ElementD2, int32_t, float, true> kernelMoeTokenUnpermuteOp;
            kernelMoeTokenUnpermuteOp.Init(shmem() + peermemInfo.offsetD, workspaceInfo.expandedRowIdx, params.probs,
                                           reinterpret_cast<GM_ADDR>(params.ptrOutput), &tilingData);
#ifndef DFFC_XS_STAGE5
            kernelMoeTokenUnpermuteOp.Process();
#endif
        }
    }

    CATLASS_DEVICE
    void CombineV2(Params const &params, BlockEpilogue2 &blockEpilogue)
    {
        blockEpilogue.InitFlag();
        BlockScheduler blockScheduler;
        int32_t syncLoopIdx = 0;
        uint32_t startCoreIdx = 0;
        uint32_t aicCoreNum = coreNum / 2;
        uint32_t aicCoreIdx = get_block_idx();
        uint32_t aivSubCoreIdx = get_subblockid();
        uint32_t preSrcExpertSum = 0;
        uint32_t n2 = params.problemShape.k();
        uint32_t k2 = params.problemShape.n() / 2;

        int64_t gmGroupOffsetC = 0;
        uint32_t aivCoreNum = coreNum;
        uint32_t aivCoreIdx = coreIdx;

        icache_preload(8);
        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentExpertM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preSrcExpertSum >= params.maxOutputSize) {
                currentExpertM = 0;
            } else if (preSrcExpertSum + currentExpertM > params.maxOutputSize) {
                currentExpertM = params.maxOutputSize - preSrcExpertSum;
            }

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                currentExpertM = currentExpertM * 2;
            }
            GemmCoord inGroupProblemShape{currentExpertM, n2, k2};  // M N K
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            uint32_t startLoopIdx =
                ((aicCoreIdx < startCoreIdx) ? (aicCoreIdx + aicCoreNum) : aicCoreIdx) - startCoreIdx;

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicCoreNum) {
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                int32_t m0 = 32;
                int32_t m_rows = (actualBlockShape.m() + m0 - 1) /
                                 m0;
                int32_t aiv_m_rows = m_rows / 2;
                if (aivSubCoreIdx == 1 && aiv_m_rows * 2 < m_rows) {
                    aiv_m_rows += 1;
                }
                uint32_t m_offset = blockCoord.m() * L1TileShape::M;  // blockOffset
                if (aivSubCoreIdx == 1) {
                    m_offset += (m_rows / 2) * m0;
                }

                for (; syncLoopIdx <= groupIdx; syncLoopIdx++) {
                    int32_t flag_id = syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
                    AscendC::CrossCoreWaitFlag<0x2>(flag_id);
                }

                for (int32_t cur_row = 0; cur_row < aiv_m_rows; cur_row++) {
                    GemmCoord realTileCoord{m_offset, blockCoord.n() * L1TileShape::N, 1};
                    uint32_t actualm = m0;
                    if (aivSubCoreIdx == 1 && cur_row == aiv_m_rows - 1) {
                        actualm = actualBlockShape.m() - (m_rows / 2) * m0 - cur_row * m0;
                    }
                    GemmCoord realTileShape{actualm, actualBlockShape.n(), 1};
                    blockEpilogue(gmCGMM2, gmC2, gmPerTokenScale2, params.ptrMAux2, realTileCoord, realTileShape, groupIdx,
                                  preSrcExpertSum, preSumBeforeRank, params.listLen);
                    m_offset += m0;
                }
            }

            preSrcExpertSum += currentExpertM;
            startCoreIdx = (startCoreIdx + coreLoops) % aicCoreNum;
        }

        for (; syncLoopIdx < params.expertPerRank; syncLoopIdx++) {
            int32_t flag_id = syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
            AscendC::CrossCoreWaitFlag<0x2>(flag_id);
        }
        blockEpilogue.Finalize();
    }

    // ---- Combine D push (cross-server only) ----
    // The combine epilogue staged the rows of every cross-server destination in
    // this rank's Out window at exactly the row offsets that destination reads
    // them from in its own In window. Each core owns one destination rank and
    // posts one AI-RDMA WRITE per (dst, group) cell (source LOCAL_OUTPUT ->
    // peer REMOTE_INPUT). Self / same-server rows were written in place by the
    // epilogue (outAt), and the V2 status posted later rides the same QP, so a
    // destination may read its D region right after CrossRankSyncV2Wait.
    CATLASS_DEVICE
    void PushDCrossServer(Params const &params)
    {
        if (!(crossCtx_.ready && params.EP > 8) || coreIdx >= params.EP) {
            return;
        }
        const int32_t dst = static_cast<int32_t>(coreIdx);
        if (!crossCtx_.CrossServer(dst)) {
            return;
        }
        const int64_t rowBytes = static_cast<int64_t>(params.problemShape.k()) *
                                 static_cast<int64_t>(sizeof(ElementD2));
        GM_ADDR srcBase = crossCtx_.LocalOut() + peermemInfo.offsetD;
        GM_ADDR dstBase = crossCtx_.In(dst) + peermemInfo.offsetD;
        // Source row walks the packed staging mirror; the destination row is
        // the same one the epilogue computed for dst's own D region.
        int64_t stageRow = static_cast<int64_t>(ptrStageBase_[dst]);
        for (int32_t g = 0; g < params.expertPerRank; ++g) {
            const int32_t rows = static_cast<int32_t>(
                tokenPerExpert(tokenPerExpertLayout(dst, params.rank, g)));
            if (rows <= 0) {
                continue;
            }
            const int64_t dstRowStart = static_cast<int64_t>(
                preSumBeforeRank(dst * params.expertPerRank + g));
            int64_t pushRows = static_cast<int64_t>(rows);
            if (dstRowStart >= static_cast<int64_t>(params.maxOutputSize)) {
                stageRow += rows;
                continue;
            }
            if (dstRowStart + pushRows > static_cast<int64_t>(params.maxOutputSize)) {
                pushRows = static_cast<int64_t>(params.maxOutputSize) - dstRowStart;
            }
            if (pushRows > 0) {
                DffcA2::AivRdmaPostSend(srcBase + stageRow * rowBytes,
                                        dstBase + dstRowStart * rowBytes,
                                        static_cast<uint64_t>(dst),
                                        static_cast<uint64_t>(pushRows) * static_cast<uint64_t>(rowBytes),
                                        crossCtx_.qp, ubRdmaLocal_, ubRdmaHead_);
                ++xsDPushCnt_;
            }
            stageRow += rows;
        }
    }

#ifdef DFFC_XS_STAGE4
    // Diagnostics (temporary): stage marks and a staging sample via the output
    // tensor, since kernel printf does not surface on this setup.
    CATLASS_DEVICE
    void XsStage4Mark(Params const &params, int32_t stage)
    {
        if (coreIdx != 0) {
            return;
        }
        __gm__ int32_t *p = reinterpret_cast<__gm__ int32_t *>(params.ptrOutput);
        p[0] = params.rank;
        p[1] = stage;
        AscendC::GlobalTensor<int32_t> g;
        g.SetGlobalBuffer(p);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(g);
    }

    CATLASS_DEVICE
    void XsStage4Dump(Params const &params)
    {
        __gm__ int32_t *p = reinterpret_cast<__gm__ int32_t *>(params.ptrOutput);
        for (int32_t d = 0; d < 4; ++d) {
            p[4 + d] = ptrStageBase_[d];
        }
        const int64_t rowBytes = static_cast<int64_t>(params.problemShape.k()) *
                                 static_cast<int64_t>(sizeof(ElementD2));
        int32_t firstXs = -1;
        for (int32_t d = 0; d < static_cast<int32_t>(params.EP); ++d) {
            if (crossCtx_.CrossServer(d)) {
                firstXs = d;
                break;
            }
        }
        if (firstXs >= 0) {
            __gm__ int32_t *src = reinterpret_cast<__gm__ int32_t *>(
                crossCtx_.LocalOut() + peermemInfo.offsetD +
                static_cast<int64_t>(ptrStageBase_[firstXs]) * rowBytes);
            p[8] = firstXs;
            p[9] = ptrStageBase_[firstXs];
            for (int32_t i = 0; i < 4; ++i) {
                p[10 + i] = src[i];
            }
        }
        AscendC::GlobalTensor<int32_t> g;
        g.SetGlobalBuffer(p);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(g);
        if (coreIdx < params.EP) {
            __gm__ int32_t *q = p + 64 + coreIdx * 16;
            q[0] = static_cast<int32_t>(coreIdx);
            q[1] = xsDPushCnt_;
            AscendC::GlobalTensor<int32_t> gq;
            gq.SetGlobalBuffer(q);
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(gq);
        }
    }
#endif

private:
    struct WorkspaceInfo {
        // GM_ADDR ptrA;
        GM_ADDR ptrPerTokenScale;
        GM_ADDR ptrcumsumMM;
        GM_ADDR ptrC;
        GM_ADDR ptrC2;
        // GM_ADDR ptrPermutedToken;
        GM_ADDR ptrPerTokenScale2;
        GM_ADDR expandedRowIdx;
        GM_ADDR ptrTokenPerExpert;
        GM_ADDR ptrA1Int4;
        GM_ADDR ptrA2Int4;
        GM_ADDR ptrCGMM1;
        GM_ADDR ptrCGMM2;
        GM_ADDR ptrSumBeforeRank;
#ifdef ENABLE_TIMER
        GM_ADDR ptrTimer;
#endif
        __gm__ float *ptrSoftFlagBase;

        CATLASS_DEVICE
        WorkspaceInfo() {}

        CATLASS_DEVICE
        WorkspaceInfo(const Params &params)
        {
            uint32_t k2 = params.problemShape.n() / 2;
            uint32_t n2 = params.problemShape.k();
            int64_t workspaceOffset = 0;
            expandedRowIdx = params.ptrWorkspace;

            workspaceOffset += AlignUp(params.problemShape.m(), 256) * params.topK * sizeof(int32_t);
            ptrcumsumMM = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);
            ptrPerTokenScale = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            ptrPerTokenScale2 = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            ptrTokenPerExpert = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);
            // GMM1 output (ptrC) and GMM2 output (ptrC2) alias — their lifetimes are separated by SYNCFLAGC2V barrier
            ptrC = params.ptrWorkspace + workspaceOffset;
            ptrC2 = params.ptrWorkspace + workspaceOffset;

            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                workspaceOffset += static_cast<int64_t>(params.maxOutputSize) * (static_cast<int64_t>(params.problemShape.n()) > static_cast<int64_t>(n2) ? static_cast<int64_t>(params.problemShape.n()) : static_cast<int64_t>(n2)) * sizeof(ElementC) * 2;
            } else {
                workspaceOffset += static_cast<int64_t>(params.maxOutputSize) * (static_cast<int64_t>(params.problemShape.n()) > static_cast<int64_t>(n2) ? static_cast<int64_t>(params.problemShape.n()) : static_cast<int64_t>(n2)) * sizeof(ElementC);
            }
            // ptrA = params.ptrWorkspace + workspaceOffset;  // 9

            // workspaceOffset += params.maxOutputSize * params.problemShape.k() * sizeof(ElementABefore);
            // ptrPermutedToken = params.ptrWorkspace + workspaceOffset;  // 10

            // workspaceOffset += params.maxOutputSize * k2 * sizeof(ElementABefore);
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                // A1Int4 and A2Int4 alias reuse (same pattern as W8A8 ptrA/ptrPermutedToken):
                // lifetimes separated by SyncAll barrier after GMM1 loop — GMM1 reads A1Int4 complete
                // before SwiGLU epilogue writes A2Int4
                ptrA1Int4 = params.ptrWorkspace + workspaceOffset;
                ptrA2Int4 = params.ptrWorkspace + workspaceOffset;

                workspaceOffset += params.maxOutputSize * (params.problemShape.k() > k2 ? params.problemShape.k() : k2);

                ptrCGMM1 = params.ptrWorkspace + workspaceOffset;
                ptrCGMM2 = params.ptrWorkspace + workspaceOffset;
            }
            ptrSumBeforeRank = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += params.EP * sizeof(int32_t) * params.expertPerRank;

#ifdef SOFT_SYNC
            ptrSoftFlagBase = reinterpret_cast<__gm__ float *>(params.ptrWorkspace + workspaceOffset);
            workspaceOffset += params.EP * sizeof(int32_t) * FLAGSTRIDE;
#endif

#ifdef ENABLE_TIMER
            ptrTimer = params.ptrWorkspace + workspaceOffset;
            workspaceOffset += AscendTimer::N_TIMING_COUNTER * sizeof(int64_t);
#endif
        }
    };

    struct PeermemInfo {
        int64_t offsetA;
        int64_t offsetPeerPerTokenScale;
        int64_t offsetPeerTokenPerExpert;
        int64_t offsetD;

        CATLASS_DEVICE
        PeermemInfo() {}

        CATLASS_DEVICE
        PeermemInfo(const Params &params, int64_t segmentSize)
        {
            offsetA = 0;
            offsetPeerPerTokenScale = offsetA + AlignUp(segmentSize / 3, 512);
            offsetD = offsetPeerPerTokenScale + MB_SIZE;
            offsetPeerTokenPerExpert = segmentSize - 2 * MB_SIZE;
        }
    };

    Arch::Resource<ArchTag> resource;

    uint32_t coreIdx;
    uint32_t coreNum;
    uint32_t nSyncSwiglu;

    Params params;
    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;

    int64_t m_prevSumBeforeRank;
    uint32_t dequantSum[16] = {0};

    // AscendC::GlobalTensor<ElementABefore> gmA;
    AscendC::GlobalTensor<ElementC> gmC;
    AscendC::GlobalTensor<ElementScale> gmS;

    AscendC::GlobalTensor<int4b_t> gmA1I4;
    AscendC::GlobalTensor<int8_t> gmA1I4_I8;
    AscendC::GlobalTensor<int8_t> gmB1_I8;
    AscendC::GlobalTensor<int4b_t> gmA2I4;
    AscendC::GlobalTensor<int8_t> gmA2I4_I8;
    AscendC::GlobalTensor<float> gmCGMM1;
    AscendC::GlobalTensor<float> gmCGMM2;

    // AscendC::GlobalTensor<ElementD1> gmPermutedToken;
    AscendC::GlobalTensor<ElementScale> gmS2;
    AscendC::GlobalTensor<ElementC> gmC2;

    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale1;
    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale2;

    AscendC::GlobalTensor<bool> gmXActiveMask;

    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    AscendC::GlobalTensor<int32_t> cumsumMM;
    AscendC::GlobalTensor<int32_t> preSumBeforeRank;
    Layout3D tokenPerExpertLayout;
    HcclShmem shmem;
    int32_t paddedExpertNumAligned;

    // A2 cross-server (KFC + AI-RDMA) transport.
    DffcA2::CrossServerCtx crossCtx_;
    // Cross-node D staging base per destination (workspace, EP int32s).
    __gm__ int32_t *ptrStageBase_{nullptr};
    // Cross-node combine D posts issued by this core (diagnostics).
    int32_t xsDPushCnt_{0};
    // Diagnostic counters (cross-server bring-up).
    int32_t xsTpeFails_{0};
    int32_t xsTpePollFails_{0};
    AscendC::LocalTensor<uint64_t> ubRdmaLocal_;
    AscendC::LocalTensor<uint32_t> ubRdmaHead_;
};

}  // namespace Catlass::Gemm::Kernel

#endif // DISPATCH_FFN_COMBINE_W4A8_KERNEL_HPP