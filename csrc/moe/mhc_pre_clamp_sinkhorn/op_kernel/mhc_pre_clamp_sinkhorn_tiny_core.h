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
 * \file mhc_pre_clamp_sinkhorn_tiny_core.h
 * \brief key=2 tiny 路径：小 batch（decode，bs<=32）推理专用精简 kernel。
 *
 * 与 key=1（MembaseKSplitCorePart1/Part2）同源的数学与 workspace 布局，结构差异：
 *  - blockDim = 8 AIC + 16 AIV（key=1 为 20+40 全量 launch）：砍固定 launch/同步税
 *  - 每核单外层 K 轮（cvLoopKSize == splitKSize == Kslice）：l1a/l1b 用作单 256KB 大槽，
 *    整个 A/B 切片一次性驻留 L1，从结构上绕开 key=1 的多轮双缓冲事件协议（B<=8 挂死区）
 *  - 内层 MMAD 块 K_L0_TINY=256（key=1 为 32）：Kslice=1792 时块数 56->7，砍 AIC scalar
 *  - Part2 fork 删除全部 needGrad 分支（路由保证推理路径）
 * 输出契约（WS 布局 mmOut[kcore][bs][128] / squareSum[kcore][align16(bs)][16]）与 key=1
 * 完全一致，Part2 的合并/epilogue 数学逐行同源。
 */

#ifndef MHC_PRE_SINKHORN_TINY_CORE_H
#define MHC_PRE_SINKHORN_TINY_CORE_H

#include "kernel_operator.h"
#include "mhc_pre_clamp_sinkhorn_base.h"
#include <cstdint>

using AscendC::BLOCK_CUBE;

namespace MhcPreClampSinkhorn {
using namespace AscendC;

struct TinyMmParams {
    uint64_t curML1;      // 本轮 M 行数（<= 32）
    uint64_t curNL1;      // N = hcMix = 24
    uint64_t singleCoreK; // 本核 K 切片长度（<= cvLoopKSize，构造上恒单轮）
    uint64_t xWsKSize;    // cvLoopKSize（x cast WS 行距）
    uint64_t nOutSize;    // mmOut WS 行距（128，512B 对齐）
    uint64_t kGmSize;     // K 全长
};

// tiny 专用 cube 计算：单 256KB L1 槽 + 单轮 + K_L0_TINY 内层块
class MhcPreClampSinkhornTinyCubeCompute {
public:
    __aicore__ inline MhcPreClampSinkhornTinyCubeCompute() {}

    // L1 槽尺寸由 tiling 派生（不用硬件常量）：A NZ = align16(mL1Size)*cvLoopKSize*4B、
    // B NZ = align16(hcMix)*cvLoopKSize*4B——与 CopyInA1/CopyInB1 的 Nd2Nz dst 范围
    // 逐字段同源（dstNzC0Stride*dValue），也与 host 侧 gate 预算公式（tiling.cpp 的
    // aNzRows/bNzRows * tinyKsize）一致：单一事实来源，kernel 申请量恒 <= gate 校验量。
    // 说明：kernel（device 代码）无平台查询 API（GetCoreMemSize 仅 host 侧），
    // runtime 的平台自适应点在 host gate；此处尺寸跟随 tiling 即跟随已 gate 的形状。
    __aicore__ inline void Init(const GlobalTensor<float> &phiGm, TPipe *tpipe,
                                const MhcPreClampSinkhornTilingData *tilingData)
    {
        phiGm_ = phiGm;

        uint64_t l1aBytes = static_cast<uint64_t>(CeilAlign(tilingData->mL1Size, BLOCK_CUBE)) *
                            static_cast<uint64_t>(tilingData->cvLoopKSize) * sizeof(float);
        uint64_t l1bBytes = static_cast<uint64_t>(CeilAlign(tilingData->hcMix, BLOCK_CUBE)) *
                            static_cast<uint64_t>(tilingData->cvLoopKSize) * sizeof(float);
        TBuf<TPosition::A1> l1aBuffer;
        tpipe->InitBuffer(l1aBuffer, static_cast<uint32_t>(l1aBytes));
        l1a_ = l1aBuffer.Get<float>();

        TBuf<TPosition::B1> l1bBuffer;
        tpipe->InitBuffer(l1bBuffer, static_cast<uint32_t>(l1bBytes));
        l1b_ = l1bBuffer.Get<float>();

        TBuf<TPosition::A2> l0aBuffer;
        tpipe->InitBuffer(l0aBuffer, L0_TINY_BUF_SIZE);
        l0a_ = l0aBuffer.Get<float>();

        TBuf<TPosition::B2> l0bBuffer;
        tpipe->InitBuffer(l0bBuffer, L0_TINY_BUF_SIZE);
        l0b_ = l0bBuffer.Get<float>();

        TBuf<TPosition::CO1> l0cBuffer;
        tpipe->InitBuffer(l0cBuffer, L0_TINY_BUF_SIZE);
        l0c_ = l0cBuffer.Get<float>();

        // 预置全部 L0/L1 槽位事件（空闲可用）；tiny 单轮只消费 slot0，slot1 的预置由
        // End() 回收——与 key=1 单轮路径同款记账（每 Set 必有匹配 Wait，无孤儿等待）
        for (int i = 0; i < L0A_BUF_NUM; i++) {
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + i);
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + i);
        }
        for (int i = 0; i < L1_BUF_NUM; i++) {
            SetFlag<HardEvent::MTE1_MTE2>(X_MTE1_MTE2_EVENT + i);
            SetFlag<HardEvent::MTE1_MTE2>(B_MTE1_MTE2_EVENT + i);
        }
    }

    // B 切片入 L1（phi 跨距列采集，Nd2Nz；单槽 l1b_[0]）
    __aicore__ inline void CopyInB1(uint64_t kGmOffset, uint64_t kL1Size, const TinyMmParams &mmParams)
    {
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = mmParams.curNL1;
        nd2nzParams.dValue = kL1Size;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = mmParams.kGmSize;
        nd2nzParams.dstNzC0Stride = (mmParams.curNL1 + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        DataCopy(l1b_[0], phiGm_[kGmOffset], nd2nzParams);
    }

    // A 切片入 L1（xCast WS，行距 xWsKSize；单槽 l1a_[0]）
    __aicore__ inline void CopyInA1(uint64_t kGmOffset, uint64_t kL1Size, const TinyMmParams &mmParams)
    {
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = mmParams.curML1;
        nd2nzParams.dValue = kL1Size;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = mmParams.xWsKSize;
        nd2nzParams.dstNzC0Stride = (mmParams.curML1 + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        DataCopy(l1a_[0], xCastGm_[kGmOffset], nd2nzParams);
    }

    // 单轮计算：A/B 已在 L1（CopyInB1 由外部先行），逐 K_L0_TINY 块 Mmad 累加后 Fixp 出 WS。
    // xCastGm_: 本轮 A 切片起点（WS 单槽 slot0）；wsA2/wsAB: squareSum / mmOut partial 落点
    __aicore__ inline void ComputeOnce(const GlobalTensor<float> &xCastGm,
                                       const GlobalTensor<float> &wsA2, const GlobalTensor<float> &wsAB,
                                       const TinyMmParams &mmParams)
    {
        xCastGm_ = xCastGm;
        // 等 l1a 槽空闲（Init 预置），随后 A 入 L1；MTE1 按序完成 B/A 两笔搬运后置
        // MTE2_MTE1 事件，MTE2 的 L0 装载统一等它（一笔事件覆盖 B+A 就绪，与 key=1 同款）
        WaitFlag<HardEvent::MTE1_MTE2>(X_MTE1_MTE2_EVENT);
        CopyInA1(0, mmParams.singleCoreK, mmParams);
        SetFlag<HardEvent::MTE2_MTE1>(MM1_MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MM1_MTE2_MTE1_EVENT);

        const uint64_t kTail = (mmParams.singleCoreK % K_L0_TINY == 0)
                                   ? K_L0_TINY
                                   : (mmParams.singleCoreK % K_L0_TINY);
        for (uint64_t kL1Offset = 0; kL1Offset < mmParams.singleCoreK; kL1Offset += K_L0_TINY) {
            const uint64_t kL0Size = (kL1Offset + K_L0_TINY <= mmParams.singleCoreK) ? K_L0_TINY : kTail;
            const bool isLastChunk = kL1Offset + kL0Size >= mmParams.singleCoreK;
            const bool isFirstChunk = kL1Offset == 0;

            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + l0aLoopIdx_ % L0A_BUF_NUM);
            LoadAToL0A(kL1Offset, kL0Size, mmParams);

            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            LoadAToL0B(kL1Offset, kL0Size, mmParams);
            MmadA2(kL0Size, isFirstChunk, isLastChunk, mmParams);
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            l0bLoopIdx_++;

            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            LoadBToL0B(kL1Offset, kL0Size, mmParams);
            MmadAB(kL0Size, isFirstChunk, isLastChunk, mmParams);
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            l0bLoopIdx_++;

            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + l0aLoopIdx_ % L0A_BUF_NUM);
            l0aLoopIdx_++;
        }
        // l1a 单槽本轮用完，标记空闲（End() 回收记账）
        SetFlag<HardEvent::MTE1_MTE2>(X_MTE1_MTE2_EVENT);
        Fixp(wsA2, wsAB, mmParams);
    }

    __aicore__ inline void WaitBL1Ready()
    {
        // l1b 槽空闲（Init 预置 / 上轮 SetBL1Free 回收）
        WaitFlag<HardEvent::MTE1_MTE2>(B_MTE1_MTE2_EVENT);
    }

    __aicore__ inline void SetBL1Free()
    {
        SetFlag<HardEvent::MTE1_MTE2>(B_MTE1_MTE2_EVENT);
        l1bLoopIdx_++;
    }

    __aicore__ inline void End()
    {
        for (int i = 0; i < L0A_BUF_NUM; i++) {
            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + i);
            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + i);
        }
        for (int i = 0; i < L1_BUF_NUM; i++) {
            WaitFlag<HardEvent::MTE1_MTE2>(X_MTE1_MTE2_EVENT + i);
            WaitFlag<HardEvent::MTE1_MTE2>(B_MTE1_MTE2_EVENT + i);
        }
    }

private:
    __aicore__ inline void LoadAToL0A(uint64_t kL1Offset, uint64_t kL0Size, const TinyMmParams &mmParams)
    {
        static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};
        LoadData3DParamsV2<float> loadData3DParams;
        loadData3DParams.l1H = CeilDiv(mmParams.curML1, BLOCK_CUBE);
        loadData3DParams.l1W = BLOCK_CUBE;
        loadData3DParams.channelSize = kL0Size;
        loadData3DParams.padList[0] = 0;
        loadData3DParams.padList[1] = 0;
        loadData3DParams.padList[2] = 0;
        loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗结果
        loadData3DParams.mExtension = CeilAlign(mmParams.curML1, BLOCK_CUBE);
        loadData3DParams.kExtension = kL0Size;
        loadData3DParams.mStartPt = 0;
        loadData3DParams.kStartPt = 0;
        loadData3DParams.strideW = 1;
        loadData3DParams.strideH = 1;
        loadData3DParams.filterW = 1;
        loadData3DParams.filterSizeW = (1 >> 8) & 255;
        loadData3DParams.filterH = 1;
        loadData3DParams.filterSizeH = (1 >> 8) & 255;
        loadData3DParams.dilationFilterW = 1;
        loadData3DParams.dilationFilterH = 1;
        loadData3DParams.enTranspose = 0;
        loadData3DParams.fMatrixCtrl = 0;
        LoadData<float, LOAD3DV2_CONFIG>(
            l0a_[(l0aLoopIdx_ % L0A_BUF_NUM) * L0AB_BUF_OFFSET],
            l1a_[CeilAlign(mmParams.curML1, static_cast<uint64_t>(BLOCK_CUBE)) * kL1Offset], loadData3DParams);
    }

    __aicore__ inline void LoadAToL0B(uint64_t kL1Offset, uint64_t kL0Size, const TinyMmParams &mmParams)
    {
        // mk nz -> m,k zz
        for (uint64_t mL0Offset = 0; mL0Offset < mmParams.curML1; mL0Offset += BLOCK_CUBE) {
            LoadData2DParams l1ToL0bParams;
            l1ToL0bParams.startIndex = 0;
            l1ToL0bParams.repeatTimes = CeilDiv(kL0Size, (uint64_t)BLOCK_CUBE >> 1);
            l1ToL0bParams.srcStride = CeilDiv(mmParams.curML1, (uint64_t)BLOCK_CUBE);
            l1ToL0bParams.dstGap = 0;
            LoadData(l0b_[(l0bLoopIdx_ % L0AB_BUF_NUM) * L0AB_BUF_OFFSET +
                          mL0Offset * CeilAlign(kL0Size, (uint64_t)BLOCK_CUBE >> 1)],
                l1a_[CeilAlign(mmParams.curML1, static_cast<uint64_t>(BLOCK_CUBE)) * kL1Offset +
                     mL0Offset * (uint64_t)(BLOCK_CUBE >> 1)],
                l1ToL0bParams);
        }
    }

    __aicore__ inline void LoadBToL0B(uint64_t kL1Offset, uint64_t kL0Size, const TinyMmParams &mmParams)
    {
        LoadData2DParams l1ToL0bParams;
        l1ToL0bParams.startIndex = 0;
        l1ToL0bParams.repeatTimes =
            CeilDiv(mmParams.curNL1, (uint64_t)BLOCK_CUBE) * CeilDiv(kL0Size, (uint64_t)BLOCK_CUBE >> 1);
        l1ToL0bParams.srcStride = 1;
        l1ToL0bParams.dstGap = 0;
        // n,k nz -> n,k nz（单槽 l1b_[0]）
        LoadData(l0b_[(l0bLoopIdx_ % L0AB_BUF_NUM) * L0AB_BUF_OFFSET],
            l1b_[CeilAlign(mmParams.curNL1, static_cast<uint64_t>(BLOCK_CUBE)) * kL1Offset], l1ToL0bParams);
    }

    __aicore__ inline void MmadA2(uint64_t kL0Size, bool isFirstChunk, bool isLastChunk, const TinyMmParams &mmParams)
    {
        SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
        WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
        for (uint64_t mL0Offset = 0; mL0Offset < mmParams.curML1; mL0Offset += BLOCK_CUBE) {
            MmadParams mmadParams;
            mmadParams.m = BLOCK_CUBE;
            mmadParams.n = BLOCK_CUBE;
            mmadParams.k = kL0Size;
            mmadParams.cmatrixInitVal = isFirstChunk;
            mmadParams.cmatrixSource = false;
            mmadParams.unitFlag = isLastChunk ? UNIT_FLAG_ENABLE_AUTO_CLOSE : UNIT_FLAG_ENABLE;
            Mmad(l0c_[mL0Offset * BLOCK_CUBE],
                l0a_[(l0aLoopIdx_ % L0A_BUF_NUM) * L0AB_BUF_OFFSET + mL0Offset * CeilAlign(kL0Size, 8)],
                l0b_[(l0bLoopIdx_ % L0AB_BUF_NUM) * L0AB_BUF_OFFSET + mL0Offset * CeilAlign(kL0Size, 8)],
                mmadParams);
        }
    }

    __aicore__ inline void MmadAB(uint64_t kL0Size, bool isFirstChunk, bool isLastChunk, const TinyMmParams &mmParams)
    {
        SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
        WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
        AscendC::SetHF32Mode(1);
        AscendC::SetHF32TransMode(1);
        MmadParams mmadParams;
        mmadParams.m = CeilAlign(mmParams.curML1, BLOCK_CUBE);
        mmadParams.n = mmParams.curNL1;
        mmadParams.k = kL0Size;
        mmadParams.cmatrixInitVal = isFirstChunk;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = isLastChunk ? UNIT_FLAG_ENABLE_AUTO_CLOSE : UNIT_FLAG_ENABLE;
        Mmad(l0c_[L0C_A2_BUF_OFFSET],
            l0a_[(l0aLoopIdx_ % L0A_BUF_NUM) * L0AB_BUF_OFFSET],
            l0b_[(l0bLoopIdx_ % L0AB_BUF_NUM) * L0AB_BUF_OFFSET],
            mmadParams);
        AscendC::SetHF32Mode(0);
    }

    __aicore__ inline void CopyOutC(const GlobalTensor<float> &workspaceGlobal, const LocalTensor<float> &c1Local,
                                    uint64_t baseM, uint64_t baseN, bool enableNz2Nd, uint64_t N)
    {
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.nSize = baseN;
        intriParams.mSize = baseM;
        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.nz2ndEn = enableNz2Nd;
        if (enableNz2Nd) {
            intriParams.dstStride = N; // NZ -> ND
            intriParams.srcStride = CeilAlign(baseM, AscendC::BLOCK_CUBE);
            AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        } else {
            intriParams.dstStride = CeilAlign(intriParams.nSize, AscendC::BLOCK_CUBE); // NZ -> NZ
            intriParams.srcStride = CeilAlign(baseM, AscendC::BLOCK_CUBE);
        }
        intriParams.unitFlag = UNIT_FLAG_ENABLE_AUTO_CLOSE;
        AscendC::DataCopy(workspaceGlobal, c1Local, intriParams);
    }

    __aicore__ inline void Fixp(const GlobalTensor<float> &wsA2, const GlobalTensor<float> &wsAB,
                                const TinyMmParams &mmParams)
    {
        // MmadA2 结果（Gram 对角线即每行平方和）：NZ->NZ [M,16]
        CopyOutC(wsA2, l0c_, mmParams.curML1, BLOCK_CUBE, false, BLOCK_CUBE);
        // MmadAB 结果（partial [M,24]）：NZ->ND，行距 nOutSize=128（512B 对齐）
        CopyOutC(wsAB, l0c_[L0C_A2_BUF_OFFSET], mmParams.curML1, mmParams.curNL1, true, mmParams.nOutSize);
    }

    GlobalTensor<float> phiGm_;
    GlobalTensor<float> xCastGm_;

    LocalTensor<float> l1a_;
    LocalTensor<float> l1b_;
    LocalTensor<float> l0a_;
    LocalTensor<float> l0b_;
    LocalTensor<float> l0c_;

    static constexpr uint64_t MM1_MTE2_MTE1_EVENT = 2;
    static constexpr uint64_t X_MTE1_MTE2_EVENT = 2;
    static constexpr uint64_t B_MTE1_MTE2_EVENT = 4;
    static constexpr uint64_t M_MTE1_EVENT_L0A = 3;
    static constexpr uint64_t M_MTE1_EVENT_L0B = 5;
    static constexpr uint64_t MTE1_M_EVENT = 2;

    static constexpr uint64_t L1_BUF_NUM = 2;
    static constexpr uint64_t L0A_BUF_NUM = 2;
    static constexpr uint64_t L0B_BUF_NUM = 2;
    static constexpr uint64_t L0AB_BUF_NUM = 2;

    static constexpr uint64_t L0AB_BUF_OFFSET = 32 * 256;
    static constexpr uint64_t L0C_A2_BUF_OFFSET = 256 * 16;

    static constexpr uint16_t UNIT_FLAG_ENABLE = 2;
    static constexpr uint16_t UNIT_FLAG_ENABLE_AUTO_CLOSE = 3;

    // L0 槽 64KB（l0a/l0b/l0c TBuf）：K_L0_TINY=256 时 B NZ 32*256*4B=32KB/槽，双槽
    // 轮转峰值 64KB。DAV_2201（910B/910_93 同架构）L0A/L0B=64KB、L0C=128KB 恒满足；
    // host 侧 gate（tiling.cpp TINY_L0_BUF_BYTES，与此前后两处同值）校验平台
    // L0A/L0B/L0C >= 64KB，不足即回退 key=1。L0 槽步进算术（L0AB_BUF_OFFSET）与
    // K_L0_TINY 联动、且 device 侧无平台查询——保留常量 + host gate 的契约形态。
    static constexpr uint64_t L0_TINY_BUF_SIZE = 64 * 1024;
    // 内层 MMAD 块大小：Kslice=1792 -> 7 块（key=1 的 32 -> 56 块是 AIC scalar 主因）。
    // L0B 32KB 槽约束 B NZ 32*K_L0*4B <= 32KB -> K_L0 <= 256，取上限
    static constexpr uint64_t K_L0_TINY = 256;

    uint64_t l1bLoopIdx_ = 0;
    uint64_t l0aLoopIdx_ = 0;
    uint64_t l0bLoopIdx_ = 0;
};

template <typename T>
class MhcPreClampSinkhornTinyPart1 {
public:
    __aicore__ inline MhcPreClampSinkhornTinyPart1() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcFn, GM_ADDR workspace,
                                const MhcPreClampSinkhornTilingData *tilingDataPtr, TPipe *pipePtr,
                                int64_t bsOffset, int64_t curBs)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T *)x + (bsOffset * tilingData->hcMult * tilingData->d));
        hcFnGm.SetGlobalBuffer((__gm__ float *)hcFn);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace);

        uint64_t curVectorBlockIdx = GetBlockIdx();
        uint64_t curCubeBlockIdx = curVectorBlockIdx;
        if ASCEND_IS_AIV {
            curCubeBlockIdx = curCubeBlockIdx / CV_RATIO;
        }
        // 与 key=1 同款 WS 布局（GetWorkspaceSize 的 kSplit 分支一致）：
        // [xCast 单核区(双缓冲) | mmOut partial | squareSum partial]
        xCastFp32BufSize_ = tilingData->mL1Size * CeilAlign(tilingData->cvLoopKSize, MM_CACHE_LINE_BYTES / sizeof(float)) * sizeof(float);
        int64_t singleCubeCoreXCastSize = xCastFp32BufSize_ * DOUBLE_BUFFER / sizeof(float);
        xCastFp32WsGm.SetGlobalBuffer((__gm__ float *)workspace + curCubeBlockIdx * singleCubeCoreXCastSize);
        uint64_t wsOffset = tilingData->cubeCoreNum * singleCubeCoreXCastSize;
        mmOutFp32WsGm.SetGlobalBuffer((__gm__ float *)workspace + wsOffset);
        mmOuterInnerSize_ = CeilAlign(N_SIZE, MM_CACHE_LINE_BYTES / sizeof(float));
        mmOutFp32BufSize_ = curBs * mmOuterInnerSize_;
        wsOffset += tilingData->cubeBlockDimK / tilingData->cubeBlockDimM * mmOutFp32BufSize_;
        squareSumFp32WsGm.SetGlobalBuffer((__gm__ float *)workspace + wsOffset);
        squareSumFp32BufSize_ = CeilAlign(curBs, BLOCK_CUBE) * BLOCK_CUBE;

        if ASCEND_IS_AIC {
            cubeCompute_.Init(hcFnGm, pipePtr, tilingData);
            return;
        }
        int64_t xQueNum = tilingData->stage1MFactor * RoundUp<T>(tilingData->cvLoopKSize);
        pipe->InitBuffer(xQue, NUM_TWO, xQueNum * sizeof(T));
        pipe->InitBuffer(mmInQue, NUM_TWO, xQueNum * sizeof(float));
    }

    __aicore__ inline void Process(int64_t bsOffset, int64_t curBs)
    {
        int64_t curML1Size = tilingData->mL1Size;
        int64_t curCubeBlockDimM = tilingData->cubeBlockDimM;

        if ASCEND_IS_AIC {
            // 预置 2 次 AIC->AIV 事件：覆盖 AIV 尾部双 Wait（空 K 核/单轮路径均平衡）
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
        }

        uint64_t curBlockIdx = GetBlockIdx();
        uint64_t curVectorBlockIdx = curBlockIdx;
        if ASCEND_IS_AIV {
            curBlockIdx = curBlockIdx / NUM_TWO;
        }
        uint64_t mBlkDimIdx = curBlockIdx / tilingData->cubeBlockDimK;
        uint64_t kBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimK;
        uint64_t kGmStartOffset = tilingData->multCoreSplitKSize * kBlkDimIdx;
        uint64_t kGmEndOffset = kGmStartOffset + tilingData->multCoreSplitKSize;
        if (kGmEndOffset > tilingData->k) {
            kGmEndOffset = tilingData->k;
        }
        for (uint64_t mGmOffset = mBlkDimIdx * curML1Size; mGmOffset < (uint64_t)curBs;
             mGmOffset += curCubeBlockDimM * curML1Size) {
            uint64_t realMSize = mGmOffset + curML1Size > (uint64_t)curBs ? (uint64_t)curBs - mGmOffset : curML1Size;
            // 单外层轮（cvLoopKSize == splitKSize，kGmEnd-kGmStart <= cvLoopKSize 恒成立）
            for (uint64_t kGmBaseOffset = kGmStartOffset; kGmBaseOffset < kGmEndOffset;
                 kGmBaseOffset += tilingData->cvLoopKSize) {
                uint64_t realKGmSize = kGmBaseOffset + tilingData->cvLoopKSize > kGmEndOffset
                                           ? kGmEndOffset - kGmBaseOffset
                                           : tilingData->cvLoopKSize;
                if ASCEND_IS_AIC {
                    TinyMmParams mmParams;
                    mmParams.curML1 = realMSize;
                    mmParams.curNL1 = N_SIZE;
                    mmParams.singleCoreK = realKGmSize;
                    mmParams.xWsKSize = tilingData->cvLoopKSize;
                    mmParams.nOutSize = mmOuterInnerSize_;
                    mmParams.kGmSize = tilingData->k;

                    cubeCompute_.WaitBL1Ready();
                    cubeCompute_.CopyInB1(kGmBaseOffset, realKGmSize, mmParams);

                    CrossCoreWaitFlag(SYNC_AIV_TO_AIC_FLAG);
                    cubeCompute_.ComputeOnce(
                        xCastFp32WsGm[cvLoopIdx_ % DOUBLE_BUFFER * xCastFp32BufSize_ / sizeof(float)],
                        squareSumFp32WsGm[kBlkDimIdx * squareSumFp32BufSize_ + mGmOffset * BLOCK_CUBE],
                        mmOutFp32WsGm[kBlkDimIdx * mmOutFp32BufSize_ + mGmOffset * mmOuterInnerSize_], mmParams);
                    cubeCompute_.SetBL1Free();
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
                } else {
                    CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
                    int64_t mVectorOffset = mGmOffset;
                    int64_t mVectorLength = (realMSize + 1) / NUM_TWO;
                    if ((curVectorBlockIdx % NUM_TWO) == 1) {
                        mVectorOffset = mGmOffset + (realMSize + 1) / NUM_TWO;
                        mVectorLength = realMSize - mVectorLength;
                    }
                    int64_t curUbLoops = (mVectorLength + tilingData->stage1MFactor - 1) / tilingData->stage1MFactor;
                    int64_t curUbMfactorTail = mVectorLength - ((curUbLoops - 1) * tilingData->stage1MFactor);
                    for (int64_t i = 0; i < curUbLoops; ++i) {
                        int64_t curUbMFactor = (i != (curUbLoops - 1)) ? tilingData->stage1MFactor : curUbMfactorTail;
                        xLocal = xQue.template AllocTensor<T>();
                        int64_t curGlobalxOffset = (mVectorOffset + i * tilingData->stage1MFactor) * tilingData->k + kGmBaseOffset;
                        CopyIn(xGm[curGlobalxOffset], xLocal, curUbMFactor, realKGmSize, tilingData->k - realKGmSize);
                        xQue.template EnQue(xLocal);
                        xLocal = xQue.template DeQue<T>();
                        xCastLocal = mmInQue.AllocTensor<float>();
                        CastTwoDim(xCastLocal, xLocal, curUbMFactor, realKGmSize);
                        xQue.template FreeTensor(xLocal);
                        mmInQue.template EnQue(xCastLocal);
                        xCastLocal = mmInQue.template DeQue<float>();
                        int64_t cutMmInOffset = cvLoopIdx_ % DOUBLE_BUFFER * xCastFp32BufSize_ / sizeof(float) +
                                                (i * tilingData->stage1MFactor + mVectorOffset) * tilingData->cvLoopKSize;
                        CopyOut(xCastLocal, xCastFp32WsGm[cutMmInOffset], curUbMFactor, realKGmSize,
                                tilingData->cvLoopKSize - realKGmSize);
                        mmInQue.FreeTensor(xCastLocal);
                    }
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_TO_AIC_FLAG);
                }
                cvLoopIdx_++;
            }
        }
        if ASCEND_IS_AIC {
            cubeCompute_.End();
        } else {
            CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
            CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
        }
        // SyncAll 必须保留：Part2 要读全部 8 个 AIC 的 partial，上方成对 flag 只保证
        // 本对 AIC->AIV 的顺序性，不提供跨对 GM 可见性（key=1 实证同款承重墙）
        SyncAll<false>();
    }

private:
    TPipe *pipe;
    const MhcPreClampSinkhornTilingData *tilingData;
    GlobalTensor<float> workspaceGm;
    GlobalTensor<float> xCastFp32WsGm;
    GlobalTensor<float> mmOutFp32WsGm;
    GlobalTensor<float> squareSumFp32WsGm;
    GlobalTensor<float> hcFnGm;
    GlobalTensor<T> xGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> mmInQue;

    LocalTensor<T> xLocal;
    LocalTensor<float> xCastLocal;

    MhcPreClampSinkhornTinyCubeCompute cubeCompute_;
    static constexpr uint64_t SYNC_AIV_TO_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_TO_AIV_FLAG = 9;
    static constexpr uint64_t SYNC_MODE2 = NUM_TWO;

    uint64_t cvLoopIdx_ = 0;
    uint64_t xCastFp32BufSize_;
    uint64_t mmOuterInnerSize_;
    uint64_t mmOutFp32BufSize_;
    uint64_t squareSumFp32BufSize_;
};

// Part2：fork 自 MembaseKSplitCorePart2，删除全部 needGrad 分支（路由保证推理路径）。
// 合并（squareSum 树归约+对角 gather / mmOut 跨核归约）与 epilogue 数学逐行同源。
template <typename T>
class MhcPreClampSinkhornTinyPart2 {
public:
    __aicore__ inline MhcPreClampSinkhornTinyPart2() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR y,
                                GM_ADDR post, GM_ADDR combFrag, GM_ADDR workspace,
                                const MhcPreClampSinkhornTilingData *tilingDataPtr, TPipe *pipePtr,
                                int64_t bsOffset)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T *)x + (bsOffset * tilingData->hcMult * tilingData->d));
        hcScaleGm.SetGlobalBuffer((__gm__ float *)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float *)hcBase);
        yGm.SetGlobalBuffer((__gm__ T *)y + (bsOffset * tilingData->d));
        postGm.SetGlobalBuffer((__gm__ float *)post + (bsOffset * tilingData->hcMult));
        combFragGm.SetGlobalBuffer((__gm__ float *)combFrag + (bsOffset * tilingData->hcMult * tilingData->hcMult));
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace);

        // InQue（布局同 key=1 Part2：[kcore][rows][...]）
        int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
        int64_t mixesQue01Size = stage1UsedCoreNum * tilingData->stage2RowFactor * tilingData->hcMultAlign * NUM_TWO * sizeof(float);
        pipe->InitBuffer(mixesQue01, NUM_TWO, mixesQue01Size);
        pipe->InitBuffer(mixesQue2, NUM_TWO,
                         stage1UsedCoreNum * tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(squareSumQue, NUM_TWO, stage1UsedCoreNum * tilingData->stage2RowFactor * SQUARE_SUM_SIZE * sizeof(float));

        int64_t xQueNum2 = tilingData->stage2RowFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor);
        pipe->InitBuffer(xQue, NUM_TWO, xQueNum2 * sizeof(T));

        // OutQue
        pipe->InitBuffer(yQue, NUM_TWO, tilingData->stage2RowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        pipe->InitBuffer(postQue, NUM_TWO, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(combFragQue, NUM_TWO,
                         tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));

        // TBuf
        pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcScaleBuf, BLOCK_SIZE);
        pipe->InitBuffer(rowBrcbBuf0, RoundUp<float>(tilingData->stage2RowFactor) * BLOCK_SIZE);
        pipe->InitBuffer(hcBrcbBuf1, RoundUp<float>(tilingData->stage2RowFactor * tilingData->hcMultAlign * NUM_TWO) * BLOCK_SIZE);
        pipe->InitBuffer(reduceBuf, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(treeReduceBuf, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(mxies01ReduceBuf, tilingData->stage2RowFactor * tilingData->hcMultAlign * NUM_TWO * sizeof(float));
        pipe->InitBuffer(mxies02ReduceBuf, tilingData->stage2RowFactor * tilingData->hcMultAlign * tilingData->hcMult * sizeof(float));
        pipe->InitBuffer(squareReduceBuf, tilingData->stage2RowFactor * SQUARE_SUM_SIZE * sizeof(float));
        pipe->InitBuffer(xCastBuf, xQueNum2 * sizeof(float));
        pipe->InitBuffer(yCastBuf, tilingData->stage2RowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(float));
        pipe->InitBuffer(rsqrtBuf, RoundUp<float>(tilingData->stage2RowFactor) * sizeof(float));
        pipe->InitBuffer(maskPatternBuf, RoundUp<uint32_t>(NUM_SIXTEEN * NUM_EIGHT) * sizeof(uint32_t));
        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        hcBase2Local = hcBaseBuf2.Get<float>();
        hcScaleLocal = hcScaleBuf.Get<float>();
        rowBrcbLocal0 = rowBrcbBuf0.Get<float>();
        hcBrcbLocal1 = hcBrcbBuf1.Get<float>();
        reduceLocal = reduceBuf.Get<float>();
        treeReduceLocal = treeReduceBuf.Get<float>();
        squareReduceLocal = squareReduceBuf.Get<float>();
        mxies01ReduceLocal = mxies01ReduceBuf.Get<float>();
        mxies02ReduceLocal = mxies02ReduceBuf.Get<float>();
        xCastLocal = xCastBuf.Get<float>();
        yCastLocal = yCastBuf.Get<float>();
        rsqrtLocal = rsqrtBuf.Get<float>();
        maskPatternLocal = maskPatternBuf.Get<uint32_t>();
        SetGatherMaskPattern(maskPatternLocal);
    }

    __aicore__ inline void Process(int64_t bsOffset, int64_t curBs)
    {
        int64_t curRowOfFormerBlock = tilingData->rowOfFormerBlock;
        int64_t curRowLoopOfFormerBlock = tilingData->rowLoopOfFormerBlock;
        int64_t curRowLoopOfTailBlock = tilingData->rowLoopOfTailBlock;
        int64_t curSecondUsedCoreNum = tilingData->secondUsedCoreNum;
        int64_t curStage2RowFactor = tilingData->stage2RowFactor;
        int64_t curTailRowFactorOfFormerBlock = tilingData->tailRowFactorOfFormerBlock;
        int64_t curTailRowFactorOfTailBlock = tilingData->tailRowFactorOfTailBlock;
        int64_t curML1Size = tilingData->mL1Size;

        if ASCEND_IS_AIV {
            int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
            int64_t stage2BlockIdx = GetBlockIdx();
            int64_t stage2UsedCoreNum = curSecondUsedCoreNum;
            if (stage2BlockIdx >= stage2UsedCoreNum) {
                return;
            }
            int64_t mmLastAxisSize = CeilAlign(tilingData->hcMix, MM_CACHE_LINE_BYTES / sizeof(float));
            int64_t xCastFp32BufSize = curML1Size * CeilAlign(tilingData->cvLoopKSize, MM_CACHE_LINE_BYTES / sizeof(float)) * sizeof(float);
            int64_t workspaceSize1 = (tilingData->cubeCoreNum * DOUBLE_BUFFER * xCastFp32BufSize) / sizeof(float);
            int64_t workspaceSize2 = CeilAlign(stage1UsedCoreNum * curBs * mmLastAxisSize * sizeof(float), WORKSPACE_ALIGN_SIZE) / sizeof(float);
            CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult * NUM_TWO], hcBase2Local, tilingData->hcMult, tilingData->hcMult);
            // 只拷实际 3 个元素，不做整块 32B 拷贝（避免越界读 hcScale 尾部）
            CopyIn(hcScaleGm, hcScaleLocal, 1, 3);
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);

            int64_t rowOuterLoop =
                (stage2BlockIdx == stage2UsedCoreNum - 1) ? curRowLoopOfTailBlock : curRowLoopOfFormerBlock;
            int64_t tailRowFactor = (stage2BlockIdx == stage2UsedCoreNum - 1) ? curTailRowFactorOfTailBlock :
                                                                        curTailRowFactorOfFormerBlock;
            int64_t xGmBlockBaseOffsetPart2 = stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMult * tilingData->d;

            for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
                int64_t xGmBsBaseOffsetPart2 = rowOuterIdx * curStage2RowFactor * tilingData->hcMult * tilingData->d;
                int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : curStage2RowFactor;
                squareSumOutLocal = squareSumQue.AllocTensor<float>();

                CopyIn(workspaceGm[workspaceSize1 + workspaceSize2 + stage2BlockIdx * curRowOfFormerBlock * SQUARE_SUM_SIZE + rowOuterIdx * curStage2RowFactor * SQUARE_SUM_SIZE],
                        squareSumOutLocal, stage1UsedCoreNum, curRowFactor * SQUARE_SUM_SIZE, CeilAlign(curBs, SQUARE_SUM_SIZE) * SQUARE_SUM_SIZE - curRowFactor * SQUARE_SUM_SIZE);
                squareSumQue.EnQue(squareSumOutLocal);
                squareSumOutLocal = squareSumQue.DeQue<float>();
                ReduceSumMidDimPerf(squareReduceLocal, squareSumOutLocal, treeReduceLocal, 1, stage1UsedCoreNum, curRowFactor * SQUARE_SUM_SIZE);
                int64_t curBsIdxForAll = stage2BlockIdx * curRowOfFormerBlock + rowOuterIdx * curStage2RowFactor;
                GatherMaskByDiagonal(rsqrtLocal, squareReduceLocal, maskPatternLocal[(curBsIdxForAll % SQUARE_SUM_SIZE) * 8], curRowFactor);
                float coeff = 1.0f / static_cast<float>(tilingData->k);
                Muls(rsqrtLocal, rsqrtLocal, coeff, curRowFactor);
                PipeBarrier<PIPE_V>();
                Adds(rsqrtLocal, rsqrtLocal, tilingData->normEps, curRowFactor);
                PipeBarrier<PIPE_V>();
                Sqrt(rsqrtLocal, rsqrtLocal, curRowFactor);
                Duplicate(rowBrcbLocal0, static_cast<float>(1.0f), curRowFactor);
                PipeBarrier<PIPE_V>();
                Div(rsqrtLocal, rowBrcbLocal0, rsqrtLocal, curRowFactor);

                mxies01Local = mixesQue01.AllocTensor<float>();
                uint64_t mixBaseOffset = workspaceSize1 + stage2BlockIdx * curRowOfFormerBlock * CeilAlign(tilingData->hcMix, WORKSPACE_ALIGN_SIZE / sizeof(float)) +
                                         rowOuterIdx * curStage2RowFactor * CeilAlign(tilingData->hcMix, WORKSPACE_ALIGN_SIZE / sizeof(float));
                CopyInWithOuterFor(workspaceGm[mixBaseOffset], mxies01Local, stage1UsedCoreNum, curRowFactor, tilingData->hcMult,
                                curBs, CeilAlign(tilingData->hcMix, WORKSPACE_ALIGN_SIZE / sizeof(float)));
                CopyInWithOuterFor(workspaceGm[mixBaseOffset + tilingData->hcMult],
                                mxies01Local[stage1UsedCoreNum * curRowFactor * tilingData->hcMultAlign],
                                stage1UsedCoreNum, curRowFactor, tilingData->hcMult, curBs,
                                CeilAlign(tilingData->hcMix, WORKSPACE_ALIGN_SIZE / sizeof(float)));

                // wk:[2, kcorenum, bs, n^2 +2n] -> mx0[2, kcorenum, curRowFactor, n]
                mixesQue01.EnQue(mxies01Local);
                mxies01Local = mixesQue01.DeQue<float>();
                ReduceSumMidDimPerf(mxies01ReduceLocal, mxies01Local, treeReduceLocal, NUM_TWO, stage1UsedCoreNum, curRowFactor * tilingData->hcMultAlign);

                ProcessPre(mxies01ReduceLocal, mxies01ReduceLocal, hcBase0Local, rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1,
                            hcScaleLocal.GetValue(0), tilingData->hcEps, curRowFactor, tilingData->hcMult);

                for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                    int64_t curDFactor =
                        (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                    xLocal = xQue.template AllocTensor<T>();
                    CopyIn(xGm[xGmBlockBaseOffsetPart2 + xGmBsBaseOffsetPart2 +
                                dLoopIdx * tilingData->dFactor],
                            xLocal, curRowFactor * tilingData->hcMult, curDFactor, tilingData->d - curDFactor);
                    xQue.template EnQue(xLocal);
                    xLocal = xQue.template DeQue<T>();
                    yLocal = yQue.template AllocTensor<T>();

                    ProcessY(yLocal, xLocal, mxies01ReduceLocal, hcBrcbLocal1, xCastLocal, yCastLocal, curRowFactor,
                                tilingData->hcMult, curDFactor);
                    xQue.template FreeTensor(xLocal);
                    yQue.template EnQue(yLocal);
                    yLocal = yQue.template DeQue<T>();
                    CopyOut(yLocal,
                            yGm[stage2BlockIdx * curRowOfFormerBlock * tilingData->d +
                                rowOuterIdx * curStage2RowFactor * tilingData->d + dLoopIdx * tilingData->dFactor],
                            curRowFactor, curDFactor, tilingData->d - curDFactor);
                    yQue.FreeTensor(yLocal);
                }
                // post
                postLocal = postQue.AllocTensor<float>();
                ProcessPost(postLocal, mxies01ReduceLocal[curRowFactor * tilingData->hcMultAlign], hcBase1Local,
                            rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1, hcScaleLocal.GetValue(1), curRowFactor,
                            tilingData->hcMult);
                mixesQue01.template FreeTensor(mxies01Local);
                postQue.EnQue(postLocal);
                postLocal = postQue.DeQue<float>();
                CopyOut(postLocal,
                        postGm[stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMult +
                                rowOuterIdx * curStage2RowFactor * tilingData->hcMult],
                        curRowFactor, tilingData->hcMult);
                postQue.FreeTensor(postLocal);

                // combFrag（mix2 归约 + Sinkhorn）
                // 按 4x(4 元素) 子块搬运，dst 行距 8 的 row-padded 结构是下游读取契约
                // （AddBAFirstDimBrcInline(hcBase2[4][8]) 等），不可折叠为 64B 大块
                mixes2Local = mixesQue2.AllocTensor<float>();
                for (int64_t i = 0; i < stage1UsedCoreNum; ++i) {
                    for (int64_t j = 0; j < curRowFactor; ++j) {
                        CopyIn(workspaceGm[workspaceSize1 + i * curBs * mmLastAxisSize + j * mmLastAxisSize + stage2BlockIdx * curRowOfFormerBlock * mmLastAxisSize + rowOuterIdx * curStage2RowFactor * mmLastAxisSize + tilingData->hcMult * NUM_TWO],
                                        mixes2Local[(i * curRowFactor + j) * tilingData->hcMult * tilingData->hcMultAlign], tilingData->hcMult, tilingData->hcMult);
                    }
                }
                mixesQue2.EnQue(mixes2Local);
                mixes2Local = mixesQue2.DeQue<float>();
                ReduceSumMidDimPerf(mxies02ReduceLocal, mixes2Local, treeReduceLocal, 1,
                                    stage1UsedCoreNum, curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);

                combFragLocal = combFragQue.AllocTensor<float>();

                MulABLastDimBrcInline<float, false>(mxies02ReduceLocal, mxies02ReduceLocal, rsqrtLocal, rowBrcbLocal0, curRowFactor,
                                                    tilingData->hcMult * tilingData->hcMultAlign);
                Muls(mxies02ReduceLocal, mxies02ReduceLocal, hcScaleLocal.GetValue(NUM_TWO),
                        curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);
                PipeBarrier<PIPE_V>();
                AddBAFirstDimBrcInline<float>(mxies02ReduceLocal, mxies02ReduceLocal, hcBase2Local, curRowFactor,
                                                tilingData->hcMult * tilingData->hcMultAlign);
                SoftmaxFP32Perf(mxies02ReduceLocal, mxies02ReduceLocal, reduceLocal, hcBrcbLocal1, curRowFactor * tilingData->hcMult,
                                tilingData->hcMult, tilingData->hcEps);

                ReduceSumMidDimPerf(reduceLocal, mxies02ReduceLocal, treeReduceLocal, curRowFactor, tilingData->hcMult, tilingData->hcMult);
                // reduceLocal 为 [curRowFactor][hcMultAlign] stride-8 布局, eps 须按行加
                for (int64_t r = 0; r < curRowFactor; r++) {
                    Adds(reduceLocal[r * tilingData->hcMultAlign], reduceLocal[r * tilingData->hcMultAlign],
                        tilingData->hcEps, tilingData->hcMult);
                }
                PipeBarrier<PIPE_V>();
                DivABABrcInline(combFragLocal, mxies02ReduceLocal, reduceLocal, curRowFactor, tilingData->hcMult,
                                tilingData->hcMult);

                for (int64_t iter = 1; iter < tilingData->iterTimes; iter++) {
                    LastDimReduceSumPerf(reduceLocal, combFragLocal, curRowFactor * tilingData->hcMult, tilingData->hcMult);
                    Adds(reduceLocal, reduceLocal, tilingData->hcEps, curRowFactor * tilingData->hcMult);

                    PipeBarrier<PIPE_V>();
                    DivABLastDimBrcInline<float, true>(combFragLocal, combFragLocal, reduceLocal, hcBrcbLocal1,
                                                        curRowFactor * tilingData->hcMult, tilingData->hcMult);

                    ReduceSumMidDimPerf(reduceLocal, combFragLocal, treeReduceLocal, curRowFactor, tilingData->hcMult, tilingData->hcMult);
                    for (int64_t r = 0; r < curRowFactor; r++) {
                        Adds(reduceLocal[r * tilingData->hcMultAlign], reduceLocal[r * tilingData->hcMultAlign],
                            tilingData->hcEps, tilingData->hcMult);
                    }

                    PipeBarrier<PIPE_V>();
                    DivABABrcInline(combFragLocal, combFragLocal, reduceLocal, curRowFactor, tilingData->hcMult,
                                    tilingData->hcMult);
                }
                mixesQue2.FreeTensor(mixes2Local);
                squareSumQue.template FreeTensor(squareSumOutLocal);

                combFragQue.EnQue(combFragLocal);
                combFragLocal = combFragQue.DeQue<float>();
                CopyOut(combFragLocal,
                        combFragGm[stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMult * tilingData->hcMult +
                                    rowOuterIdx * curStage2RowFactor * tilingData->hcMult * tilingData->hcMult],
                        curRowFactor * tilingData->hcMult, tilingData->hcMult);
                combFragQue.FreeTensor(combFragLocal);
            }
        }
    }

private:
    TPipe *pipe;
    const MhcPreClampSinkhornTilingData *tilingData;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<float> workspaceGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;

    TQue<QuePosition::VECIN, 1> mixesQue01;
    TQue<QuePosition::VECIN, 1> mixesQue2;
    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECOUT, 1> combFragQue;
    TQue<QuePosition::VECIN, 1> squareSumQue;

    TBuf<QuePosition::VECCALC> hcBaseBuf0;
    TBuf<QuePosition::VECCALC> hcBaseBuf1;
    TBuf<QuePosition::VECCALC> hcBaseBuf2;
    TBuf<QuePosition::VECCALC> hcScaleBuf;
    TBuf<QuePosition::VECCALC> rowBrcbBuf0;
    TBuf<QuePosition::VECCALC> hcBrcbBuf1;
    TBuf<QuePosition::VECCALC> reduceBuf;
    TBuf<QuePosition::VECCALC> treeReduceBuf;
    TBuf<QuePosition::VECCALC> rsqrtBuf;
    TBuf<QuePosition::VECCALC> squareReduceBuf;
    TBuf<QuePosition::VECCALC> mxies01ReduceBuf;
    TBuf<QuePosition::VECCALC> mxies02ReduceBuf;
    TBuf<QuePosition::VECCALC> xCastBuf;
    TBuf<QuePosition::VECCALC> yCastBuf;
    TBuf<QuePosition::VECCALC> maskPatternBuf;

    LocalTensor<float> mxies01Local;
    LocalTensor<float> mixes2Local;
    LocalTensor<float> rsqrtLocal;
    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> postLocal;
    LocalTensor<float> combFragLocal;
    LocalTensor<float> hcBase0Local;
    LocalTensor<float> hcBase1Local;
    LocalTensor<float> hcBase2Local;
    LocalTensor<float> hcScaleLocal;
    LocalTensor<float> rowBrcbLocal0;
    LocalTensor<float> hcBrcbLocal1;
    LocalTensor<float> reduceLocal;
    LocalTensor<float> treeReduceLocal;
    LocalTensor<float> squareReduceLocal;
    LocalTensor<float> mxies01ReduceLocal;
    LocalTensor<float> mxies02ReduceLocal;
    LocalTensor<float> xCastLocal;
    LocalTensor<float> yCastLocal;
    LocalTensor<float> squareSumOutLocal;
    LocalTensor<uint32_t> maskPatternLocal;
};

} // namespace MhcPreClampSinkhorn

#endif // MHC_PRE_SINKHORN_TINY_CORE_H
