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
 * \file mhc_post_kernel.h
 * \brief MhcPost kernel（移植自 hermes arch22，USE_PERMANENT_X 模板参数改为 tiling 运行时字段）
 * Formula: x_{l+1} = (H_l^res)^T * x_l + h_l^out * H_t^post
 * (H^T × x) 以转置下标读 H（hRes[j*n+i]），无需 GM 转置——这是相对 CANN aclnnHcPost
 * （host 先发 10.8µs Transpose kernel）的核心差异。
 */

#ifndef MHC_POST_KERNEL_H
#define MHC_POST_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"

namespace MhcPost {
using namespace AscendC;

constexpr uint32_t MHC_POST_DOUBLE_BUFFER_DEPTH = 2;

template <typename T>
class MhcPostKernel {
public:
    __aicore__ inline MhcPostKernel(TPipe *tPipe, const MhcPostTilingData *__restrict tilingData)
        : pipe_(tPipe), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost, GM_ADDR output);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInHOut(int64_t bsIdx, int64_t dIdx, int64_t dNum);
    __aicore__ inline void ComputeCopyOut(int64_t bsIdx, int64_t dIdx, int64_t dNum);
    __aicore__ inline void ComputeCopyOutAllX(int64_t bsIdx, int64_t dIdx, int64_t dNum);
    __aicore__ inline void CopyOutTile(int64_t bsIdx, int64_t dIdx, int64_t nI, int64_t dNum);
    __aicore__ inline void CopyInX(int64_t bsIdx, int64_t dIdx, int64_t nJ, int64_t dNum);

private:
    TPipe *pipe_;
    const MhcPostTilingData *__restrict tilingData_;

    TQue<QuePosition::VECIN, MHC_POST_DOUBLE_BUFFER_DEPTH> hOutTileQueue_;
    TQue<QuePosition::VECIN, MHC_POST_DOUBLE_BUFFER_DEPTH> xTileQueue_;
    TQue<QuePosition::VECOUT, MHC_POST_DOUBLE_BUFFER_DEPTH> outputTileQueue_;

    TBuf<QuePosition::VECCALC> hOutF32Buf_;
    TBuf<QuePosition::VECCALC> xF32Buf_;
    TBuf<QuePosition::VECCALC> outF32Buf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> hOutGm_;
    GlobalTensor<float> hResGm_;
    GlobalTensor<float> hPostGm_;
    GlobalTensor<T> outputGm_;

    int64_t xFactor_ = 1;
    int64_t curCoreItemCount_ = 0;
    int64_t itemStart_ = 0;
    uint32_t blockIdx_ = 0;
};

template <typename T>
__aicore__ inline void MhcPostKernel<T>::Init(GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost,
                                              GM_ADDR output)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    if (blockIdx_ < tilingData_->usedCoreNum - 1) {
        curCoreItemCount_ = tilingData_->normalCoreProcessNum;
    } else {
        curCoreItemCount_ = tilingData_->tailCoreProcessNum;
    }
    itemStart_ = blockIdx_ * tilingData_->normalCoreProcessNum;

    xFactor_ = (tilingData_->usePermanentX == 1) ? tilingData_->n : 1;

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    hOutGm_.SetGlobalBuffer((__gm__ T *)hOut);
    hResGm_.SetGlobalBuffer((__gm__ float *)hRes);
    hPostGm_.SetGlobalBuffer((__gm__ float *)hPost);
    outputGm_.SetGlobalBuffer((__gm__ T *)output);

    pipe_->InitBuffer(hOutTileQueue_, MHC_POST_DOUBLE_BUFFER_DEPTH, tilingData_->dInner * sizeof(T));
    pipe_->InitBuffer(xTileQueue_, MHC_POST_DOUBLE_BUFFER_DEPTH, xFactor_ * tilingData_->dInner * sizeof(T));
    pipe_->InitBuffer(outputTileQueue_, MHC_POST_DOUBLE_BUFFER_DEPTH, tilingData_->dInner * sizeof(T));

    pipe_->InitBuffer(hOutF32Buf_, tilingData_->dInner * sizeof(float));
    pipe_->InitBuffer(xF32Buf_, xFactor_ * tilingData_->dInner * sizeof(float));
    pipe_->InitBuffer(outF32Buf_, tilingData_->dInner * sizeof(float));
}

template <typename T>
__aicore__ inline void MhcPostKernel<T>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    for (int64_t itemIdx = 0; itemIdx < curCoreItemCount_; itemIdx++) {
        int64_t globalItemIdx = itemStart_ + itemIdx;
        int64_t bsIdx = globalItemIdx / tilingData_->dOuter;
        int64_t dIdx = globalItemIdx - bsIdx * tilingData_->dOuter;
        int64_t dNum = (dIdx < tilingData_->dOuter - 1) ? tilingData_->dInner : tilingData_->dTail;

        CopyInHOut(bsIdx, dIdx, dNum);
        if (tilingData_->usePermanentX == 1) {
            CopyInX(bsIdx, dIdx, 0, dNum);
            ComputeCopyOutAllX(bsIdx, dIdx, dNum);
        } else {
            ComputeCopyOut(bsIdx, dIdx, dNum);
        }
    }
}

template <typename T>
__aicore__ inline void MhcPostKernel<T>::CopyInHOut(int64_t bsIdx, int64_t dIdx, int64_t dNum)
{
    int64_t hOutOffset = bsIdx * tilingData_->d + dIdx * tilingData_->dInner;
    LocalTensor<T> hOutTileLocal = hOutTileQueue_.AllocTensor<T>();

    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(dNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(hOutTileLocal, hOutGm_[hOutOffset], copyParams, {false, 0, 0, 0});

    hOutTileQueue_.EnQue(hOutTileLocal);
}

template <typename T>
__aicore__ inline void MhcPostKernel<T>::ComputeCopyOut(int64_t bsIdx, int64_t dIdx, int64_t dNum)
{
    int64_t hPostBase = bsIdx * tilingData_->n;
    int64_t hResBase = bsIdx * tilingData_->n * tilingData_->n;

    LocalTensor<T> hOutTile = hOutTileQueue_.DeQue<T>();

    LocalTensor<float> hOutF32 = hOutF32Buf_.Get<float>();
    LocalTensor<float> outF32 = outF32Buf_.Get<float>();
    LocalTensor<float> xF32 = xF32Buf_.Get<float>();

    Cast(hOutF32, hOutTile, RoundMode::CAST_NONE, dNum);

    for (int64_t i = 0; i < tilingData_->n; i++) {
        LocalTensor<T> outputTile = outputTileQueue_.AllocTensor<T>();

        Muls(outF32, hOutF32, hPostGm_.GetValue(hPostBase + i), dNum);
        for (int64_t j = 0; j < tilingData_->n; j++) {
            CopyInX(bsIdx, dIdx, j, dNum);
            LocalTensor<T> xTile = xTileQueue_.DeQue<T>();
            Cast(xF32, xTile, RoundMode::CAST_NONE, dNum);
            // 转置下标读 H：x_{l+1}[i] += H[j,i] * x_l[j]
            Axpy(outF32, xF32, hResGm_.GetValue(hResBase + j * tilingData_->n + i), dNum);
            xTileQueue_.FreeTensor(xTile);
        }

        Cast(outputTile, outF32, RoundMode::CAST_RINT, dNum);
        outputTileQueue_.EnQue(outputTile);
        CopyOutTile(bsIdx, dIdx, i, dNum);
    }

    hOutTileQueue_.FreeTensor(hOutTile);
}

template <typename T>
__aicore__ inline void MhcPostKernel<T>::ComputeCopyOutAllX(int64_t bsIdx, int64_t dIdx, int64_t dNum)
{
    int64_t hPostBase = bsIdx * tilingData_->n;
    int64_t hResBase = bsIdx * tilingData_->n * tilingData_->n;
    int64_t dNumAlign = (dIdx < tilingData_->dOuter - 1) ? dNum : tilingData_->dTailAlign;
    LocalTensor<T> hOutTile = hOutTileQueue_.DeQue<T>();
    LocalTensor<T> xTile = xTileQueue_.DeQue<T>();

    LocalTensor<float> hOutF32 = hOutF32Buf_.Get<float>();
    LocalTensor<float> xF32 = xF32Buf_.Get<float>();
    LocalTensor<float> outF32 = outF32Buf_.Get<float>();

    Cast(hOutF32, hOutTile, RoundMode::CAST_NONE, dNum);
    Cast(xF32, xTile, RoundMode::CAST_NONE, tilingData_->n * dNumAlign);

    for (int64_t i = 0; i < tilingData_->n; i++) {
        LocalTensor<T> outputTile = outputTileQueue_.AllocTensor<T>();

        Muls(outF32, hOutF32, hPostGm_.GetValue(hPostBase + i), dNum);
        for (int64_t j = 0; j < tilingData_->n; j++) {
            PipeBarrier<PIPE_V>();
            Axpy(outF32, xF32[j * dNumAlign], hResGm_.GetValue(hResBase + j * tilingData_->n + i), dNum);
            PipeBarrier<PIPE_V>();
        }

        Cast(outputTile, outF32, RoundMode::CAST_RINT, dNum);
        outputTileQueue_.EnQue(outputTile);
        CopyOutTile(bsIdx, dIdx, i, dNum);
    }

    hOutTileQueue_.FreeTensor(hOutTile);
    xTileQueue_.FreeTensor(xTile);
}

template <typename T>
__aicore__ inline void MhcPostKernel<T>::CopyInX(int64_t bsIdx, int64_t dIdx, int64_t nJ, int64_t dNum)
{
    int64_t dStart = dIdx * tilingData_->dInner;
    int64_t xBase = bsIdx * tilingData_->n * tilingData_->d + nJ * tilingData_->d;
    int64_t xOffset = xBase + dStart;
    LocalTensor<T> xTileLocal = xTileQueue_.AllocTensor<T>();

    if (tilingData_->usePermanentX == 1) {
        DataCopyExtParams copyParams = {static_cast<uint16_t>(tilingData_->n),
                                        static_cast<uint32_t>(dNum * sizeof(T)),
                                        static_cast<uint32_t>((tilingData_->d - dNum) * sizeof(T)), 0, 0};
        DataCopyPad(xTileLocal, xGm_[xOffset], copyParams, {false, 0, 0, 0});
    } else {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(dNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(xTileLocal, xGm_[xOffset], copyParams, {false, 0, 0, 0});
    }

    xTileQueue_.EnQue(xTileLocal);
}

template <typename T>
__aicore__ inline void MhcPostKernel<T>::CopyOutTile(int64_t bsIdx, int64_t dIdx, int64_t nI, int64_t dNum)
{
    int64_t dStart = dIdx * tilingData_->dInner;
    int64_t outputBase = bsIdx * tilingData_->n * tilingData_->d + nI * tilingData_->d;
    int64_t outputOffset = outputBase + dStart;
    LocalTensor<T> outputTile = outputTileQueue_.DeQue<T>();

    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(dNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm_[outputOffset], outputTile, copyParams);

    outputTileQueue_.FreeTensor(outputTile);
}

} // namespace MhcPost

#endif // MHC_POST_KERNEL_H
