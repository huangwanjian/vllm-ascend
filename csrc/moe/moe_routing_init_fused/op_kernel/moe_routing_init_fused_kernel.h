/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. Do not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_routing_init_fused_kernel.h
 * \brief MoeRoutingInitFused kernel：小 batch（activeNum = S*k <= 1024）MoE 派发融合。
 *
 * 替代 CANN MoeInitRoutingV3 + host 侧 torch.abs + .to(int64) + topk_weights.to(dtype) 四个 kernel。
 * 语义与 MoeInitRoutingV3 对齐（参照 in-tree moe_v2_init_routing_fullload）：
 *  - expanded_row_idx[i]（i = token_row*k + slot，源位置索引）= 目标行号（本算子不 drop，恒非负）
 *  - expert 段内序 = 源位置升序（counting-sort 稳定性 = CANN Sort 稳定序）
 *  - expert_tokens[e] = 该 expert 行数（count 模式，int64，GMM group_list）
 * 结构（照 MoeV2FullLoad 范式）：跨核零依赖——每个参与核冗余计算 counting-sort 派发映射
 * （结果留 UB，不落 GM、不用 SyncAll/workspace 中转），core0 额外写 expanded_row_idx/expert_tokens
 * 输出；全核按源位置 i 分区 gather（hidden 行 GM→UB→GM 中继，TQueBind 惯用法）。
 * 坑位记录：core0 写 GM 映射 + 其余核 GM 裸读（GetValue）+ 部分核 SyncAll 的方案已验证不可行
 * （S<40 全核数时 SyncAll 挂死；S=40 GM 裸读返回错值）。
 */
#ifndef MOE_ROUTING_INIT_FUSED_KERNEL_H
#define MOE_ROUTING_INIT_FUSED_KERNEL_H

#include "kernel_operator.h"

namespace MoeRoutingInitFused {
using namespace AscendC;

__aicore__ inline int64_t RifCeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

// 自满足屏障（照 moe_v2_common.h 的 SetWaitFlag）：同一事件 Set 后立即 Wait，
// 用于 MTE2 搬入后、MTE3 搬出前的数据可见性排序。TQueBind 不走 EnQue/DeQue。
template <HardEvent event>
__aicore__ inline void RifSetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

template <typename T>
class MoeRoutingInitFusedKernel {
public:
    __aicore__ inline MoeRoutingInitFusedKernel(){};

    __aicore__ inline void Init(GM_ADDR hidden, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR sortedHidden,
                                GM_ADDR expandedRowIdx, GM_ADDR expertTokens, GM_ADDR topkWeightsOut,
                                GM_ADDR userWorkspace, const MoeRoutingInitFusedTilingData *tilingData,
                                TPipe *tPipe)
    {
        (void)userWorkspace; // 跨核零依赖：不使用 workspace
        pipe_ = tPipe;
        tilingData_ = tilingData;
        blockIdx_ = GetBlockIdx();

        tokenNum_ = tilingData->tokenNum;
        k_ = tilingData->k;
        expertNum_ = tilingData->expertNum;
        cols_ = tilingData->cols;
        activeNum_ = tilingData->activeNum;
        needCoreNum_ = tilingData->needCoreNum;
        perCoreRows_ = tilingData->perCoreRows;

        // 本核 gather 责任区（按源位置 i = row*k + slot 切，照 moe_v2_gather_out）
        myRowStart_ = perCoreRows_ * blockIdx_;
        if (myRowStart_ > activeNum_) {
            myRowStart_ = activeNum_;
        }
        int64_t myRowCount = (blockIdx_ == needCoreNum_ - 1) ? tilingData->lastCoreRows : perCoreRows_;
        myRowCount_ = (myRowStart_ + myRowCount > activeNum_) ? (activeNum_ - myRowStart_) : myRowCount;

        hiddenGm_.SetGlobalBuffer((__gm__ T *)hidden, tokenNum_ * cols_);
        topkIdsGm_.SetGlobalBuffer((__gm__ int32_t *)topkIds, activeNum_);
        sortedHiddenGm_.SetGlobalBuffer((__gm__ T *)sortedHidden, activeNum_ * cols_);
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, activeNum_);
        expertTokensGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokens, expertNum_);
        // topk_weights 纯透传：kernel 零工作（host 侧以输入别名作输出）

        int64_t activeNumAlign32 = RifCeilDiv(activeNum_ * static_cast<int64_t>(sizeof(int32_t)), 32) * 8;
        int64_t colsAlign = RifCeilDiv(cols_ * static_cast<int64_t>(sizeof(T)), 32) * 8;

        // 派发映射（每核冗余算，结果留 UB）：ids 搬入队列 + rowIdx/counts/cursor 本地缓冲
        pipe_->InitBuffer(idsQue_, 1, activeNumAlign32 * sizeof(int32_t));
        pipe_->InitBuffer(rowIdxBuf_, static_cast<uint32_t>(activeNumAlign32 * sizeof(int32_t)));
        pipe_->InitBuffer(countsBuf_, static_cast<uint32_t>(RifCeilDiv(expertNum_, 8) * 8 * sizeof(int64_t))); // int64 8 元素 32B 对齐
        pipe_->InitBuffer(cursorBuf_, static_cast<uint32_t>(RifCeilDiv(expertNum_, 8) * 8 * sizeof(int32_t)));

        // gather 阶段：一行 T 的 GM->UB->GM 中继（双缓冲）
        pipe_->InitBuffer(gatherQue_, 2, colsAlign * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ < needCoreNum_) {
            ComputeDispatchMapLocal();
            if (blockIdx_ == 0) {
                WriteIndicesAndCounts();
            }
            GatherRows();
        }
    }

private:
    // counting-sort 派发映射（每个参与核冗余执行，结果留 UB）：
    // histogram -> 前缀 -> scatter，产出 rowIdx[i] = dst（源位置 i 的目标行号）
    __aicore__ inline void ComputeDispatchMapLocal()
    {
        LocalTensor<int32_t> idsLocal = idsQue_.AllocTensor<int32_t>();
        DataCopyExtParams copyInParams{1, static_cast<uint32_t>(activeNum_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(idsLocal, topkIdsGm_, copyInParams, padParams);
        idsQue_.EnQue(idsLocal);
        idsLocal = idsQue_.DeQue<int32_t>();
        // 标量读前置同步（照 moe_v2_gather_out CopyOut 的 MTE2_S 惯用法）
        RifSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        LocalTensor<int64_t> countsLocal = countsBuf_.Get<int64_t>();
        LocalTensor<int32_t> cursorLocal = cursorBuf_.Get<int32_t>();
        LocalTensor<int32_t> rowIdxLocal = rowIdxBuf_.Get<int32_t>();
        // histogram（int64 直接产，消 .to(int64)）
        for (int64_t e = 0; e < expertNum_; e++) {
            countsLocal.SetValue(e, static_cast<int64_t>(0));
        }
        for (int64_t i = 0; i < activeNum_; i++) {
            int64_t e = idsLocal.GetValue(i);
            countsLocal.SetValue(e, countsLocal.GetValue(e) + static_cast<int64_t>(1));
        }
        // 前缀和 -> 游标（base）
        int64_t running = 0;
        for (int64_t e = 0; e < expertNum_; e++) {
            cursorLocal.SetValue(e, static_cast<int32_t>(running));
            running += countsLocal.GetValue(e);
        }
        // scatter：expanded_row_idx[i] = dst
        for (int64_t i = 0; i < activeNum_; i++) {
            int64_t e = idsLocal.GetValue(i);
            int32_t dst = cursorLocal.GetValue(e);
            cursorLocal.SetValue(e, dst + 1);
            rowIdxLocal.SetValue(i, dst);
        }
        idsQue_.FreeTensor(idsLocal);
    }

    // 仅 core0：派发映射与计数落 GM（标量写 -> MTE3 出，照 moe_v2 S_MTE3 惯用法）
    __aicore__ inline void WriteIndicesAndCounts()
    {
        LocalTensor<int32_t> rowIdxLocal = rowIdxBuf_.Get<int32_t>();
        LocalTensor<int64_t> countsLocal = countsBuf_.Get<int64_t>();
        RifSetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams idxParams{1, static_cast<uint32_t>(activeNum_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(expandedRowIdxGm_, rowIdxLocal, idxParams);
        DataCopyExtParams expertParams{1, static_cast<uint32_t>(expertNum_ * sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(expertTokensGm_, countsLocal, expertParams);
    }

    // 全核并行：按源位置 i 分区 gather hidden 行（GM -> UB -> GM）。
    // 照 moe_v2_gather_out：同一源行（i/k 相同的连续 k 个槽位）只搬入一次、
    // 散出 k 次；TQueBind 惯用法 Alloc -> 搬入 -> RifSetWaitFlag<MTE2_MTE3>
    // -> 搬出 -> Free（TQueBind 不走 EnQue/DeQue——那会等一个永不置位的事件，挂死）。
    __aicore__ inline void GatherRows()
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        LocalTensor<int32_t> rowIdxLocal = rowIdxBuf_.Get<int32_t>();
        int64_t i = myRowStart_;
        int64_t iEnd = myRowStart_ + myRowCount_;
        while (i < iEnd) {
            int64_t srcRow = i / k_;
            LocalTensor<T> rowLocal = gatherQue_.AllocTensor<T>();
            DataCopyPad(rowLocal, hiddenGm_[srcRow * cols_], copyParams, padParams);
            RifSetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            while (i < iEnd && i / k_ == srcRow) {
                int32_t dst = rowIdxLocal.GetValue(i);
                DataCopyPad(sortedHiddenGm_[static_cast<int64_t>(dst) * cols_], rowLocal, copyParams);
                i++;
            }
            gatherQue_.FreeTensor(rowLocal);
        }
    }

private:
    TPipe *pipe_;
    const MoeRoutingInitFusedTilingData *tilingData_;
    int64_t blockIdx_;

    int64_t tokenNum_;
    int64_t k_;
    int64_t expertNum_;
    int64_t cols_;
    int64_t activeNum_;
    int64_t needCoreNum_;
    int64_t perCoreRows_;
    int64_t myRowStart_;
    int64_t myRowCount_;

    GlobalTensor<T> hiddenGm_;
    GlobalTensor<int32_t> topkIdsGm_;
    GlobalTensor<T> sortedHiddenGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int64_t> expertTokensGm_;

    TQue<QuePosition::VECIN, 1> idsQue_;
    TBuf<TPosition::VECCALC> rowIdxBuf_;
    TBuf<TPosition::VECCALC> countsBuf_;
    TBuf<TPosition::VECCALC> cursorBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> gatherQue_;
};

} // namespace MoeRoutingInitFused
#endif // MOE_ROUTING_INIT_FUSED_KERNEL_H
