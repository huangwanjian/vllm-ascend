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
 * \file mhc_pre_clamp_sinkhorn.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_clamp_sinkhorn_m_k_split_core.h"
#include "mhc_pre_clamp_sinkhorn_m_split_core.h"
#include "mhc_pre_clamp_sinkhorn_tiny_core.h"
#include "mhc_pre_clamp_sinkhorn_base.h"
using namespace MhcPreClampSinkhorn;

using namespace AscendC;

extern "C" __global__ __aicore__ void mhc_pre_clamp_sinkhorn(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                             GM_ADDR hin, GM_ADDR hPost, GM_ADDR hRes,
                                             GM_ADDR hPre, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
                                             GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR hResLogits,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }

    GET_TILING_DATA_WITH_STRUCT(MhcPreClampSinkhornTilingData, tiling_data_in, tiling);
    const MhcPreClampSinkhornTilingData *__restrict tilingData = &tiling_data_in;

    if (TILING_KEY_IS(0)) {
        // -----------split m--------------
        TPipe pipe;
        MhcPreClampSinkhorn::MhcPreClampSinkhornStage1<DTYPE_X> op1;
        op1.cubeCompute_.mm1_.Init(&tilingData->mm1TilingData, &pipe);
        op1.cubeCompute_.mm1_.SetSubBlockIdx(0);
        op1.Init(x, phi, invRms, hcBeforeNorm, userWs, tilingData, &pipe);
        op1.Process();
        pipe.Destroy();
            
        TPipe pipeStage2;
        MhcPreClampSinkhorn::MhcPreClampSinkhornStage2<DTYPE_X> op2;
        op2.Init(x, alpha, bias, hin, hPost, hRes,
                    hPre, hcBeforeNorm, invRms, sumOut, normOut, hResLogits,
                    userWs, tilingData, &pipeStage2);
        op2.Process(false);
        pipeStage2.Destroy();
    } else if (TILING_KEY_IS(1)) {
        // -----------split m k------------
        for (int64_t bsLoopIdx = 0; bsLoopIdx < tilingData->bsLoop; bsLoopIdx++) {
            int64_t curBsOffset = bsLoopIdx * tilingData->curBsSplit;
            int64_t curBs = (bsLoopIdx == tilingData->bsLoop - 1) ? tilingData->tailBs : tilingData->curBsSplit;
            bool isTailBsLoop = (bsLoopIdx == tilingData->bsLoop - 1) && (tilingData->tailBs != tilingData->curBsSplit);
            TPipe pipe;
            MhcPreClampSinkhorn::MhcPreClampSinkhornMembaseKSplitCorePart1<DTYPE_X> op;
            op.Init(x, phi, userWs, tilingData, &pipe, curBsOffset, curBs, isTailBsLoop);
            op.Process(curBsOffset, curBs, isTailBsLoop);
            pipe.Destroy();

            TPipe pipeStage2;
            MhcPreClampSinkhorn::MhcPreClampSinkhornMembaseKSplitCorePart2<DTYPE_X> op2;
            op2.Init(x, alpha, bias, hin, hPost, hRes,
                        hPre, hcBeforeNorm, invRms, sumOut, normOut, hResLogits,
                        userWs, tilingData, &pipeStage2, curBsOffset, isTailBsLoop);
            op2.Process(curBsOffset, curBs, isTailBsLoop);
            pipeStage2.Destroy();
            if (bsLoopIdx + 1 < tilingData->bsLoop) {
                // 仅迭代间需要全核同步（防下轮 Part1 复写 workspace 与本轮 Part2 读竞争）。
                // 末轮后无后续消费者，kernel 结束框架自带各核 join（key=0 路径同构无尾同步）。
                SyncAll<false>(); // cv全部同步
            }
        }
    } else if (TILING_KEY_IS(2)) {
        // -----------tiny decode (bs<=32, !needGrad)：8 AIC + 16 AIV 精简 launch-----------
        // bsLoop 恒 1（路由保证 bs <= 32 < BS_SPLIT_THRESHOLD），循环保留与 key=1 同构
        for (int64_t bsLoopIdx = 0; bsLoopIdx < tilingData->bsLoop; bsLoopIdx++) {
            int64_t curBsOffset = bsLoopIdx * tilingData->curBsSplit;
            int64_t curBs = (bsLoopIdx == tilingData->bsLoop - 1) ? tilingData->tailBs : tilingData->curBsSplit;

            if (tilingData->tinyFuse == 0) {
                // ===== fuse=off（MHC_PRE_TINY_FUSE 默认档）：原路径，逐行不动 =====
                TPipe pipe;
                MhcPreClampSinkhorn::MhcPreClampSinkhornTinyPart1<DTYPE_X> op;
                op.Init(x, phi, userWs, tilingData, &pipe, curBsOffset, curBs);
                op.Process(curBsOffset, curBs);
                pipe.Destroy();

                TPipe pipeStage2;
                MhcPreClampSinkhorn::MhcPreClampSinkhornTinyPart2<DTYPE_X> op2;
                op2.Init(x, alpha, bias, hin, hPost, hRes, userWs, tilingData, &pipeStage2, curBsOffset);
                op2.Process(curBsOffset, curBs);
                pipeStage2.Destroy();
            } else {
                // ===== fuse=on（MHC_PRE_TINY_FUSE=on，方案 perf/TINY_FUSE_DESIGN.md §3 路径 A'）=====
                // SyncAll 保留且外提（跨对 GM 可见性墙不动，AIC 到达序不变），AIV 把
                // Part2 全部 prologue 前置到 barrier 等待阴影：
                //   AIC : Init → phi Nd2Nz → 等 AIV cast → Mmad → Fixp → End ──────┐
                //   AIV : Init → cast x → SetFlag(AIV→AIC) → [MTE3 排空防护] →    │
                //         pipe1.Destroy → pipe2 ctor+Init → 常量 CopyIn(+MTE2_V)  │
                //         → [fork A 支 x 首块预取] → 尾部双 Wait ── SyncAll ←──────┘
                //         → Part2 Process/ProcessFork（Init 已完成、常量已在 UB、x 已在途）
                // 两 pipe 顺序存活（pipe1.Destroy 先于 pipe2 ctor），UB 不叠加。
                TPipe pipe;
                MhcPreClampSinkhorn::MhcPreClampSinkhornTinyPart1<DTYPE_X> op;
                op.Init(x, phi, userWs, tilingData, &pipe, curBsOffset, curBs);
                // deferTailSync=true：跳过 AIV 尾部双 Wait 与尾部 SyncAll（后置到 prologue
                // 之后，使 prologue 落进等待阴影；flag 记账不变）
                op.Process(curBsOffset, curBs, true);
                // MTE3 in-flight 冲突防护（方案 §3 要点 3 / 简报硬约束 4）：等本核
                // MTE3（xCastLocal→WS 读源）排空再覆写 pipe2 的 UB；须在 pipe.Destroy()
                // 之前（FetchEventID 依赖当前 TPipe 存活）
                op.WaitMte3DrainForFuse();
                pipe.Destroy();

                if ASCEND_IS_AIV {
                    TPipe pipeStage2;
                    MhcPreClampSinkhorn::MhcPreClampSinkhornTinyPart2<DTYPE_X> op2;
                    op2.Init(x, alpha, bias, hin, hPost, hRes, userWs, tilingData, &pipeStage2, curBsOffset);
                    // v1 常量前置 + v2 fork A 支 x 首块预取（见 tiny_core.h PrologueFuse）
                    op2.PrologueFuse(curBs);
                    // AIV 尾部双 Wait 原样保留（位置后置）：等 AIC post-compute flag 期间
                    // prologue 的 MTE2 搬运在途——这就是被填进的等待阴影
                    op.WaitAivTailFlags();
                    // 常量 MTE2->V 等待后置收尾（flag 已在阴影里置位，瞬时通过）
                    op2.WaitPrologueConsts();
                    SyncAll<false>();
                    op2.Process(curBsOffset, curBs);
                    pipeStage2.Destroy();
                } else {
                    // AIC：Part1 后半程无独立工作，关闭 pipe2 ctor/Init/Process/Destroy
                    // （设计 §3 要点 4，卫生性改动；AIC 不在 post-barrier 关键路径上）
                    SyncAll<false>();
                }
            }
            if (bsLoopIdx + 1 < tilingData->bsLoop) {
                SyncAll<false>();
            }
        }
    }
    // #endif
}