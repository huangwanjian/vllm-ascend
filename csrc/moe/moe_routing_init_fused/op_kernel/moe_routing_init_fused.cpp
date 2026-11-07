/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. Do not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_routing_init_fused.cpp
 * \brief
 */

#include "moe_routing_init_fused_kernel.h"

#define TILING_KEY_ROUTING_INIT_FUSED_FULLLOAD 0

using namespace AscendC;
using namespace MoeRoutingInitFused;

extern "C" __global__ __aicore__ void moe_routing_init_fused(GM_ADDR hidden, GM_ADDR topkIds, GM_ADDR topkWeights,
                                                             GM_ADDR sortedHidden, GM_ADDR expandedRowIdx,
                                                             GM_ADDR expertTokens, GM_ADDR topkWeightsOut,
                                                             GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }

    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    TPipe tPipe;
    if (TILING_KEY_IS(TILING_KEY_ROUTING_INIT_FUSED_FULLLOAD)) {
        GET_TILING_DATA_WITH_STRUCT(MoeRoutingInitFusedTilingData, tilingData, tiling);
        const MoeRoutingInitFusedTilingData *__restrict t = &tilingData;
        MoeRoutingInitFusedKernel<DTYPE_HIDDEN> op;
        op.Init(hidden, topkIds, topkWeights, sortedHidden, expandedRowIdx, expertTokens, topkWeightsOut, userWS, t,
                &tPipe);
        op.Process();
    }
}
