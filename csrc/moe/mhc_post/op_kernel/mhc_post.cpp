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
 * \file mhc_post.cpp
 * \brief MhcPost kernel entry（arch22）
 * Formula: x_{l+1} = (H_l^res)^T * x_l + h_l^out * H_t^post
 */

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "mhc_post_kernel.h"

using namespace AscendC;
using namespace MhcPost;

extern "C" __global__ __aicore__ void mhc_post(GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost,
                                               GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // 本算子不使用 workspace（tiling 侧申请为 0），不做判空提前返回
    (void)workspace;
    GET_TILING_DATA_WITH_STRUCT(MhcPostTilingData, tiling_data_in, tiling);
    const MhcPostTilingData *__restrict tilingData = &tiling_data_in;

    TPipe tPipe;
    MhcPostKernel<DTYPE_X> op(&tPipe, tilingData);
    op.Init(x, hRes, hOut, hPost, y);
    op.Process();
    tPipe.Destroy();
}
