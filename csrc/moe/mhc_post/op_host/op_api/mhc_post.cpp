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
 * \brief mhc_post L0 实现（移植自 hermes，消除 CANN aclnnHcPost 的 host 侧 Transpose kernel）
 */

#include "mhc_post.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(MhcPost);

static const aclTensor *MhcPostAiCore(const aclTensor *x, const aclTensor *hRes, const aclTensor *hOut,
                                      const aclTensor *hPost, aclTensor *y, aclOpExecutor *executor)
{
    L0_DFX(MhcPostAiCore, x, hRes, hOut, hPost, y);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MhcPost,
        OP_INPUT(x, hRes, hOut, hPost),
        OP_OUTPUT(y),
        OP_EMPTY_ARG);

    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MhcPost ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return y;
}

const aclTensor *MhcPost(const aclTensor *x, const aclTensor *hRes, const aclTensor *hOut,
                         const aclTensor *hPost, aclTensor *y, aclOpExecutor *executor)
{
    return MhcPostAiCore(x, hRes, hOut, hPost, y, executor);
}

} // namespace l0op
