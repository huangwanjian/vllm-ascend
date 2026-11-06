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
 * \file aclnn_mhc_post.cpp
 * \brief aclnn_mhc_post（替换 CANN aclnnHcPost：kernel 直接按转置下标读 H_res，无 host 侧 Transpose）
 */

#include "aclnn_mhc_post.h"
#include "mhc_post.h"
#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn/aclnn_base.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_X_SUPPORT_LIST = {op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> DTYPE_F32_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static bool CheckNotNull(const aclTensor *x, const aclTensor *hRes, const aclTensor *hOut,
                         const aclTensor *hPost, const aclTensor *y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(hRes, return false);
    OP_CHECK_NULL(hOut, return false);
    OP_CHECK_NULL(hPost, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *x, const aclTensor *hRes, const aclTensor *hOut,
                            const aclTensor *hPost, const aclTensor *y)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_X_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(hRes, DTYPE_F32_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(hOut, DTYPE_X_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(hPost, DTYPE_F32_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, DTYPE_X_SUPPORT_LIST, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *hRes, const aclTensor *hOut,
                               const aclTensor *hPost, const aclTensor *y)
{
    CHECK_RET(CheckNotNull(x, hRes, hOut, hPost, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, hRes, hOut, hPost, y), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPostGetWorkspaceSize(
    const aclTensor *x, const aclTensor *hRes, const aclTensor *hOut, const aclTensor *hPost,
    aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "workspaceSize or executor is null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    L2_DFX_PHASE_1(aclnnMhcPost, DFX_IN(x, hRes, hOut, hPost), DFX_OUT(y));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, hRes, hOut, hPost, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空张量直接返回成功
    if (x->IsEmpty() || y->IsEmpty()) {
        OP_LOGW("[aclnnMhcPost] Input tensor is empty, skip computation and return success.");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor *xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *hResContiguous = l0op::Contiguous(hRes, uniqueExecutor.get());
    CHECK_RET(hResContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *hOutContiguous = l0op::Contiguous(hOut, uniqueExecutor.get());
    CHECK_RET(hOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *hPostContiguous = l0op::Contiguous(hPost, uniqueExecutor.get());
    CHECK_RET(hPostContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *yContiguous = l0op::Contiguous(y, uniqueExecutor.get());
    CHECK_RET(yContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *kernelOut = l0op::MhcPost(xContiguous, hResContiguous, hOutContiguous, hPostContiguous,
                                               const_cast<aclTensor *>(yContiguous), uniqueExecutor.get());
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPost(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPost);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
