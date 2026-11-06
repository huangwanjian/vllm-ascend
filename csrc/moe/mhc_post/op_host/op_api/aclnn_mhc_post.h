#ifndef OP_API_INC_LEVEL2_ACLNN_MHC_POST_H_
#define OP_API_INC_LEVEL2_ACLNN_MHC_POST_H_
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#ifdef __cplusplus
extern "C" {
#endif
ACLNN_API aclnnStatus aclnnMhcPostGetWorkspaceSize(
    const aclTensor *x, const aclTensor *hRes, const aclTensor *hOut, const aclTensor *hPost,
    aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnMhcPost(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif // OP_API_INC_LEVEL2_ACLNN_MHC_POST_H_
