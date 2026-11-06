#ifndef OP_API_INC_LEVEL0_OP_MHC_POST_OP_H_
#define OP_API_INC_LEVEL0_OP_MHC_POST_OP_H_

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief MhcPost L0 kernel interface
 * Function: Implements the MHC post mapping
 * Formula: x_{l+1} = (H_l^res)^T * x_l + h_l^out * H_t^post
 *
 * @param [in] x: Input residual tensor, shape is [T, n, D] or [B, S, n, D], dtype is BF16/FP16
 * @param [in] hRes: Doubly-stochastic matrix from Sinkhorn, shape is [T, n, n] or [B, S, n, n], dtype is FP32
 * @param [in] hOut: Atten/MLP output, shape is [T, D] or [B, S, D], dtype is BF16/FP16
 * @param [in] hPost: Post coefficients, shape is [T, n] or [B, S, n], dtype is FP32
 * @param [out] y: Output tensor, same shape/dtype as x
 * @param [in] executor: Op executor
 * @return aclTensor*: Output tensor y
 */
const aclTensor *MhcPost(const aclTensor *x, const aclTensor *hRes, const aclTensor *hOut,
                         const aclTensor *hPost, aclTensor *y, aclOpExecutor *executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_MHC_POST_OP_H_
