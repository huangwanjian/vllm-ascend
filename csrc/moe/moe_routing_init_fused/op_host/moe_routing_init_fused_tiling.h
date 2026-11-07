/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. Do not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_routing_init_fused_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_ROUTING_INIT_FUSED_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_ROUTING_INIT_FUSED_H

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/error_log.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeRoutingInitFusedTilingData)
TILING_DATA_FIELD_DEF(int64_t, tokenNum);      // S：派发前 token 数
TILING_DATA_FIELD_DEF(int64_t, k);             // topk
TILING_DATA_FIELD_DEF(int64_t, expertNum);     // E
TILING_DATA_FIELD_DEF(int64_t, cols);          // K：hidden 列数
TILING_DATA_FIELD_DEF(int64_t, activeNum);     // S*k：派发行数
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);   // gather 阶段使用核数
TILING_DATA_FIELD_DEF(int64_t, perCoreRows);   // 每核 gather 行数（按 dst 行切）
TILING_DATA_FIELD_DEF(int64_t, lastCoreRows);  // 尾核行数
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeRoutingInitFused, MoeRoutingInitFusedTilingData)

struct MoeRoutingInitFusedCompileInfo {};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_ROUTING_INIT_FUSED_H
