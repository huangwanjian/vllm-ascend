/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. Do not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_routing_init_fused_tiling_base.cpp
 * \brief
 */
#include "moe_routing_init_fused_tiling.h"
#include "register/op_def_registry.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/error_log.h"
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {
static ge::graphStatus TilingForMoeRoutingInitFused(gert::TilingContext *context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForMoeRoutingInitFused(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeRoutingInitFused)
    .Tiling(TilingForMoeRoutingInitFused)
    .TilingParse<MoeRoutingInitFusedCompileInfo>(TilingPrepareForMoeRoutingInitFused);

} // namespace optiling
