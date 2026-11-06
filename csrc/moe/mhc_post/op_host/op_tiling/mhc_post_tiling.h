/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details on details on copying and modification.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_post_tiling.h
 * \brief MhcPost tiling header
 * 移植自 hermes/ops-transformer mhc_post（arch22），去除模板参数与框架依赖。
 * Formula: x_{l+1} = (H_l^res)^T * x_l + h_l^out * H_t^post
 */

#pragma once

#include <cstdint>
#include <register/op_def_registry.h>
#include <tiling/tiling_api.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MhcPostTilingData)
TILING_DATA_FIELD_DEF(int64_t, n);                   // hc 流数（=4）
TILING_DATA_FIELD_DEF(int64_t, d);                   // 特征维度
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);         // 实际使用 AIV 核数
TILING_DATA_FIELD_DEF(int64_t, normalCoreProcessNum); // 前核处理 item 数
TILING_DATA_FIELD_DEF(int64_t, tailCoreProcessNum);  // 尾核处理 item 数
TILING_DATA_FIELD_DEF(int64_t, dInner);              // d 维单块大小
TILING_DATA_FIELD_DEF(int64_t, dOuter);              // d 维块数
TILING_DATA_FIELD_DEF(int64_t, dTail);               // d 维尾块实际元素数
TILING_DATA_FIELD_DEF(int64_t, dTailAlign);          // d 维尾块对齐元素数
TILING_DATA_FIELD_DEF(int64_t, usePermanentX);       // 1=x 整块常驻 UB（原模板参数，改运行时）
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcPost, MhcPostTilingData)

struct MhcPostCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class MhcPostTiling {
public:
    explicit MhcPostTiling(gert::TilingContext* tilingContext) : context_(tilingContext) {}
    ~MhcPostTiling() = default;

    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus PostTiling();

private:
    ge::graphStatus CheckParam();

    gert::TilingContext *context_ = nullptr;
    uint64_t aivCoreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t s_ = 0;   // totalItems（bs*seq 或 T）
    int64_t n_ = 0;
    int64_t d_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t normalCoreProcessNum_ = 0;
    int64_t tailCoreProcessNum_ = 0;
    int64_t dInner_ = 0;
    int64_t dOuter_ = 0;
    int64_t dTail_ = 0;
    int64_t dTailAlign_ = 0;
    int64_t usePermanentX_ = 0;

    MhcPostTilingData tilingData_{};
};

} // namespace optiling
