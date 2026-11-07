# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        moe_routing_init_fused_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME MoeRoutingInitFused
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE moe_routing_init_fused ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
