/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre_clamp_sinkhorn_base.h
 * \brief
 */

#ifndef MHC_PRE_SINKHORN_VECTOR_BASE_H
#define MHC_PRE_SINKHORN_VECTOR_BASE_H

#include "kernel_operator.h"
#include <cstdint>

namespace MhcPreClampSinkhorn {
using namespace AscendC;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t DEFAULT_BLOCK_STRIDE = 1;
constexpr int32_t DEFAULT_REPEAT_STRIDE = 8;
constexpr int32_t ONE_REPEAT_BLOCK_NUMS = 8;
constexpr int32_t REPEAT_SIZE = 256;
constexpr int32_t MAX_REPEAT_STRIDE = 255;
constexpr int32_t WORKSPACE_ALIGN_SIZE = 512;
constexpr uint32_t ONE = 1;
constexpr uint32_t SHIFT_COEFF = 17;
constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int32_t CV_RATIO = 2;
constexpr uint64_t N_SIZE = 24;
constexpr uint64_t NUM_TWO = 2;
constexpr uint64_t NUM_EIGHT = 8;
constexpr uint64_t NUM_SIXTEEN = 16;
constexpr uint64_t SQUARE_SUM_SIZE = 16;
constexpr uint64_t MM_CACHE_LINE_BYTES = 512;
constexpr uint64_t MAX_BS_PER_LOOP = 32;
constexpr uint64_t ELEMENTS_SIZE_PER_BLOCK = BLOCK_SIZE / sizeof(float);

__aicore__ inline int32_t CeilDiv(int32_t a, int32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int32_t CeilAlign(int32_t a, int32_t b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num)
{
    int32_t elemNum = BLOCK_SIZE / sizeof(T);
    return CeilAlign(num, elemNum);
}

__aicore__ inline void SetGatherMaskPattern(const LocalTensor<uint32_t>& maskPattern)
{
    uint32_t base = ONE;
    int32_t length = 8;
    for (uint32_t i = 0; i < NUM_SIXTEEN; i++) {
        Duplicate(maskPattern[i * length], base, length);
        base = (base << 1);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GatherMaskByDiagonal(const LocalTensor<float>& output, const LocalTensor<float>& input,
                                            const LocalTensor<uint32_t> maskPattern, uint16_t dim0)
{
    uint32_t totalCount = 16 * 16;
    uint32_t maskCount = 16;
    uint16_t src0StriRepeat = 2; // 16/8(block)=2
    uint32_t remainCount = (dim0 % 16) * 16;
    uint32_t loopCount = dim0 / 16;
    uint64_t rsvdCnt = 0;
    for (uint32_t loopIdx = 0; loopIdx < loopCount; loopIdx++) {
        GatherMask(output[loopIdx * maskCount], input[loopIdx * totalCount], maskPattern, true, totalCount, {1, 1, 0, 0}, rsvdCnt);
    }
    GatherMask(output[loopCount * maskCount], input[loopCount * totalCount], maskPattern, true, maskCount, {1, dim0, src0StriRepeat, 1}, rsvdCnt);
    PipeBarrier<PIPE_V>();
}


template <typename T, bool needBrc = true>
__aicore__ inline void MulABLastDimBrcInline(const LocalTensor<T> &output, const LocalTensor<T> &input0,
                                             const LocalTensor<T> &input1, const LocalTensor<T> &tmpBuffer,
                                             const int32_t curRowNum, const int32_t curColNum)
{
    if constexpr (needBrc) {
        uint32_t repeatTimes = CeilDiv(curRowNum, ONE_REPEAT_BLOCK_NUMS);
        Brcb(tmpBuffer, input1, repeatTimes, {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE});
    }
    PipeBarrier<PIPE_V>();
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    if (curColNum <= elemInOneBlock) {
        Mul(output, input0, tmpBuffer, curRowNum * curColNumAlign);
    } else {
        int32_t numRepeatPerLine = curColNum / elemInOneRepeat;
        int32_t numRemainPerLine = curColNum % elemInOneRepeat;
        int32_t dstRepStridePerLine = CeilDiv(curColNum, elemInOneBlock);
        BinaryRepeatParams instrParams;

        if (numRepeatPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE || curRowNum < numRepeatPerLine) {
                // 在Col方向开Repeat, 并且Repeat小于255
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Mul(output[i * curColNumAlign], input0[i * curColNumAlign], tmpBuffer[i * elemInOneBlock],
                        elemInOneRepeat, numRepeatPerLine, instrParams);
                }
            } else {
                // 在Row方向开Repeat
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                instrParams.src1RepStride = 1;
                for (uint32_t i = 0; i < numRepeatPerLine; i++) {
                    Mul(output[i * elemInOneRepeat], input0[i * elemInOneRepeat], tmpBuffer, elemInOneRepeat, curRowNum,
                        instrParams);
                }
            }
        }

        if (numRemainPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE) {
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = 0;
                instrParams.src0RepStride = 0;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Mul(output[numRepeatPerLine * elemInOneRepeat + i * curColNumAlign],
                        input0[numRepeatPerLine * elemInOneRepeat + i * curColNumAlign], tmpBuffer[i * elemInOneBlock],
                        numRemainPerLine, 1, instrParams);
                }
            } else {
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                // repeatTimes=curRowNum 沿行走且逐行换系数, src1RepStride 必须为 1;
                // rf=1 时 repeatTimes=1, 与旧值 0 等价, 行为不变
                instrParams.src1RepStride = 1;
                Mul(output[numRepeatPerLine * elemInOneRepeat], input0[numRepeatPerLine * elemInOneRepeat], tmpBuffer,
                    numRemainPerLine, curRowNum, instrParams);
            }
        }
    }
    PipeBarrier<PIPE_V>();
}


template <typename T, bool needBrc = true>
__aicore__ inline void MulABLastDimBrcInline2(const LocalTensor<T> &output, const LocalTensor<T> &input0,
                                             const LocalTensor<T> &input1, const LocalTensor<T> &tmpBuffer,
                                             const int32_t curRowNum, const int32_t curColNum, const int32_t numN)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    if constexpr (needBrc) {
        uint32_t repeatTimes = CeilDiv(curRowNum * CeilDiv(elemInOneBlock, numN), ONE_REPEAT_BLOCK_NUMS);
        Brcb(tmpBuffer, input1, repeatTimes, {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE}); // n-> n c
    }

    PipeBarrier<PIPE_V>();
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    if (curColNum <= elemInOneBlock) {
        if (curRowNum == numN) {
            // rf=1 legacy path: tmpBuffer 前 curRowNum 块即各 x 行系数, 与旧版行为一致
            Mul(output, input0, tmpBuffer, curRowNum * curColNumAlign);
        } else {
            // rf>1: Brcb 输出按组布局, 组 g 系数 c 在块 (g*elemInOneBlock + c),
            // 块 (g*elemInOneBlock + numN..elemInOneBlock-1) 为 padding, 逐行取系数块
            for (uint32_t g = 0; g < static_cast<uint32_t>(curRowNum) / numN; g++) {
                for (uint32_t c = 0; c < static_cast<uint32_t>(numN); c++) {
                    uint32_t row = g * numN + c;
                    Mul(output[row * curColNumAlign], input0[row * curColNumAlign],
                        tmpBuffer[(g * elemInOneBlock + c) * elemInOneBlock], curColNumAlign);
                }
            }
        }
    } else {
        int32_t numRepeatPerLine = curColNum / elemInOneRepeat;
        int32_t numRemainPerLine = curColNum % elemInOneRepeat;
        int32_t dstRepStridePerLine = CeilDiv(curColNum, elemInOneBlock);
        BinaryRepeatParams instrParams;

        if (numRepeatPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE || curRowNum < numRepeatPerLine) {
                // 在Col方向开Repeat, 并且Repeat小于255
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Mul(output[i * curColNumAlign], input0[i * curColNumAlign], tmpBuffer[(i / numN * elemInOneBlock + i % numN) * elemInOneBlock],
                        elemInOneRepeat, numRepeatPerLine, instrParams);
                }
            } else {
                // 在Row方向开Repeat: rf>1 时按组循环, 组内 src1 以 RepStride=1 线性走
                // 系数块 (g*elemInOneBlock + c), 避免线性走整表读到组间 padding 块
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                instrParams.src1RepStride = 1;
                for (uint32_t g = 0; g < static_cast<uint32_t>(curRowNum) / numN; g++) {
                    for (uint32_t i = 0; i < numRepeatPerLine; i++) {
                        Mul(output[g * numN * curColNumAlign + i * elemInOneRepeat],
                            input0[g * numN * curColNumAlign + i * elemInOneRepeat],
                            tmpBuffer[g * elemInOneBlock * elemInOneBlock], elemInOneRepeat, numN, instrParams);
                    }
                }
            }
        }
        // 非128对齐场景;
        if (numRemainPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE) {
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = 0;
                instrParams.src0RepStride = 0;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Mul(output[numRepeatPerLine * elemInOneRepeat + i * curColNumAlign],
                        input0[numRepeatPerLine * elemInOneRepeat + i * curColNumAlign],
                        tmpBuffer[(i / numN * elemInOneBlock + i % numN) * elemInOneBlock],
                        numRemainPerLine, 1, instrParams);
                }
            } else {
                // remain-row-direction: 与 row-direction 同理按组循环
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                instrParams.src1RepStride = 1;
                for (uint32_t g = 0; g < static_cast<uint32_t>(curRowNum) / numN; g++) {
                    Mul(output[g * numN * curColNumAlign + numRepeatPerLine * elemInOneRepeat],
                        input0[g * numN * curColNumAlign + numRepeatPerLine * elemInOneRepeat],
                        tmpBuffer[g * elemInOneBlock * elemInOneBlock], numRemainPerLine, numN, instrParams);
                }
            }
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, bool needBrc = true>
__aicore__ inline void SubABLastDimBrcInline(const LocalTensor<T> &output, const LocalTensor<T> &input0,
                                             const LocalTensor<T> &input1, const LocalTensor<T> &tmpBuffer,
                                             const int32_t curRowNum, const int32_t curColNum)
{
    if constexpr (needBrc) {
        uint32_t repeatTimes = CeilDiv(curRowNum, ONE_REPEAT_BLOCK_NUMS);
        Brcb(tmpBuffer, input1, repeatTimes, {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE});
    }
    PipeBarrier<PIPE_V>();
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    if (curColNum <= elemInOneBlock) {
        Sub(output, input0, tmpBuffer, curRowNum * curColNumAlign);
    } else {
        int32_t numRepeatPerLine = curColNum / elemInOneRepeat;
        int32_t numRemainPerLine = curColNum % elemInOneRepeat;
        int32_t dstRepStridePerLine = CeilDiv(curColNum, elemInOneBlock);
        BinaryRepeatParams instrParams;

        if (numRepeatPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE || curRowNum < numRepeatPerLine) {
                // 在Col方向开Repeat, 并且Repeat小于255
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Sub(output[i * curColNumAlign], input0[i * curColNumAlign], tmpBuffer[i * elemInOneBlock],
                        elemInOneRepeat, numRepeatPerLine, instrParams);
                }
            } else {
                // 在Row方向开Repeat
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                instrParams.src1RepStride = 1;
                for (uint32_t i = 0; i < numRepeatPerLine; i++) {
                    Sub(output[i * elemInOneRepeat], input0[i * elemInOneRepeat], tmpBuffer, elemInOneRepeat, curRowNum,
                        instrParams);
                }
            }
        }

        if (numRemainPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE) {
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = 0;
                instrParams.src0RepStride = 0;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Sub(output[numRepeatPerLine * elemInOneRepeat + i * curColNumAlign],
                        input0[numRepeatPerLine * elemInOneRepeat + i * curColNumAlign], tmpBuffer[i * elemInOneBlock],
                        numRemainPerLine, 1, instrParams);
                }
            } else {
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                // repeatTimes=curRowNum 沿行走且逐行换系数, src1RepStride 必须为 1;
                // rf=1 时 repeatTimes=1, 与旧值 0 等价, 行为不变
                instrParams.src1RepStride = 1;
                Sub(output[numRepeatPerLine * elemInOneRepeat], input0[numRepeatPerLine * elemInOneRepeat], tmpBuffer,
                    numRemainPerLine, curRowNum, instrParams);
            }
        }
    }
    PipeBarrier<PIPE_V>();
}


template <typename T, bool needBrc = true>
__aicore__ inline void DivABLastDimBrcInline(const LocalTensor<T> &output, const LocalTensor<T> &input0,
                                             const LocalTensor<T> &input1, const LocalTensor<T> &tmpBuffer,
                                             const int32_t curRowNum, const int32_t curColNum)
{
    if constexpr (needBrc) {
        uint32_t repeatTimes = CeilDiv(curRowNum, ONE_REPEAT_BLOCK_NUMS);
        Brcb(tmpBuffer, input1, repeatTimes, {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE});
    }
    PipeBarrier<PIPE_V>();
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    if (curColNum <= elemInOneBlock) {
        Div(output, input0, tmpBuffer, curRowNum * curColNumAlign);
    } else {
        int32_t numRepeatPerLine = curColNum / elemInOneRepeat;
        int32_t numRemainPerLine = curColNum % elemInOneRepeat;
        int32_t dstRepStridePerLine = CeilDiv(curColNum, elemInOneBlock);
        BinaryRepeatParams instrParams;

        if (numRepeatPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE || curRowNum < numRepeatPerLine) {
                // 在Col方向开Repeat, 并且Repeat小于255
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Div(output[i * curColNumAlign], input0[i * curColNumAlign], tmpBuffer[i * elemInOneBlock],
                        elemInOneRepeat, numRepeatPerLine, instrParams);
                }
            } else {
                // 在Row方向开Repeat
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                instrParams.src1RepStride = 1;
                for (uint32_t i = 0; i < numRepeatPerLine; i++) {
                    Div(output[i * elemInOneRepeat], input0[i * elemInOneRepeat], tmpBuffer, elemInOneRepeat, curRowNum,
                        instrParams);
                }
            }
        }

        if (numRemainPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE) {
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = 0;
                instrParams.src0RepStride = 0;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Div(output[numRepeatPerLine * elemInOneRepeat + i * curColNumAlign],
                        input0[numRepeatPerLine * elemInOneRepeat + i * curColNumAlign], tmpBuffer[i * elemInOneBlock],
                        numRemainPerLine, 1, instrParams);
                }
            } else {
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                // repeatTimes=curRowNum 沿行走且逐行换系数, src1RepStride 必须为 1;
                // rf=1 时 repeatTimes=1, 与旧值 0 等价, 行为不变
                instrParams.src1RepStride = 1;
                Div(output[numRepeatPerLine * elemInOneRepeat], input0[numRepeatPerLine * elemInOneRepeat], tmpBuffer,
                    numRemainPerLine, curRowNum, instrParams);
            }
        }
    }
    PipeBarrier<PIPE_V>();
}


template <typename T>
__aicore__ inline void AddBAFirstDimBrcInline(const LocalTensor<T> &output, const LocalTensor<T> &input0,
                                              const LocalTensor<T> &input1, const int32_t curRowNum,
                                              const int32_t curColNum)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    int32_t numRepeatPerLine = curColNum / elemInOneRepeat;
    int32_t numRemainPerLine = curColNum % elemInOneRepeat;
    int32_t dstRepStridePerLine = CeilDiv(curColNum, elemInOneBlock);
    BinaryRepeatParams instrParams;
    if (numRepeatPerLine > 0) {
        if (dstRepStridePerLine > MAX_REPEAT_STRIDE || curRowNum < numRepeatPerLine) {
            // 在Col方向开Repeat, 并且Repeat小于255
            instrParams.dstBlkStride = 1;
            instrParams.src0BlkStride = 1;
            instrParams.src1BlkStride = 1;
            instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
            instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
            instrParams.src1RepStride = DEFAULT_REPEAT_STRIDE;
            for (uint32_t i = 0; i < curRowNum; i++) {
                Add(output[i * curColNumAlign], input0[i * curColNumAlign], input1, elemInOneRepeat, numRepeatPerLine,
                    instrParams);
            }
        } else {
            // 在Row方向开Repeat
            instrParams.dstBlkStride = 1;
            instrParams.src0BlkStride = 1;
            instrParams.src1BlkStride = 1;
            instrParams.dstRepStride = dstRepStridePerLine;
            instrParams.src0RepStride = dstRepStridePerLine;
            instrParams.src1RepStride = 0;
            for (uint32_t i = 0; i < numRepeatPerLine; i++) {
                Add(output[i * elemInOneRepeat], input0[i * elemInOneRepeat], input1[i * elemInOneRepeat],
                    elemInOneRepeat, curRowNum, instrParams);
            }
        }
    }

    if (numRemainPerLine > 0) {
        if (dstRepStridePerLine > MAX_REPEAT_STRIDE) {
            instrParams.dstBlkStride = 1;
            instrParams.src0BlkStride = 1;
            instrParams.src1BlkStride = 1;
            instrParams.dstRepStride = 0;
            instrParams.src0RepStride = 0;
            instrParams.src1RepStride = 0;
            for (uint32_t i = 0; i < curRowNum; i++) {
                Add(output[numRepeatPerLine * elemInOneRepeat], input0[numRepeatPerLine * elemInOneRepeat], input1,
                    numRemainPerLine, 1, instrParams);
            }
        } else {
            instrParams.dstBlkStride = 1;
            instrParams.src0BlkStride = 1;
            instrParams.src1BlkStride = 1;
            instrParams.dstRepStride = dstRepStridePerLine;
            instrParams.src0RepStride = dstRepStridePerLine;
            instrParams.src1RepStride = 0;
            Add(output[numRepeatPerLine * elemInOneRepeat], input0[numRepeatPerLine * elemInOneRepeat], input1,
                numRemainPerLine, curRowNum, instrParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void CalcDenominator(const LocalTensor<T> &output, const LocalTensor<T> &input,
                                       const uint32_t calCount)
{
    Muls(output, input, static_cast<T>(-1.0), calCount);
    PipeBarrier<PIPE_V>();
    Exp(output, output, calCount);
    PipeBarrier<PIPE_V>();
    Adds(output, output, static_cast<T>(1.0), calCount);
    PipeBarrier<PIPE_V>();
}

// 暂时不处理repeat超限场景
template <typename T>
__aicore__ inline void SigmoidPerf(const LocalTensor<T> &output, const LocalTensor<T> &input,
                                   const LocalTensor<T> &tmpBuffer, const int64_t calCount)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    Duplicate(tmpBuffer, static_cast<T>(1.0), elemInOneBlock);
    CalcDenominator(output, input, calCount);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    int32_t numRepeatPerLine = calCount / elemInOneRepeat;
    int32_t numRemainPerLine = calCount % elemInOneRepeat;
    BinaryRepeatParams instrParams;
    instrParams.dstBlkStride = 1;
    instrParams.src0BlkStride = 0;
    instrParams.src1BlkStride = 1;
    instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
    instrParams.src0RepStride = 0;
    instrParams.src1RepStride = DEFAULT_REPEAT_STRIDE;
    Div(output, tmpBuffer, output, elemInOneRepeat, numRepeatPerLine, instrParams);
    if (numRemainPerLine != 0) {
        Div(output[numRepeatPerLine * elemInOneRepeat], tmpBuffer, output[numRepeatPerLine * elemInOneRepeat],
            numRemainPerLine, 1, instrParams);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ProcessPre(const LocalTensor<float> &preLocal, const LocalTensor<float> &mixLocal,
                                  const LocalTensor<float> &hcBaseLocal, const LocalTensor<float> &rsqrtLocal,
                                  const LocalTensor<float> &tmpBuffer0, const LocalTensor<float> &tmpBuffer1,
                                  float scale, float eps, const int32_t curRowNum, const int32_t curColNum)
{
    int32_t curColNumAlign = RoundUp<float>(curColNum);
    MulABLastDimBrcInline<float, true>(mixLocal, mixLocal, rsqrtLocal, tmpBuffer0, curRowNum, curColNum);
    Muls(mixLocal, mixLocal, scale, curRowNum * curColNumAlign);
    PipeBarrier<PIPE_V>();
    AddBAFirstDimBrcInline<float>(mixLocal, mixLocal, hcBaseLocal, curRowNum, curColNum);
    SigmoidPerf(preLocal, mixLocal, tmpBuffer1, curRowNum * curColNumAlign);
    Adds(preLocal, preLocal, eps, curRowNum * curColNumAlign);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ReduceSumARAPerf(const LocalTensor<float> &output, const LocalTensor<float> &input,
                                        const uint32_t dim0, const uint32_t dim1, const uint32_t dim2)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(float);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(float);
    uint32_t dim2Align = RoundUp<float>(dim2);

    // 拷贝第一个R到output上
    DataCopyParams copyParams;
    copyParams.blockCount = dim0;
    copyParams.blockLen = dim2Align / elemInOneBlock;
    copyParams.srcStride = (dim1 - 1) * (dim2Align / elemInOneBlock);
    copyParams.dstStride = 0;
    DataCopy(output, input, copyParams);
    PipeBarrier<PIPE_V>();
    uint32_t dim2RepeatTimes = dim2 / elemInOneRepeat;
    uint32_t dim2Reminder = dim2 % elemInOneRepeat;
    // 沿着dim2方向开repeat
    BinaryRepeatParams instrParams;
    instrParams.dstBlkStride = 1;
    instrParams.src0BlkStride = 1;
    instrParams.src1BlkStride = 1;
    instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
    instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
    instrParams.src1RepStride = DEFAULT_REPEAT_STRIDE;
    for (uint32_t i = 0; i < dim0; i++) {
        for (uint32_t j = 1; j < dim1; j++) {
            Add(output[i * dim2Align], output[i * dim2Align], input[i * dim1 * dim2Align + j * dim2Align],
                elemInOneRepeat, dim2RepeatTimes, instrParams);
            if (dim2Reminder != 0) {
                Add(output[i * dim2Align + dim2RepeatTimes * elemInOneRepeat],
                    output[i * dim2Align + dim2RepeatTimes * elemInOneRepeat],
                    input[i * dim1 * dim2Align + j * dim2Align + +dim2RepeatTimes * elemInOneRepeat], dim2Reminder, 1,
                    instrParams);
            }
            PipeBarrier<PIPE_V>();
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T0, typename T1>
__aicore__ inline void CastTwoDim(const LocalTensor<T0> &output, const LocalTensor<T1> &input, const uint32_t dim0,
                                  const uint32_t dim1)
{
    uint32_t dim1AlignT0 = RoundUp<T0>(dim1);
    uint32_t dim1AlignT1 = RoundUp<T1>(dim1);
    if constexpr (IsSameType<T1, bfloat16_t>::value && IsSameType<T0, float>::value) {
        for (uint32_t i = 0; i < dim0; i++) {
            Cast(output[i * dim1AlignT0], input[i * dim1AlignT1], AscendC::RoundMode::CAST_NONE, dim1);
        }
    } else {
        for (uint32_t i = 0; i < dim0; i++) {
            Cast(output[i * dim1AlignT0], input[i * dim1AlignT1], AscendC::RoundMode::CAST_RINT, dim1);
        }
    }
    PipeBarrier<PIPE_V>();
}

// dim0:bs dim1:n dim2:c
template <typename T>
__aicore__ void inline ProcessY(const LocalTensor<T> &yLocal, const LocalTensor<T> &xLocal,
                                const LocalTensor<float> &mix01Local, const LocalTensor<float> &hcBrcbLocal1,
                                const LocalTensor<float> &xCastLocal, const LocalTensor<float> &yCastLocal,
                                const uint32_t dim0, const uint32_t dim1, const uint32_t dim2)
{
    CastTwoDim(xCastLocal, xLocal, dim0 * dim1, dim2);
    MulABLastDimBrcInline2<float, true>(xCastLocal, xCastLocal, mix01Local, hcBrcbLocal1, dim0 * dim1, dim2, dim1);
    ReduceSumARAPerf(yCastLocal, xCastLocal, dim0, dim1, dim2);
    CastTwoDim(yLocal, yCastLocal, dim0, dim2);
}


__aicore__ inline void ProcessPost(const LocalTensor<float> &postLocal, const LocalTensor<float> &mixLocal,
                                   const LocalTensor<float> &hcBaseLocal, const LocalTensor<float> &rsqrtLocal,
                                   const LocalTensor<float> &tmpBuffer0, const LocalTensor<float> &tmpBuffer1,
                                   float scale, const int32_t curRowNum, const int32_t curColNum)
{
    int32_t curColNumAlign = RoundUp<float>(curColNum);
    MulABLastDimBrcInline<float, false>(mixLocal, mixLocal, rsqrtLocal, tmpBuffer0, curRowNum, curColNum);
    Muls(mixLocal, mixLocal, scale, curRowNum * curColNumAlign);
    PipeBarrier<PIPE_V>();
    AddBAFirstDimBrcInline<float>(mixLocal, mixLocal, hcBaseLocal, curRowNum, curColNum);
    SigmoidPerf(postLocal, mixLocal, tmpBuffer1, curRowNum * curColNumAlign);
    Muls(postLocal, postLocal, static_cast<float>(2.0f), curRowNum * curColNumAlign);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void LastDimReduceMaxPerf(const LocalTensor<float> &output, const LocalTensor<float> &input,
                                            const uint32_t curRowNum, const uint32_t curColNum)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(float);
    WholeReduceMax(output, input, curColNum, curRowNum, 1, 1, CeilDiv(curColNum, elemInOneBlock),
                   ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void LastDimReduceSumPerf(const LocalTensor<float> &output, const LocalTensor<float> &input,
                                            const uint32_t curRowNum, const uint32_t curColNum)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(float);
    WholeReduceSum(output, input, curColNum, curRowNum, 1, 1, CeilDiv(curColNum, elemInOneBlock));
    PipeBarrier<PIPE_V>();
}

// 中间轴(dim1)归约:dim1==4 时树形 (in[0]+in[3])+(in[1]+in[2]),写不同 buffer 可背靠背流水,
// PipeBarrier 3→2 (vs ReduceSumARAPerf 顺序)。其他 dim1 fallback 顺序归约保安全。
// 同步自源仓 yayake/mhc_pre_clamp_sinkhorn 5c9dd77 (commit 428bb15, 大 shape false -3.8%)。
__aicore__ inline void ReduceSumMidDimPerf(const LocalTensor<float> &output, const LocalTensor<float> &input,
                                           const LocalTensor<float> &transposeBuf, const uint32_t dim0,
                                           const uint32_t dim1, const uint32_t dim2)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(float);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(float);
    uint32_t dim2Align = RoundUp<float>(dim2);

    // Copy first dim1 slice (j=0) to output, same as original.
    DataCopyParams copyParams;
    copyParams.blockCount = dim0;
    copyParams.blockLen = dim2Align / elemInOneBlock;
    copyParams.srcStride = (dim1 - 1) * (dim2Align / elemInOneBlock);
    copyParams.dstStride = 0;
    DataCopy(output, input, copyParams);
    PipeBarrier<PIPE_V>();

    uint32_t dim2RepeatTimes = dim2 / elemInOneRepeat;
    uint32_t dim2Reminder = dim2 % elemInOneRepeat;
    BinaryRepeatParams instrParams;
    instrParams.dstBlkStride = 1;
    instrParams.src0BlkStride = 1;
    instrParams.src1BlkStride = 1;
    instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
    instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
    instrParams.src1RepStride = DEFAULT_REPEAT_STRIDE;

    // Lambda to perform Add with repeat/reminder split, matching original pattern.

    if (dim1 == 4) {
        // Tree reduction: (in[0]+in[3]) + (in[1]+in[2])
        // Reduces PipeBarriers from 3 (sequential j=1,2,3) to 2.
        // Step 1 (parallel pair): transposeBuf = in[1]+in[2], output += in[3]
        for (uint32_t i = 0; i < dim0; i++) {
            uint32_t base = i * dim1 * dim2Align;
            uint32_t dstOff = i * dim2Align;
            if (dim2RepeatTimes > 0) {
                Add(transposeBuf[dstOff], input[base + dim2Align], input[base + 2 * dim2Align],
                    elemInOneRepeat, dim2RepeatTimes, instrParams);
            }
            if (dim2Reminder != 0) {
                uint32_t remOff = dim2RepeatTimes * elemInOneRepeat;
                Add(transposeBuf[dstOff + remOff], input[base + dim2Align + remOff],
                    input[base + 2 * dim2Align + remOff], dim2Reminder, 1, instrParams);
            }
        }
        for (uint32_t i = 0; i < dim0; i++) {
            uint32_t base = i * dim1 * dim2Align;
            uint32_t dstOff = i * dim2Align;
            if (dim2RepeatTimes > 0) {
                Add(output[dstOff], output[dstOff], input[base + 3 * dim2Align],
                    elemInOneRepeat, dim2RepeatTimes, instrParams);
            }
            if (dim2Reminder != 0) {
                uint32_t remOff = dim2RepeatTimes * elemInOneRepeat;
                Add(output[dstOff + remOff], output[dstOff + remOff],
                    input[base + 3 * dim2Align + remOff], dim2Reminder, 1, instrParams);
            }
        }
        PipeBarrier<PIPE_V>();
        // Step 2: output += transposeBuf
        for (uint32_t i = 0; i < dim0; i++) {
            uint32_t dstOff = i * dim2Align;
            if (dim2RepeatTimes > 0) {
                Add(output[dstOff], output[dstOff], transposeBuf[dstOff],
                    elemInOneRepeat, dim2RepeatTimes, instrParams);
            }
            if (dim2Reminder != 0) {
                uint32_t remOff = dim2RepeatTimes * elemInOneRepeat;
                Add(output[dstOff + remOff], output[dstOff + remOff],
                    transposeBuf[dstOff + remOff], dim2Reminder, 1, instrParams);
            }
        }
        PipeBarrier<PIPE_V>();
    } else if (dim1 > 4) {
        // General pairwise tree, in-place on input slices 1..dim1-1 (live set starts at
        // slice index 1, stride doubles each round). PipeBarriers: ceil(log2(dim1-1)) + 1,
        // vs sequential (dim1-1) barriers per dim0 row. NOTE: destroys input slices
        // 1..dim1-1 — caller must not reuse the input tensor afterwards (queue tensors
        // that get FreeTensor'd right after the reduce are safe).
        uint32_t nLive = dim1 - 1;
        uint32_t stride = 1; // live slices reside at input slice indices 1 + k * stride
        while (nLive > 1) {
            uint32_t pairs = nLive / 2;
            for (uint32_t i = 0; i < dim0; i++) {
                for (uint32_t p = 0; p < pairs; p++) {
                    uint32_t dstOff = (i * dim1 + 1 + (2 * p) * stride) * dim2Align;
                    uint32_t srcOff = (i * dim1 + 1 + (2 * p + 1) * stride) * dim2Align;
                    if (dim2RepeatTimes > 0) {
                        Add(input[dstOff], input[dstOff], input[srcOff], elemInOneRepeat,
                            dim2RepeatTimes, instrParams);
                    }
                    if (dim2Reminder != 0) {
                        Add(input[dstOff + dim2RepeatTimes * elemInOneRepeat],
                            input[dstOff + dim2RepeatTimes * elemInOneRepeat],
                            input[srcOff + dim2RepeatTimes * elemInOneRepeat], dim2Reminder, 1, instrParams);
                    }
                }
            }
            // odd tail slice (index 1 + (nLive-1)*stride) stays live at 1 + pairs*2*stride
            PipeBarrier<PIPE_V>();
            nLive = pairs + (nLive % 2);
            stride *= 2;
        }
        // fold the last live slice (index 1) into output (output already holds slice 0)
        for (uint32_t i = 0; i < dim0; i++) {
            uint32_t srcOff = (i * dim1 + 1) * dim2Align;
            if (dim2RepeatTimes > 0) {
                Add(output[i * dim2Align], output[i * dim2Align], input[srcOff], elemInOneRepeat,
                    dim2RepeatTimes, instrParams);
            }
            if (dim2Reminder != 0) {
                Add(output[i * dim2Align + dim2RepeatTimes * elemInOneRepeat],
                    output[i * dim2Align + dim2RepeatTimes * elemInOneRepeat],
                    input[srcOff + dim2RepeatTimes * elemInOneRepeat], dim2Reminder, 1, instrParams);
            }
        }
        PipeBarrier<PIPE_V>();
    } else {
        // Fallback: original sequential reduction for non-standard dim1
        for (uint32_t i = 0; i < dim0; i++) {
            uint32_t base = i * dim1 * dim2Align;
            uint32_t dstOff = i * dim2Align;
            for (uint32_t j = 1; j < dim1; j++) {
                if (dim2RepeatTimes > 0) {
                    Add(output[dstOff], output[dstOff], input[base + j * dim2Align],
                        elemInOneRepeat, dim2RepeatTimes, instrParams);
                }
                if (dim2Reminder != 0) {
                    uint32_t remOff = dim2RepeatTimes * elemInOneRepeat;
                    Add(output[dstOff + remOff], output[dstOff + remOff],
                        input[base + j * dim2Align + remOff], dim2Reminder, 1, instrParams);
                }
                PipeBarrier<PIPE_V>();
            }
        }
        PipeBarrier<PIPE_V>();
    }
}


// 暂时只支持R轴小于64,既curColNum不能超过64
__aicore__ inline void SoftmaxFP32Perf(const LocalTensor<float> &output, const LocalTensor<float> &input,
                                       const LocalTensor<float> &tmpReduceBuffer,
                                       const LocalTensor<float> tmpBrcbBuffer, const int32_t curRowNum,
                                       const int32_t curColNum, float eps)
{
    LastDimReduceMaxPerf(tmpReduceBuffer, input, curRowNum, curColNum);
    SubABLastDimBrcInline<float, true>(output, input, tmpReduceBuffer, tmpBrcbBuffer, curRowNum, curColNum);
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    Exp(output, output, curRowNum * curColNumAlign);
    PipeBarrier<PIPE_V>();
    LastDimReduceSumPerf(tmpReduceBuffer, output, curRowNum, curColNum);
    DivABLastDimBrcInline<float, true>(output, output, tmpReduceBuffer, tmpBrcbBuffer, curRowNum, curColNum);
    // 新版下面这个eps不要加了
    // if (eps != 0.0f) {
    //     Adds(output, output, eps, curRowNum * curColNumAlign);
    // }
    PipeBarrier<PIPE_V>();
}

// (bs, hc_mult, hc_mult) = (bs, hc_mult, hc_mult) +  (bs, 1, hc_mult)

template <typename T>
__aicore__ inline void DivABABrcInline(const LocalTensor<T> &output, const LocalTensor<T> &input0,
                                       const LocalTensor<T> &input1, const uint32_t dim0, const uint32_t dim1,
                                       const uint32_t dim2)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    uint32_t dim2Align = RoundUp<T>(dim2);
    uint32_t dim2RepeatTimes = dim2 / elemInOneRepeat;
    uint32_t dim2Reminder = dim2 % elemInOneRepeat;
    uint32_t dim2RepeatStride = CeilDiv(dim2, elemInOneBlock);
    // 在dim1方向开repeat
    BinaryRepeatParams instrParams;
    if (dim1 >= dim2RepeatTimes) {
        instrParams.dstBlkStride = 1;
        instrParams.src0BlkStride = 1;
        instrParams.src1BlkStride = 1;
        instrParams.dstRepStride = dim2RepeatStride;
        instrParams.src0RepStride = dim2RepeatStride;
        instrParams.src1RepStride = 0;
        for (uint32_t i = 0; i < dim0; i++) {
            for (uint32_t j = 0; j < dim2RepeatTimes; j++) {
                Div(output[i * dim1 * dim2Align + j * elemInOneRepeat],
                    input0[i * dim1 * dim2Align + j * elemInOneRepeat], input1[i * dim2Align + j * elemInOneRepeat],
                    elemInOneRepeat, dim1, instrParams);
            }
            if (dim2Reminder != 0) {
                Div(output[i * dim1 * dim2Align + dim2RepeatTimes * elemInOneRepeat],
                    input0[i * dim1 * dim2Align + dim2RepeatTimes * elemInOneRepeat],
                    input1[i * dim2Align + dim2RepeatTimes * elemInOneRepeat], dim2Reminder, dim1, instrParams);
            }
        }
    } else {
        // 在dim2方向开repeat
        instrParams.dstBlkStride = 1;
        instrParams.src0BlkStride = 1;
        instrParams.src1BlkStride = 1;
        instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
        instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
        instrParams.src1RepStride = DEFAULT_REPEAT_STRIDE;
        for (uint32_t i = 0; i < dim0; i++) {
            for (uint32_t j = 0; j < dim1; j++) {
                Div(output[i * dim1 * dim2Align + j * dim2Align], input0[i * dim1 * dim2Align + j * dim2Align],
                    input1[i * dim2Align], dim2);
            }
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void CopyIn(const GlobalTensor<T> &inputGm, const LocalTensor<T> &inputTensor, const uint16_t nBurst,
                              const uint32_t copyLen, uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = dstStride / elemInOneBlock;
    DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
}

// (bs, hc_mult, hc_mult) --> (bs, hc_mult, hc_mult_align)
// 两分支对应两种 GM 布局（由调用方 tiling 路径决定，不是同一 gather 的循环序选择）：
//  - outerLoop <= nBurst（key=0，GM 行主序 [rows, hcMix]）：全部块的 src/dst 下标同为
//    i*nBurst+j 的等差数列，总块数不超上限时整圈循环可合为单条 CopyIn（超限退回循环）。
//  - 否则（key=1，GM 为 [kcore, bs, hcMix] 的转置 gather）：src 下标 i+gmFirstDim*j 与
//    dst 下标 i+nBurst*j 是两个不同线性式，单条 DataCopy 要求同一下标驱动 src/dst，
//    故仅 gmFirstDim == nBurst 才可合并；当前调用点（bs=4 vs rowFactor=2）不满足，
//    循环即 910B 下限（SetLoopModePara/MultiCopy 多维 DMA 仅 950 支持）。
// 前置条件：gmLastDim >= copyLen（防无符号下溢）。
template <typename T>
__aicore__ inline void CopyInWithOuterFor(const GlobalTensor<T> &inputGm, const LocalTensor<T> &inputTensor, const uint16_t outerLoop,
                                          const uint16_t nBurst, const uint32_t copyLen, const uint32_t gmFirstDim,
                                          const uint32_t gmLastDim)
{
    uint32_t ubLastDimAlign = RoundUp<T>(copyLen);
    if (outerLoop <= nBurst) {
        // MTE 单条指令 blockCount 保守上限（DataCopyParams 文档范围 [0,4095]）
        constexpr uint32_t COPY_MAX_BLOCK_COUNT = 4095;
        uint32_t totalBlocks = static_cast<uint32_t>(outerLoop) * nBurst;
        if (totalBlocks <= COPY_MAX_BLOCK_COUNT) {
            CopyIn(inputGm, inputTensor, static_cast<uint16_t>(totalBlocks), copyLen, gmLastDim - copyLen);
        } else {
            for (uint32_t i = 0; i < outerLoop; i++) {
                CopyIn(inputGm[i * nBurst * gmLastDim], inputTensor[i * nBurst * ubLastDimAlign], nBurst, copyLen,
                       gmLastDim - copyLen);
            }
        }
    } else {
        uint32_t srcStride = (gmLastDim - copyLen) + (gmFirstDim - 1) * gmLastDim;
        uint32_t dstStride = (nBurst - 1) * ubLastDimAlign;
        for (uint32_t i = 0; i < nBurst; i++) {
            CopyIn(inputGm[i * gmLastDim], inputTensor[i * ubLastDimAlign], outerLoop, copyLen, srcStride, dstStride);
        }
    }
}

template <typename T>
__aicore__ inline void CopyOut(const LocalTensor<T> &outputTensor, const GlobalTensor<T> &outputGm,
                               const uint16_t nBurst, const uint32_t copyLen, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad(outputGm, outputTensor, dataCopyParams);
}

__aicore__ inline void VToMTE3Sync()
{
    event_t eventIDVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVtoMTE3);
}

__aicore__ inline void MTE3ToVSync()
{
    event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
}

} // namespace MhcPreClampSinkhorn

#endif