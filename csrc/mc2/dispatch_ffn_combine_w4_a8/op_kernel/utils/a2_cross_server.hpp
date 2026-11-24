/**
 * A2 cross-server transport for dispatch_ffn_combine.
 *
 * A2 has no HCCL symmetric window through GetHcclContext (hcclCommRegister
 * returns "not supported"), so the operator uses CANN's KFC CommContext:
 *   Mc2MoeContext { u32 epRankId; u32 rankSizePerServer; u64 kfcContextAddr;
 *                   u64 epHcclBuffer_[1024]; ... }
 * epHcclBuffer_[rank] is an IPC-mapped peer window that only covers the local
 * server. Cross-server traffic must go through the AI-RDMA QPs described by
 * HcclA2CombineOpParam::aiRMAInfo, exactly like the product A2 MoE ops and the
 * MegaMoe A2 kernel.
 *
 * Key constraint of the AI-RDMA WQE layout: the local source segment must be
 * registered as LOCAL_OUTPUT (windowsOut) and the remote destination as
 * REMOTE_INPUT (the peer's windowsIn). A2 supports remote WRITE only; there is
 * no remote read path.
 */
#ifndef DFFC_A2_CROSS_SERVER_HPP
#define DFFC_A2_CROSS_SERVER_HPP

#include "kernel_operator.h"

// Self-contained: some translation units include this header before
// get_tensor_addr.hpp / hccl_shmem.hpp define the macro.
#ifndef FORCE_INLINE_AICORE
#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
#endif
// GetSystemCycle() (used for the 5s flag/QP timeouts) lives in the sys-var
// interface; without this include the call may resolve to a stub returning 0.
#include "basic_api/kernel_operator_sys_var_intf.h"
#include "moe_distribute_base.h"

namespace DffcA2 {

constexpr int32_t kServerRankSize = 8;      // A2: 8 die per server
constexpr uint32_t kMaxRankNum = 1024;      // Mc2MoeContext::epHcclBuffer_ size

// Mirror of Mc2Aclnn::Mc2MoeContext (mc2_moe_context.h). Defined locally so
// the op does not depend on the CANN private include path.
struct Mc2MoeContextLayout {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr;
    uint64_t epHcclBuffer[kMaxRankNum];
};

// Mirror of CANN's MemDetails (hccl_inner_def.h): size/addr/key.
struct A2MemDetail {
    uint64_t size;
    uint64_t addr;
    uint32_t key;
    uint32_t rsv;
};

// Mirror of AscendC::IbVerbsData (120 bytes): one entry per rank when
// HcclA2CombineOpParam::multiFlag != 0. `local*` are this rank's windows,
// `remote*` are the peer windows mapped into this process; both are directly
// addressable by MTE (this is how the product A2 ops copy peer GM data).
struct A2IbvData {
    A2MemDetail remoteInput;
    A2MemDetail remoteOutput;
    A2MemDetail localInput;
    A2MemDetail localOutput;
    uint8_t res[24];
};

FORCE_INLINE_AICORE void FillRdmaSendWqe(GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId,
                                         uint64_t messageLen, __gm__ uint8_t *wqeAddr, uint64_t curHead,
                                         uint64_t shift,
                                         decltype(HcclAiRMAInfo::memPtr) memInfoTable,
                                         decltype(HcclAiRMAInfo::sizeOfAiRMAMem) sizeOfMemDetail,
                                         HcclAiRMAMemType destType = HcclAiRMAMemType::REMOTE_INPUT)
{
    uint64_t ownBit = (curHead >> shift) & 1U;
    uint32_t byte4 = 3U;
    byte4 |= ((~ownBit) << 7U) & (1U << 7U);
    byte4 |= 1U << 8U;

    *(__gm__ uint32_t *)(wqeAddr) = byte4;
    *(__gm__ uint32_t *)(wqeAddr + 4) = messageLen;
    *(__gm__ uint32_t *)(wqeAddr + 8) = 0;
    *(__gm__ uint32_t *)(wqeAddr + 12) = 1U << 24U;
    *(__gm__ uint32_t *)(wqeAddr + 16) = 0;
    __gm__ HcclAiRMAMemInfo *memDetail = (__gm__ HcclAiRMAMemInfo *)(memInfoTable + sizeOfMemDetail * destRankId);
    *(__gm__ uint32_t *)(wqeAddr + 20) =
        ((__gm__ A2MemDetail *)(memDetail->memDetailPtr +
                                memDetail->sizeOfMemDetails *
                                    static_cast<uint32_t>(destType)))
            ->key;
    *(__gm__ uint64_t *)(wqeAddr + 24) = (uint64_t)(destDmaAddr);

    __gm__ uint8_t *sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
    *(__gm__ uint32_t *)(sgeAddr) = messageLen;
    *(__gm__ uint32_t *)(sgeAddr + sizeof(uint32_t)) =
        ((__gm__ A2MemDetail *)(memDetail->memDetailPtr +
                                memDetail->sizeOfMemDetails *
                                    static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT)))
            ->key;
    *(__gm__ uint64_t *)(sgeAddr + 2 * sizeof(uint32_t)) = (uint64_t)(srcDmaAddr);

    cacheWriteThrough(wqeAddr, sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
}

FORCE_INLINE_AICORE void PublishRdmaSendHead(__gm__ HcclAiRMAWQ *qpCtxEntry, uint64_t curHead,
                                             decltype(HcclAiRMAWQ::headAddr) curHardwareHead,
                                             AscendC::LocalTensor<uint64_t> &ubLocal,
                                             AscendC::LocalTensor<uint32_t> &ubLocalHead)
{
    uint64_t doorBellInfo = 0;
    doorBellInfo |= qpCtxEntry->wqn;
    doorBellInfo |= 0UL << 24UL;
    doorBellInfo |= (curHead % 65536UL) << 32UL;
    doorBellInfo |= static_cast<uint64_t>(qpCtxEntry->sl) << 48UL;

    __gm__ uint64_t *doorBellAddr = (__gm__ uint64_t *)(qpCtxEntry->dbAddr);
    AscendC::PipeBarrier<PIPE_ALL>();

    ubLocal.SetValue(0, doorBellInfo);
    AscendC::GlobalTensor<uint64_t> dbGlobalTensor;
    dbGlobalTensor.SetGlobalBuffer(doorBellAddr);
    AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(dbGlobalTensor, ubLocal, copyParams);
    AscendC::PipeBarrier<PIPE_ALL>();

    ubLocalHead.SetValue(0, static_cast<uint32_t>(curHead));
    AscendC::GlobalTensor<uint32_t> headGlobalTensor;
    headGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)curHardwareHead);
    AscendC::DataCopyExtParams copyParamsHead{1, 1 * sizeof(uint32_t), 0, 0, 0};
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(headGlobalTensor, ubLocalHead, copyParamsHead);
    AscendC::PipeBarrier<PIPE_ALL>();
}

// One AI-RDMA WRITE from local windowsOut to the peer's windowsIn.
// Returns false (instead of trapping) when the QP does not drain in 5s.
FORCE_INLINE_AICORE bool AivRdmaPostSendTry(GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId,
                                            uint64_t messageLen, __gm__ HcclAiRMAInfo *qpInfo,
                                            AscendC::LocalTensor<uint64_t> &ubLocal,
                                            AscendC::LocalTensor<uint32_t> &ubLocalHead,
                                            HcclAiRMAMemType destType = HcclAiRMAMemType::REMOTE_INPUT)
{
    auto qpNum = qpInfo->qpNum;
    auto qpCtxEntry =
        (__gm__ HcclAiRMAWQ *)(qpInfo->sqPtr + destRankId * qpNum * static_cast<uint64_t>(qpInfo->sizeOfAiRMAWQ));
    auto memInfoTable = qpInfo->memPtr;
    auto sizeOfMemDetail = qpInfo->sizeOfAiRMAMem;
    auto sqBaseAddr = qpCtxEntry->bufAddr;
    auto wqeSize = qpCtxEntry->wqeSize;
    auto curHardwareHead = qpCtxEntry->headAddr;
    cacheWriteThrough((__gm__ uint8_t *)curHardwareHead, 8);
    uint64_t curHead = *(__gm__ uint32_t *)(curHardwareHead);
    auto curHardwareTailAddr = qpCtxEntry->tailAddr;
    uint64_t shift = 15U;
    auto qpDepth = qpCtxEntry->depth;

    AscendC::PipeBarrier<PIPE_ALL>();

    // 5s bound instead of spinning forever if the QP is wedged.
    constexpr uint64_t kTimeoutCycles = 5ULL * 1000ULL * 1000ULL * 50ULL;
    const int64_t waitStart = AscendC::GetSystemCycle();
    while (true) {
        cacheWriteThrough((__gm__ uint8_t *)curHardwareTailAddr, 8);
        if ((curHead - *(__gm__ uint32_t *)(curHardwareTailAddr)) < qpDepth - 1) {
            break;
        }
        if (static_cast<uint64_t>(AscendC::GetSystemCycle() - waitStart) >= kTimeoutCycles) {
            return false;
        }
    }

    __gm__ uint8_t *wqeAddr = (__gm__ uint8_t *)(sqBaseAddr + wqeSize * (curHead % qpDepth));
    FillRdmaSendWqe(srcDmaAddr, destDmaAddr, destRankId, messageLen, wqeAddr, curHead, shift, memInfoTable,
                    sizeOfMemDetail, destType);
    AscendC::PipeBarrier<PIPE_ALL>();
    curHead++;
    PublishRdmaSendHead(qpCtxEntry, curHead, curHardwareHead, ubLocal, ubLocalHead);
    return true;
}

FORCE_INLINE_AICORE void AivRdmaPostSend(GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId,
                                         uint64_t messageLen, __gm__ HcclAiRMAInfo *qpInfo,
                                         AscendC::LocalTensor<uint64_t> &ubLocal,
                                         AscendC::LocalTensor<uint32_t> &ubLocalHead)
{
    if (!AivRdmaPostSendTry(srcDmaAddr, destDmaAddr, destRankId, messageLen, qpInfo, ubLocal, ubLocalHead)) {
        AscendC::Trap();
    }
}

// MTE2 flag wait. Never poll a peer-written cache line with scalar
// gm_dcci+gm_load: on A2 that starves the sender's AI-RDMA QP (see the
// MegaMoe cross-server fix). The 32B DMA keeps the line out of this core's
// D-Cache entirely.
FORCE_INLINE_AICORE void WaitFlagMte2(GM_ADDR flagAddr, int32_t expect, AscendC::LocalTensor<int32_t> &flagUb,
                                      uint64_t timeoutCycles = 5ULL * 1000ULL * 1000ULL * 50ULL)
{
    AscendC::GlobalTensor<int32_t> flagGm;
    flagGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flagAddr));
    constexpr uint32_t kCheckMask = 1024U - 1U;
    const int64_t waitStart = AscendC::GetSystemCycle();
    uint32_t poll = 0;
    while (true) {
        AscendC::DataCopy(flagUb, flagGm, 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        if (flagUb.GetValue(0) == expect) {
            return;
        }
        if ((++poll & kCheckMask) == 0U &&
            static_cast<uint64_t>(AscendC::GetSystemCycle() - waitStart) >= timeoutCycles) {
            AscendC::Trap();
            return;
        }
    }
}

// Parsed A2 cross-server context. Everything is read from the KFC
// Mc2MoeContext the op already receives; no host or Python change needed.
struct CrossServerCtx {
    __gm__ Mc2MoeContextLayout *mc2Ctx{nullptr};
    __gm__ HcclA2CombineOpParam *win{nullptr};
    __gm__ HcclAiRMAInfo *qp{nullptr};
    int32_t rank{-1};
    int32_t rankSize{0};
    int32_t serverId{-1};
    uint64_t winSize{0};
    int32_t multiFlag{-1};
    bool ready{false};
    // multiFlag != 0: per-rank IbVerbsData entries (120B each) whose
    // remoteInput/remoteOutput addresses are valid on every rank.
    __gm__ A2IbvData *ibv{nullptr};

    FORCE_INLINE_AICORE void Init(GM_ADDR context, int32_t rankSizeIn)
    {
        mc2Ctx = reinterpret_cast<__gm__ Mc2MoeContextLayout *>(context);
        rank = static_cast<int32_t>(mc2Ctx->epRankId);
        rankSize = rankSizeIn;
        serverId = rank / kServerRankSize;
        if (mc2Ctx->kfcContextAddr == 0ULL) {
            return;
        }
        win = reinterpret_cast<__gm__ HcclA2CombineOpParam *>(mc2Ctx->kfcContextAddr);
        multiFlag = static_cast<int32_t>(win->multiFlag);
        winSize = win->winSize;
        qp = reinterpret_cast<__gm__ HcclAiRMAInfo *>(win->aiRMAInfo);
        if (win->multiFlag != 0U) {
            // Multi-flag mode (the A2 KFC default): peer windows live in the
            // per-rank IbVerbsData table, not in windowsIn/Out[].
            if (win->data == nullptr) {
                return;
            }
            ibv = reinterpret_cast<__gm__ A2IbvData *>(win->data);
            ready = true;
            return;
        }
        ready = (qp != nullptr);
    }

    FORCE_INLINE_AICORE bool CrossServer(int32_t dstRank) const
    {
        return (dstRank / kServerRankSize) != serverId;
    }

    FORCE_INLINE_AICORE GM_ADDR LocalIn() const
    {
        return (ibv != nullptr) ? reinterpret_cast<GM_ADDR>(ibv[rank].localInput.addr)
                                : reinterpret_cast<GM_ADDR>(win->windowsIn[rank]);
    }

    FORCE_INLINE_AICORE GM_ADDR LocalOut() const
    {
        return (ibv != nullptr) ? reinterpret_cast<GM_ADDR>(ibv[rank].localOutput.addr)
                                : reinterpret_cast<GM_ADDR>(win->windowsOut[rank]);
    }

    FORCE_INLINE_AICORE GM_ADDR In(int32_t r) const
    {
        if (ibv != nullptr) {
            return (r == rank) ? reinterpret_cast<GM_ADDR>(ibv[r].localInput.addr)
                               : reinterpret_cast<GM_ADDR>(ibv[r].remoteInput.addr);
        }
        return reinterpret_cast<GM_ADDR>(win->windowsIn[r]);
    }

    FORCE_INLINE_AICORE GM_ADDR Out(int32_t r) const
    {
        if (ibv != nullptr) {
            return (r == rank) ? reinterpret_cast<GM_ADDR>(ibv[r].localOutput.addr)
                               : reinterpret_cast<GM_ADDR>(ibv[r].remoteOutput.addr);
        }
        return reinterpret_cast<GM_ADDR>(win->windowsOut[r]);
    }
};

} // namespace DffcA2

#endif // DFFC_A2_CROSS_SERVER_HPP
