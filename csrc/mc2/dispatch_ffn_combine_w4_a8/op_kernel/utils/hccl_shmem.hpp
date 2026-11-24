#ifndef SYNC_UTIL_HPP
#define SYNC_UTIL_HPP


#include "kernel_operator.h"
#include "const_args.hpp"
#include "a2_cross_server.hpp"

#ifdef HCCL_COMM
#include "moe_distribute_base.h"
using namespace AscendC::HcclContextDef;

#else
#include "shmem_api.h"
#endif

#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
constexpr int32_t MAX_RANK_SIZE = 32;
constexpr int32_t SHMEM_MEM = 700 * MB_SIZE;

constexpr uint16_t SEND_SYNC_EVENT_ID = 9;
constexpr uint16_t RECV_SYNC_EVENT_ID = 10;

constexpr uint32_t SELF_STATE_OFFSET = 256 * 1024;
constexpr uint32_t STATE_OFFSET = 512;

FORCE_INLINE_AICORE void AicSyncAll() {
    AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(8);
    AscendC::CrossCoreWaitFlag<0x0>(8);
}

template<typename T>
FORCE_INLINE_AICORE void gm_store(__gm__ T *addr, T val) {
    *((__gm__ T *)addr) = val;
}

template<typename T>
FORCE_INLINE_AICORE T gm_load(__gm__ T *cache) {
    return *((__gm__ T *)cache);
}

template<typename T>
FORCE_INLINE_AICORE void gm_dcci(__gm__ T * addr) {
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(addr));

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

FORCE_INLINE_AICORE int32_t gm_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    do {
        gm_dcci((__gm__ uint8_t *)sig_addr);
        if (*sig_addr == cmp_val) {
            return *sig_addr;
        }
        if (*sig_addr == cmp_val + 1) {
            return *sig_addr;
        }
    } while (true);
    return -1;
}

FORCE_INLINE_AICORE bool gm_signal_wait_until_ne_try(__gm__ int32_t *sig_addr, int32_t cmp_val,
                                                     uint64_t timeoutCycles) {
    constexpr uint32_t kCheckMask = 1024U - 1U;
    const int64_t waitStart = AscendC::GetSystemCycle();
    uint32_t poll = 0;
    do {
        AscendC::LocalTensor<int32_t> ub;
        ub.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
        ub.address_.bufferAddr = 0;
        AscendC::GlobalTensor<int32_t> sig;
        sig.SetGlobalBuffer(sig_addr);
        AscendC::DataCopy(ub, sig, 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        if (ub(0) != cmp_val) {
            return true;
        }
        if (((++poll & kCheckMask) == 0U) &&
            static_cast<uint64_t>(AscendC::GetSystemCycle() - waitStart) >= timeoutCycles) {
            return false;
        }
    } while (true);
    return false;
}

FORCE_INLINE_AICORE void gm_signal_wait_until_ne(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    // Bounded wait: a cross-server producer that never lands must surface as a
    // Trap (~5s) instead of an unbounded spin (234s watchdog / silent hang).
    constexpr uint64_t kTimeoutCycles = 5ULL * 1000ULL * 1000ULL * 50ULL;
    constexpr uint32_t kCheckMask = 1024U - 1U;
    const int64_t waitStart = AscendC::GetSystemCycle();
    uint32_t poll = 0;
    do {
        AscendC::LocalTensor<int32_t> ub;
        ub.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
        ub.address_.bufferAddr = 0;
        AscendC::GlobalTensor<int32_t> sig;
        sig.SetGlobalBuffer(sig_addr);
        AscendC::DataCopy(ub, sig, 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        if (ub(0) != cmp_val) {
            return;
        }
        if ((++poll & kCheckMask) == 0U &&
            static_cast<uint64_t>(AscendC::GetSystemCycle() - waitStart) >= kTimeoutCycles) {
            AscendC::Trap();
            return;
        }
    } while (true);
    return;
}


class HcclShmem {
public:
    #ifdef HCCL_COMM    // HCCL needs to initialize the HCCL context
        __gm__ HcclOpResParamCustom *WinContext_{nullptr};
        Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
        AscendC::LocalTensor<int32_t> ub;
        // A2 (Ascend910B) has no HCCL symmetric window registered through
        // GetHcclContext. Instead, CANN's KFC CommContext (
        // cann_ops_transformer get_symm_buffer_for_mega_moe) exposes one
        // registered CCL buffer per rank in epHcclBuffer_[rank], with a
        // symmetric address mapping across ranks. layout:
        //   uint32 epRankId; uint32 rankSizePerServer; uint64 kfcContextAddr;
        //   uint64 epHcclBuffer_[rank]; ...
        __gm__ uint64_t *kfcCtx_{nullptr};
        bool useKfc_{false};
        FORCE_INLINE_AICORE
        HcclShmem(){
            auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
            WinContext_ = (__gm__ HcclOpResParamCustom *)contextGM0;

            m_rank = WinContext_->localUsrRankId;
            m_rankSize = WinContext_->rankSize;
            m_segmentSize = WinContext_->winSize;
        }
        FORCE_INLINE_AICORE
        void initKfc(GM_ADDR context, int32_t rankSize, uint64_t segmentSize) {
            kfcCtx_ = reinterpret_cast<__gm__ uint64_t *>(context);
            m_rank = static_cast<int32_t>(kfcCtx_[0]);
            m_rankSize = rankSize;
            m_segmentSize = static_cast<size_t>(segmentSize);
            useKfc_ = true;
        }
    #else
        FORCE_INLINE_AICORE
        HcclShmem(){
            m_segmentSize = SHMEM_MEM;
        }
        FORCE_INLINE_AICORE 
        void initShmem(GM_ADDR symmetricPtr_, size_t rank, size_t rankSize) {
            symmetricPtr = symmetricPtr_;
            m_rank = rank;
            m_rankSize = rankSize;
        }
    #endif

    FORCE_INLINE_AICORE
    GM_ADDR operator() () const {   // No parameters: return pointer to local peermem
        #ifdef HCCL_COMM
            if (useKfc_) {
                return reinterpret_cast<GM_ADDR>(kfcCtx_[2 + m_rank]);
            }
            return (GM_ADDR)(WinContext_->localWindowsIn);
        #else
            return reinterpret_cast<GM_ADDR>(shmem_ptr(symmetricPtr, m_rank));
        #endif
    }

    FORCE_INLINE_AICORE
    GM_ADDR operator() (int32_t index) const {  // With index parameter: return pointer to the base address of remote peermem
        #ifdef HCCL_COMM
            if (xs_ != nullptr && xs_->ready) {
                return xs_->In(index);
            }
            if (useKfc_) {
                return reinterpret_cast<GM_ADDR>(kfcCtx_[2 + index]);
            }
            return (GM_ADDR)((index == m_rank) ? WinContext_->localWindowsIn :
                                    ((HcclRankRelationResV2Custom *)(WinContext_->remoteRes[index].nextDevicePtr))->windowsIn);
        #else
            return reinterpret_cast<GM_ADDR>(shmem_ptr(symmetricPtr, index));
        #endif
    }

    FORCE_INLINE_AICORE
    GM_ADDR operator () (int64_t offset, int32_t rankId) const  {
        #ifdef HCCL_COMM
            if (offset < 0 || offset >= m_segmentSize) {
                return nullptr;
            }
            if (rankId < 0 || rankId >= m_rankSize) {
                return nullptr;
            }
            if (xs_ != nullptr && xs_->ready) {
                return xs_->In(rankId) + offset;
            }
            if (useKfc_) {
                return reinterpret_cast<GM_ADDR>(kfcCtx_[2 + rankId]) + offset;
            }
            return (GM_ADDR)((rankId == m_rank) ? WinContext_->localWindowsIn :
                                    ((HcclRankRelationResV2Custom *)(WinContext_->remoteRes[rankId].nextDevicePtr))->windowsIn) + offset;
        #else
            return reinterpret_cast<GM_ADDR>(shmem_ptr((symmetricPtr + offset), rankId));
        #endif
    }



    // Cross-node (EP spans servers). Only then is D staged in the local Out
    // window and pushed with AI-RDMA (REMOTE_OUTPUT).
    FORCE_INLINE_AICORE
    bool CrossNode() const {
        return xs_ != nullptr && xs_->ready && xs_->rankSize > DffcA2::kServerRankSize;
    }

    // Local Out window base.
    FORCE_INLINE_AICORE
    GM_ADDR outBase() const {
        if (CrossNode()) {
            return xs_->LocalOut();
        }
        return (*this)();
    }

    // True when rankId lives on another server and can only be reached with
    // AI-RDMA (no IPC mapping, no remote read).
    FORCE_INLINE_AICORE
    bool IsCrossServer(int32_t rankId) const {
        return xs_ != nullptr && xs_->ready && xs_->CrossServer(rankId);
    }

    // Destination of a combine (D) write. Self and same-server peers keep the
    // original IPC peer-window write. Cross-server peers cannot be written by
    // MTE, so those rows are staged in this rank's Out window (the AI-RDMA
    // source key LOCAL_OUTPUT only covers windowsOut) and forwarded to the
    // peer In window by the D push pass that runs after the epilogue.
    FORCE_INLINE_AICORE
    GM_ADDR outAt(int64_t offset, int32_t rankId) const {
        if (xs_ != nullptr && xs_->ready) {
            if (xs_->CrossServer(rankId)) {
                return xs_->LocalOut() + offset;
            }
            return xs_->In(rankId) + offset;
        }
        return (*this)(offset, rankId);
    }

    FORCE_INLINE_AICORE
    size_t SegmentSize() const {
        return m_segmentSize;
    }

    FORCE_INLINE_AICORE
    void setSegmentSize(size_t size) {
        m_segmentSize = size;
    }

    // A2 cross-server transport (KFC + AI-RDMA). When set, rank-to-rank state
    // posted by CrossRankSyncV2Set goes over RoCE for remote servers instead of
    // a direct (invalid) write to the peer window.
    FORCE_INLINE_AICORE
    void setCrossServer(DffcA2::CrossServerCtx *ctx, AscendC::LocalTensor<uint64_t> ubLocal,
                        AscendC::LocalTensor<uint32_t> ubHead) {
        xs_ = ctx;
        ubRdmaLocal_ = ubLocal;
        ubRdmaHead_ = ubHead;
    }

    // Cross-server V2 status staging: a 64B GM slot pair in the Out window
    // holding 32B blocks whose word0 is float 1.0 (state == 1) and 0.0
    // (state == 0). Posting from constant slots keeps the per-call work to a
    // single AI-RDMA write; only word0 of each block is consumed by the peer.
    FORCE_INLINE_AICORE
    void setV2Stage(GM_ADDR slot1) {
        xsV2Slot1_ = slot1;
    }

    // Temporary bring-up diagnostics (DFFC_XS_STAGE5): when dbg != nullptr the
    // V2 wait becomes bounded and both sides dump what they observed.
    FORCE_INLINE_AICORE
    void setV2Debug(GM_ADDR dbg, bool bounded) {
        xsV2Dbg_ = dbg;
        xsV2Bounded_ = bounded;
    }

    // Write one diagnostic record of 8 int32 at out + off (64B-aligned slot).
    FORCE_INLINE_AICORE
    void V2DiagWrite(int32_t off, int32_t a, int32_t b, int32_t c, int32_t d,
                     int32_t e, int32_t f) {
        if (xsV2Dbg_ == nullptr) {
            return;
        }
        __gm__ int32_t *p = reinterpret_cast<__gm__ int32_t *>(xsV2Dbg_) + off;
        p[0] = a;
        p[1] = b;
        p[2] = c;
        p[3] = d;
        p[4] = e;
        p[5] = f;
        AscendC::GlobalTensor<int32_t> g;
        g.SetGlobalBuffer(p);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(g);
    }

    FORCE_INLINE_AICORE
    int32_t RankSize() const {
        return m_rankSize;
    }


    FORCE_INLINE_AICORE
    ~HcclShmem() {
    }


    FORCE_INLINE_AICORE
    void CrossRankSync() {
        uint64_t flag_offset = (m_segmentSize - MB_SIZE) / sizeof(int32_t);
        __gm__ int32_t* sync_counter = (__gm__ int32_t*)(*this)() + flag_offset;
        __gm__ int32_t* sync_base = (__gm__ int32_t*)(*this)() + flag_offset + 2048;
        int count = gm_load(sync_base) + 1;
        int vec_id = AscendC::GetBlockIdx();
        int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        for(int i = vec_id; i < m_rankSize; i += vec_size) {
            __gm__ int32_t* sync_remote = (__gm__ int32_t*)((*this)(i)) + flag_offset + m_rank * 16;
            gm_store(sync_remote, count);
            gm_dcci((__gm__ uint8_t*)sync_remote);
            auto sync_check = sync_counter + i * 16;
            gm_signal_wait_until_eq_for_barrier(sync_check, count);
        }

        AscendC::SyncAll<true>();
        gm_store(sync_base, count);
    }


    FORCE_INLINE_AICORE
    void InitStatusTargetSum()
    {
        using namespace AscendC;
        uint64_t flag_offset = (m_segmentSize - MB_SIZE) + SELF_STATE_OFFSET;
        //uint64_t self_state_offset = (m_segmentSize - 2 * MB_SIZE);
        // ep state
        //uint32_t coreIdx = get_block_idx();;
        uint32_t coreIdx = GetBlockIdx();
        GlobalTensor<int32_t> selfStatusTensor;
        selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)((*this)() + flag_offset));
        __asm__ __volatile__("");
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfStatusTensor[coreIdx * UB_ALIGN]);
        __asm__ __volatile__("");
        int32_t state = selfStatusTensor(coreIdx * UB_ALIGN);
        if (state == 0) {
            sumTarget_ = static_cast<float>(1.0);
            selfStatusTensor(coreIdx * UB_ALIGN) = 0x3F800000;  // 1.0f
            epStateValue_ = 0x3F800000;                          // 1.0f
        } else {
            sumTarget_ = static_cast<float>(0.0);
            selfStatusTensor(coreIdx * UB_ALIGN) = 0;
            epStateValue_ = 0;
        }
        __asm__ __volatile__("");
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfStatusTensor[coreIdx * UB_ALIGN]);
        __asm__ __volatile__("");
    }

    FORCE_INLINE_AICORE
    void CrossRankSyncV2Set(AscendC::LocalTensor<int32_t> ctrBuffer) {
        //subblockid = 0
        uint32_t stateOffset_ =  STATE_OFFSET;
        // uint32_t epStateOffsetOnWin_ = m_rank * stateOffset_;
        
        uint64_t flag_offset = (m_segmentSize - MB_SIZE) + m_rank * stateOffset_;
        //uint64_t flag_offset = (m_segmentSize - MB_SIZE);
        int vec_size = get_block_num();
        int vec_id = get_block_idx();
 
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(RECV_SYNC_EVENT_ID);
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SEND_SYNC_EVENT_ID);
        AscendC::CrossCoreWaitFlag(SEND_SYNC_EVENT_ID);
        pipe_barrier(PIPE_ALL);
 
        ctrBuffer.SetValue(0, epStateValue_);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        int32_t xsPostsTotal = 0;
        int32_t xsPostFails = 0;
        for (uint32_t dstEpIdx = vec_id; dstEpIdx < m_rankSize; dstEpIdx += vec_size) {
            if (xs_ != nullptr && xs_->ready &&
                xs_->CrossServer(static_cast<int32_t>(dstEpIdx))) {
                // Cross-server: the peer In window is RoCE-only. Post the
                // 32B state from a constant staging slot; the D push pass ran
                // before this (SyncAll) and used the same QP, so the data
                // lands before the state and the receiver may read D as soon
                // as it sees every rank's state.
                if (xsV2Slot1_ != nullptr) {
                    GM_ADDR slot = xsV2Slot1_ + ((epStateValue_ == 0) ? 32 : 0);
                    if (xsV2Dbg_ != nullptr) {
                        if (!DffcA2::AivRdmaPostSendTry(
                                slot, xs_->In(static_cast<int32_t>(dstEpIdx)) + flag_offset,
                                static_cast<uint64_t>(dstEpIdx), 32ULL, xs_->qp, ubRdmaLocal_,
                                ubRdmaHead_)) {
                            ++xsPostFails;
                        }
                    } else {
                        DffcA2::AivRdmaPostSend(slot,
                                                xs_->In(static_cast<int32_t>(dstEpIdx)) + flag_offset,
                                                static_cast<uint64_t>(dstEpIdx), 32ULL, xs_->qp,
                                                ubRdmaLocal_, ubRdmaHead_);
                    }
                }
                ++xsPostsTotal;
                continue;
            }
            AscendC::GlobalTensor<int32_t> gmDstStates;
            gmDstStates.SetGlobalBuffer((__gm__ int32_t*)((*this)(flag_offset, dstEpIdx)));
            DataCopy(gmDstStates, ctrBuffer, 8);
            ++xsPostsTotal;
        }
        if (xsV2Dbg_ != nullptr) {
            int32_t stageWord0 = 0;
            int32_t slotOff = 0;
            if (xsV2Slot1_ != nullptr) {
                stageWord0 = *reinterpret_cast<__gm__ int32_t *>(
                    xsV2Slot1_ + ((epStateValue_ == 0) ? 32 : 0));
                slotOff = static_cast<int32_t>(xsV2Slot1_ - xs_->LocalOut());
            }
            V2DiagWrite(2048 + vec_id * 16, xsPostsTotal, xsPostFails, stageWord0,
                        static_cast<int32_t>(epStateValue_), slotOff,
                        static_cast<int32_t>(flag_offset));
        }
        AscendC::CrossCoreWaitFlag(RECV_SYNC_EVENT_ID);
    }

    FORCE_INLINE_AICORE
    void CrossRankSyncV2Wait(AscendC::LocalTensor<float> statusTensor, AscendC::LocalTensor<float> gatherMaskOutTensor,
        AscendC::LocalTensor<uint32_t> gatherTmpTensor, AscendC::LocalTensor<float> statusSumOutTensor) {

        uint64_t flag_offset = (m_segmentSize - MB_SIZE);
        int vec_size = get_block_num();
        int vec_id = get_block_idx();
        uint32_t stateOffset_ =  STATE_OFFSET;

        uint32_t sendRankNum_ = m_rankSize / vec_size;
        uint32_t remainderRankNum = m_rankSize % vec_size;
        uint32_t startRankId_ = sendRankNum_ * vec_id;
        if (vec_id < remainderRankNum) {
            sendRankNum_++;
            startRankId_ += vec_id;
        } else {
            startRankId_ += remainderRankNum;
        }
        uint32_t endRankId_ = startRankId_ + sendRankNum_;
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SEND_SYNC_EVENT_ID);

        AscendC::GlobalTensor<float> epStatusSpaceGlobalTensor_;
        epStatusSpaceGlobalTensor_.SetGlobalBuffer((__gm__ float *)((*this)() + flag_offset));

        if (startRankId_ < m_rankSize) {
            AscendC::PipeBarrier<PIPE_ALL>();
            gatherTmpTensor.SetValue(0, 1);
            uint32_t mask = 1;  // gatherMask + sum
            uint64_t rsvdCnt = 0;
            // DataCopyParams intriParams{static_cast<uint16_t>(sendRankNum_), 1,
            //                         static_cast<uint16_t>((moeSendNum_ > 512) ? 7 : 15), 0}; 
            AscendC::DataCopyParams intriParams{static_cast<uint16_t>(sendRankNum_), 1,
                                    static_cast<uint16_t>(15), 0}; 

            float sumOfFlag = static_cast<float>(-1.0);
            float minTarget = (sumTarget_ * sendRankNum_) - (float)0.5;
            float maxTarget = (sumTarget_ * sendRankNum_) + (float)0.5;
            AscendC::SumParams sumParams{1, sendRankNum_, sendRankNum_};

            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

            uint32_t pollCnt = 0;
            const int64_t waitStart = AscendC::GetSystemCycle();
            while ((sumOfFlag < minTarget) || (sumOfFlag > maxTarget)) {
                AscendC::DataCopy<float>(statusTensor, epStatusSpaceGlobalTensor_[startRankId_ * stateOffset_ / sizeof(float)],
                                intriParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                GatherMask(gatherMaskOutTensor, statusTensor, gatherTmpTensor, true, mask,
                        {1, (uint16_t)sendRankNum_, 1, 0}, rsvdCnt);

                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                sumOfFlag = statusSumOutTensor.GetValue(0);
                ++pollCnt;
                if (xsV2Bounded_ && ((pollCnt & 1023U) == 0U) &&
                    static_cast<uint64_t>(AscendC::GetSystemCycle() - waitStart) >=
                        10ULL * 1000ULL * 1000ULL * 50ULL) {
                    break;   // diagnostic build only: never hang, report instead
                }
            }
            if (xsV2Dbg_ != nullptr) {
                V2DiagWrite(32 + vec_id * 16, static_cast<int32_t>(sumOfFlag * 1000.0f),
                            static_cast<int32_t>(startRankId_), static_cast<int32_t>(sendRankNum_),
                            static_cast<int32_t>(minTarget * 1000.0f),
                            static_cast<int32_t>(statusTensor.GetValue(0) * 1000.0f),
                            static_cast<int32_t>(pollCnt));
            }
        }

        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(RECV_SYNC_EVENT_ID);
        AscendC::CrossCoreWaitFlag(RECV_SYNC_EVENT_ID);

        //unpermute
        AscendC::CrossCoreWaitFlag(SEND_SYNC_EVENT_ID);
    }


    FORCE_INLINE_AICORE
    __gm__ int32_t* SyncBaseAddr() {
        uint64_t flag_offset = (m_segmentSize - MB_SIZE) / sizeof(int32_t);
        return (__gm__ int32_t*)(*this)() + flag_offset + 2048;
    }

private:
    GM_ADDR symmetricPtr;
    int32_t m_rank;
    int32_t m_rankSize;
    size_t m_segmentSize;
    float sumTarget_{0.0};
    int32_t epStateValue_;
    // Optional A2 cross-server transport (KFC context + AI-RDMA QPs).
    DffcA2::CrossServerCtx *xs_{nullptr};
    AscendC::LocalTensor<uint64_t> ubRdmaLocal_;
    AscendC::LocalTensor<uint32_t> ubRdmaHead_;
    // Cross-server V2 status staging slot (word0 float 1.0); +32B holds the
    // state == 0 variant.
    GM_ADDR xsV2Slot1_{nullptr};
    // Diagnostics only (nullptr in the functional build).
    GM_ADDR xsV2Dbg_{nullptr};
    bool xsV2Bounded_{false};
};


#endif
