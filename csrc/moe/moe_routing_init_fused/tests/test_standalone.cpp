/**
 * MoeRoutingInitFused standalone 精度 + 性能测试（纯 ACL，无 torch 依赖）。
 *
 * 参考实现跑在宿主机（与 kernel 相同的 counting-sort 语义）：
 *   counts[e] = #{i : ids[i]==e}; base[e] = 前缀和; dst[i] = base[e] + 组内序;
 *   rowIdx[i] = dst[i]; srcMap[dst] = i/k; sorted[dst] = hidden[i/k]
 * 对拍三项（全部要求精确）：expanded_row_idx、expert_tokens(int64)、sorted_hidden
 * （gather 搬运，bf16 位精确）。性能：warmup 后 N 次含同步调用取均值。
 *
 * 编译（示例，路径按实际安装位置）：
 *   g++ -O2 -std=c++17 test_standalone.cpp -o /tmp/test_rif \
 *     -I$ASCEND_HOME_PATH/include \
 *     -I<pkg>/vendors/custom_transformer/op_api/include \
 *     -L<pkg>/vendors/custom_transformer/op_api/lib -lcust_opapi \
 *     -L$ASCEND_HOME_PATH/lib64 -lascendcl \
 *     -Wl,-rpath,<pkg>/vendors/custom_transformer/op_api/lib
 * 运行：LD_LIBRARY_PATH 含包 op_api/lib；ASCEND_CUSTOM_OPP_PATH 指向包 vendors/<name>
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_routing_init_fused.h"

#define CHECK_RET(cond, expr) do { if (!(cond)) { expr; } } while (0)

static int64_t ShapeSize(const std::vector<int64_t> &s)
{
    int64_t n = 1;
    for (auto d : s) n *= d;
    return n;
}

static uint16_t Fp32ToBf16(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, 4);
    uint32_t lsb = (u >> 16) & 1;
    return static_cast<uint16_t>((u + 0x7FFFu + lsb) >> 16);
}

static float Bf16ToFp32(uint16_t v)
{
    uint32_t u = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static aclTensor *MakeTensor(void *dev, const std::vector<int64_t> &shape, aclDataType dt)
{
    return aclCreateTensor(shape.data(), shape.size(), dt, nullptr, 0, ACL_FORMAT_ND,
                           shape.data(), shape.size(), dev);
}

struct CaseResult {
    bool ok = true;
    int64_t maxIdxDiff = 0;
    int64_t badSortedRows = 0;
};

// skew: 0=均匀随机 1=全部挤 expert0（极端倾斜） 2=90% 挤 expert0
static CaseResult RunCase(int64_t S, int64_t K, int64_t E, int64_t kTop, aclrtStream stream,
                          std::mt19937 &rng, int64_t *outMicros, int skew = 0)
{
    int64_t active = S * kTop;
    // host 数据
    std::vector<float> hidden(S * K);
    std::uniform_real_distribution<float> hd(-2.0f, 2.0f);
    for (auto &v : hidden) v = hd(rng);
    std::vector<int32_t> ids(active);
    std::uniform_int_distribution<int> ed(0, static_cast<int>(E) - 1);
    for (auto &v : ids) v = ed(rng);
    if (skew == 1) {
        for (auto &v : ids) v = 0;
    } else if (skew == 2) {
        for (auto &v : ids) v = ((rng() % 10) == 0) ? ed(rng) : 0;
    }

    // 宿主机参考（与 kernel 相同的确定性语义）
    std::vector<int64_t> refCounts(E, 0);
    for (int64_t i = 0; i < active; i++) refCounts[ids[i]]++;
    std::vector<int64_t> base(E, 0);
    int64_t running = 0;
    for (int64_t e = 0; e < E; e++) {
        base[e] = running;
        running += refCounts[e];
    }
    std::vector<int64_t> cursor(E);
    for (int64_t e = 0; e < E; e++) cursor[e] = base[e];
    std::vector<int32_t> refRowIdx(active);
    std::vector<int32_t> refSrcMap(active);
    for (int64_t i = 0; i < active; i++) {
        int64_t e = ids[i];
        int32_t dst = static_cast<int32_t>(cursor[e]);
        cursor[e]++;
        refRowIdx[i] = dst;
        refSrcMap[dst] = static_cast<int32_t>(i / kTop);
    }

    // device 输入
    std::vector<uint16_t> hiddenBf16(hidden.size());
    for (size_t i = 0; i < hidden.size(); i++) hiddenBf16[i] = Fp32ToBf16(hidden[i]);
    std::vector<uint16_t> weightsBf16(active, 0x3F80); // 1.0（透传，kernel 不消费）
    void *dHidden = nullptr, *dIds = nullptr, *dWeights = nullptr;
    void *dSorted = nullptr, *dRowIdx = nullptr, *dCounts = nullptr, *dWeightsOut = nullptr;
    aclrtMalloc(&dHidden, hiddenBf16.size() * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dIds, active * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dWeights, active * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dSorted, active * K * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dRowIdx, active * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dCounts, E * 8, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dWeightsOut, active * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dHidden, hiddenBf16.size() * 2, hiddenBf16.data(), hiddenBf16.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dIds, active * 4, ids.data(), active * 4, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dWeights, active * 2, weightsBf16.data(), active * 2, ACL_MEMCPY_HOST_TO_DEVICE);

    aclTensor *tHidden = MakeTensor(dHidden, {S, K}, ACL_BF16);
    aclTensor *tIds = MakeTensor(dIds, {S, kTop}, ACL_INT32);
    aclTensor *tWeights = MakeTensor(dWeights, {S, kTop}, ACL_BF16);
    aclTensor *tSorted = MakeTensor(dSorted, {active, K}, ACL_BF16);
    aclTensor *tRowIdx = MakeTensor(dRowIdx, {active}, ACL_INT32);
    aclTensor *tCounts = MakeTensor(dCounts, {E}, ACL_INT64);
    aclTensor *tWeightsOut = MakeTensor(dWeightsOut, {S, kTop}, ACL_BF16);

    // 两段式调用
    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    printf("  [%ld] calling GWS\n", (long)S); fflush(stdout);
    aclnnStatus st = aclnnMoeRoutingInitFusedGetWorkspaceSize(
        tHidden, tIds, tWeights, E, tSorted, tRowIdx, tCounts, tWeightsOut, &wsSize, &exec);
    CaseResult r;
    CHECK_RET(st == ACL_SUCCESS, printf("S=%ld GetWorkspaceSize failed: %d\n", (long)S, st); r.ok = false; return r);
    void *ws = nullptr;
    if (wsSize > 0) {
        aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    printf("  [%ld] GWS ok ws=%llu, calling run\n", (long)S, (unsigned long long)wsSize); fflush(stdout);
    st = aclnnMoeRoutingInitFused(ws, wsSize, exec, stream);
    printf("  [%ld] run returned %d, syncing\n", (long)S, (int)st); fflush(stdout);
    CHECK_RET(st == ACL_SUCCESS, printf("S=%ld aclnn run failed: %d\n", (long)S, st); r.ok = false; return r);
    if (outMicros != nullptr) {
        // 计时：warmup 已含一次；测 N 轮含同步
        const int iters = 100;
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; it++) {
            uint64_t wsz2 = 0;
            aclOpExecutor *ex2 = nullptr;
            aclnnMoeRoutingInitFusedGetWorkspaceSize(tHidden, tIds, tWeights, E, tSorted, tRowIdx, tCounts,
                                                     tWeightsOut, &wsz2, &ex2);
            aclnnMoeRoutingInitFused(ws, wsz2, ex2, stream);
            aclrtSynchronizeStream(stream);
        }
        auto t1 = std::chrono::steady_clock::now();
        *outMicros = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / iters;
        aclrtSynchronizeStream(stream);
    } else {
        printf("  [%ld] sync done\n", (long)S); fflush(stdout);
    }

    // 回收比对
    std::vector<int32_t> gotRowIdx(active);
    std::vector<int64_t> gotCounts(E);
    std::vector<uint16_t> gotSorted(active * K);
    aclrtMemcpy(gotRowIdx.data(), active * 4, dRowIdx, active * 4, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(gotCounts.data(), E * 8, dCounts, E * 8, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(gotSorted.data(), active * K * 2, dSorted, active * K * 2, ACL_MEMCPY_DEVICE_TO_HOST);

    for (int64_t i = 0; i < active; i++) {
        if (gotRowIdx[i] != refRowIdx[i]) {
            r.maxIdxDiff++;
        }
    }
    for (int64_t e = 0; e < E; e++) {
        if (gotCounts[e] != refCounts[e]) {
            r.ok = false;
        }
    }
    for (int64_t j = 0; j < active; j++) {
        int32_t src = refSrcMap[j];
        if (std::memcmp(&gotSorted[j * K], &hiddenBf16[src * K], K * 2) != 0) {
            r.badSortedRows++;
        }
    }
    if (r.maxIdxDiff != 0 || r.badSortedRows != 0) {
        r.ok = false;
    }

    aclDestroyTensor(tHidden);
    aclDestroyTensor(tIds);
    aclDestroyTensor(tWeights);
    aclDestroyTensor(tSorted);
    aclDestroyTensor(tRowIdx);
    aclDestroyTensor(tCounts);
    aclDestroyTensor(tWeightsOut);
    if (ws) aclrtFree(ws);
    aclrtFree(dHidden);
    aclrtFree(dIds);
    aclrtFree(dWeights);
    aclrtFree(dSorted);
    aclrtFree(dRowIdx);
    aclrtFree(dCounts);
    aclrtFree(dWeightsOut);
    return r;
}

int main(int argc, char **argv)
{
    aclrtContext ctx;
    aclrtStream stream;
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, printf("aclInit failed\n"); return -1);
    int devId = (argc > 1) ? atoi(argv[1]) : 1;
    CHECK_RET(aclrtSetDevice(devId) == ACL_SUCCESS, printf("setDevice failed\n"); return -1);
    aclrtCreateContext(&ctx, devId);
    aclrtSetCurrentContext(ctx);
    aclrtCreateStream(&stream);

    const int64_t K = 3584;
    const int64_t E = 64;
    const int64_t kTop = 4;
    std::mt19937 rng(42);

    bool allOk = true;
    printf("%-8s %-10s %-14s %s\n", "S", "acc", "avg us/call", "detail"); fflush(stdout);
    for (int64_t S : {1, 2, 5, 8, 10, 16, 32, 40}) {
        int64_t micros = 0;
        CaseResult r = RunCase(S, K, E, kTop, stream, rng, &micros);
        allOk = allOk && r.ok;
        printf("%-8ld %-10s %-14ld idxDiff=%ld badRows=%ld\n", (long)S, r.ok ? "PASS" : "FAIL",
               (long)micros, (long)r.maxIdxDiff, (long)r.badSortedRows); fflush(stdout);
    }
    // 倾斜分布（不计时）：全挤 expert0 与 90% 挤 expert0
    for (int skew : {1, 2}) {
        for (int64_t S : {1, 8, 40}) {
            CaseResult r = RunCase(S, K, E, kTop, stream, rng, nullptr, skew);
            allOk = allOk && r.ok;
            printf("skew%d-%-3ld %-10s %-14s idxDiff=%ld badRows=%ld\n", skew, (long)S,
                   r.ok ? "PASS" : "FAIL", "-", (long)r.maxIdxDiff, (long)r.badSortedRows);
            fflush(stdout);
        }
    }
    printf("RESULT: %s\n", allOk ? "PASS" : "FAIL"); fflush(stdout);

    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(devId);
    aclFinalize();
    return allOk ? 0 : 1;
}
