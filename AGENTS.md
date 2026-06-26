# Workspace: vllm-ascend

vLLM hardware plugin for Huawei Ascend NPU. Python + C++ (csrc/).

**Current focus**: GDN (Gated Delta Net) operator optimization for Qwen3.5-35B on **Ascend 950 (A5)**.

## GDN Optimization Context

**Key files**:
- Triton kernels: `vllm_ascend/ops/triton/fla/` (solve_tril.py, wy_fast.py, l2norm.py, l2norm_fused.py)
- AscendC kernels: `csrc/attention/` (recurrent_gated_delta_rule/) and `csrc/moe/` (chunk_gated_delta_rule_fwd_h/, chunk_fwd_o/)
- Integration: `vllm_ascend/ops/gdn.py` (GDN attention layer)
- Profiling data: `profling/` (msprof outputs, kernel_details.csv)
- Benchmarks: root-level `benchmark_*.py` scripts
- Full report: `GDN_optimization_report.md`

**Known blockers**:
- BTHD→BHTD layout conversion: 8 `.transpose().contiguous()` calls per prefill, ~16MB wasted copies
- Decode launch overhead: 5 kernels × 15us launch = 75us/step, actual compute only ~13us (79% of decode time)
- ops-transformer's `chunk_gated_delta_rule` fused operator does NOT support Ascend 950

**Completed optimizations**:
- `chunk_size=128`: Python/Triton layer updated (was 64), C++ kernel still uses 64-token blocks internally (tiling processor splits 128→64 sub-blocks, matching flash-linear-attention-npu approach)
- `recompute_w_u_fwd`: BK/BV=128 hardcoded (was 64), 1.49x speedup, ~35ms saved
- `l2norm_fwd_fused`: q/k l2norm merged into single kernel, 1.63x speedup, ~2080ms saved on decode
- RGDR A5 PipeBarrier merge: Sub+Muls combined (6→5 PipeBarriers), ~6ms saved (pending compilation)
- RGDR A5 ProcessKQ fusion: state update + q projection fused (already in A5 codebase)
- `solve_tril`: autotune tested (NTASKS/warps/stages), no improvement over baseline

**Abandoned optimizations**:
- solve_tril 8x8 split: Triton on NPU doesn't support 3D tensor slicing
- solve_tril 64x64 direct: slower in production config (708us vs 446us)
- g_buf precompute: kernel launch 285us >> GetValue 32us
- GetValue elimination: no benefit in decode (seqLen too small, same call count)

**vllm-ascend vs ops-transformer RGDR**:
- vllm-ascend faster by ~15-20% (MTP3) or ~25-30% (MTP5) in decode scenarios
- Key advantages: fewer PipeBarriers (5 vs 8), ProcessKQ fusion, specialized ReduceSum paths
- ops-transformer advantages: double-buffering, async CopyIn

**MTP3 vs MTP5**: MTP5 recommended for 12-concurrency scenarios (vllm-ascend advantage larger)

**Deployment** (spec decode enabled):
```bash
vllm serve /home/weights/Qwen3.5-35B-A3B-W8A8-MXFP8-FULL-QUANT \
  --served-model-name "qwen35" \
  --host 0.0.0.0 --port 18888 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 16384 --max-num-batched-tokens 16384 \
  --enable-chunked-prefill --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 --quantization ascend \
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,16,20,24,28,32,36,40,44,48], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --trust-remote-code --async-scheduling \
  --additional-config '{"enable_cpu_binding":true}' \
  --no-enable-prefix-caching
```

## Commands

```bash
# Install
pip install -e .[dev]

# Lint (pre-commit; markdownlint only runs with 'ci' arg)
bash format.sh ci

# Type check
bash tools/mypy.sh

# Single test
pytest -sv tests/ut/<path>::<test_name>

# mypy targets: vllm_ascend, examples, tests
```

## Critical rules

- **Logger**: Use `from vllm.logger import logger`. Never use `init_logger(__name__)` in `vllm_ascend/` — logs are silently dropped (pre-commit enforces this).
- **Env vars**: All env vars centralized in `vllm_ascend/envs.py` via `env_variables` dict. New vars should use `VLLM_ASCEND_*` prefix. Never hardcode env var names outside `envs.py`.
- **Forbidden imports**: `pickle`/`cloudpickle` blocked (pre-commit `check-forbidden-imports`).
- **Long functions**: New functions over 100 lines must have comments (pre-commit `check-long-functions`).
- **Boolean context managers**: No boolean ops (`and`/`or`/`not`) in `with` statements (pre-commit `check-boolean-context-manager`).
- **Commits**: Must be signed off (`git commit -s`). Pre-commit auto-appends `Signed-off-by`.
- **PRs**: Created from fork, not main repo. Title format: `[Type][Module] Description`.

## Architecture

- **Plugin pattern**: Patches upstream vLLM classes via monkey-patching in `vllm_ascend/patch/` (platform/ and worker/).
- **Model runners**: `vllm_ascend/worker/model_runner_v1.py` (v1), `vllm_ascend/worker/v2/model_runner.py` (v2), `vllm_ascend/_310p/model_runner_310p.py` (310P).
- **NPU gotcha**: `tensor.item()` on device tensors causes NPU→CPU sync, blocks `AsyncScheduler`. Avoid in hot paths.
- **Tests**: `tests/ut/` (unit), `tests/e2e/` (integration), `tests/e2e/nightly/` (benchmarks). Require NPU hardware.
- **Pinned deps**: `torch==2.10.0`, `torch-npu==2.10.0`, `triton-ascend==3.2.1`, `transformers==5.5.4`.

## AscendC Kernel Compilation

C++ kernels in `csrc/` are compiled via CMake during `pip install -e .`. To rebuild specific kernels:

```bash
cd csrc
export PATH=/usr/local/python3.11.14/bin:$PATH  # ensure Python with torch is in PATH
bash build.sh --pkg --soc=ascend910b --ops=chunk_gated_delta_rule_fwd_h,chunk_fwd_o
```

**Gotchas**:
- Requires `patch` tool (`yum install patch` on openEuler)
- CANN 8.5.1 doesn't have `ASCEND950` enum — comment out references in `common/src/tiling_base/tiling_util.cpp:27`
- Build output: `build_out/*.run` package, install with `./build_out/*.run`
- For A5 (Ascend 950): use `--soc=ascend950` flag
- Compilation logs: `/tmp/rgdr_build*.log`

**RGDR A5-specific optimizations** (in `csrc/attention/recurrent_gated_delta_rule/op_kernel/arch35/`):
- `recurrent_gated_delta_rule.h`: Main kernel with ProcessKQ fusion, specialized ReduceSum paths
- `vf_vec_mul_mat.h`: Vector-matrix multiplication using `__simd_vf__` intrinsics
- `vf_outer_add.h`: Outer product + add fusion using `DIST_BRC_B32` broadcast

**PipeBarrier optimization**: A5 codebase has 5 PipeBarriers (vs 8 in A2). Sub+Muls merged into single barrier. Further reduction requires careful data dependency analysis.
