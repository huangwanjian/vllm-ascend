#!/usr/bin/env python
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the details in the License.
# -----------------------------------------------------------------------------------------------------------

"""特定算子 kernel 的 fwk 引擎重编覆盖（修复编译引擎代差导致的数值错误二进制）。

背景
----
csrc vendored 的 cmake/scripts/util/ascendc_impl_build.py 是旧版，生成的编译
.py 驱动老 tbe.tikcpp 引擎；CANN 自带 fwk driver（ascendc_compile_kernel.py）
驱动新 asc_op_compile_base 引擎。两引擎对同一份 kernel 源产出不同二进制：
老引擎对 MhcPreClampSinkhorn 的 tiny fork 路径产出数值错误二进制（golden 可
复现 hRes 确定性损坏，MHC_PRE_TINY_FORK=on 时 spec acceptance 崩跌），新引擎
正确（与 mhc_pre_clamp_sinkhorn standalone 仓构建逐字节一致，可 sha256 比对）。

做法
----
标准构建（老引擎产物落盘 build/binary/<soc>/bin/<op>/）之后运行本脚本：对
FWK_ENGINE_OPS 中的每个算子，用 CANN fwk driver 以 in-tree 的源码 / exc
ops-info ini / custom_opc_options.ini / tiling lib / ops-info json 重编全部
变体（base + relocatable，变体集由同一 ini 决定，与标准构建一致），并原位
覆盖同名产物。随后由 build.sh 重新执行 package 目标，产出含正确 kernel 的
.run。带混合引擎闸门：bin 目录若残留未被 fwk 产物覆盖的 .o/.json 直接报错，
避免新旧引擎产物混包的静默数值损坏。

注意：包内随行的 dynamic .py（老引擎版）不改动——serving 走预编译二进制选
择（binary_info_config），不触发运行时动态编译；待上游整体升级 vendored 编
译脚本（ascendc_impl_build.py 家族）后一并消除。

用法：python3 scripts/util/compile_fwk_kernel.py（在 csrc/ 目录下，需已
source CANN set_env.sh；由 build.sh 的 --ops/--pkg 路径自动调用）。
环境变量 VLLM_ASCEND_FWK_ENGINE_OPS 可追加算子（snake 名逗号分隔），
off/none/0 关闭本脚本。
"""

import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

# 需要用 CANN fwk 引擎重编 kernel 的算子（snake 目录名 -> Camel 算子类型名）。
FWK_ENGINE_OPS = {
    # 老引擎对该算子 tiny fork 路径产出数值错误二进制，必须走新引擎。
    "mhc_pre_clamp_sinkhorn": "MhcPreClampSinkhorn",
}

DRIVER_TIMEOUT = 3600


def _fail(msg):
    print("[compile_fwk_kernel] ERROR: %s" % msg, file=sys.stderr)
    sys.exit(1)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snake_to_camel(name):
    return "".join(part.capitalize() for part in name.split("_"))


def _find_driver():
    home = os.environ.get("ASCEND_HOME_PATH") or "/usr/local/Ascend/ascend-toolkit/latest"
    candidates = [
        os.path.join(home, "compiler", "tikcpp", "ascendc_kernel_cmake", "fwk_modules",
                     "util", "ascendc_compile_kernel.py"),
        os.path.join(home, "%s-linux" % os.uname().machine, "tikcpp",
                     "ascendc_kernel_cmake", "fwk_modules", "util",
                     "ascendc_compile_kernel.py"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    _fail("CANN fwk driver ascendc_compile_kernel.py not found under %s" % home)


def _find_op_source(csrc_root, op_snake):
    # 只在 csrc 一级源码目录下查找，排除 build/output 等产物目录
    for entry in sorted(os.listdir(csrc_root)):
        cand = os.path.join(csrc_root, entry, op_snake, "op_kernel", op_snake + ".cpp")
        if os.path.isfile(cand):
            return cand
    _fail("kernel source not found for op %s (expect %s/*/%s/op_kernel/%s.cpp)"
          % (op_snake, csrc_root, op_snake, op_snake))


def _find_tiling_lib(build_dir):
    for cand in (os.path.join(build_dir, "libcust_opmaster_rt2.0.so"),
                 os.path.join(build_dir, "liboptiling.so"),
                 os.path.join(build_dir, "compat", "liboptiling.so")):
        if os.path.isfile(cand):
            return cand
    _fail("tiling library not found under %s" % build_dir)


def _recompile_op(driver, csrc_root, build_dir, soc, op_snake, op_camel):
    bin_dir = os.path.join(build_dir, "binary", soc, "bin", op_snake)
    if not (os.path.isdir(bin_dir) and any(n.endswith(".o") for n in os.listdir(bin_dir))):
        print("[compile_fwk_kernel] skip %s/%s: not built in this tree" % (soc, op_snake))
        return

    exc_ini = os.path.join(build_dir, "autogen", "exc", "aic-%s-ops-info.ini" % soc)
    fallback_ini = os.path.join(build_dir, "autogen", "aic-%s-ops-info.ini" % soc)
    cfg_ini = exc_ini if os.path.isfile(exc_ini) else fallback_ini
    if not os.path.isfile(cfg_ini):
        _fail("ops info ini not found for soc %s (tried %s / %s)" % (soc, exc_ini, fallback_ini))
    with open(cfg_ini) as f:
        if op_camel not in f.read():
            _fail("op %s not present in %s" % (op_camel, cfg_ini))

    ops_json = os.path.join(build_dir, "autogen", "aic-%s-ops-info.json" % soc)
    if not os.path.isfile(ops_json):
        ops_json = os.path.join(build_dir, "custom", "op_impl", "ai_core", "tbe", "config",
                                soc, "aic-%s-ops-info.json" % soc)
    if not os.path.isfile(ops_json):
        _fail("ops info json not found for soc %s" % soc)

    src_file = _find_op_source(csrc_root, op_snake)
    tiling_lib = _find_tiling_lib(build_dir)
    old_files = set(os.listdir(bin_dir))

    staging = tempfile.mkdtemp(prefix="fwk_kernel_")
    try:
        autogen_dir = os.path.join(staging, "autogen")
        tbe_dir = os.path.join(staging, "tbe")
        out_dir = os.path.join(staging, "binary")
        dynamic_dir = os.path.join(out_dir, "dynamic")
        os.makedirs(autogen_dir)
        os.makedirs(tbe_dir)
        os.makedirs(dynamic_dir)
        shutil.copy(cfg_ini, autogen_dir)
        # custom_opc_options.ini 携带 per-op relocatable 等编译配置，保证 fwk
        # 产出变体集（base + relocatable）与标准构建一致。
        opc_ini = os.path.join(build_dir, "autogen", "custom_opc_options.ini")
        if os.path.isfile(opc_ini):
            shutil.copy(opc_ini, autogen_dir)
        shutil.copy(ops_json, tbe_dir)

        cmd = [sys.executable, driver,
               "--op-type=%s" % op_camel,
               "--src-file=%s" % src_file,
               "--compute-unit=%s" % soc,
               "--compile-options=",
               "--debug-config=",
               "--config-ini=%s" % os.path.join(autogen_dir, os.path.basename(cfg_ini)),
               "--tiling-lib=%s" % tiling_lib,
               "--output-path=%s" % out_dir,
               "--dynamic-dir=%s" % dynamic_dir,
               "--enable-binary=TRUE",
               "--json-file=%s" % os.path.join(tbe_dir, os.path.basename(ops_json)),
               "--target-name=ascendc_kernels",
               "--auto-gen-path=%s" % autogen_dir,
               "--build-tool=make"]
        proc = subprocess.run(cmd, cwd=staging, capture_output=True, text=True,
                              timeout=DRIVER_TIMEOUT)
        if proc.returncode != 0:
            print(proc.stdout[-4000:], file=sys.stderr)
            print(proc.stderr[-4000:], file=sys.stderr)
            _fail("fwk driver failed for %s/%s" % (soc, op_camel))

        produced_dir = os.path.join(out_dir, soc, op_snake)
        if not os.path.isdir(produced_dir):
            _fail("fwk driver produced no artifacts for %s/%s" % (soc, op_camel))
        produced = set(os.listdir(produced_dir))
        if not any(n.endswith(".o") for n in produced):
            _fail("fwk driver produced no kernel binary for %s/%s" % (soc, op_camel))
        for name in sorted(produced):
            shutil.copy(os.path.join(produced_dir, name), os.path.join(bin_dir, name))

        # 混合引擎闸门：bin 目录不允许残留未被 fwk 产物覆盖的 .o/.json，
        # 否则新旧引擎产物混包 = 静默数值损坏，必须报错停下。
        leftover = [n for n in sorted(os.listdir(bin_dir))
                    if (n.endswith(".o") or n.endswith(".json")) and n not in produced]
        if leftover:
            _fail("mixed engine artifacts in %s (not covered by fwk output): %s"
                  % (bin_dir, leftover))
        extra = sorted(produced - old_files)
        if extra:
            print("[compile_fwk_kernel] warn: %s/%s new artifacts absent in "
                  "standard build: %s" % (soc, op_snake, extra))
        for name in sorted(os.listdir(bin_dir)):
            if name.endswith(".o"):
                print("[compile_fwk_kernel] %s/%s %s md5=%s"
                      % (soc, op_snake, name, _md5(os.path.join(bin_dir, name))))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main():
    extra = os.environ.get("VLLM_ASCEND_FWK_ENGINE_OPS", "").strip()
    if extra.lower() in ("off", "none", "0"):
        print("[compile_fwk_kernel] disabled by VLLM_ASCEND_FWK_ENGINE_OPS=%s" % extra)
        return
    ops = dict(FWK_ENGINE_OPS)
    for name in [n.strip() for n in extra.split(",") if n.strip()]:
        ops[name] = _snake_to_camel(name)

    csrc_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    build_dir = os.path.join(csrc_root, "build")
    if not os.path.isdir(build_dir):
        _fail("build directory not found: %s" % build_dir)
    driver = _find_driver()

    soc_dirs = [d for d in sorted(glob.glob(os.path.join(build_dir, "binary", "*")))
                if os.path.isdir(d)]
    if not soc_dirs:
        _fail("no build/binary/<soc> directories under %s" % build_dir)
    for soc_dir in soc_dirs:
        soc = os.path.basename(soc_dir)
        for op_snake, op_camel in sorted(ops.items()):
            _recompile_op(driver, csrc_root, build_dir, soc, op_snake, op_camel)
    print("[compile_fwk_kernel] done")


if __name__ == "__main__":
    main()
