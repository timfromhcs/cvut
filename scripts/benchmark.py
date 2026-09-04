#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Central Benchmark Runner for CVUT (CUDA-to-Vulkan Universal Translator)
Measures initialization, memory bandwidth, compute throughput, and lifter latency.
"""

import sys
import os
import platform
import subprocess
import time

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)

    is_win = platform.system() == "Windows"
    exe_ext = ".exe" if is_win else ""

    env = os.environ.copy()
    build_bin = os.path.join(root, "build", "bin")
    build_lib = os.path.join(root, "build", "lib")
    if is_win:
        env["PATH"] = f"{build_bin};{build_lib};" + env.get("PATH", "")
    else:
        env["PATH"] = f"{build_bin}:{build_lib}:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = f"{build_lib}:{build_bin}:" + env.get("LD_LIBRARY_PATH", "")

    bench_bin = os.path.join("build", "bin", f"benchmark{exe_ext}")
    if not os.path.exists(bench_bin):
        print(f"Error: {bench_bin} not found. Run ./scripts/build first.")
        sys.exit(1)

    print("============================================================")
    print(" Running CVUT Performance & Hardware Benchmark Suite         ")
    print("============================================================")

    # 1. Measure lifter compilation latency
    lifter_bin = os.path.join("build", "bin", f"sass_lifter{exe_ext}")
    if os.path.exists(lifter_bin):
        cubin_vadd = os.path.join("tests", "fixtures", "sm80_vector_add.cubin")
        t0 = time.perf_counter()
        ITERS = 20
        for _ in range(ITERS):
            subprocess.run([lifter_bin, "--input", cubin_vadd, "--output", "build/shaders/temp_bench.spv"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        lifter_avg_ms = ((time.perf_counter() - t0) / ITERS) * 1000.0
        print(f"[BENCHMARK] SASS-to-SPIR-V Translation: {lifter_avg_ms:.3f} ms/binary ({1000.0/lifter_avg_ms:.0f} binaries/s)")

    # 2. Run runtime and kernel benchmarks
    res = subprocess.run([bench_bin], env=env)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
