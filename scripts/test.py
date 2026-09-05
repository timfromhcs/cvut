#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Central Test Runner for CVUT (CUDA-to-Vulkan Universal Translator)
Runs all unit, integration, functional, and e2e test layers across platforms.
"""

import sys
import os
import platform
import subprocess
import time

def log(msg):
    print(f"[CVUT TEST] {msg}")

def run_test_step(name, cmd, cwd, env):
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
    log(f"Running [{name}]: {cmd_str}")
    start = time.perf_counter()
    res = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    
    if res.returncode == 0:
        log(f"  --> PASSED ({elapsed:.3f}s)")
        if res.stdout:
            for line in res.stdout.strip().splitlines()[-4:]:
                print(f"      {line}")
        return True, elapsed, ""
    else:
        log(f"  --> FAILED (exit code {res.returncode}, {elapsed:.3f}s)")
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr)
        return False, elapsed, res.stderr or res.stdout

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)

    is_win = platform.system() == "Windows"
    exe_ext = ".exe" if is_win else ""

    # Setup environment PATH
    env = os.environ.copy()
    build_bin = os.path.join(root, "build", "bin")
    build_lib = os.path.join(root, "build", "lib")
    if is_win:
        env["PATH"] = f"{build_bin};{build_lib};" + env.get("PATH", "")
    else:
        env["PATH"] = f"{build_bin}:{build_lib}:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = f"{build_lib}:{build_bin}:" + env.get("LD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = f"{build_lib}:{build_bin}:" + env.get("DYLD_LIBRARY_PATH", "")

    test_plan = [
        ("Unit: sass_lifter Rust tests", ["cargo", "test", "--manifest-path", "src/lifter/Cargo.toml"]),
        ("Negative: invalid API inputs (GPU-gated tiers)", [os.path.join("build", "bin", f"t02_test{exe_ext}")]),
        ("Memory: BDA allocator & sync", [os.path.join("build", "bin", f"t01_test{exe_ext}")]),
        ("Hardening: Heap fragmentation & concurrency", [os.path.join("build", "bin", f"stress_test{exe_ext}")]),
        ("Compute: vector_add (1M floats)", [os.path.join("build", "bin", f"run_test{exe_ext}"), "--case=vector_add", "--elements=1048576"]),
        ("Compute: matrix_transpose (2048x2048)", [os.path.join("build", "bin", f"run_test{exe_ext}"), "--case=matrix_transpose", "--dim=2048"]),
        ("Compute: gemm_fp16 (1024x1024x1024)", [os.path.join("build", "bin", f"run_test{exe_ext}"), "--case=gemm_fp16", "--m=1024", "--n=1024", "--k=1024"]),
        ("Lifting: sm80_reduction SASS to SPIR-V execution", [os.path.join("build", "bin", f"run_sass_test{exe_ext}"), "--cubin=tests/fixtures/sm80_reduction_pure_sass.cubin"]),
        ("Driver E2E: Memory & compute dispatch", [os.path.join("build", "bin", f"driver_api_e2e{exe_ext}")]),
        ("Graphics: Offscreen rasterizer & Adler-32 checksum", [os.path.join("build", "bin", f"graphic_interop_test{exe_ext}")]),
        ("Telemetry: NVML Introspection", [os.path.join("build", "bin", f"nvml_test{exe_ext}")]),
        ("CLI: nvidia-smi query", [os.path.join("build", "bin", f"nvidia-smi{exe_ext}")]),
    ]

    print("============================================================")
    print(" CVUT Central Automated Test Execution                      ")
    print("============================================================")

    results = []
    total_start = time.perf_counter()
    all_passed = True

    for name, cmd in test_plan:
        passed, elapsed, err_msg = run_test_step(name, cmd, root, env)
        results.append((name, passed, elapsed))
        if not passed:
            all_passed = False

    total_time = time.perf_counter() - total_start

    print("============================================================")
    print(" Test Execution Summary                                     ")
    print("============================================================")
    for name, passed, elapsed in results:
        status = "PASSED" if passed else "FAILED"
        print(f" {status:6} | {elapsed:7.3f}s | {name}")
    print("------------------------------------------------------------")
    print(f"Total time: {total_time:.3f}s | Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("============================================================")

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
