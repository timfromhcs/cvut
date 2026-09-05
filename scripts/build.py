#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Central Build Script for CVUT (CUDA-to-Vulkan Universal Translator)
Cross-platform: Windows, Linux, macOS
"""

import sys
import os
import platform
import subprocess
import shutil
import glob

def log(msg):
    print(f"[CVUT BUILD] {msg}")

def run_cmd(cmd, check=True, env=None):
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
    log(f"Running: {cmd_str}")
    res = subprocess.run(cmd, shell=isinstance(cmd, str), env=env)
    if check and res.returncode != 0:
        log(f"Command failed with exit code {res.returncode}: {cmd_str}")
        sys.exit(res.returncode)
    return res.returncode

def find_vulkan():
    system = platform.system()
    vk_inc = None
    vk_lib_dir = None
    vk_lib = None

    if system == "Windows":
        vulkan_sdk = os.environ.get("VULKAN_SDK")
        if not vulkan_sdk or not os.path.exists(vulkan_sdk):
            candidates = sorted(glob.glob(r"C:\VulkanSDK\*"), reverse=True)
            if candidates:
                vulkan_sdk = candidates[0]
        if vulkan_sdk and os.path.exists(vulkan_sdk):
            vk_inc = os.path.join(vulkan_sdk, "Include")
            vk_lib_dir = os.path.join(vulkan_sdk, "Lib")
            vk_lib = "vulkan-1"
        else:
            vk_lib = "vulkan-1"
    elif system == "Darwin":
        # macOS Homebrew
        brew = shutil.which("brew")
        if brew:
            vk_headers = subprocess.check_output([brew, "--prefix", "vulkan-headers"]).decode().strip()
            vk_loader = subprocess.check_output([brew, "--prefix", "vulkan-loader"]).decode().strip()
            vk_inc = os.path.join(vk_headers, "include")
            vk_lib_dir = os.path.join(vk_loader, "lib")
        vk_lib = "vulkan"
    else:
        # Linux
        vk_lib = "vulkan"

    return vk_inc, vk_lib_dir, vk_lib

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)

    system = platform.system()
    is_win = (system == "Windows")
    is_mac = (system == "Darwin")
    exe_ext = ".exe" if is_win else ""
    
    log(f"Target OS: {system} ({platform.machine()})")

    # Create directories
    os.makedirs("build/bin", exist_ok=True)
    os.makedirs("build/lib", exist_ok=True)
    os.makedirs("build/shaders", exist_ok=True)
    os.makedirs("dist/bin", exist_ok=True)
    os.makedirs("dist/lib", exist_ok=True)
    os.makedirs("dist/include", exist_ok=True)
    os.makedirs("dist/shaders", exist_ok=True)
    os.makedirs("dist/fixtures", exist_ok=True)

    # 1. Generate SASS fixtures
    log("1. Generating SASS test fixtures...")
    run_cmd([sys.executable, "scripts/generate_fixtures.py"])

    # 2. Build sass_lifter (Rust)
    log("2. Building sass_lifter (Rust release)...")
    run_cmd(["cargo", "build", "--release", "--manifest-path", "src/lifter/Cargo.toml"])
    
    lifter_bin = f"src/lifter/target/release/sass_lifter{exe_ext}"
    if os.path.exists(lifter_bin):
        shutil.copy2(lifter_bin, f"build/bin/sass_lifter{exe_ext}")
        shutil.copy2(lifter_bin, f"dist/bin/sass_lifter{exe_ext}")
    else:
        log(f"Warning: {lifter_bin} not found!")

    # 3. Compile compute shaders to SPIR-V
    log("3. Compiling compute shaders to SPIR-V...")
    glslang = shutil.which("glslangValidator") or "glslangValidator"
    run_cmd([glslang, "-V", "src/shaders/matrix_transpose.comp", "-o", "build/shaders/matrix_transpose.spv"])
    run_cmd([glslang, "-V", "src/shaders/gemm_fp16.comp", "-o", "build/shaders/gemm_fp16.spv"])
    run_cmd([glslang, "-V", "src/shaders/reduction.comp", "-o", "build/shaders/reduction.spv"])
    run_cmd([glslang, "-V", "src/shaders/mandelbrot_rgba8.comp", "-o", "build/shaders/mandelbrot_rgba8.spv"])

    # Lift fixtures using sass_lifter
    lifter_cmd = os.path.join("build", "bin", f"sass_lifter{exe_ext}")
    run_cmd([lifter_cmd, "--input", "tests/fixtures/sm80_vector_add.cubin", "--output", "build/shaders/vector_add.spv"])
    run_cmd([lifter_cmd, "--input", "tests/fixtures/sm80_reduction_pure_sass.cubin", "--output", "build/shaders/reduction_lifted.spv"])

    for spv in glob.glob("build/shaders/*.spv"):
        shutil.copy2(spv, "dist/shaders/")

    # 4. Prepare compiler and flags
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++") or "clang++"
    vk_inc, vk_lib_dir, vk_lib = find_vulkan()

    cxx_flags = ["-std=c++20"]
    if vk_inc and os.path.exists(vk_inc):
        cxx_flags.append(f"-I{vk_inc}")
    cxx_flags.append("-Isrc/runtime")

    ld_flags = []
    if vk_lib_dir and os.path.exists(vk_lib_dir):
        ld_flags.append(f"-L{vk_lib_dir}")
    ld_flags.append(f"-l{vk_lib}")

    log(f"Using C++ Compiler: {cxx}")

    # 5. Compile Runtime Libraries
    log("4. Compiling CUDA Runtime library...")
    if is_win:
        cudart_out = "build/lib/cudart64_12.dll"
        run_cmd([cxx] + cxx_flags + ["-shared", "src/runtime/cuda_runtime.cpp"] + ld_flags + ["-o", cudart_out])
        # Also copy / create cudart.dll symlink/copy for compatibility
        shutil.copy2(cudart_out, "build/lib/cudart.dll")
        shutil.copy2(cudart_out, "build/bin/cudart64_12.dll")
        shutil.copy2(cudart_out, "build/bin/cudart.dll")
        if os.path.exists("build/lib/cudart64_12.lib"):
            shutil.copy2("build/lib/cudart64_12.lib", "build/bin/cudart64_12.lib")
            shutil.copy2("build/lib/cudart64_12.lib", "build/lib/cudart.lib")
            shutil.copy2("build/lib/cudart64_12.lib", "build/bin/cudart.lib")
    elif is_mac:
        cudart_out = "build/lib/libcudart.dylib"
        run_cmd([cxx] + cxx_flags + ["-shared", "-fPIC", "src/runtime/cuda_runtime.cpp"] + ld_flags + ["-o", cudart_out])
    else:
        cudart_out = "build/lib/libcudart.so"
        run_cmd([cxx] + cxx_flags + ["-shared", "-fPIC", "src/runtime/cuda_runtime.cpp"] + ld_flags + ["-o", cudart_out])

    log("5. Compiling CUDA Driver API library...")
    if is_win:
        nvcuda_out = "build/bin/nvcuda.dll"
        run_cmd([cxx] + cxx_flags + ["-shared", "-static", "src/runtime/nvcuda_driver.cpp", "src/runtime/cuda_runtime.cpp"] + ld_flags + ["-o", nvcuda_out])
        shutil.copy2(nvcuda_out, "build/lib/nvcuda.dll")
        if os.path.exists("build/bin/nvcuda.lib"):
            shutil.copy2("build/bin/nvcuda.lib", "build/lib/nvcuda.lib")
    elif is_mac:
        nvcuda_out = "build/lib/libcuda.dylib"
        run_cmd([cxx] + cxx_flags + ["-shared", "-fPIC", "src/runtime/nvcuda_driver.cpp", "src/runtime/cuda_runtime.cpp"] + ld_flags + ["-o", nvcuda_out])
    else:
        nvcuda_out = "build/lib/libcuda.so"
        run_cmd([cxx] + cxx_flags + ["-shared", "-fPIC", "src/runtime/nvcuda_driver.cpp", "src/runtime/cuda_runtime.cpp"] + ld_flags + ["-o", nvcuda_out])

    log("6. Compiling NVML & NVIDIA-SMI CLI...")
    if is_win:
        nvml_out = "build/bin/nvml.dll"
        run_cmd([cxx] + cxx_flags + ["-shared", "src/runtime/nvml_shim.cpp"] + ld_flags + ["-o", nvml_out])
        shutil.copy2(nvml_out, "build/lib/nvml.dll")
        if os.path.exists("build/bin/nvml.lib"):
            shutil.copy2("build/bin/nvml.lib", "build/lib/nvml.lib")
        smi_out = "build/bin/nvidia-smi.exe"
        run_cmd([cxx] + cxx_flags + ["src/tools/nvidia_smi.cpp", "-Lbuild/bin", "-lnvml", "-o", smi_out])
    elif is_mac:
        nvml_out = "build/lib/libnvidia-ml.dylib"
        run_cmd([cxx] + cxx_flags + ["-shared", "-fPIC", "src/runtime/nvml_shim.cpp"] + ld_flags + ["-o", nvml_out])
        smi_out = "build/bin/nvidia-smi"
        run_cmd([cxx] + cxx_flags + ["src/tools/nvidia_smi.cpp", "-Lbuild/lib", "-lnvidia-ml", "-o", smi_out])
    else:
        nvml_out = "build/lib/libnvidia-ml.so"
        run_cmd([cxx] + cxx_flags + ["-shared", "-fPIC", "src/runtime/nvml_shim.cpp"] + ld_flags + ["-o", nvml_out])
        smi_out = "build/bin/nvidia-smi"
        run_cmd([cxx] + cxx_flags + ["src/tools/nvidia_smi.cpp", "-Lbuild/lib", "-lnvidia-ml", "-o", smi_out])

    # 6. Compile Test Suite
    log("7. Compiling test validation harness & benchmark...")
    test_link_runtime = ["-Lbuild/lib", "-Lbuild/bin"]
    if is_win:
        test_link_runtime += ["-lcudart64_12"]
    elif is_mac:
        test_link_runtime += ["-lcudart"]
    else:
        test_link_runtime += ["-lcudart"]

    test_targets = [
        ("tests/t01_memory_test.cpp", f"build/bin/t01_test{exe_ext}", test_link_runtime + ld_flags),
        ("tests/t02_negative_test.cpp", f"build/bin/t02_test{exe_ext}", ["-Lbuild/bin", "-Lbuild/lib", "-lnvcuda"] if is_win else ["-Lbuild/lib", "-lcuda"]),
        ("tests/stress_test.cpp", f"build/bin/stress_test{exe_ext}", test_link_runtime + ld_flags),
        ("tests/run_test.cpp", f"build/bin/run_test{exe_ext}", test_link_runtime + ld_flags),
        ("tests/run_sass_test.cpp", f"build/bin/run_sass_test{exe_ext}", test_link_runtime + ld_flags),
        ("tests/driver_api_e2e.cpp", f"build/bin/driver_api_e2e{exe_ext}", ["-Lbuild/bin", "-Lbuild/lib", "-lnvcuda"] if is_win else ["-Lbuild/lib", "-lcuda"]),
        ("tests/graphic_interop_test.cpp", f"build/bin/graphic_interop_test{exe_ext}", (["-Lbuild/bin", "-Lbuild/lib", "-lnvcuda"] if is_win else ["-Lbuild/lib", "-lcuda"]) + ld_flags),
        ("tests/nvml_test.cpp", f"build/bin/nvml_test{exe_ext}", ["-Lbuild/bin", "-Lbuild/lib", "-lnvml"] if is_win else ["-Lbuild/lib", "-lnvidia-ml"]),
        ("tests/benchmark.cpp", f"build/bin/benchmark{exe_ext}", (["-Lbuild/bin", "-Lbuild/lib", "-lnvcuda", "-lcudart64_12"] if is_win else ["-Lbuild/lib", "-lcuda", "-lcudart"]) + ld_flags),
    ]

    for src, out, extra_flags in test_targets:
        if os.path.exists(src):
            log(f"Compiling {src} -> {out}...")
            run_cmd([cxx] + cxx_flags + [src] + extra_flags + ["-o", out])

    # 7. Copy Distribution Files
    log("8. Populating dist/ directory...")
    for ext in ["*.dll", "*.so", "*.dylib", "*.lib"]:
        for f in glob.glob(f"build/lib/{ext}") + glob.glob(f"build/bin/{ext}"):
            shutil.copy2(f, "dist/lib/")
    for ext in ["*"]:
        for f in glob.glob(f"build/bin/*{exe_ext}"):
            if not f.endswith(".dll") and not f.endswith(".lib") and not f.endswith(".exp"):
                shutil.copy2(f, "dist/bin/")
    for h in glob.glob("src/runtime/*.h"):
        shutil.copy2(h, "dist/include/")
    for cubin in glob.glob("tests/fixtures/*.cubin"):
        shutil.copy2(cubin, "dist/fixtures/")

    log("SUCCESS: Build completed cleanly!")

if __name__ == "__main__":
    main()
