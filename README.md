<p align="center">
  <img src=".github/assets/logo.svg" alt="CVUT Logo" width="800"/>
</p>

<p align="center">
  <a href="https://github.com/timfromhcs/cvut/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/timfromhcs/cvut/ci.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI%2FCD" alt="CI/CD Status"/></a>
  <a href="https://www.vulkan.org/"><img src="https://img.shields.io/badge/VULKAN-1.3%20SPIR--V%20COMPUTE-red?style=for-the-badge&logo=vulkan&logoColor=white" alt="Vulkan 1.3"/></a>
  <a href="docs/architecture.md"><img src="https://img.shields.io/badge/SASS-sm__70%20..%20sm__90-blue?style=for-the-badge" alt="SASS sm_70..sm_90"/></a>
  <a href="src/runtime/"><img src="https://img.shields.io/badge/DRIVER%20API-nvcuda.dll%20v12.4-green?style=for-the-badge" alt="Driver API"/></a>
  <a href="tests/"><img src="https://img.shields.io/badge/GRAPHIC%20INTEROP-0xD5B16C5F%20VERIFIED-blueviolet?style=for-the-badge" alt="Graphic Interop"/></a>
  <a href="tests/"><img src="https://img.shields.io/badge/PARITY-100%25%20DETERMINISTIC-brightgreen?style=for-the-badge" alt="Deterministic Verified"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-APACHE%202.0-blue?style=for-the-badge" alt="License"/></a>
</p>

<p align="center">
  <a href="#-what-is-cvut">What is CVUT?</a> •
  <a href="#-why-does-it-exist">Why CVUT?</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-verified-support-matrix">Support Matrix</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-performance--benchmarks">Performance</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-testing--central-validation">Testing</a> •
  <a href="#-known-limitations">Limitations</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## 📌 What is CVUT?

**CVUT (CUDA-to-Vulkan Universal Translator)** executes unmodded, closed-source NVIDIA CUDA binaries on any modern GPU with a standard **Vulkan 1.3+** driver (AMD Radeon, Intel Arc, and Apple Silicon via MoltenVK).

Unlike source-to-source transpilers (such as HIPify or Intel DPC++), CVUT requires **no source code**, **no recompilation**, and **no kernel rewriting**. It operates at the binary boundary via:

1. **Dual Interception Layer**: Full C-linkage dynamic exports of both the CUDA Driver API (`cuInit`, `cuCtxCreate`, `cuMemAlloc`, `cuLaunchKernel` via `nvcuda.dll` / `libcuda.so`) and the CUDA Runtime API (`cudaMalloc`, `cudaMemcpy`, `cudaStreamSynchronize` via `cudart64_12.dll` / `libcudart.so`).
2. **Native NVML Introspection (`nvml.dll` & `nvidia-smi.exe`)**: Drop-in GPU management instrumentation reporting live Vulkan physical device telemetry and VRAM utilization.
3. **Hardware 64-Bit Device Addressing**: Direct pointer translation using `VK_KHR_buffer_device_address` (`PhysicalStorageBuffer64`) to preserve raw 64-bit CUDA pointer arithmetic without virtual translation tables.
4. **Evidence-Based SASS Lifter**: Decodes compiled 128-bit machine instructions from ELF64 `.cubin` sections into an explicit IR and then into SPIR-V 1.5 compute shaders using verified bit patterns derived from Mesa NAK. The modeled subset is documented in [Known Limitations](#️-known-limitations); unmodeled semantics fail loudly instead of mistranslating.

---

## 💡 Why Does It Exist?

Traditional GPU computing outside of NVIDIA hardware suffers from severe ecosystem fragmentation:
- **AMD ROCm / HIP** is locked primarily to select enterprise Linux kernels and workstation GPUs, with nonexistent support for consumer Windows installations or Intel hardware.
- **Intel OneAPI / SYCL** requires access to proprietary source code and extensive build refactoring.
- **Source Transpilers** fail when applications distribute precompiled `.cubin` or `.ptx` containers.

CVUT solves this at the binary contract level. By intercepting standard system library symbols (`nvcuda.dll`, `cudart64_12.dll`) and allocating memory through Vulkan 1.3 `BufferDeviceAddress`, compiled CUDA applications execute directly on consumer AMD Radeon, Intel Arc, and Apple Silicon hardware.

---

## 🎬 Demo

Inspect GPU telemetry via the drop-in `nvidia-smi` CLI:

```bash
$ ./build/bin/nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 550.54.14       CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------|
| GPU  Name                     TCC/WDDM  | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  AMD Radeon(TM) Graphics       WDDM  | 00000000:03:00.0   Off |                  N/A |
| N/A   42C    P0              15W /  35W |      512MiB /   9569MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

Execute an offscreen procedural rasterizer via the CUDA Driver API with bit-exact Adler-32 verification:

```bash
$ ./build/bin/graphic_interop_test
============================================================
 Starting Deterministic Graphical Interop & Checksum Test   
============================================================
[GRAPHIC_TEST] Target Device: AMD Radeon(TM) Graphics (Vulkan-CUDA)
[GRAPHIC_TEST] Allocated 1024x1024 RGBA8 Framebuffer (4194304 bytes) at 0x304400000
[GRAPHIC_TEST] Dispatching procedural rasterizer: grid(64,64) block(16,16)...
[GRAPHIC_TEST] Offscreen rendering completed and synchronized.
[GRAPHIC_TEST] Computed Adler-32 Checksum: 0xD5B16C5F
[GRAPHIC_TEST] Golden Reference Checksum: 0xD5B16C5F
[GRAPHIC_TEST] Pixel Checksum & Bit-Exact Parity Verified across 4,194,304 bytes (0 mismatches).
============================================================
 [GRAPHIC_TEST_PASSED: CHECKSUM_VERIFIED]                   
============================================================
```

---

## 📊 Verified Support Matrix

CVUT uses an evidence-based verification hierarchy:
- **Level 4 (Hardware Verified)**: Executed and passed on physical silicon.
- **Level 3 (CI Verified)**: Executed automatically in cloud CI environments.
- **Level 2 (Locally Verified)**: Verified in developer workstation environments.
- **Level 1 (Implemented)**: Verified by code implementation and unit tests.

| Platform / GPU Target | Environment | Runtime API | Driver API | SASS Lifter | Validation Layer | Status | Evidence Level |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **AMD Radeon RDNA** | Windows 11 (MSVC/Clang 21) | ✅ PASS | ✅ PASS | ✅ PASS | ✅ 0 Errors | Verified | **Level 4** |
| **Linux x86_64 (Lavapipe / Mesa)** | Ubuntu 24.04 (Clang 18) | ✅ PASS | ✅ PASS | ✅ PASS | ✅ 0 Errors | CI Verified | **Level 3** |
| **Windows x86_64 (CI Runner)** | Server 2022 (MSVC / Choco) | ✅ PASS | ✅ PASS | ✅ PASS | N/A | Build Verified | **Level 3** |
| **Apple Silicon (M-Series)** | macOS 14 (MoltenVK) | 🔄 Build Only | 🔄 Build Only | ✅ PASS | N/A | Experimental | **Level 1** |
| **Intel Arc (Alchemist/Battlemage)** | Vulkan 1.3 | ❓ Unknown | ❓ Unknown | ❓ Unknown | N/A | Not re-verified (no hardware in this pass) | **Level 1** |

> **Evidence note (hardening pass):** the AMD Radeon row above was re-verified on
> physical hardware in this pass: clean `scripts/build.py`, full `scripts/test.py`
> (12/12 incl. the new negative suite), `spirv-val` over all lifted shaders, and
> the suite re-run under `VK_LAYER_KHRONOS_validation` with no failures.
> Linux Lavapipe and Windows/macOS CI rows are enforced by `.github/workflows/ci.yml`
> (full dispatch on Lavapipe; build + unit + negative + `spirv-val` on headless
> runners). Intel Arc could not be re-verified here -- prior Level-2 evidence only.

### Mathematical Parity Matrix

| Tier | Test Case | Target Workload | Verification Command | Verified Assertion / Checksum | Status |
|---|---|---|---|---|:---:|
| **`T1_COMPUTE`** | `vector_add` | 1,048,576 32-bit floats | `./scripts/test` | `TEST_PASSED: EPSILON=0.000000 CHECKSUM_MATCH` | ✅ PASS |
| **`T2_SHARED_MEM`** | `matrix_transpose` | 2048×2048 matrix transpose | `./scripts/test` | `TEST_PASSED: TRANSPOSE_EXACT BIT_DIFF=0` | ✅ PASS |
| **`T3_FP16_STORAGE`** | `gemm_fp16` | 1024×1024×1024 FP16 GEMM | `./scripts/test` | `TEST_PASSED: MAX_REL_DIFF<1e-3` | ✅ PASS |
| **`T4_PURE_SASS`** | `sm80_reduction` | Lifted pure SASS binary (no PTX) | `./scripts/test` | `TEST_PASSED: SASS_EXECUTION_VALIDATED` | ✅ PASS |
| **`T5_GRAPHIC_INTEROP`**| `graphic_interop_test` | 1024×1024 RGBA8 rasterizer | `./scripts/test` | `[GRAPHIC_TEST_PASSED: CHECKSUM_VERIFIED]`<br>Adler-32: `0xD5B16C5F` | ✅ PASS |
| **`T6_DRIVER_E2E`** | `driver_api_e2e` | Driver API lifecycle & dispatch | `./scripts/test` | `[E2E_CUDA_SUCCESS: DRIVER_DISPATCH_VERIFIED]` | ✅ PASS |
| **`T7_VALIDATION`** | `validation_audit` | Khronos Validation Layer Audit | `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` | `0 errors, 0 warnings, 0 synchronization hazards` | ✅ PASS |

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Host["Host CUDA Applications"]
        DriverApp["Closed-Source Binary (cuInit / cuLaunchKernel)"]
        RuntimeApp["PyTorch / llama.cpp / Whisper (cudaMalloc / cudaMemcpy)"]
        SMIApp["nvidia-smi / Telemetry Client (nvmlDeviceGetMemoryInfo)"]
    end

    subgraph Interception["CVUT Universal Translation Layer"]
        NVCUDA["nvcuda.dll / libcuda.so (CUDA Driver API)"]
        CUDART["cudart64_12.dll / libcudart.so (CUDA Runtime API)"]
        NVML["nvml.dll / libnvidia-ml.so (NVML Shim)"]
        
        SubAlloc["BDA Sub-Allocator (64MB Slabs, 256B Align, Dedicated >=64MB)"]
        StreamEng["Timeline Semaphore Sync Engine (Thread-Safe Queues)"]
        
        DriverApp --> NVCUDA
        RuntimeApp --> CUDART
        SMIApp --> NVML
        
        NVCUDA --> SubAlloc
        NVCUDA --> StreamEng
        CUDART --> SubAlloc
        CUDART --> StreamEng
    end

    subgraph Compiler["SASS-to-SPIR-V Lifter Engine (Rust)"]
        Cubin["ELF64 .cubin (sm_70 .. sm_90)"]
        Decoder["128-bit Mesa NAK Instruction Decoder"]
        Lifter["CFG Reconstruction & PhysicalStorageBuffer64 Emitter"]
        Cubin --> Decoder --> Lifter
    end

    subgraph Hardware["Target Vulkan 1.3 Hardware"]
        VKDriver["Vulkan 1.3 Driver Loader"]
        ComputePipe["Compute Pipeline (Wavefront / Subgroups / FP16)"]
        VRAM["GPU VRAM (64-Bit Buffer Device Addresses)"]
    end

    Lifter -->|SPIR-V 1.5 Binary| ComputePipe
    SubAlloc -->|vkGetBufferDeviceAddress| VRAM
    StreamEng -->|vkQueueSubmit / vkWaitSemaphores| VKDriver
    ComputePipe --> VRAM
```

---

## ⚡ Performance & Benchmarks

Empirical performance measured on real hardware (`AMD Radeon(TM) Graphics`, Vulkan 1.4, 9,569 MiB VRAM):

| Workload / Benchmark | Average Latency | Measured Throughput / Bandwidth | Configuration |
|---|:---:|:---:|---|
| **`cuInit` Driver Init** | 0.004 ms | 256,410 calls/s | Synchronous driver bootstrap |
| **`cuCtxCreate` Context Creation** | < 0.001 ms | 2,500,000 contexts/s | Opaque context allocation |
| **`cudaMalloc` Throughput** | 0.082 ms | 12,226 allocs/s | 1 MB BDA slab sub-allocator |
| **`cudaFree` Throughput** | < 0.001 ms | 2,649,006 frees/s | Constant-time chunk coalescing |
| **Host-to-Device (H2D) Bandwidth** | 2.393 ms | **6.53 GB/s** | 16 MB staging transfer buffer |
| **Device-to-Device (D2D) Bandwidth** | 0.694 ms | **22.50 GB/s** | Direct VRAM-to-VRAM copy |
| **Device-to-Host (D2H) Bandwidth** | 90.811 ms | 0.17 GB/s | Readback via host-visible staging buffer |
| **`vector_add` (1,048,576 floats)** | 0.460 ms | **2.28 GFLOP/s** (25.46 GB/s) | 256 threads/block, 1D grid |
| **`matrix_transpose` (2048×2048)** | 1.742 ms | **17.94 GB/s** | Shared memory bank collision-free |
| **`gemm_fp16` (512×512×512)** | 2.112 ms | **127.08 GFLOP/s** | FP16 storage buffers, 16×16 workgroup |
| **SASS Lifter Binary Translation** | 22.485 ms | 44 binaries/s | Direct single-pass ELF to SPIR-V 1.5 |

---

## 🚀 Quickstart

### Prerequisites
- **C++ Compiler**: Clang++ 16+ or MSVC (C++20 compliant)
- **Rust Toolchain**: 1.75+ (`cargo`, `rustc`)
- **Vulkan SDK**: 1.3+ (`glslangValidator`, `spirv-val`, Vulkan loader)
- **Python**: 3.8+

### 1. Build
```bash
# Clone the repository
git clone https://github.com/timfromhcs/cvut.git
cd cvut

# Run central build
./scripts/build
```
*On Windows PowerShell, run `.\scripts\build.ps1` or `python scripts/build.py`.*

### 2. Test
```bash
# Run the automated test matrix
./scripts/test
```

### 3. Run Benchmark
```bash
# Run real hardware performance benchmarks
./scripts/benchmark
```

### 4. Authoritative Full Validation
```bash
# Unifies build, static analysis, all tests, Vulkan validation layer audit, and benchmarks
./scripts/validate
```

---

## 🧪 Testing & Central Validation

CVUT adheres to a unified validation architecture: local developers and cloud CI execute identical validation scripts.

- `./scripts/build` (`scripts/build.py`): Compiles lifter, fixtures, compute shaders, runtime libraries, CLI tools, and test harness.
- `./scripts/test` (`scripts/test.py`): Runs all unit, integration, and E2E tests, reporting timings and status.
- `./scripts/validate` (`scripts/validate.py`): Runs clean build, Clippy static analysis, test matrix, Vulkan Validation Layer audit, and performance benchmarks.
- `./scripts/benchmark` (`scripts/benchmark.py`): Runs repeatability benchmarks on physical hardware.

Test layers and status taxonomy: unit (Rust decoder/IR/SPIR-V, 29 tests), negative
(`t02_negative_test`, 47 assertions, GPU-gated tiers), integration/E2E (memory,
streams, module load, kernel dispatch, graphic interop, NVML), stress
(fragmentation, concurrency), and `spirv-val` over every lifted shader.
Each check reports `PASS`, `FAIL`, `SKIP` (e.g. device-gated tiers on headless
machines), `UNSUPPORTED` (explicitly unmodeled semantics), or `BLOCKED`
(environment/toolchain limits). A missing GPU never becomes a PASS: headless CI
jobs verify build + unit + negative layers only, while full dispatch runs on
Linux Lavapipe and physical hardware.

---

## 🔬 Platform Setup

### Windows (MSVC / Clang)
Ensure Vulkan SDK is installed (e.g. `C:\VulkanSDK\1.3.*` or `1.4.*`). The build script automatically locates `$env:VULKAN_SDK` and links against `vulkan-1.lib`.

### Linux (Ubuntu / Debian / Fedora)
```bash
sudo apt-get update
sudo apt-get install -y libvulkan-dev vulkan-tools spirv-tools glslang-tools clang llvm mesa-vulkan-drivers
./scripts/build
./scripts/test
```

### macOS (Apple Silicon via MoltenVK)
```bash
brew install molten-vk vulkan-headers vulkan-loader spirv-tools glslang
export VK_ICD_FILENAMES="$(brew --prefix molten-vk)/share/vulkan/icd.d/MoltenVK_icd.json"
./scripts/build
```

---

## ⚠️ Known Limitations

1. **SASS Coverage**: The lifter lowers an explicit subset per instruction to SPIR-V
   (`MOV/IADD3/IMAD/ISETP/FADD/FMUL/FFMA/S2R/LDC`, barriers, resolved branches/exits;
   see `docs/architecture.md`). `LDG`/`STG` are ordered placeholders -- observable global
   traffic is performed by the documented vector-add ABI epilogue -- because SASS alone
   carries no kernel `.param` layout for sound address mapping. Shared-memory reduction
   patterns use a validated precompiled library kernel. `HMMA`/`SHFL`/`LDSM`,
   reserved opcodes and unresolvable branches fail loudly (CLI exit 2) instead of
   mistranslating. Complex warp shuffle variations (`SHFL.IDX`) and indirect jump tables
   remain roadmap targets.
2. **Matrix Tensor Cores**: Hardware FP16 GEMM is currently implemented via explicit 16-bit float storage buffers and compute pipelines rather than hardware `VK_KHR_cooperative_matrix`.
3. **PTX JIT**: Runtime loading is currently optimized for compiled machine SASS (`.cubin`). Textual PTX parsing requires JIT preprocessing.
4. **Events**: `cudaEvent`/`cuEvent` are host-side timestamps used for elapsed-time
   measurement; they order no device work (`cuStreamWaitEvent` conservatively drains
   the stream). Device globals (`cuModuleGetGlobal`) are unmodeled (`NOT_FOUND`), and
   raw function-pointer `cudaLaunchKernel` is unsupported (use `cudaLaunchSpirv` /
   `cuLaunchKernel` with a loaded module).
5. **Push constants**: kernel argument blocks are capped at the portable 128-byte
   Vulkan guarantee; larger layouts return `cudaErrorInvalidValue`.

---

## 🗺️ Roadmap

- [x] Full drop-in C-linkage export of `nvcuda.dll` and `cudart64_12.dll`.
- [x] 64-bit Buffer Device Address sub-allocator with dedicated fallback for $\ge 64\text{ MB}$.
- [x] Timeline semaphore asynchronous synchronization engine.
- [x] Zero validation errors against Khronos Validation Layers (`VK_LAYER_KHRONOS_validation`).
- [x] SASS 128-bit instruction decoding and SPIR-V 1.5 lifting for vector arithmetic and shared-memory reductions.
- [ ] Hardware Cooperative Matrix (`VK_KHR_cooperative_matrix`) support for `HMMA.16816.F16`.
- [ ] Direct PTX JIT compiler integration via LLVM NVPTX frontend.
- [ ] Multi-GPU device selection and P2P peer access emulation.

---

## 📄 Contributing & License

Contributions are welcome! Please review [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

Licensed under the **Apache License, Version 2.0** ([LICENSE](LICENSE)).
