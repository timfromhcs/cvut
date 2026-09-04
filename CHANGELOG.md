# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Centralized validation architecture (`scripts/build.py`, `scripts/test.py`, `scripts/validate.py`, `scripts/benchmark.py`) with unified POSIX and PowerShell command entry points.
- Real-time empirical hardware benchmark suite (`tests/benchmark.cpp`) measuring initialization latency, BDA allocation throughput, H2D/D2D/D2H bandwidth, and compute kernel throughput.
- Full NVML telemetry integration test suite (`tests/nvml_test.cpp`).
- SASS lifter unit test suite covering instruction bit decoding, special registers, ELF64 header parsing, and SPIR-V 1.5 binary generation.
- Dynamic detection and lifting of SASS instruction sequences in `sass_lifter` for both vector arithmetic and shared-memory reduction kernels.

### Fixed
- Fixed Vulkan validation layer specification violation `VUID-VkDeviceCreateInfo-pNext-02830` in `src/runtime/cuda_runtime.cpp` caused by duplicate feature chaining in `VkDeviceCreateInfo::pNext`.
- Fixed 16-bit storage buffer capability validation warning in `gemm_fp16` by adding `VkPhysicalDeviceVulkan11Features::storageBuffer16BitAccess`.
- Fixed push constant range size alignment (multiple of 4 bytes) satisfying `VUID-VkPushConstantRange-size-00298`.
- Fixed silent fallback in `cuModuleGetFunction` to return `CUDA_ERROR_NOT_FOUND` when shaders are missing.
- Fixed `run_sass_test.cpp` to execute the actual lifted SPIR-V output (`reduction_lifted.spv`) rather than precompiled source shaders.
- Fixed `cudaGetDeviceProperties` to report real VRAM from physical device memory heaps rather than a hardcoded default.
- Fixed static destruction crash on process exit by decoupling teardown from global C++ runtime static destructors.

## [0.1.0] - 2026-09-04

### Added
- Initial release of CVUT (CUDA-to-Vulkan Universal Translator).
- Drop-in Driver API (`nvcuda.dll`) and Runtime API (`cudart64_12.dll`) interception.
- Zero-overhead 64MB BDA slab sub-allocator with 256-byte alignment.
- NVML shim (`nvml.dll`) and `nvidia-smi` CLI utility.
- Basic SASS instruction decoder and SPIR-V compute pipeline dispatcher.
- Initial validation tests: `vector_add`, `matrix_transpose`, `gemm_fp16`, and graphical interop rasterizer.
