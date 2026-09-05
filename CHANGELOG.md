# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Real SASS intermediate representation (`src/lifter/src/ir.rs`): every decoded
  instruction lowers to exactly one explicit IR node; control-flow graphs are
  built from instruction addresses with safe branch-target resolution.
- Fallible lifter entry point (`Lifter::try_lift`): unsupported opcodes
  (HMMA, SHFL, LDSM, reserved/unknown), unresolvable branches and invalid
  operands return structured errors instead of silently emitting a fixed shader.
  The CLI exits non-zero with a diagnostic on unsupported semantics.
- SPIR-V opcode ground-truth regression tests (verified against `spirv-as`):
  OpConvertUToF=112, OpConvertFToU=109, OpFMul=133, OpINotEqual=171,
  OpLogicalNot=168, OpControlBarrier=224; plus a no-constants-in-function test.
- Bounds-safe ELF64/cubin parser: ELF class/endianness/version checks, checked
  arithmetic for all section-table ranges, string-table and `.text` validation,
  malformed-input regression tests and deterministic no-panic fuzz tests.
- Negative API test binary (`tests/t02_negative_test.cpp`, 47 assertions):
  null/invalid/overflow inputs for runtime and driver APIs, with GPU-gated
  tiers so headless runners report SKIP instead of fake PASS.
- Runtime allocation-range query (`cudaGetAllocRange`) backing a real
  `cuMemGetAddressRange`; non-blocking stream query (`cudaStreamQuery` /
  `cuStreamQuery` via semaphore counters); recorded-state event query.
- CI: SHA-pinned third-party actions, `cargo audit` security job, exact-artifact
  packaging verification, `spirv-val` over all shipped shaders, and honestly
  scoped Windows/macOS jobs (build + unit + negative, no hardware dispatch).

### Fixed
- Wrong SPIR-V opcode numbers in the generic emitter (58/82/83/131/174/226)
  that produced invalid modules; verified with `spirv-val`.
- `OpConstant`/`OpTypeArray` emitted inside the function body (spec violation).
- Unsound wild PhysicalStorageBuffer LDG/STG lowering from uninitialized
  address registers (caused `vector_add`/`driver_api_e2e` mismatches); global
  memory ops are now ordered placeholders with the transfer performed by the
  documented ABI epilogue. Shared-memory indices are clamped into bounds.
- Fixture generator bit-overwrite bug (`set_bits` OR-only) that produced
  unresolvable branch targets; regenerated `tests/fixtures/*.cubin`.
- `cudaMemset` null-pointer fake success; `cudaLaunchKernel` fake success (now
  `cudaErrorNotSupported`); `cuModuleGetGlobal` fake success (now
  `CUDA_ERROR_NOT_FOUND`); `cuStreamWaitEvent`/`cuEventQuery` fake success.
- Out-of-range `cudaMemcpy` (now validated against allocation bounds),
  `malloc`/pitch/set `N` integer-overflow paths, unchecked file IO in module
  loading and SPIR-V loading (size caps, magic check, push-constant limit).
- Single-slot thread-local context replaced with a real context stack;
  primary contexts are ref-counted instead of leaked per retain.
- Vulkan device features assumed unconditionally; BDA/timeline are now probed
  (required) and Int64/Float64/Float16 enabled only when reported.
- CI failure masking: removed Windows `$LASTEXITCODE = 0` reset and Linux
  `|| cargo test` fallback; validation-layer audit now gates when installed
  and reports SKIP when absent instead of warning-and-passing.

### Security
- ELF/cubin and SASS inputs treated as untrusted: no panics on malformed
  bytes (fuzz-tested), structured errors only.
- Device-pointer range validation on every copy/fill path; use-after-free and
  double-free still rejected via the allocation registry.

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
