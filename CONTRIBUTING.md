# Contributing to CUDA-to-Vulkan Universal Translator

Thank you for your interest in contributing! This project aims to provide a zero-overhead, open-standard alternative to proprietary GPU runtimes by lifting NVIDIA SASS machine code directly to SPIR-V and executing CUDA code on standard Vulkan 1.3 hardware (AMD Radeon, Intel Arc, Apple Silicon via MoltenVK, and ARM Mali).

---

## Code of Conduct

All contributors and maintainers are expected to follow friendly, respectful, and evidence-based collaboration practices.

---

## Development Principles & Invariants

All contributions must strictly adhere to the project core invariants:

1. **No Silent Placeholders**: No stub functions, mock implementations, `TODO` comments, or silent fallbacks. All functions must either fully implement the expected behavior or explicitly return an appropriate error code (e.g. `VK_ERROR_FEATURE_NOT_PRESENT`, `cudaErrorNotSupported`, `CUDA_ERROR_NOT_FOUND`). Explicitly documented transitional lowerings (see `docs/architecture.md`, e.g. SASS memory-op placeholders and the reduction library kernel) are permitted only with fail-loud behavior for anything outside the documented subset.
2. **Evidence-Based SASS Decoding**: All 128-bit SASS opcodes and bitfields (for Volta, Turing, Ampere, Ada, and Hopper) must be derived from verified hardware documentation and Mesa NAK source code (`src/nouveau/compiler/nak/encode/`).
3. **Automated Verification**: Any pull request must pass the automated verification gates:
   ```bash
   python scripts/build.py
   python scripts/test.py
   python scripts/validate.py
   ```
   plus the required GitHub Actions workflows (see `.github/workflows/ci.yml`).
   Legacy packaging helpers (`scripts/build_dist.sh`, `scripts/test_clean_install.sh`,
   `scripts/install.sh`) are maintained best-effort and are not CI gates.

---

## Building and Testing Locally

### Prerequisites
- **Rust Toolchain** (1.75+): `rustc`, `cargo`
- **C++ Compiler** (C++20 compliant): `clang++` or `g++`
- **Vulkan SDK** (1.3+): `glslangValidator`, `spirv-val`, Vulkan headers & loader
- **Python** (3.8+)

### Running the Test Suite
```bash
# 1. Compile release lifter and runtime
cargo build --release --manifest-path src/lifter/Cargo.toml
clang++ -std=c++20 -shared -fPIC src/runtime/cuda_runtime.cpp -lvulkan -o build/lib/libcudart.so

# 2. Run Tiered Verification
./build/bin/run_test --case=vector_add --elements=1048576
./build/bin/run_test --case=matrix_transpose --dim=2048
./build/bin/run_test --case=gemm_fp16 --m=1024 --n=1024 --k=1024
./build/bin/run_sass_test --cubin=tests/fixtures/sm80_reduction_pure_sass.cubin
```
