#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " Building CUDA-to-Vulkan Universal Translator Distribution  "
echo "============================================================"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p build/bin build/lib build/shaders dist/bin dist/lib dist/include dist/shaders dist/fixtures

echo "[BUILD_DIST] 1. Generating SASS test fixtures..."
python scripts/generate_fixtures.py

echo "[BUILD_DIST] 2. Building sass_lifter (Rust release)..."
cargo build --release --manifest-path src/lifter/Cargo.toml
cp -f src/lifter/target/release/sass_lifter* dist/bin/
cp -f src/lifter/target/release/sass_lifter* build/bin/ || true

echo "[BUILD_DIST] 3. Compiling compute shaders to SPIR-V..."
glslangValidator -V src/shaders/matrix_transpose.comp -o build/shaders/matrix_transpose.spv
glslangValidator -V src/shaders/gemm_fp16.comp -o build/shaders/gemm_fp16.spv
glslangValidator -V src/shaders/reduction.comp -o build/shaders/reduction.spv
glslangValidator -V src/shaders/mandelbrot_rgba8.comp -o build/shaders/mandelbrot_rgba8.spv
build/bin/sass_lifter --input tests/fixtures/sm80_vector_add.cubin --output build/shaders/vector_add.spv

cp -f build/shaders/*.spv dist/shaders/

echo "[BUILD_DIST] 4. Compiling CUDA Runtime (cudart64_12.dll)..."
clang++ -std=c++20 -shared -fPIC src/runtime/cuda_runtime.cpp -lvulkan -o build/lib/cudart64_12.dll
cp -f build/lib/cudart64_12.* dist/lib/ || true

echo "[BUILD_DIST] 5. Compiling CUDA Driver API (nvcuda.dll)..."
clang++ -std=c++20 -shared -fPIC -static src/runtime/nvcuda_driver.cpp src/runtime/cuda_runtime.cpp -lvulkan -o build/bin/nvcuda.dll
cp -f build/bin/nvcuda.* dist/lib/ || true

echo "[BUILD_DIST] 6. Compiling NVML & NVIDIA-SMI CLI..."
clang++ -std=c++20 -shared src/runtime/nvml_shim.cpp -lvulkan -o build/bin/nvml.dll
clang++ -std=c++20 src/tools/nvidia_smi.cpp -Lbuild/bin -lnvml -o build/bin/nvidia-smi.exe
cp -f build/bin/nvml.* dist/lib/ || true
cp -f build/bin/nvidia-smi.exe dist/bin/ || true

echo "[BUILD_DIST] 7. Compiling test validation harness..."
clang++ -std=c++20 tests/t01_memory_test.cpp -Isrc/runtime -Lbuild/lib -lcudart64_12 -lvulkan -o build/bin/t01_test
clang++ -std=c++20 tests/stress_test.cpp -Isrc/runtime -Lbuild/lib -lcudart64_12 -lvulkan -o build/bin/stress_test
clang++ -std=c++20 tests/run_test.cpp -Isrc/runtime -Lbuild/lib -lcudart64_12 -lvulkan -o build/bin/run_test
clang++ -std=c++20 tests/run_sass_test.cpp -Isrc/runtime -Lbuild/lib -lcudart64_12 -lvulkan -o build/bin/run_sass_test
clang++ -std=c++20 tests/driver_api_e2e.cpp -Lbuild/bin -lnvcuda -o build/bin/driver_api_e2e.exe
clang++ -std=c++20 tests/graphic_interop_test.cpp -Isrc/runtime -Lbuild/bin -lnvcuda -lvulkan -o build/bin/graphic_interop_test.exe

cp -f build/bin/run_test* dist/bin/ || true
cp -f build/bin/run_sass_test* dist/bin/ || true
cp -f build/bin/stress_test* dist/bin/ || true
cp -f build/bin/driver_api_e2e* dist/bin/ || true
cp -f build/bin/graphic_interop_test* dist/bin/ || true

echo "[BUILD_DIST] 8. Copying headers and fixtures..."
cp -f src/runtime/cuda_runtime.h dist/include/
cp -f src/runtime/cuda.h dist/include/
cp -f src/runtime/nvml.h dist/include/
cp -f tests/fixtures/*.cubin dist/fixtures/

echo "[BUILD_DIST] 9. Generating distribution tarball..."
tar -czf dist/cuda-vulkan-translator.tar.gz -C dist bin lib include shaders fixtures

echo "[BUILD_DIST] SUCCESS: Distribution package built at dist/cuda-vulkan-translator.tar.gz"
