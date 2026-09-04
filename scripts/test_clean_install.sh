#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " Testing Clean Installation and Verification Harness        "
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

TEST_PREFIX="/tmp/cuda_vk_clean_test_$$"
mkdir -p "${TEST_PREFIX}"

echo "[TEST_CLEAN_INSTALL] 1. Installing to sandbox: ${TEST_PREFIX}..."
"${SCRIPT_DIR}/install.sh" --prefix="${TEST_PREFIX}"

echo "[TEST_CLEAN_INSTALL] 2. Verifying directory structure and files..."
test -f "${TEST_PREFIX}/bin/sass_lifter" || test -f "${TEST_PREFIX}/bin/sass_lifter.exe"
test -f "${TEST_PREFIX}/bin/run_test" || test -f "${TEST_PREFIX}/bin/run_test.exe"
test -f "${TEST_PREFIX}/include/cuda/cuda_runtime.h"
test -f "${TEST_PREFIX}/share/cuda-vulkan/shaders/vector_add.spv"
test -f "${TEST_PREFIX}/share/cuda-vulkan/fixtures/sm80_vector_add.cubin"
test -f "${TEST_PREFIX}/share/cuda-vulkan/install_manifest.txt"

echo "[TEST_CLEAN_INSTALL] 3. Testing installed sass_lifter binary..."
"${TEST_PREFIX}/bin/sass_lifter" --input "${TEST_PREFIX}/share/cuda-vulkan/fixtures/sm80_vector_add.cubin" --output "${TEST_PREFIX}/test_out.spv"
spirv-val "${TEST_PREFIX}/test_out.spv"
rm -f "${TEST_PREFIX}/test_out.spv"

echo "[TEST_CLEAN_INSTALL] 4. Testing compilation against installed runtime headers and lib..."
cat << 'EOF' > "${TEST_PREFIX}/test_client.cpp"
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int devCount = 0;
    cudaError_t err = cudaGetDeviceCount(&devCount);
    if (err != cudaSuccess || devCount <= 0) return 1;
    void* ptr = nullptr;
    err = cudaMalloc(&ptr, 1024);
    if (err != cudaSuccess || !ptr) return 1;
    err = cudaFree(ptr);
    if (err != cudaSuccess) return 1;
    return 0;
}
EOF

# Ensure DLL can be found during test client execution
cp -f "${TEST_PREFIX}/lib"/* "${TEST_PREFIX}/bin/" 2>/dev/null || true

clang++ -std=c++20 -I"${TEST_PREFIX}/include/cuda" "${TEST_PREFIX}/test_client.cpp" -L"${TEST_PREFIX}/lib" -lcudart -o "${TEST_PREFIX}/bin/test_client"
"${TEST_PREFIX}/bin/test_client"
rm -f "${TEST_PREFIX}/test_client.cpp" "${TEST_PREFIX}/bin/test_client"*

echo "[TEST_CLEAN_INSTALL] 5. Running uninstaller..."
"${SCRIPT_DIR}/uninstall.sh" --prefix="${TEST_PREFIX}"

echo "[TEST_CLEAN_INSTALL] 6. Verifying clean removal..."
if [ -f "${TEST_PREFIX}/include/cuda/cuda_runtime.h" ]; then
    echo "ERROR: File cuda_runtime.h still exists after uninstall!"
    exit 1
fi
if [ -f "${TEST_PREFIX}/share/cuda-vulkan/install_manifest.txt" ]; then
    echo "ERROR: Manifest still exists after uninstall!"
    exit 1
fi

rm -rf "${TEST_PREFIX}" 2>/dev/null || true

echo "============================================================"
echo " SUCCESS: Clean Install & Uninstall Verified (Exit Code 0)  "
echo "============================================================"
