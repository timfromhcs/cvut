#!/usr/bin/env bash
set -euo pipefail

PREFIX="/usr/local"

for arg in "$@"; do
    case $arg in
        --prefix=*)
            PREFIX="${arg#*=}"
            shift
            ;;
    esac
done

echo "Uninstalling CUDA-to-Vulkan Universal Translator from ${PREFIX}..."

MANIFEST="${PREFIX}/share/cuda-vulkan/install_manifest.txt"

if [ -f "${MANIFEST}" ]; then
    while IFS= read -r file; do
        if [ -f "${file}" ]; then
            rm -f "${file}"
        fi
    done < "${MANIFEST}"
fi

# Clean up empty directories
rm -rf "${PREFIX}/share/cuda-vulkan" 2>/dev/null || true
rmdir "${PREFIX}/include/cuda" 2>/dev/null || true
rmdir "${PREFIX}/bin" 2>/dev/null || true
rmdir "${PREFIX}/lib" 2>/dev/null || true
rmdir "${PREFIX}/include" 2>/dev/null || true
rmdir "${PREFIX}/share" 2>/dev/null || true
rmdir "${PREFIX}" 2>/dev/null || true

echo "Uninstallation successfully completed from ${PREFIX}."
