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

echo "Installing CUDA-to-Vulkan Universal Translator to ${PREFIX}..."

mkdir -p "${PREFIX}/bin" "${PREFIX}/lib" "${PREFIX}/include/cuda" "${PREFIX}/share/cuda-vulkan/shaders" "${PREFIX}/share/cuda-vulkan/fixtures"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST="${PREFIX}/share/cuda-vulkan/install_manifest.txt"
> "${MANIFEST}"

copy_file() {
    local src="$1"
    local dst="$2"
    if [ -f "$src" ]; then
        cp -f "$src" "$dst"
        echo "$dst/$(basename "$src")" >> "${MANIFEST}"
    fi
}

for f in "${ROOT_DIR}/dist/bin"/*; do
    copy_file "$f" "${PREFIX}/bin"
done

for f in "${ROOT_DIR}/dist/lib"/*; do
    copy_file "$f" "${PREFIX}/lib"
done

for f in "${ROOT_DIR}/dist/include"/*; do
    copy_file "$f" "${PREFIX}/include/cuda"
done

for f in "${ROOT_DIR}/dist/shaders"/*; do
    copy_file "$f" "${PREFIX}/share/cuda-vulkan/shaders"
done

for f in "${ROOT_DIR}/dist/fixtures"/*; do
    copy_file "$f" "${PREFIX}/share/cuda-vulkan/fixtures"
done

echo "${MANIFEST}" >> "${MANIFEST}"

echo "Installation successfully completed at ${PREFIX}."
