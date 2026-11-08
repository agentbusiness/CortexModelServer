#!/bin/sh
# Plugin build (run by cortex-pack in backend/): llama-server from CortexAgent's vendored llama.cpp
# with CMake, then the Rust backend. Everything llama-server needs lands in target/release/llama/.
#
# CortexModelServer patch: CortexAgent's vendored tree (vendor/llama.cpp, unslothai/llama.cpp `468ad58`
# b11876, byte for byte) carries only the sources CortexAgent builds: the CPU, BLAS and Metal backends, and the server
# core without upstream's server.cpp, tools/ CMake files or UI. So the executable is assembled by
# agent-server/CMakeLists.txt (the vendored tree added as CortexAgent's engine shim adds it), and only
# macOS (Metal, one static executable) is built. ct's Linux CUDA/Vulkan path needs the CUDA and
# Vulkan backends, which this tree does not have.
set -eu
cd "$(dirname "$0")"
here="$(pwd)"
out="$here/target/release/llama"
case "$(uname -s)" in
  Darwin) ;;
  *)
    echo "model-llama-agent builds on macOS only: CortexAgent's llama.cpp carries the Metal, BLAS and CPU backends, not CUDA or Vulkan" >&2
    exit 1
    ;;
esac
build="$here/target/llama-build"
# CFLAGS / CXXFLAGS / LDFLAGS (scripts/build-env.sh) on every configure, not only a new build folder's.
cmake -S "$here/agent-server" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="${CFLAGS:-}" -DCMAKE_CXX_FLAGS="${CXXFLAGS:-}" -DCMAKE_EXE_LINKER_FLAGS="${LDFLAGS:-}" \
  -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF
jobs="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
rm -rf "$out"
mkdir -p "$out"
cmake --build "$build" --config Release --target llama-server -j "$jobs"
cp "$build/bin/llama-server" "$out/"
cargo build --release
