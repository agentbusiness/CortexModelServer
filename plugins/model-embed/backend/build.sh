#!/bin/sh
# Plugin build (run by cortex-pack in backend/). CortexModelServer patch: the plain `cargo build` the
# plugin had, and on Linux a second build with CUDA in target/release/cuda/.
#
# candle links NVIDIA's libraries at build time, so a CUDA executable does not start on a machine
# without the driver. The package carries both: the plain one is the plugin's executable and becomes
# the CUDA one when the driver and NVIDIA's libraries are here (`become_cuda_build` in src/main.rs).
# The libraries are the ones model-llama's package ships, so the Linux release carries them once.
set -eu
cd "$(dirname "$0")"
out="target/release/cuda"
cargo build --release
rm -rf "$out"
mkdir -p "$out"
case "$(uname -s)" in
  Linux)
    # The release container has no GPU to ask. candle's kernels are PTX, which runs on this level
    # (sm_75, the oldest CUDA 13 supports) and every newer GPU.
    CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-75}" cargo build --release --features cuda --target-dir target/with-cuda
    cp target/with-cuda/release/cc-model-embed "$out/"
    ;;
esac
