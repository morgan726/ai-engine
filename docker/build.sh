#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${root_dir}/build-docker"

cmake -S "${root_dir}" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DADK_BUILD_TRT_YOLO="${ADK_BUILD_TRT_YOLO:-OFF}" \
  -DADK_BUILD_MSSD="${ADK_BUILD_MSSD:-ON}" \
  -DADK_BUILD_EXAMPLES="${ADK_BUILD_EXAMPLES:-ON}" \
  -DADK_BUILD_TESTS="${ADK_BUILD_TESTS:-OFF}" \
  -DADK_CUDA_ARCHITECTURES="${ADK_CUDA_ARCHITECTURES:-89}" \
  -DAI_ENABLE_OPENVINO="${AI_ENABLE_OPENVINO:-ON}" \
  -DAI_BUILD_TENSORRT_BACKEND="${AI_BUILD_TENSORRT_BACKEND:-ON}"
cmake --build "${build_dir}" -j"${JOBS:-$(nproc)}"
"${build_dir}/bin/backend_probe"
if [[ -x "${build_dir}/bin/scheduler_probe" ]]; then
  "${build_dir}/bin/scheduler_probe" 2 10
fi
