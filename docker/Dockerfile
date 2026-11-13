# TensorRT 24.01 is based on Ubuntu 22.04 and CUDA 12.x. The image is
# intended for the RTX 4060 Ti (SM 8.9); override TRT_TAG for another matrix.
ARG TRT_TAG=24.01-py3
ARG TRT_BASE_IMAGE=nvcr.io/nvidia/tensorrt:${TRT_TAG}
FROM ${TRT_BASE_IMAGE}

ARG BUILD_TRT_YOLO=OFF
ARG CUDA_ARCHITECTURES=89
ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git gnupg pkg-config ninja-build build-essential cmake \
      libssl-dev libgoogle-glog-dev libgflags-dev libglib2.0-dev \
      libopencv-dev libavcodec-dev libavformat-dev libavutil-dev \
      libswscale-dev libswresample-dev ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Intel CPU/iGPU runtime. The adapter is optional at configure time and is
# enabled automatically when this package exports an OpenVINO CMake target.
RUN wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
      gpg --dearmor -o /usr/share/keyrings/intel-archive.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/intel-archive.gpg] https://apt.repos.intel.com/openvino/2023 ubuntu22 main" \
      > /etc/apt/sources.list.d/intel-openvino.list && \
    apt-get update && apt-get install -y --no-install-recommends openvino && \
    rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000
RUN groupadd -r -f -g "${GID}" ai && \
    useradd -o -r -l -u "${UID}" -g "${GID}" -ms /bin/bash ai && \
    mkdir -p /workspace && chown -R ai:ai /workspace

ENV PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics
WORKDIR /workspace
COPY --chown=ai:ai . /workspace
USER ai

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DADK_BUILD_TRT_YOLO=${BUILD_TRT_YOLO} -DADK_BUILD_MSSD=ON \
      -DADK_BUILD_EXAMPLES=ON -DADK_BUILD_TESTS=OFF \
      -DADK_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
      -DAI_ENABLE_OPENVINO=ON -DAI_BUILD_TENSORRT_BACKEND=ON && \
    cmake --build build -j"$(nproc)" --target backend_probe scheduler_probe trt_scheduler_demo mssd priority_demo && \
    if [ "${BUILD_TRT_YOLO}" = "ON" ]; then cmake --build build -j"$(nproc)" --target trt_yolo; fi && \
    build/bin/backend_probe

CMD ["/bin/bash"]
