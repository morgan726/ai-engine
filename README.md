# AI Engine

本项目沿用 `InferServer → Scheduler → Processor → Runner/API` 的分层，增加了统一的 Intel/NVIDIA 后端选择层。CPU、OpenVINO 和 TensorRT 通过同一个 `BackendRegistry` 暴露；OpenVINO/TensorRT 适配器可以在对应运行时存在时注册，公共代码不包含厂商头文件。没有 GPU 时仍可用 CPU reference backend 做流水线和内存生命周期 smoke test。

## Docker 编译

TensorRT 开发镜像基于 Ubuntu 22.04/NVIDIA CUDA 12，适合 RTX 4060 Ti。它包含 TensorRT、OpenCV、FFmpeg、glog/gflags 和 OpenVINO 运行时：

```bash
docker build -t ai-engine-trt -f docker/Dockerfile .
docker run --rm -it --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
  -v "$PWD":/workspace ai-engine-trt
```

如果机器已有兼容的 TensorRT/CUDA Ubuntu 22.04 基础镜像，可通过
`TRT_BASE_IMAGE` 避免重复拉取 NGC 大镜像，例如：

```bash
docker build --build-arg TRT_BASE_IMAGE=easy_deploy_base_dev:nvidia_gpu_tensorrt_trt10_u2204 \
  -t ai-engine-trt -f docker/Dockerfile .
```

默认镜像会编译 `backend_probe`、`scheduler_probe` 和 MSSD 示例。旧版自定义 TensorRT YOLO plugin 需要打开时执行（4060 Ti 使用 SM 89；当前主机若确实是 RTX 5070 Ti，应按 CUDA/驱动支持传入对应架构）：

```bash
docker build --build-arg BUILD_TRT_YOLO=ON --build-arg CUDA_ARCHITECTURES=89 \
  -t ai-engine-trt -f docker/Dockerfile .
```

DeepStream 使用 NVIDIA 官方兼容镜像单独构建，避免把 CUDA/TensorRT 版本矩阵混装：

```bash
docker build -t ai-engine-deepstream -f docker/Dockerfile.deepstream .
docker run --rm -it --gpus all --net=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
  -v "$PWD":/workspace ai-engine-deepstream
```

容器内也可以直接运行 `docker/build.sh`。它使用 Release、Ninja 和宿主机的全部 CPU 线程，并在编译结束后运行后端 smoke test。脚本默认编译统一 TensorRT plan 适配器，旧的 TensorRT YOLO plugin 保持关闭；该 plugin 使用旧版 TensorRT API，如确有兼容源码和 SDK，再显式设置 `ADK_BUILD_TRT_YOLO=ON`。

## 线程池运行

`InferenceScheduler` 使用项目的 `PriorityThreadPool` 调度请求，数值较大的优先级先从等待队列取出；已经运行的任务不会被抢占。每个 `Request` 可以携带自己的 `BackendConfig`，因此同一个队列可以路由 CPU、Intel OpenVINO 和 NVIDIA TensorRT。每个线程按配置懒加载并独占后端实例及其推理上下文，默认两个线程，可通过 `Options::workers` 调整。`Submit` 返回 `future`，可取得状态和输出；`Shutdown(true)` 等待已提交的任务完成，`Shutdown(false)` 让尚未开始的任务返回取消状态。动态组批和序列任务仍未实现；不同模型或 SDK 会在每个 worker 中各保留一份实例，增加线程数会增加内存或显存占用。

先验证异构线程池路由：探针按 CPU、OpenVINO、TensorRT 轮换请求。未设置模型环境变量时，CPU 请求成功，两个厂商请求会返回模型加载错误，但仍会显示 SDK 路由；这用于检查调度路径，不代表厂商模型推理成功。

```bash
docker run --rm --gpus all ai-engine-trt /workspace/build/bin/scheduler_probe 2 10
```

给探针提供真实模型时，使用 `AI_ENGINE_OPENVINO_MODEL`、`AI_ENGINE_OPENVINO_DEVICE`、`AI_ENGINE_TENSORRT_MODEL` 和 `AI_ENGINE_TENSORRT_DEVICE`；输入张量仍需与模型签名匹配：

```bash
docker run --rm --gpus all \
  -e AI_ENGINE_OPENVINO_MODEL=/workspace/models/model.xml \
  -e AI_ENGINE_OPENVINO_DEVICE=CPU \
  -e AI_ENGINE_TENSORRT_MODEL=/workspace/models/model.engine \
  -e AI_ENGINE_TENSORRT_DEVICE=0 \
  -v "$PWD/models":/workspace/models:ro \
  ai-engine-trt /workspace/build/bin/scheduler_probe 2 12
```

针对 YOLO TensorRT plan，可以直接使用线程池推理入口。它将图片缩放为 `1x3x640x640 FP32`，提交到 TensorRT 后端并打印原始输出 tensor：

```bash
docker run --rm --gpus all \
  --mount type=bind,src="$PWD/weights",dst=/workspace/weights,readonly \
  --mount type=bind,src="$PWD/data",dst=/workspace/data,readonly \
  ai-engine-trt /workspace/build/bin/trt_scheduler_demo \
  /workspace/weights/yolov5-6.0/yolov5s-kFLOAT-batch1.engine \
  /workspace/data/image/dog.jpg 2 4 0
```

参数依次是 `engine 文件`、`图片`、`线程数`、`请求数`、`CUDA device_id`。每个 worker 会加载一份 TensorRT engine 和 execution context。这个 plan 必须使用与运行容器兼容的 TensorRT 版本和 GPU 架构生成；仓库中的示例 plan 显示由 TensorRT 8.6.1.6 生成，并包含 Ampere/Ada 的 SM 目标。在 TensorRT 10 容器或 RTX 5070 Ti（Compute Capability 12.0）上运行失败时，需要用当前容器的 `trtexec` 从 ONNX 重新生成：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/workspace/models/yolov5s.onnx \
  --saveEngine=/workspace/models/yolov5s-trt10.engine \
  --fp16
```

然后把 `yolov5s-trt10.engine` 传给 `trt_scheduler_demo`。只有 `backend_probe` 输出 `tensorrt runtime=yes factory=yes` 只能说明 SDK 已加载，不能代表具体 plan 可以反序列化；`trt_scheduler_demo` 返回 `status=success` 才表示真正完成了 TensorRT 推理。

运行实际的 MSSD 图片检测，两个线程处理四次请求，结果保存在宿主机 `output/`：

```bash
mkdir -p output
docker run --rm --gpus all \
  --mount type=bind,src="$PWD/weights",dst=/workspace/weights,readonly \
  --mount type=bind,src="$PWD/data",dst=/workspace/data,readonly \
  --mount type=bind,src="$PWD/output",dst=/results \
  --workdir /results ai-engine-trt \
  /workspace/build/bin/mssd /workspace/data/image/dog.jpg 2 4
```

命令最后两个数字分别是线程数和请求数；多请求产生 `output_0.jpg` 等文件。MSSD 示例目前使用 OpenCV DNN 默认 CPU 后端，`--gpus all` 不意味着它使用 TensorRT。真实 OpenVINO/TensorRT 模型需通过 `InferenceScheduler` 配置模型路径，并按照模型的输入类型和 shape 构造张量；`backend_probe` 只检查运行时和工厂是否存在。每个 worker 都可能加载每种配置的一份模型，增大线程数会增加内存或显存占用，应实测吞吐量后调整。

## 本地无 GPU 编译检查

```bash
cmake -S . -B build -DADK_BUILD_TRT_YOLO=OFF \
  -DADK_BUILD_EXAMPLES=OFF -DADK_BUILD_TESTS=OFF
cmake --build build -j
build/bin/backend_probe
```

典型输出是 `Auto → CPU`。在安装 OpenVINO 或 TensorRT 共享库后，构建系统会编译对应适配器，自动选择顺序为 OpenVINO、TensorRT、CPU。`backend_probe` 会列出运行时发现结果并验证张量尺寸、字节数和输出生命周期。

## CMake 选项

| 选项 | 默认值 | 作用 |
| --- | --- | --- |
| `ADK_BUILD_TRT_YOLO` | `OFF` | 编译现有 CUDA/TensorRT YOLO plugin 目标 |
| `ADK_BUILD_MSSD` | `ON` | 编译 OpenCV DNN MobileNet-SSD 示例 |
| `ADK_BUILD_EXAMPLES` | `OFF` | 编译调度器示例，需要 glog/gflags/OpenCV 开发包 |
| `ADK_BUILD_TESTS` | `OFF` | 编译旧版编解码测试 |
| `AI_ENABLE_OPENVINO` | `ON` | 保留 OpenVINO 后端构建开关 |
| `AI_BUILD_TENSORRT_BACKEND` | `ADK_BUILD_TRT_YOLO` | 编译 TensorRT serialized plan 适配器 |

Tensor 数据使用显式 shape 和字节数校验；任务和后端实例使用 `shared_ptr`，设备/输出缓存由各自 deleter 负责释放。视频编解码仍沿用 OpenCV/FFmpeg 与现有 `StreamRunner`，DeepStream 场景建议直接使用其 NVDEC/NVENC/GStreamer pipeline。
