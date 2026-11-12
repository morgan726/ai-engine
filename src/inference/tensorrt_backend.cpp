#include "inference/backend.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ai {
namespace inference {
namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) last_message_ = message == nullptr ? "" : message;
  }
  std::string last_message_;
};

template <typename T>
struct Destroyer {
  void operator()(T* value) const noexcept {
    if (value == nullptr) return;
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
    // TensorRT 10 removed the legacy destroy() virtual methods.  The
    // interfaces now have public virtual destructors and are released with
    // delete.  Keep the old path for TensorRT 8/9 images still used by the
    // project.
    delete value;
#else
    value->destroy();
#endif
  }
};

using RuntimePtr = std::unique_ptr<nvinfer1::IRuntime, Destroyer<nvinfer1::IRuntime>>;
using EnginePtr = std::unique_ptr<nvinfer1::ICudaEngine, Destroyer<nvinfer1::ICudaEngine>>;
using ContextPtr = std::unique_ptr<nvinfer1::IExecutionContext, Destroyer<nvinfer1::IExecutionContext>>;

struct DeviceBuffer {
  void* ptr{nullptr};
  ~DeviceBuffer() {
    if (ptr != nullptr) cudaFree(ptr);
  }
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr != nullptr) cudaFree(ptr);
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }
};

bool ConvertType(nvinfer1::DataType source, TensorType* target) noexcept {
  if (target == nullptr) return false;
  switch (source) {
    case nvinfer1::DataType::kUINT8:
      *target = TensorType::kUInt8;
      return true;
    case nvinfer1::DataType::kHALF:
      *target = TensorType::kFloat16;
      return true;
    case nvinfer1::DataType::kFLOAT:
      *target = TensorType::kFloat32;
      return true;
    case nvinfer1::DataType::kINT32:
      *target = TensorType::kInt32;
      return true;
    case nvinfer1::DataType::kINT64:
      *target = TensorType::kInt64;
      return true;
    default:
      return false;
  }
}

bool ConvertShape(const nvinfer1::Dims& source, std::vector<int64_t>* target) noexcept {
  if (target == nullptr || source.nbDims < 0 || source.nbDims > nvinfer1::Dims::MAX_DIMS) return false;
  target->clear();
  target->reserve(source.nbDims);
  for (int index = 0; index < source.nbDims; ++index) {
    if (source.d[index] <= 0) return false;
    target->push_back(source.d[index]);
  }
  return true;
}

class TensorRTBackend final : public IInferenceBackend {
 public:
  BackendKind Kind() const noexcept override { return BackendKind::kTensorRT; }
  const char* Name() const noexcept override { return "tensorrt"; }
  bool IsAvailable() const noexcept override { return true; }

  BackendStatus Load(const BackendConfig& config) noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    UnloadLocked();
    if (config.model_path.empty() || config.device_id < 0) return BackendStatus::kInvalidArgument;
    std::ifstream stream(config.model_path, std::ios::binary | std::ios::ate);
    if (!stream) return BackendStatus::kInvalidArgument;
    const std::streamsize size = stream.tellg();
    if (size <= 0) return BackendStatus::kInvalidArgument;
    stream.seekg(0, std::ios::beg);
    std::vector<char> plan(static_cast<size_t>(size));
    if (!stream.read(plan.data(), size)) return BackendStatus::kInvalidArgument;

    if (cudaSetDevice(config.device_id) != cudaSuccess) return BackendStatus::kUnavailable;
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) return BackendStatus::kRuntimeError;
    engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine_) return BackendStatus::kRuntimeError;
    context_.reset(engine_->createExecutionContext());
    if (!context_ || cudaStreamCreate(&stream_) != cudaSuccess) {
      UnloadLocked();
      return BackendStatus::kRuntimeError;
    }

    for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
      const char* name = engine_->getIOTensorName(index);
      if (name == nullptr) continue;
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        input_names_.emplace_back(name);
      } else {
        output_names_.emplace_back(name);
      }
    }
    if (input_names_.empty() || output_names_.empty()) {
      UnloadLocked();
      return BackendStatus::kRuntimeError;
    }
    input_types_.resize(input_names_.size());
    config_ = config;
    loaded_ = true;
    return BackendStatus::kSuccess;
  }

  BackendStatus Infer(const std::vector<Tensor>& inputs,
                      std::vector<Tensor>* outputs) noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_ || !context_) return BackendStatus::kNotLoaded;
    if (outputs == nullptr || inputs.size() != input_names_.size()) return BackendStatus::kInvalidArgument;
    if (cudaSetDevice(config_.device_id) != cudaSuccess) return BackendStatus::kUnavailable;

    try {
      std::vector<DeviceBuffer> device_buffers;
      device_buffers.reserve(inputs.size() + output_names_.size());
      std::vector<Tensor> result;
      result.reserve(output_names_.size());
      for (size_t index = 0; index < inputs.size(); ++index) {
        const Tensor& input = inputs[index];
        if (!input.IsValid() || !ConvertType(engine_->getTensorDataType(input_names_[index].c_str()), &input_types_[index])) {
          return BackendStatus::kInvalidArgument;
        }
        if (input.type != input_types_[index]) return BackendStatus::kInvalidArgument;
        nvinfer1::Dims dims{};
        dims.nbDims = static_cast<int>(input.shape.size());
        if (dims.nbDims < 0 || dims.nbDims > nvinfer1::Dims::MAX_DIMS) return BackendStatus::kInvalidArgument;
        for (int dim = 0; dim < dims.nbDims; ++dim) {
          if (input.shape[dim] <= 0 ||
              input.shape[dim] > std::numeric_limits<int32_t>::max()) {
            return BackendStatus::kInvalidArgument;
          }
          dims.d[dim] = static_cast<int32_t>(input.shape[dim]);
        }
        if (!context_->setInputShape(input_names_[index].c_str(), dims)) return BackendStatus::kInvalidArgument;
        device_buffers.emplace_back();
        if (cudaMalloc(&device_buffers.back().ptr, input.data.size()) != cudaSuccess ||
            cudaMemcpyAsync(device_buffers.back().ptr, input.data.data(), input.data.size(), cudaMemcpyHostToDevice,
                            stream_) != cudaSuccess ||
            !context_->setTensorAddress(input_names_[index].c_str(), device_buffers.back().ptr)) {
          return BackendStatus::kRuntimeError;
        }
      }

      for (const std::string& name : output_names_) {
        const nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
        Tensor output;
        if (!ConvertType(engine_->getTensorDataType(name.c_str()), &output.type) || !ConvertShape(dims, &output.shape)) {
          return BackendStatus::kNotImplemented;
        }
        const size_t bytes = output.ExpectedBytes();
        if (bytes == 0) return BackendStatus::kRuntimeError;
        device_buffers.emplace_back();
        if (cudaMalloc(&device_buffers.back().ptr, bytes) != cudaSuccess ||
            !context_->setTensorAddress(name.c_str(), device_buffers.back().ptr)) {
          return BackendStatus::kRuntimeError;
        }
        output.data.resize(bytes);
        result.emplace_back(std::move(output));
      }

      if (!context_->enqueueV3(stream_) || cudaStreamSynchronize(stream_) != cudaSuccess) {
        return BackendStatus::kRuntimeError;
      }
      size_t buffer_index = inputs.size();
      for (auto& output : result) {
        if (cudaMemcpy(output.data.data(), device_buffers[buffer_index++].ptr, output.data.size(),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          return BackendStatus::kRuntimeError;
        }
      }
      outputs->swap(result);
      return BackendStatus::kSuccess;
    } catch (...) {
      return BackendStatus::kRuntimeError;
    }
  }

  void Unload() noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    UnloadLocked();
  }

 private:
  void UnloadLocked() noexcept {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    stream_ = nullptr;
    context_.reset();
    engine_.reset();
    runtime_.reset();
    input_names_.clear();
    output_names_.clear();
    input_types_.clear();
    loaded_ = false;
    config_ = BackendConfig{};
  }

  Logger logger_;
  RuntimePtr runtime_{nullptr};
  EnginePtr engine_{nullptr};
  ContextPtr context_{nullptr};
  cudaStream_t stream_{nullptr};
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<TensorType> input_types_;
  BackendConfig config_;
  std::mutex mutex_;
  bool loaded_{false};
};

}  // namespace

std::shared_ptr<IInferenceBackend> CreateTensorRTBackend(const BackendConfig&) {
  return std::make_shared<TensorRTBackend>();
}

}  // namespace inference
}  // namespace ai
