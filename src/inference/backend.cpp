#include "inference/backend.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace ai {
namespace inference {
#if defined(AI_ENGINE_WITH_OPENVINO)
// Defined in openvino_backend.cpp.  Keeping this declaration here lets the
// registry install the adapter without exposing OpenVINO headers in backend.h.
std::shared_ptr<IInferenceBackend> CreateOpenVINOBackend(const BackendConfig& config);
#endif
#if defined(AI_ENGINE_WITH_TENSORRT)
std::shared_ptr<IInferenceBackend> CreateTensorRTBackend(const BackendConfig& config);
#endif
namespace {

bool IsKnownBackend(BackendKind kind) noexcept {
  return kind == BackendKind::kCpu || kind == BackendKind::kOpenVINO || kind == BackendKind::kTensorRT;
}

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool HasSharedLibrary(const std::vector<std::string>& candidates) noexcept {
#if defined(__unix__) || defined(__APPLE__)
  for (const auto& candidate : candidates) {
    void* handle = dlopen(candidate.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      dlclose(handle);
      return true;
    }
  }
#else
  (void)candidates;
#endif
  return false;
}

std::string EnvOrEmpty(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string(value);
}

class CpuReferenceBackend final : public IInferenceBackend {
 public:
  BackendKind Kind() const noexcept override { return BackendKind::kCpu; }
  const char* Name() const noexcept override { return "cpu-reference"; }
  bool IsAvailable() const noexcept override { return true; }

  BackendStatus Load(const BackendConfig& config) noexcept override {
    if (!config.model_path.empty()) {
      // This backend is a scheduler/pipeline reference implementation.  It
      // must not silently claim to execute an ONNX/IR/engine file.
      loaded_ = false;
      return BackendStatus::kNotImplemented;
    }
    loaded_ = true;
    return BackendStatus::kSuccess;
  }

  BackendStatus Infer(const std::vector<Tensor>& inputs,
                      std::vector<Tensor>* outputs) noexcept override {
    if (!loaded_) return BackendStatus::kNotLoaded;
    if (outputs == nullptr || inputs.empty()) return BackendStatus::kInvalidArgument;
    for (const auto& input : inputs) {
      if (!input.IsValid()) return BackendStatus::kInvalidArgument;
    }
    // The reference backend intentionally performs a deterministic copy.  It
    // makes pipeline/scheduler tests runnable on machines without a vendor
    // runtime while vendor adapters provide actual model execution.
    *outputs = inputs;
    return BackendStatus::kSuccess;
  }

  void Unload() noexcept override { loaded_ = false; }

 private:
  bool loaded_{false};
};

}  // namespace

const char* BackendKindName(BackendKind kind) noexcept {
  switch (kind) {
    case BackendKind::kAuto:
      return "auto";
    case BackendKind::kCpu:
      return "cpu";
    case BackendKind::kOpenVINO:
      return "openvino";
    case BackendKind::kTensorRT:
      return "tensorrt";
  }
  return "unknown";
}

bool ParseBackendKind(const std::string& name, BackendKind* kind) noexcept {
  if (kind == nullptr) return false;
  const std::string value = Lower(name);
  if (value == "auto") {
    *kind = BackendKind::kAuto;
  } else if (value == "cpu" || value == "host") {
    *kind = BackendKind::kCpu;
  } else if (value == "openvino" || value == "ov" || value == "intel") {
    *kind = BackendKind::kOpenVINO;
  } else if (value == "tensorrt" || value == "trt" || value == "nvidia") {
    *kind = BackendKind::kTensorRT;
  } else {
    return false;
  }
  return true;
}

const char* BackendStatusName(BackendStatus status) noexcept {
  switch (status) {
    case BackendStatus::kSuccess:
      return "success";
    case BackendStatus::kInvalidArgument:
      return "invalid_argument";
    case BackendStatus::kUnavailable:
      return "unavailable";
    case BackendStatus::kNotLoaded:
      return "not_loaded";
    case BackendStatus::kNotImplemented:
      return "not_implemented";
    case BackendStatus::kRuntimeError:
      return "runtime_error";
  }
  return "unknown";
}

size_t TensorTypeSize(TensorType type) noexcept {
  switch (type) {
    case TensorType::kUInt8:
      return sizeof(uint8_t);
    case TensorType::kFloat16:
      return sizeof(uint16_t);
    case TensorType::kFloat32:
      return sizeof(float);
    case TensorType::kInt32:
      return sizeof(int32_t);
    case TensorType::kInt64:
      return sizeof(int64_t);
  }
  return 0;
}

size_t Tensor::ElementCount() const noexcept {
  size_t count = 1;
  // An empty shape denotes a scalar in OpenVINO and most tensor runtimes.
  if (shape.empty()) return 1;
  for (int64_t dim : shape) {
    if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / count) return 0;
    count *= static_cast<size_t>(dim);
  }
  return count;
}

size_t Tensor::ExpectedBytes() const noexcept {
  const size_t type_size = TensorTypeSize(type);
  const size_t count = ElementCount();
  if (type_size == 0 || count == 0 || count > std::numeric_limits<size_t>::max() / type_size) return 0;
  return count * type_size;
}

bool Tensor::IsValid() const noexcept {
  const size_t expected = ExpectedBytes();
  return expected != 0 && expected == data.size();
}

BackendRegistry& BackendRegistry::Instance() noexcept {
  static BackendRegistry registry;
  return registry;
}

BackendRegistry::BackendRegistry() {
  factories_.emplace_back(BackendKind::kCpu,
                          [](const BackendConfig&) { return std::make_shared<CpuReferenceBackend>(); });
#if defined(AI_ENGINE_WITH_OPENVINO)
  factories_.emplace_back(BackendKind::kOpenVINO, &CreateOpenVINOBackend);
#endif
#if defined(AI_ENGINE_WITH_TENSORRT)
  factories_.emplace_back(BackendKind::kTensorRT, &CreateTensorRTBackend);
#endif
}

bool BackendRegistry::Register(BackendKind kind, BackendFactory factory, bool replace) {
  if (!IsKnownBackend(kind) || !factory) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = std::find_if(factories_.begin(), factories_.end(), [kind](const auto& entry) {
    return entry.first == kind;
  });
  if (iter != factories_.end()) {
    if (!replace) return false;
    iter->second = std::move(factory);
    return true;
  }
  factories_.emplace_back(kind, std::move(factory));
  return true;
}

bool BackendRegistry::Unregister(BackendKind kind) {
  if (!IsKnownBackend(kind) || kind == BackendKind::kCpu) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  const auto old_size = factories_.size();
  factories_.erase(std::remove_if(factories_.begin(), factories_.end(), [kind](const auto& entry) {
                    return entry.first == kind;
                  }),
                  factories_.end());
  return factories_.size() != old_size;
}

bool BackendRegistry::HasFactory(BackendKind kind) const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return std::any_of(factories_.begin(), factories_.end(), [kind](const auto& entry) {
    return entry.first == kind;
  });
}

BackendKind BackendRegistry::Select(BackendKind requested) const noexcept {
  if (requested != BackendKind::kAuto) {
    return HasFactory(requested) && IsRuntimeAvailable(requested) ? requested : BackendKind::kAuto;
  }
  constexpr BackendKind preference[] = {BackendKind::kOpenVINO, BackendKind::kTensorRT, BackendKind::kCpu};
  for (BackendKind candidate : preference) {
    if (HasFactory(candidate) && IsRuntimeAvailable(candidate)) return candidate;
  }
  return BackendKind::kAuto;
}

std::shared_ptr<IInferenceBackend> BackendRegistry::Create(const BackendConfig& config) const {
  const BackendKind selected = Select(config.kind);
  if (selected == BackendKind::kAuto) return nullptr;
  BackendFactory factory;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = std::find_if(factories_.begin(), factories_.end(), [selected](const auto& entry) {
      return entry.first == selected;
    });
    if (iter == factories_.end()) return nullptr;
    factory = iter->second;
  }
  BackendConfig effective = config;
  effective.kind = selected;
  return factory(effective);
}

std::vector<BackendAvailability> BackendRegistry::Discover() const {
  std::vector<BackendAvailability> result;
  constexpr BackendKind kinds[] = {BackendKind::kCpu, BackendKind::kOpenVINO, BackendKind::kTensorRT};
  result.reserve(sizeof(kinds) / sizeof(kinds[0]));
  for (BackendKind kind : kinds) {
    BackendAvailability entry;
    entry.kind = kind;
    entry.factory_registered = HasFactory(kind);
    entry.runtime_available = IsRuntimeAvailable(kind);
    entry.detail = RuntimeAvailabilityDetail(kind);
    result.emplace_back(std::move(entry));
  }
  return result;
}

bool IsRuntimeAvailable(BackendKind kind) noexcept {
  if (kind == BackendKind::kCpu) return true;
  if (kind == BackendKind::kOpenVINO) {
    const std::string override_path = EnvOrEmpty("AI_ENGINE_OPENVINO_LIB");
    if (!override_path.empty() && HasSharedLibrary({override_path})) return true;
    return HasSharedLibrary({"libopenvino.so", "libopenvino.so.2023.0", "libopenvino.so.2022.1",
                             "libopenvino_c.so", "libopenvino_c.so.2022"});
  }
  if (kind == BackendKind::kTensorRT) {
    const std::string override_path = EnvOrEmpty("AI_ENGINE_TENSORRT_LIB");
    if (!override_path.empty() && HasSharedLibrary({override_path})) return true;
    return HasSharedLibrary({"libnvinfer.so", "libnvinfer.so.10", "libnvinfer.so.8"});
  }
  return false;
}

std::string RuntimeAvailabilityDetail(BackendKind kind) {
  if (kind == BackendKind::kCpu) return "built-in reference backend";
  if (kind == BackendKind::kOpenVINO) {
    return IsRuntimeAvailable(kind) ? "OpenVINO shared library detected" : "OpenVINO shared library not found";
  }
  if (kind == BackendKind::kTensorRT) {
    return IsRuntimeAvailable(kind) ? "TensorRT shared library detected" : "TensorRT shared library not found";
  }
  return "automatic selection has no runtime";
}

std::shared_ptr<IInferenceBackend> CreateBackend(const BackendConfig& config) {
  return BackendRegistry::Instance().Create(config);
}

bool RunBackendSmokeTest(std::string* error) noexcept {
  auto fail = [error](const char* message) noexcept {
    if (error != nullptr) *error = message;
    return false;
  };
  try {
    BackendKind parsed = BackendKind::kAuto;
    if (!ParseBackendKind("TensorRT", &parsed) || parsed != BackendKind::kTensorRT) {
      return fail("backend name parser failed");
    }

    Tensor input;
    input.type = TensorType::kFloat32;
    input.shape = {1, 2};
    input.data.resize(input.ExpectedBytes());
    if (!input.IsValid()) return fail("tensor validation failed");

    BackendConfig config;
    config.kind = BackendKind::kCpu;
    auto backend = CreateBackend(config);
    if (!backend || !backend->IsAvailable()) return fail("cpu backend unavailable");
    if (backend->Load(config) != BackendStatus::kSuccess) return fail("cpu backend load failed");

    std::vector<Tensor> outputs;
    if (backend->Infer({input}, &outputs) != BackendStatus::kSuccess || outputs.size() != 1 ||
        outputs.front().data != input.data || outputs.front().shape != input.shape) {
      backend->Unload();
      return fail("cpu backend inference failed");
    }
    backend->Unload();
    if (error != nullptr) error->clear();
    return true;
  } catch (...) {
    return fail("unexpected exception in backend smoke test");
  }
}

bool RegisterOpenVINOBackend(BackendFactory factory, bool replace) {
  return BackendRegistry::Instance().Register(BackendKind::kOpenVINO, std::move(factory), replace);
}

bool RegisterTensorRTBackend(BackendFactory factory, bool replace) {
  return BackendRegistry::Instance().Register(BackendKind::kTensorRT, std::move(factory), replace);
}

}  // namespace inference
}  // namespace ai
