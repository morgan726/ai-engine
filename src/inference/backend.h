#ifndef AI_ENGINE_INFERENCE_BACKEND_H_
#define AI_ENGINE_INFERENCE_BACKEND_H_

// Backend selection is intentionally kept independent from the model manager.
// The original model manager is tied to MagicMind, while applications on this
// project may run OpenVINO on Intel or TensorRT on NVIDIA.  This small API lets
// callers select a runtime without pulling either vendor SDK into common code.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace ai {
namespace inference {

enum class BackendKind : uint8_t {
  kAuto = 0,
  kCpu = 1,
  kOpenVINO = 2,
  kTensorRT = 3,
};

const char* BackendKindName(BackendKind kind) noexcept;
bool ParseBackendKind(const std::string& name, BackendKind* kind) noexcept;

enum class BackendStatus : uint8_t {
  kSuccess = 0,
  kInvalidArgument,
  kUnavailable,
  kNotLoaded,
  kNotImplemented,
  kRuntimeError,
};

const char* BackendStatusName(BackendStatus status) noexcept;

// Tensor storage used by backend adapters.  `data` is host memory owned by the
// Tensor.  Vendor-specific adapters may use the metadata to upload it to a
// device and copy output back before returning from Infer().
enum class TensorType : uint8_t {
  kUInt8 = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kInt32 = 3,
  kInt64 = 4,
};

size_t TensorTypeSize(TensorType type) noexcept;

struct Tensor {
  TensorType type{TensorType::kFloat32};
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;

  size_t ElementCount() const noexcept;
  size_t ExpectedBytes() const noexcept;
  bool IsValid() const noexcept;
};

// Runtime configuration shared by all backends.  Fields that a backend does
// not use are ignored, which keeps configuration files portable between Intel
// and NVIDIA hosts.
struct BackendConfig {
  BackendKind kind{BackendKind::kAuto};
  std::string model_path;
  std::string device{"AUTO"};  // OpenVINO device string, e.g. CPU/GPU/AUTO
  int device_id{0};             // TensorRT CUDA device ordinal
  int num_streams{1};
  int num_threads{0};           // zero means runtime default
  bool enable_fp16{true};
};

class IInferenceBackend {
 public:
  virtual ~IInferenceBackend() = default;

  virtual BackendKind Kind() const noexcept = 0;
  virtual const char* Name() const noexcept = 0;
  virtual bool IsAvailable() const noexcept = 0;

  virtual BackendStatus Load(const BackendConfig& config) noexcept = 0;
  virtual BackendStatus Infer(const std::vector<Tensor>& inputs,
                              std::vector<Tensor>* outputs) noexcept = 0;
  virtual void Unload() noexcept = 0;
};

using BackendFactory = std::function<std::shared_ptr<IInferenceBackend>(const BackendConfig&)>;

struct BackendAvailability {
  BackendKind kind{BackendKind::kAuto};
  bool runtime_available{false};
  bool factory_registered{false};
  std::string detail;
};

// Thread-safe process-wide registry.  Vendor adapters are registered by the
// executable when their SDK is present; the registry itself has no vendor
// header or link dependency.  A CPU reference backend is registered by
// default and is useful for smoke tests and graceful CPU fallback.
class BackendRegistry {
 public:
  static BackendRegistry& Instance() noexcept;

  bool Register(BackendKind kind, BackendFactory factory, bool replace = false);
  bool Unregister(BackendKind kind);
  bool HasFactory(BackendKind kind) const noexcept;

  // Select an explicitly requested backend or pick the first available
  // registered backend for kAuto (OpenVINO, TensorRT, then CPU).
  BackendKind Select(BackendKind requested = BackendKind::kAuto) const noexcept;
  std::shared_ptr<IInferenceBackend> Create(const BackendConfig& config) const;
  std::vector<BackendAvailability> Discover() const;

 private:
  BackendRegistry();
  BackendRegistry(const BackendRegistry&) = delete;
  BackendRegistry& operator=(const BackendRegistry&) = delete;

  mutable std::mutex mutex_;
  std::vector<std::pair<BackendKind, BackendFactory>> factories_;
};

// Runtime probes are deliberately best effort.  They only inspect whether the
// vendor shared library can be loaded and never make a CUDA/OpenVINO call.
bool IsRuntimeAvailable(BackendKind kind) noexcept;
std::string RuntimeAvailabilityDetail(BackendKind kind);

std::shared_ptr<IInferenceBackend> CreateBackend(const BackendConfig& config);

// Runs a dependency-free registry/CPU smoke test.  This is useful from a
// container health check and from unit tests before vendor SDKs are installed.
// The optional error string receives a short diagnostic on failure.
bool RunBackendSmokeTest(std::string* error = nullptr) noexcept;

// Registers a vendor adapter without exposing vendor headers to callers.  The
// adapter implementation can be built in a separate translation unit or
// plugin and invoke these functions during startup.
bool RegisterOpenVINOBackend(BackendFactory factory, bool replace = false);
bool RegisterTensorRTBackend(BackendFactory factory, bool replace = false);

}  // namespace inference
}  // namespace ai

#endif  // AI_ENGINE_INFERENCE_BACKEND_H_
