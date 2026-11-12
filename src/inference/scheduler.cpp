#include "inference/scheduler.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace ai {
namespace inference {
namespace {

// BackendConfig is a value object. Requests are copied into queued
// invocations, so a stable key lets every thread cache one loaded model per
// hardware/device/configuration without sharing vendor execution state.
struct BackendKey {
  BackendKind kind{BackendKind::kAuto};
  std::string model_path;
  std::string device;
  int device_id{0};
  int num_streams{1};
  int num_threads{0};
  bool enable_fp16{true};

  static BackendKey From(const BackendConfig& config) {
    BackendKey key;
    key.kind = config.kind;
    key.model_path = config.model_path;
    key.device = config.device;
    key.device_id = config.device_id;
    key.num_streams = config.num_streams;
    key.num_threads = config.num_threads;
    key.enable_fp16 = config.enable_fp16;
    return key;
  }

  bool operator==(const BackendKey& other) const noexcept {
    return kind == other.kind && device_id == other.device_id && num_streams == other.num_streams &&
           num_threads == other.num_threads && enable_fp16 == other.enable_fp16 && model_path == other.model_path &&
           device == other.device;
  }
};

struct BackendKeyHash {
  size_t operator()(const BackendKey& key) const noexcept {
    size_t hash = std::hash<int>{}(static_cast<int>(key.kind));
    const auto combine = [&hash](size_t value) noexcept {
      hash ^= value + static_cast<size_t>(0x9e3779b9U) + (hash << 6U) + (hash >> 2U);
    };
    combine(std::hash<std::string>{}(key.model_path));
    combine(std::hash<std::string>{}(key.device));
    combine(std::hash<int>{}(key.device_id));
    combine(std::hash<int>{}(key.num_streams));
    combine(std::hash<int>{}(key.num_threads));
    combine(std::hash<bool>{}(key.enable_fp16));
    return hash;
  }
};

struct BackendSlot {
  std::shared_ptr<IInferenceBackend> backend;
  BackendKind kind{BackendKind::kAuto};
  BackendStatus status{BackendStatus::kUnavailable};
  std::string detail;

  ~BackendSlot() {
    // The cache lives in the worker thread's thread-local runtime. Unload
    // before releasing the shared_ptr so SDK resources are deterministic.
    if (backend != nullptr) backend->Unload();
  }
};

struct WorkerRuntime {
  std::unordered_map<BackendKey, std::unique_ptr<BackendSlot>, BackendKeyHash> backends;

  BackendSlot* Get(const BackendConfig& config) {
    const BackendKey key = BackendKey::From(config);
    const auto found = backends.find(key);
    if (found != backends.end()) return found->second.get();

    auto slot = std::make_unique<BackendSlot>();
    slot->kind = config.kind;
    slot->backend = CreateBackend(config);
    if (slot->backend == nullptr) {
      slot->status = BackendStatus::kUnavailable;
      slot->detail = "backend factory or runtime unavailable";
    } else {
      slot->kind = slot->backend->Kind();
      slot->detail = slot->backend->Name() == nullptr ? BackendKindName(slot->kind) : slot->backend->Name();
      slot->status = slot->backend->Load(config);
      if (slot->status != BackendStatus::kSuccess) {
        slot->detail += ": ";
        slot->detail += BackendStatusName(slot->status);
      }
    }

    auto inserted = backends.emplace(key, std::move(slot));
    return inserted.first->second.get();
  }
};

thread_local std::unique_ptr<WorkerRuntime> current_worker;

bool InitWorker() noexcept {
  try {
    current_worker = std::make_unique<WorkerRuntime>();
    return true;
  } catch (...) {
    current_worker.reset();
    return false;
  }
}

bool IsDefaultRequestConfig(const BackendConfig& config) noexcept {
  return config.kind == BackendKind::kAuto && config.model_path.empty() && config.device == "AUTO" &&
         config.device_id == 0 && config.num_streams == 1 && config.num_threads == 0 && config.enable_fp16;
}

BackendConfig ResolveConfig(const InferenceScheduler::Options& options,
                            const InferenceScheduler::Request& request) {
  // Preserve Options::backend for callers that do not need per-request
  // routing. Any non-default request config is authoritative.
  return IsDefaultRequestConfig(request.backend) ? options.backend : request.backend;
}

}  // namespace

InferenceScheduler::InferenceScheduler(Options options) : options_(std::move(options)) {
  worker_count_ = std::max<size_t>(1, options_.workers);
  pool_ = std::make_unique<infer_server::PriorityThreadPool>(&InitWorker, static_cast<int>(worker_count_));
}

InferenceScheduler::~InferenceScheduler() { Shutdown(); }

InferenceScheduler::Future InferenceScheduler::Submit(Request request) {
  auto invocation = std::make_shared<Invocation>();
  invocation->request = std::move(request);
  Future future = invocation->promise.get_future();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (invocation->request.id == 0) invocation->request.id = next_request_id_++;
    if (stopping_ || pool_ == nullptr) {
      Complete(invocation, ErrorResult(invocation->request, BackendStatus::kRuntimeError,
                                       "scheduler is stopping"));
      return future;
    }
    pending_.fetch_add(1, std::memory_order_relaxed);
    try {
      pool_->VoidPush(invocation->request.priority, [this, invocation]() { Execute(invocation); });
    } catch (...) {
      pending_.fetch_sub(1, std::memory_order_relaxed);
      Complete(invocation, ErrorResult(invocation->request, BackendStatus::kRuntimeError,
                                       "thread pool rejected request"));
    }
  }
  return future;
}

void InferenceScheduler::Shutdown(bool drain) noexcept {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopping_) {
      stopped_condition_.wait(lock, [this] { return stopped_; });
      return;
    }
    stopping_ = true;
    cancel_queued_.store(!drain, std::memory_order_release);
  }
  // PriorityThreadPool cannot resolve promises for tasks removed by
  // Stop(false). Always let callbacks run and use cancel_queued_ to complete
  // them as cancelled, preserving a usable future for every Submit call.
  if (pool_ != nullptr) pool_->Stop(true);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
  }
  stopped_condition_.notify_all();
}

bool InferenceScheduler::IsStopping() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return stopping_;
}

InferenceScheduler::Result InferenceScheduler::ErrorResult(const Request& request, BackendStatus status,
                                                           const char* detail) {
  Result result;
  result.id = request.id;
  result.status = status;
  result.detail = detail == nullptr ? "" : detail;
  return result;
}

void InferenceScheduler::Complete(const std::shared_ptr<Invocation>& invocation, Result result) noexcept {
  try {
    invocation->promise.set_value(std::move(result));
  } catch (...) {
    // A caller may abandon its future. Never terminate a worker because of a
    // broken promise or an exception from a user-owned future state.
  }
}

void InferenceScheduler::Execute(std::shared_ptr<Invocation> invocation) noexcept {
  active_.fetch_add(1, std::memory_order_relaxed);
  Result result;
  result.id = invocation->request.id;
  if (cancel_queued_.load(std::memory_order_acquire)) {
    result.status = BackendStatus::kRuntimeError;
    result.detail = "cancelled by shutdown";
  } else if (current_worker == nullptr) {
    result.status = BackendStatus::kUnavailable;
    result.detail = "thread-local backend cache unavailable";
  } else {
    try {
      const BackendConfig config = ResolveConfig(options_, invocation->request);
      BackendSlot* slot = current_worker->Get(config);
      if (slot == nullptr) {
        result.status = BackendStatus::kUnavailable;
        result.detail = "backend cache insertion failed";
      } else {
        result.backend = slot->kind;
        result.status = slot->status;
        result.detail = slot->detail;
        if (result.status == BackendStatus::kSuccess && slot->backend != nullptr) {
          result.status = slot->backend->Infer(invocation->request.inputs, &result.outputs);
          if (result.status != BackendStatus::kSuccess) {
            result.detail += ": ";
            result.detail += BackendStatusName(result.status);
          }
        }
      }
    } catch (...) {
      result.status = BackendStatus::kRuntimeError;
      result.outputs.clear();
      result.detail = "backend scheduling or inference threw an exception";
    }
  }
  Complete(invocation, std::move(result));
  active_.fetch_sub(1, std::memory_order_relaxed);
  pending_.fetch_sub(1, std::memory_order_relaxed);
}

}  // namespace inference
}  // namespace ai
