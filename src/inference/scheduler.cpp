#include "inference/scheduler.h"

#include <algorithm>
#include <utility>

namespace ai {
namespace inference {
namespace {

struct WorkerRuntime {
  std::shared_ptr<IInferenceBackend> backend;
  BackendStatus load_status{BackendStatus::kUnavailable};
  BackendKind kind{BackendKind::kAuto};
  std::string name;

  ~WorkerRuntime() {
    if (backend != nullptr) backend->Unload();
  }
};

thread_local std::unique_ptr<WorkerRuntime> current_worker;

bool InitWorker(const BackendConfig& config) noexcept {
  try {
    current_worker = std::make_unique<WorkerRuntime>();
    current_worker->backend = CreateBackend(config);
    if (current_worker->backend == nullptr) return false;
    current_worker->kind = current_worker->backend->Kind();
    current_worker->name = current_worker->backend->Name() == nullptr ? "" : current_worker->backend->Name();
    current_worker->load_status = current_worker->backend->Load(config);
    return current_worker->load_status == BackendStatus::kSuccess;
  } catch (...) {
    if (current_worker != nullptr) {
      current_worker->load_status = BackendStatus::kRuntimeError;
      current_worker->name = "backend initialization threw an exception";
    }
    return false;
  }
}

}  // namespace

InferenceScheduler::InferenceScheduler(Options options) : options_(std::move(options)) {
  worker_count_ = std::max<size_t>(1, options_.workers);
  const BackendConfig config = options_.backend;
  pool_ = std::make_unique<infer_server::PriorityThreadPool>(
      [config]() noexcept { return InitWorker(config); }, static_cast<int>(worker_count_));
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
  // Stop(true) is deliberate for both modes. In non-draining mode queued
  // tasks observe cancel_queued_ and resolve their promises immediately;
  // joining keeps pool workers and their thread-local backend instances safe.
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
    // The caller may have abandoned its future. Do not terminate a worker.
  }
}

void InferenceScheduler::Execute(std::shared_ptr<Invocation> invocation) noexcept {
  active_.fetch_add(1, std::memory_order_relaxed);
  Result result;
  result.id = invocation->request.id;
  if (cancel_queued_.load(std::memory_order_acquire)) {
    result.status = BackendStatus::kRuntimeError;
    result.detail = "cancelled by shutdown";
  } else if (current_worker == nullptr || current_worker->backend == nullptr) {
    result.status = BackendStatus::kUnavailable;
    result.detail = "backend factory unavailable";
  } else {
    result.backend = current_worker->kind;
    result.detail = current_worker->name;
    result.status = current_worker->load_status;
    if (result.status == BackendStatus::kSuccess) {
      try {
        result.status = current_worker->backend->Infer(invocation->request.inputs, &result.outputs);
      } catch (...) {
        result.status = BackendStatus::kRuntimeError;
        result.outputs.clear();
        result.detail = "backend inference threw an exception";
      }
    }
  }
  Complete(invocation, std::move(result));
  active_.fetch_sub(1, std::memory_order_relaxed);
  pending_.fetch_sub(1, std::memory_order_relaxed);
}

}  // namespace inference
}  // namespace ai
