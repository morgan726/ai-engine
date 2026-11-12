#ifndef AI_ENGINE_INFERENCE_SCHEDULER_H_
#define AI_ENGINE_INFERENCE_SCHEDULER_H_

#include "inference/backend.h"
#include "utils/thread_pool.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ai {
namespace inference {

// Request scheduler backed by the project's priority thread pool. Each pool
// worker creates one private backend instance, so TensorRT execution contexts
// and CUDA streams are never shared by concurrent requests.
class InferenceScheduler final {
 public:
  struct Options {
    BackendConfig backend;
    size_t workers{2};
    bool drain_on_shutdown{true};
  };

  struct Request {
    uint64_t id{0};
    int64_t priority{0};
    std::vector<Tensor> inputs;
  };

  struct Result {
    uint64_t id{0};
    BackendKind backend{BackendKind::kAuto};
    BackendStatus status{BackendStatus::kRuntimeError};
    std::vector<Tensor> outputs;
    std::string detail;
  };

  using Future = std::future<Result>;

  explicit InferenceScheduler(Options options);
  ~InferenceScheduler();

  InferenceScheduler(const InferenceScheduler&) = delete;
  InferenceScheduler& operator=(const InferenceScheduler&) = delete;

  Future Submit(Request request);

  // Stops accepting work and joins all pool workers. Non-draining shutdown
  // marks queued tasks cancelled; the pool still joins them so every future is
  // completed instead of becoming a broken promise.
  void Shutdown(bool drain) noexcept;
  void Shutdown() noexcept { Shutdown(options_.drain_on_shutdown); }

  size_t WorkerCount() const noexcept { return pool_ == nullptr ? 0 : worker_count_; }
  // Number of submitted requests that have not completed (including active).
  size_t Pending() const noexcept { return pending_.load(std::memory_order_relaxed); }
  size_t Active() const noexcept { return active_.load(std::memory_order_relaxed); }
  bool IsStopping() const noexcept;

 private:
  struct Invocation {
    Request request;
    std::promise<Result> promise;
  };

  void Execute(std::shared_ptr<Invocation> invocation) noexcept;
  static Result ErrorResult(const Request& request, BackendStatus status, const char* detail);
  void Complete(const std::shared_ptr<Invocation>& invocation, Result result) noexcept;

  Options options_;
  std::unique_ptr<infer_server::PriorityThreadPool> pool_;
  mutable std::mutex mutex_;
  std::condition_variable stopped_condition_;
  std::atomic<size_t> pending_{0};
  std::atomic<size_t> active_{0};
  uint64_t next_request_id_{1};
  size_t worker_count_{0};
  bool stopping_{false};
  bool stopped_{false};
  std::atomic<bool> cancel_queued_{false};
};

}  // namespace inference
}  // namespace ai

#endif  // AI_ENGINE_INFERENCE_SCHEDULER_H_
