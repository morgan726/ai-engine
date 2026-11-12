#include "inference/scheduler.h"

#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  size_t workers = 4;
  size_t requests = 16;
  if (argc > 1) workers = static_cast<size_t>(std::strtoul(argv[1], nullptr, 10));
  if (argc > 2) requests = static_cast<size_t>(std::strtoul(argv[2], nullptr, 10));
  if (workers == 0 || requests == 0) {
    std::cerr << "workers and requests must be positive\n";
    return 2;
  }

  ai::inference::InferenceScheduler::Options options;
  options.workers = workers;
  options.backend.kind = ai::inference::BackendKind::kCpu;
  ai::inference::InferenceScheduler scheduler(options);

  std::vector<ai::inference::InferenceScheduler::Future> futures;
  futures.reserve(requests);
  for (size_t index = 0; index < requests; ++index) {
    ai::inference::Tensor input;
    input.type = ai::inference::TensorType::kFloat32;
    input.shape = {1, 4};
    input.data.resize(input.ExpectedBytes(), static_cast<uint8_t>(index));
    ai::inference::InferenceScheduler::Request request;
    request.priority = static_cast<int64_t>(index % 3);
    request.inputs.emplace_back(std::move(input));
    futures.emplace_back(scheduler.Submit(std::move(request)));
  }

  size_t successful = 0;
  for (auto& future : futures) {
    const auto result = future.get();
    if (result.status == ai::inference::BackendStatus::kSuccess && result.outputs.size() == 1 &&
        result.outputs.front().data.size() == sizeof(float) * 4) {
      ++successful;
    }
  }
  scheduler.Shutdown(true);
  ai::inference::InferenceScheduler::Request rejected;
  auto rejected_future = scheduler.Submit(std::move(rejected));
  const auto rejected_result = rejected_future.get();
  std::cout << "scheduler workers=" << workers << " requests=" << requests << " successful=" << successful
            << " pending=" << scheduler.Pending() << " active=" << scheduler.Active()
            << " post_shutdown_status=" << ai::inference::BackendStatusName(rejected_result.status) << '\n';
  return successful == requests && rejected_result.status == ai::inference::BackendStatus::kRuntimeError ? 0 : 1;
}
