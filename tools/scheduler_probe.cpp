#include "inference/scheduler.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

const char* EnvOrEmpty(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? "" : value;
}

int ParseIntOr(const char* value, int fallback) {
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  return end == value || *end != '\0' ? fallback : static_cast<int>(parsed);
}

}  // namespace

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
    // Route one shared queue to all three backend kinds.  The optional model
    // paths allow the same probe to run real SDK inference; without them the
    // vendor requests still exercise routing and return a clear load error.
    if (index % 3 == 1) {
      request.backend.kind = ai::inference::BackendKind::kOpenVINO;
      request.backend.model_path = EnvOrEmpty("AI_ENGINE_OPENVINO_MODEL");
      request.backend.device = EnvOrEmpty("AI_ENGINE_OPENVINO_DEVICE");
      if (request.backend.device.empty()) request.backend.device = "CPU";
    } else if (index % 3 == 2) {
      request.backend.kind = ai::inference::BackendKind::kTensorRT;
      request.backend.model_path = EnvOrEmpty("AI_ENGINE_TENSORRT_MODEL");
      request.backend.device_id = ParseIntOr(std::getenv("AI_ENGINE_TENSORRT_DEVICE"), 0);
    } else {
      request.backend.kind = ai::inference::BackendKind::kCpu;
    }
    request.inputs.emplace_back(std::move(input));
    futures.emplace_back(scheduler.Submit(std::move(request)));
  }

  size_t successful = 0;
  size_t routed_cpu = 0;
  size_t routed_openvino = 0;
  size_t routed_tensorrt = 0;
  for (auto& future : futures) {
    const auto result = future.get();
    if (result.backend == ai::inference::BackendKind::kCpu &&
        result.status == ai::inference::BackendStatus::kSuccess && result.outputs.size() == 1 &&
        result.outputs.front().data.size() == sizeof(float) * 4) {
      ++successful;
    }
    if (result.backend == ai::inference::BackendKind::kCpu) ++routed_cpu;
    if (result.backend == ai::inference::BackendKind::kOpenVINO) ++routed_openvino;
    if (result.backend == ai::inference::BackendKind::kTensorRT) ++routed_tensorrt;
  }
  scheduler.Shutdown(true);
  ai::inference::InferenceScheduler::Request rejected;
  auto rejected_future = scheduler.Submit(std::move(rejected));
  const auto rejected_result = rejected_future.get();
  std::cout << "scheduler workers=" << workers << " requests=" << requests << " cpu_success=" << successful
            << " routed_cpu=" << routed_cpu << " routed_openvino=" << routed_openvino
            << " routed_tensorrt=" << routed_tensorrt
            << " pending=" << scheduler.Pending() << " active=" << scheduler.Active()
            << " post_shutdown_status=" << ai::inference::BackendStatusName(rejected_result.status) << '\n';
  const size_t expected_cpu = (requests + 2) / 3;
  const size_t expected_openvino = (requests + 1) / 3;
  const size_t expected_tensorrt = requests / 3;
  return successful == expected_cpu && routed_cpu == expected_cpu && routed_openvino == expected_openvino &&
         routed_tensorrt == expected_tensorrt &&
         rejected_result.status == ai::inference::BackendStatus::kRuntimeError
             ? 0
             : 1;
}
