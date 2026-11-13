#include "inference/scheduler.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool ParsePositive(const char* value, size_t* output) {
  if (value == nullptr || output == nullptr || *value == '-') return false;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0) return false;
  *output = static_cast<size_t>(parsed);
  return true;
}

bool ParseNonNegative(const char* value, int* output) {
  if (value == nullptr || output == nullptr || *value == '-') return false;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed > 128) return false;
  *output = static_cast<int>(parsed);
  return true;
}

bool MakeYoloInput(const std::string& image_path, ai::inference::Tensor* tensor) {
  if (tensor == nullptr) return false;
  constexpr int kInputHeight = 640;
  constexpr int kInputWidth = 640;
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "cannot read image: " << image_path << '\n';
    return false;
  }

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(kInputWidth, kInputHeight));
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

  tensor->type = ai::inference::TensorType::kFloat32;
  tensor->shape = {1, 3, kInputHeight, kInputWidth};
  tensor->data.resize(tensor->ExpectedBytes());
  auto* destination = reinterpret_cast<float*>(tensor->data.data());
  std::vector<cv::Mat> channels = {
      cv::Mat(kInputHeight, kInputWidth, CV_32F, destination),
      cv::Mat(kInputHeight, kInputWidth, CV_32F, destination + kInputHeight * kInputWidth),
      cv::Mat(kInputHeight, kInputWidth, CV_32F, destination + 2 * kInputHeight * kInputWidth)};
  cv::split(resized, channels);
  return tensor->IsValid();
}

std::string ShapeString(const ai::inference::Tensor& tensor) {
  std::string result = "[";
  for (size_t index = 0; index < tensor.shape.size(); ++index) {
    if (index != 0) result += ',';
    result += std::to_string(tensor.shape[index]);
  }
  result += ']';
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 6) {
    std::cerr << "usage: trt_scheduler_demo <engine.plan> <image> [workers] [requests] [device_id]\n";
    return 2;
  }

  size_t workers = 2;
  size_t requests = 1;
  int device_id = 0;
  if (argc >= 4 && !ParsePositive(argv[3], &workers)) {
    std::cerr << "workers must be positive\n";
    return 2;
  }
  if (argc >= 5 && !ParsePositive(argv[4], &requests)) {
    std::cerr << "requests must be positive\n";
    return 2;
  }
  if (argc >= 6 && !ParseNonNegative(argv[5], &device_id)) {
    std::cerr << "device_id must be non-negative\n";
    return 2;
  }

  ai::inference::Tensor input;
  if (!MakeYoloInput(argv[2], &input)) return 1;

  ai::inference::InferenceScheduler::Options options;
  options.workers = workers;
  options.backend.kind = ai::inference::BackendKind::kTensorRT;
  options.backend.model_path = argv[1];
  options.backend.device_id = device_id;
  ai::inference::InferenceScheduler scheduler(options);

  std::vector<ai::inference::InferenceScheduler::Future> futures;
  futures.reserve(requests);
  for (size_t index = 0; index < requests; ++index) {
    ai::inference::InferenceScheduler::Request request;
    request.id = index + 1;
    request.priority = static_cast<int64_t>(requests - index);
    request.backend = options.backend;
    request.inputs.push_back(input);
    futures.emplace_back(scheduler.Submit(std::move(request)));
  }

  size_t successful = 0;
  for (auto& future : futures) {
    const auto result = future.get();
    std::cout << "request=" << result.id << " backend="
              << ai::inference::BackendKindName(result.backend) << " status="
              << ai::inference::BackendStatusName(result.status) << " detail=" << result.detail;
    if (!result.outputs.empty()) {
      std::cout << " outputs=" << result.outputs.size();
      for (const auto& output : result.outputs) {
        std::cout << " " << ShapeString(output) << "/" << output.data.size() << "B";
      }
    }
    std::cout << '\n';
    if (result.status == ai::inference::BackendStatus::kSuccess) ++successful;
  }
  scheduler.Shutdown(true);
  std::cout << "trt_scheduler workers=" << workers << " requests=" << requests
            << " successful=" << successful << '\n';
  return successful == requests ? 0 : 1;
}
