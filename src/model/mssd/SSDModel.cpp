#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#if defined(AI_MSSD_USE_THREAD_POOL)
#include "utils/thread_pool.h"
#endif

namespace {

const char* kClasses[] = {"background", "station_m", "station_l_30", "station_l_45",
                          "station_r_30", "station_r_45", "background", "station_m",
                          "station_l_30", "station_l_45", "station_r_30", "background",
                          "station_m", "station_l_30", "station_l_45", "station_r_30"};

const char* EnvOr(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value == nullptr || *value == '\0' ? fallback : value;
}

struct MssdRuntime {
  std::string model_txt;
  std::string model_bin;
  cv::dnn::Net net;
};

#if defined(AI_MSSD_USE_THREAD_POOL)
thread_local std::unique_ptr<MssdRuntime> thread_runtime;
#endif

cv::dnn::Net* GetNet(const std::string& model_txt, const std::string& model_bin) {
#if defined(AI_MSSD_USE_THREAD_POOL)
  if (thread_runtime == nullptr || thread_runtime->model_txt != model_txt ||
      thread_runtime->model_bin != model_bin || thread_runtime->net.empty()) {
    auto runtime = std::make_unique<MssdRuntime>();
    runtime->model_txt = model_txt;
    runtime->model_bin = model_bin;
    runtime->net = cv::dnn::readNetFromCaffe(model_txt, model_bin);
    thread_runtime = std::move(runtime);
  }
  return &thread_runtime->net;
#else
  static cv::dnn::Net net;
  if (net.empty()) net = cv::dnn::readNetFromCaffe(model_txt, model_bin);
  return &net;
#endif
}

bool RunMssd(const std::string& image_file, const std::string& model_txt, const std::string& model_bin,
             const std::string& output_file) {
  cv::dnn::Net* net = GetNet(model_txt, model_bin);
  if (net == nullptr || net->empty()) {
    std::cerr << "Can't load network: " << model_txt << " / " << model_bin << '\n';
    return false;
  }

  cv::Mat image = cv::imread(image_file);
  if (image.empty()) {
    std::cerr << "Can't read image: " << image_file << '\n';
    return false;
  }
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(300, 300));
  const cv::Mat input = cv::dnn::blobFromImage(
      resized, 0.007843, cv::Size(300, 300), cv::Scalar(127.5, 127.5, 127.5), false);
  net->setInput(input, "data");
  const cv::Mat detection = net->forward("detection_out");
  if (detection.dims < 4 || detection.size[2] <= 0 || detection.size[3] < 7) {
    std::cerr << "Unexpected detection_out shape\n";
    return false;
  }

  const cv::Mat detection_mat(detection.size[2], detection.size[3], CV_32F,
                              const_cast<float*>(detection.ptr<float>()));
  constexpr float kConfidenceThreshold = 0.2F;
  for (int row = 0; row < detection_mat.rows; ++row) {
    const float confidence = detection_mat.at<float>(row, 2);
    if (confidence <= kConfidenceThreshold) continue;
    const int class_id = static_cast<int>(detection_mat.at<float>(row, 1));
    if (class_id < 0 || class_id >= static_cast<int>(sizeof(kClasses) / sizeof(kClasses[0]))) continue;

    const int left = static_cast<int>(detection_mat.at<float>(row, 3) * image.cols);
    const int top = static_cast<int>(detection_mat.at<float>(row, 4) * image.rows);
    const int right = static_cast<int>(detection_mat.at<float>(row, 5) * image.cols);
    const int bottom = static_cast<int>(detection_mat.at<float>(row, 6) * image.rows);
    const cv::Rect object(left, top, right - left, bottom - top);
    cv::rectangle(image, object, cv::Scalar(0, 255, 0), 2);
    std::cout << kClasses[class_id] << ": " << confidence << '\n';
    cv::putText(image, std::string(kClasses[class_id]) + ": " + std::to_string(confidence),
                cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
  }
  return cv::imwrite(output_file, image);
}

bool ParsePositive(const char* value, size_t* result) {
  if (value == nullptr || result == nullptr) return false;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (*value == '-' || end == value || *end != '\0' || parsed == 0 ||
      parsed > static_cast<unsigned long>(std::numeric_limits<int>::max())) return false;
  *result = static_cast<size_t>(parsed);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_dir = EnvOr("MSSD_MODEL_DIR", "/workspace/weights/mssd");
  const std::string model_txt = model_dir + "/MobileNetSSD_deploy.prototxt";
  const std::string model_bin = model_dir + "/MobileNetSSD_deploy.caffemodel";
  const std::string image_file = argc > 1 ? argv[1] : "/workspace/data/dog.jpg";

#if defined(AI_MSSD_USE_THREAD_POOL)
  size_t workers = 2;
  size_t requests = 1;
  if (argc > 2 && !ParsePositive(argv[2], &workers)) {
    std::cerr << "workers must be a positive integer\n";
    return 2;
  }
  if (argc > 3 && !ParsePositive(argv[3], &requests)) {
    std::cerr << "requests must be a positive integer\n";
    return 2;
  }
  infer_server::PriorityThreadPool pool([] { return true; }, static_cast<int>(workers));
  std::vector<std::future<bool>> futures;
  futures.reserve(requests);
  for (size_t index = 0; index < requests; ++index) {
    const std::string output = requests == 1 ? "output.jpg" : "output_" + std::to_string(index) + ".jpg";
    const int64_t priority = static_cast<int64_t>(index % 3);
    futures.emplace_back(pool.Push(priority, [=] {
      return RunMssd(image_file, model_txt, model_bin, output);
    }));
  }
  bool success = true;
  for (auto& future : futures) {
    try {
      success = future.get() && success;
    } catch (const std::exception& error) {
      std::cerr << "MSSD task failed: " << error.what() << '\n';
      success = false;
    }
  }
  pool.Stop(true);
  return success ? 0 : 1;
#else
  return RunMssd(image_file, model_txt, model_bin, "output.jpg") ? 0 : 1;
#endif
}
