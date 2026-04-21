#include "commands.h"

#include <opencv2/opencv.hpp>

namespace ai {

void MssdCommand::Execute(infer_server::PackagePtr) {
  // Simple runnable placeholder: load one frame or image and run preloaded net if available
  cv::Mat frame;
  cv::VideoCapture cap;
  if (src_.rfind("rtsp://", 0) == 0 || src_.rfind("http", 0) == 0 || src_.rfind("/", 0) == 0) {
    cap.open(src_);
    if (cap.isOpened()) cap >> frame;
  } else {
    frame = cv::imread(src_);
  }
  if (frame.empty()) return;

  // Placeholder: real inference is performed through the framework; here we just touch OpenCV
  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(300, 300));
  (void)resized;
}

void TrtYoloCommand::Execute(infer_server::PackagePtr) {
  // Placeholder hook: in a full integration, call into src/model/trt_yolo binaries/libraries
}

} // namespace ai
