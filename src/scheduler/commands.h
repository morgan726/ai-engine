#ifndef AI_SCHEDULER_COMMANDS_H_
#define AI_SCHEDULER_COMMANDS_H_

#include <string>
#include <memory>
#include <glog/logging.h>
#include "scheduler.h"

namespace ai {

// MSSD via OpenCV DNN (CPU/GPU depending on build)
class MssdCommand : public ICommand {
 public:
  explicit MssdCommand(const std::string& image_or_video) : src_(image_or_video) {}
  void Execute(infer_server::PackagePtr) override;
 private:
  std::string src_;
};

// TRT-YOLO placeholder command: integrate with existing TRT code path if linked
class TrtYoloCommand : public ICommand {
 public:
  explicit TrtYoloCommand(const std::string& image_or_video) : src_(image_or_video) {}
  void Execute(infer_server::PackagePtr) override;
 private:
  std::string src_;
};

}  // namespace ai

#endif  // AI_SCHEDULER_COMMANDS_H_

