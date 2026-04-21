#include <glog/logging.h>
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>

#include "scheduler/scheduler.h"
#include "scheduler/commands.h"
#include "scheduler/strategies/round_robin.h"

DEFINE_string(input, "", "image or video path (or stream URI)");
DEFINE_int32(dev_id, 0, "device id");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  // Wire scheduler
  auto& sched = ai::PriorityScheduler::Instance();
  // Let scheduler create its own small pool lazily

  // Register commands (MSSD and TRT-YOLO placeholders)
  sched.Register("mssd", std::make_shared<ai::MssdCommand>(FLAGS_input), /*base_priority*/7);
  sched.Register("trtyolo", std::make_shared<ai::TrtYoloCommand>(FLAGS_input), /*base_priority*/5);
  sched.SetStrategy(std::make_shared<ai::RoundRobinStrategy>());

  // Create a dummy package and enqueue a few tasks
  for (int i = 0; i < 4; ++i) {
    auto pkg = infer_server::Package::Create(1, "prio");
    sched.Enqueue("task_" + std::to_string(i), std::move(pkg), (i % 2) ? 7 : 5);
  }

  LOG(INFO) << "Priority demo enqueued tasks.";
  return 0;
}
