#include <gtest/gtest.h>
#include <atomic>
#include "scheduler/scheduler.h"

using namespace ai;

class DummyCommand : public ICommand {
 public:
  void Execute(infer_server::PackagePtr) override { ++counter; }
  static std::atomic<int> counter;
};

std::atomic<int> DummyCommand::counter{0};

TEST(Scheduler, EnqueueAndRun) {
  auto& sched = PriorityScheduler::Instance();
  auto cmd = std::make_shared<DummyCommand>();
  // local thread pool just for test
  auto tp = new infer_server::PriorityThreadPool([](){return true;}, 2);
  sched.SetThreadPool(tp);
  sched.Register("dummy", cmd, 5);
  auto pkg = infer_server::Package::Create(1, "t");
  sched.Enqueue("tag", std::move(pkg), 5);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_GE(DummyCommand::counter.load(), 1);
  delete tp;
}

