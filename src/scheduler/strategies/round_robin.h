#ifndef AI_SCHEDULER_STRATEGIES_ROUND_ROBIN_H_
#define AI_SCHEDULER_STRATEGIES_ROUND_ROBIN_H_

#include <atomic>
#include <vector>
#include <memory>

#include "scheduler/scheduler.h"

namespace ai {

class RoundRobinStrategy : public IStrategy {
 public:
  RoundRobinStrategy() : idx_(0) {}
  CommandPtr Select(const std::vector<CommandPtr>& commands, const infer_server::PackagePtr&) override {
    if (commands.empty()) return nullptr;
    auto i = idx_.fetch_add(1) % commands.size();
    return commands[i];
  }

 private:
  std::atomic<size_t> idx_;
};

}  // namespace ai

#endif  // AI_SCHEDULER_STRATEGIES_ROUND_ROBIN_H_
