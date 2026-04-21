#ifndef AI_SCHEDULER_STRATEGIES_HIGHEST_PRIORITY_H_
#define AI_SCHEDULER_STRATEGIES_HIGHEST_PRIORITY_H_

#include <vector>
#include <memory>
#include <glog/logging.h>
#include "scheduler/scheduler.h"

namespace ai {

// Placeholder: this strategy simply returns first command; real impl could
// read per-command priority and pick the highest.
class HighestPriorityStrategy : public IStrategy {
 public:
  CommandPtr Select(const std::vector<CommandPtr>& commands, const infer_server::PackagePtr&) override {
    if (commands.empty()) return nullptr;
    return commands.front();
  }
};

}  // namespace ai

#endif  // AI_SCHEDULER_STRATEGIES_HIGHEST_PRIORITY_H_
