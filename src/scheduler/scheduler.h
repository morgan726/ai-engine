#ifndef AI_SCHEDULER_SCHEDULER_H_
#define AI_SCHEDULER_SCHEDULER_H_

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <map>

#include "inference/infer_server.h"
#include "core/priority.h"
#include "utils/thread_pool.h"
#include "utils/batcher.h"

namespace ai {

// Command interface for model inference
class ICommand {
 public:
  virtual ~ICommand() = default;
  virtual void Execute(infer_server::PackagePtr request) = 0;
};

using CommandPtr = std::shared_ptr<ICommand>;

// Strategy to choose which model/command to run
class IStrategy {
 public:
  virtual ~IStrategy() = default;
  virtual CommandPtr Select(const std::vector<CommandPtr>& commands,
                            const infer_server::PackagePtr& pkg) = 0;
};

// Observer for task lifecycle
class ISchedulerObserver {
 public:
  virtual ~ISchedulerObserver() = default;
  virtual void OnEnqueue(const std::string& tag) {}
  virtual void OnStart(const std::string& tag) {}
  virtual void OnFinish(const std::string& tag, infer_server::Status st) {}
};

// Producer-Consumer scheduler with priority and batching
class PriorityScheduler {
 public:
  static PriorityScheduler& Instance();

  void SetThreadPool(infer_server::PriorityThreadPool* tp);
  void SetStrategy(std::shared_ptr<IStrategy> strategy);
  void AddObserver(std::shared_ptr<ISchedulerObserver> obs);

  // Register a model command with a logical name
  void Register(const std::string& name, CommandPtr cmd, int base_priority = 5);

  // Enqueue one request with tag and base priority (0..9)
  void Enqueue(const std::string& tag, infer_server::PackagePtr pkg, int base_priority = 5);

 private:
  PriorityScheduler() = default;

  struct Entry {
    CommandPtr cmd;
    int base{5};
  };

  struct Item { std::string tag; infer_server::PackagePtr pkg; int base; };
  using Batch = std::vector<Item>;
  void Consume(Batch&& batch);

  std::map<std::string, Entry> registry_;
  std::shared_ptr<IStrategy> strategy_;
  std::vector<std::shared_ptr<ISchedulerObserver>> observers_;

  std::unique_ptr<infer_server::Batcher<Item>> batcher_;
  std::unique_ptr<infer_server::PriorityThreadPool> owned_tp_;
  infer_server::PriorityThreadPool* tp_{nullptr};
};

}  // namespace ai

#endif  // AI_SCHEDULER_SCHEDULER_H_
