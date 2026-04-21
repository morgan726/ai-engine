#include "scheduler.h"

#include <glog/logging.h>

namespace ai {

using infer_server::Batcher;
using infer_server::Priority;
using infer_server::PriorityThreadPool;
using infer_server::Status;
using infer_server::PackagePtr;

PriorityScheduler& PriorityScheduler::Instance() {
  static PriorityScheduler inst;
  return inst;
}

void PriorityScheduler::SetThreadPool(PriorityThreadPool* tp) { tp_ = tp; }

void PriorityScheduler::SetStrategy(std::shared_ptr<IStrategy> strategy) { strategy_ = std::move(strategy); }

void PriorityScheduler::AddObserver(std::shared_ptr<ISchedulerObserver> obs) { observers_.push_back(std::move(obs)); }

void PriorityScheduler::Register(const std::string& name, CommandPtr cmd, int base_priority) {
  registry_[name] = Entry{std::move(cmd), base_priority};
}

void PriorityScheduler::Enqueue(const std::string& tag, PackagePtr pkg, int base_priority) {
  if (!batcher_) {
    auto notify = [this](std::vector<Item>&& items) { Consume(std::move(items)); };
    batcher_.reset(new Batcher<Item>(notify, 10, 32));
  }
  for (auto& o : observers_) o->OnEnqueue(tag);
  batcher_->AddItem(Item{tag, std::move(pkg), base_priority});
}

void PriorityScheduler::Consume(Batch&& batch) {
  if (!tp_) {
    // fallback: create a small owned priority thread pool
    if (!owned_tp_) {
      owned_tp_.reset(new PriorityThreadPool([](){return true;}, std::max(1u, std::thread::hardware_concurrency()/2)));
    }
    tp_ = owned_tp_.get();
  }
  for (auto& it : batch) {
    const std::string& tag = it.tag;
    PackagePtr& pkg = it.pkg;

    // select command
    CommandPtr chosen = nullptr;
    if (strategy_) {
      std::vector<CommandPtr> cmds; cmds.reserve(registry_.size());
      for (auto& kv : registry_) cmds.push_back(kv.second.cmd);
      chosen = strategy_->Select(cmds, pkg);
    }
    if (!chosen && !registry_.empty()) {
      chosen = registry_.begin()->second.cmd;
    }
    if (!chosen) {
      LOG(ERROR) << "No command registered for scheduling";
      continue;
    }

    // compute priority
    int base = it.base;
    auto itreg = registry_.begin();
    for (auto& kv : registry_) {
      if (kv.second.cmd == chosen) { itreg = registry_.find(kv.first); break; }
    }
    if (itreg != registry_.end()) base = itreg->second.base;
    auto prio = infer_server::Priority(base).Get(0);

    for (auto& o : observers_) o->OnStart(tag);
    tp_->VoidPush(prio, [this, chosen, pkg, tag]() {
      Status st = Status::SUCCESS;
      try {
        chosen->Execute(pkg);
      } catch (...) {
        st = Status::ERROR_BACKEND;
      }
      for (auto& o : observers_) o->OnFinish(tag, st);
    });
  }
}

}  // namespace ai
