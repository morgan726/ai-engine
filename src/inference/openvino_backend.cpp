#include "inference/backend.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include <openvino/openvino.hpp>

namespace ai {
namespace inference {
namespace {

bool ToOpenVINOType(TensorType type, ov::element::Type* result) noexcept {
  if (result == nullptr) return false;
  switch (type) {
    case TensorType::kUInt8:
      *result = ov::element::u8;
      return true;
    case TensorType::kFloat16:
      *result = ov::element::f16;
      return true;
    case TensorType::kFloat32:
      *result = ov::element::f32;
      return true;
    case TensorType::kInt32:
      *result = ov::element::i32;
      return true;
    case TensorType::kInt64:
      *result = ov::element::i64;
      return true;
  }
  return false;
}

bool FromOpenVINOType(const ov::element::Type& type, TensorType* result) noexcept {
  if (result == nullptr) return false;
  if (type == ov::element::u8) {
    *result = TensorType::kUInt8;
  } else if (type == ov::element::f16) {
    *result = TensorType::kFloat16;
  } else if (type == ov::element::f32) {
    *result = TensorType::kFloat32;
  } else if (type == ov::element::i32) {
    *result = TensorType::kInt32;
  } else if (type == ov::element::i64) {
    *result = TensorType::kInt64;
  } else {
    return false;
  }
  return true;
}

bool ToOpenVINOShape(const Tensor& tensor, ov::Shape* result) noexcept {
  if (result == nullptr) return false;
  result->clear();
  result->reserve(tensor.shape.size());
  for (const int64_t dimension : tensor.shape) {
    if (dimension <= 0 || static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max()) return false;
    result->push_back(static_cast<size_t>(dimension));
  }
  return true;
}

class OpenVINOBackend final : public IInferenceBackend {
 public:
  BackendKind Kind() const noexcept override { return BackendKind::kOpenVINO; }
  const char* Name() const noexcept override { return "openvino"; }
  bool IsAvailable() const noexcept override { return true; }

  BackendStatus Load(const BackendConfig& config) noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    UnloadLocked();
    if (config.model_path.empty()) return BackendStatus::kInvalidArgument;

    try {
      core_ = std::make_unique<ov::Core>();
      model_ = core_->read_model(config.model_path);
      if (!model_) return BackendStatus::kRuntimeError;

      const std::string device = config.device.empty() ? "AUTO" : config.device;
      compiled_model_ = std::make_unique<ov::CompiledModel>(core_->compile_model(model_, device));
      infer_request_ = std::make_unique<ov::InferRequest>(compiled_model_->create_infer_request());
      config_ = config;
      loaded_ = compiled_model_ != nullptr && infer_request_ != nullptr;
      return loaded_ ? BackendStatus::kSuccess : BackendStatus::kRuntimeError;
    } catch (const ov::Exception&) {
      UnloadLocked();
      return BackendStatus::kRuntimeError;
    } catch (const std::exception&) {
      UnloadLocked();
      return BackendStatus::kRuntimeError;
    } catch (...) {
      UnloadLocked();
      return BackendStatus::kRuntimeError;
    }
  }

  BackendStatus Infer(const std::vector<Tensor>& inputs,
                      std::vector<Tensor>* outputs) noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_ || !infer_request_ || !compiled_model_) return BackendStatus::kNotLoaded;
    if (outputs == nullptr || inputs.size() != compiled_model_->inputs().size()) {
      return BackendStatus::kInvalidArgument;
    }

    try {
      const auto& model_inputs = compiled_model_->inputs();
      for (size_t index = 0; index < inputs.size(); ++index) {
        const Tensor& input = inputs[index];
        ov::element::Type type;
        ov::Shape shape;
        if (!input.IsValid() || !ToOpenVINOType(input.type, &type) || !ToOpenVINOShape(input, &shape)) {
          return BackendStatus::kInvalidArgument;
        }
        ov::Tensor tensor(type, shape, const_cast<uint8_t*>(input.data.data()));
        infer_request_->set_tensor(model_inputs[index], tensor);
      }

      infer_request_->infer();

      std::vector<Tensor> result;
      const auto& model_outputs = compiled_model_->outputs();
      result.reserve(model_outputs.size());
      for (const auto& output : model_outputs) {
        const ov::Tensor tensor = infer_request_->get_tensor(output);
        Tensor result_tensor;
        if (!FromOpenVINOType(tensor.get_element_type(), &result_tensor.type)) {
          return BackendStatus::kNotImplemented;
        }
        const ov::Shape& shape = tensor.get_shape();
        result_tensor.shape.reserve(shape.size());
        for (const size_t dimension : shape) {
          if (dimension > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            return BackendStatus::kRuntimeError;
          }
          result_tensor.shape.push_back(static_cast<int64_t>(dimension));
        }
        const size_t bytes = tensor.get_byte_size();
        const void* source = tensor.data();
        if (bytes != result_tensor.ExpectedBytes() || (bytes != 0 && source == nullptr)) {
          return BackendStatus::kRuntimeError;
        }
        result_tensor.data.resize(bytes);
        if (bytes != 0) std::memcpy(result_tensor.data.data(), source, bytes);
        result.emplace_back(std::move(result_tensor));
      }
      outputs->swap(result);
      return BackendStatus::kSuccess;
    } catch (const ov::Exception&) {
      return BackendStatus::kRuntimeError;
    } catch (const std::exception&) {
      return BackendStatus::kRuntimeError;
    } catch (...) {
      return BackendStatus::kRuntimeError;
    }
  }

  void Unload() noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    UnloadLocked();
  }

 private:
  void UnloadLocked() noexcept {
    infer_request_.reset();
    compiled_model_.reset();
    model_.reset();
    core_.reset();
    loaded_ = false;
    config_ = BackendConfig{};
  }

  mutable std::mutex mutex_;
  std::unique_ptr<ov::Core> core_;
  std::shared_ptr<ov::Model> model_;
  std::unique_ptr<ov::CompiledModel> compiled_model_;
  std::unique_ptr<ov::InferRequest> infer_request_;
  BackendConfig config_;
  bool loaded_{false};
};

}  // namespace

std::shared_ptr<IInferenceBackend> CreateOpenVINOBackend(const BackendConfig&) {
  return std::make_shared<OpenVINOBackend>();
}

}  // namespace inference
}  // namespace ai
