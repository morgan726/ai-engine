#include "inference/scheduler.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct ModelSpec {
  std::string name;
  std::string task;
  std::string engine;
  std::string input_kind;
  std::vector<int64_t> shape;
  ai::inference::TensorType type{ai::inference::TensorType::kFloat32};
  float scale{1.0F};
  std::string layout{"nchw"};
};

struct JobArg {
  std::string model;
  std::string input;
  std::string engine_override;
};

struct Arguments {
  std::string manifest;
  std::vector<JobArg> jobs;
  size_t workers{2};
  size_t requests{1};
  int device_id{0};
  bool list{false};
};

std::string Trim(const std::string& value) {
  const size_t begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return {};
  const size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

std::vector<std::string> Split(const std::string& value, char separator) {
  std::vector<std::string> result;
  std::string field;
  std::stringstream stream(value);
  while (std::getline(stream, field, separator)) result.emplace_back(Trim(field));
  if (!value.empty() && value.back() == separator) result.emplace_back();
  return result;
}

bool ParsePositive(const std::string& value, size_t* output) {
  if (output == nullptr || value.empty() || value.front() == '-') return false;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0' || parsed == 0) return false;
  *output = static_cast<size_t>(parsed);
  return true;
}

bool ParseNonNegative(const std::string& value, int* output) {
  if (output == nullptr || value.empty() || value.front() == '-') return false;
  char* end = nullptr;
  const long parsed = std::strtol(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0' || parsed < 0 || parsed > 128) return false;
  *output = static_cast<int>(parsed);
  return true;
}

bool ParseShape(const std::string& value, std::vector<int64_t>* shape) {
  if (shape == nullptr) return false;
  shape->clear();
  for (const auto& field : Split(value, ',')) {
    if (field.empty()) return false;
    char* end = nullptr;
    const long long parsed = std::strtoll(field.c_str(), &end, 10);
    if (end == field.c_str() || *end != '\0' || parsed <= 0) return false;
    shape->push_back(static_cast<int64_t>(parsed));
  }
  return !shape->empty();
}

bool ParseScale(const std::string& value, float* scale) {
  if (scale == nullptr || value.empty()) return false;
  char* end = nullptr;
  const float parsed = std::strtof(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0' || !std::isfinite(parsed)) return false;
  *scale = parsed;
  return true;
}

bool ParseTensorType(const std::string& value, ai::inference::TensorType* type) {
  if (type == nullptr) return false;
  if (value == "uint8") {
    *type = ai::inference::TensorType::kUInt8;
  } else if (value == "float16" || value == "half") {
    *type = ai::inference::TensorType::kFloat16;
  } else if (value == "float32" || value == "float") {
    *type = ai::inference::TensorType::kFloat32;
  } else if (value == "int32") {
    *type = ai::inference::TensorType::kInt32;
  } else if (value == "int64") {
    *type = ai::inference::TensorType::kInt64;
  } else {
    return false;
  }
  return true;
}

const char* TensorTypeName(ai::inference::TensorType type) {
  switch (type) {
    case ai::inference::TensorType::kUInt8:
      return "uint8";
    case ai::inference::TensorType::kFloat16:
      return "float16";
    case ai::inference::TensorType::kFloat32:
      return "float32";
    case ai::inference::TensorType::kInt32:
      return "int32";
    case ai::inference::TensorType::kInt64:
      return "int64";
  }
  return "unknown";
}

std::string ShapeString(const std::vector<int64_t>& shape) {
  std::string result = "[";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) result += ',';
    result += std::to_string(shape[index]);
  }
  result += ']';
  return result;
}

class ModelCatalog {
 public:
  bool Load(const fs::path& path, std::string* error) {
    std::ifstream stream(path);
    if (!stream) {
      if (error != nullptr) *error = "cannot open manifest: " + path.string();
      return false;
    }
    std::string line;
    size_t line_number = 0;
    while (std::getline(stream, line)) {
      ++line_number;
      line = Trim(line);
      if (line.empty() || line.front() == '#') continue;
      const auto fields = Split(line, '|');
      if (fields.size() != 8) {
        if (error != nullptr) {
          *error = "manifest line " + std::to_string(line_number) +
                   " must contain 8 pipe-separated fields";
        }
        return false;
      }
      ModelSpec spec;
      spec.name = fields[0];
      spec.task = fields[1];
      spec.engine = fields[2];
      spec.input_kind = fields[3];
      spec.layout = fields[7];
      if (spec.name.empty() || spec.engine.empty() ||
          (spec.input_kind != "image" && spec.input_kind != "tensor") ||
          (spec.layout != "nchw" && spec.layout != "nhwc")) {
        if (error != nullptr) *error = "invalid manifest fields at line " + std::to_string(line_number);
        return false;
      }
      if (!ParseShape(fields[4], &spec.shape) || !ParseTensorType(fields[5], &spec.type) ||
          !ParseScale(fields[6], &spec.scale)) {
        if (error != nullptr) *error = "invalid shape, type or scale at line " + std::to_string(line_number);
        return false;
      }
      if (Find(spec.name) != nullptr) {
        if (error != nullptr) *error = "duplicate model in manifest: " + spec.name;
        return false;
      }
      models_.emplace_back(std::move(spec));
    }
    return !models_.empty();
  }

  const ModelSpec* Find(const std::string& name) const {
    for (const auto& model : models_) {
      if (model.name == name) return &model;
    }
    return nullptr;
  }

  const std::vector<ModelSpec>& Models() const { return models_; }

 private:
  std::vector<ModelSpec> models_;
};

fs::path FindManifest(const std::string& requested) {
  if (!requested.empty()) return requested;
  const std::vector<fs::path> candidates = {
      "configs/tensorrt_models.conf", "/workspace/configs/tensorrt_models.conf"};
  for (const auto& candidate : candidates) {
    std::error_code error;
    if (fs::is_regular_file(candidate, error)) return candidate;
  }
  return candidates.front();
}

fs::path ResolveEngine(const fs::path& manifest, const std::string& engine) {
  fs::path path(engine);
  if (path.is_absolute()) return path;
  const std::vector<fs::path> candidates = {
      path, manifest.parent_path() / path, manifest.parent_path().parent_path() / path,
      fs::path("/workspace") / path};
  for (const auto& candidate : candidates) {
    std::error_code error;
    if (fs::is_regular_file(candidate, error)) return candidate;
  }
  return path;
}

// IEEE-754 binary32 to binary16 conversion without relying on a compiler
// extension. TensorRT input tensors commonly remain float32 even when the
// engine uses FP16 kernels, but this is needed for engines with half inputs.
uint16_t FloatToHalf(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16U) & 0x8000U;
  const uint32_t exponent = (bits >> 23U) & 0xffU;
  const uint32_t mantissa = bits & 0x7fffffU;
  if (exponent == 0xffU) {
    return static_cast<uint16_t>(sign | (mantissa == 0 ? 0x7c00U : 0x7e00U));
  }
  const int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
  if (half_exponent >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00U);
  if (half_exponent <= 0) {
    if (half_exponent < -10) return static_cast<uint16_t>(sign);
    const uint32_t normalized = mantissa | 0x800000U;
    const uint32_t shifted = normalized >> static_cast<uint32_t>(14 - half_exponent);
    const uint32_t round_bit = (normalized >> static_cast<uint32_t>(13 - half_exponent)) & 1U;
    return static_cast<uint16_t>(sign | shifted + round_bit);
  }
  const uint32_t rounded = mantissa + 0x1000U;
  if (rounded & 0x800000U) {
    return static_cast<uint16_t>(sign | static_cast<uint32_t>(half_exponent + 1) << 10U);
  }
  return static_cast<uint16_t>(sign | static_cast<uint32_t>(half_exponent) << 10U | (rounded >> 13U));
}

bool ImageDimensions(const ModelSpec& spec, int64_t* batch, int64_t* channels, int64_t* height,
                     int64_t* width) {
  if (batch == nullptr || channels == nullptr || height == nullptr || width == nullptr) return false;
  if (spec.shape.size() == 4) {
    *batch = spec.shape[0];
    if (spec.layout == "nhwc") {
      *height = spec.shape[1];
      *width = spec.shape[2];
      *channels = spec.shape[3];
    } else {
      *channels = spec.shape[1];
      *height = spec.shape[2];
      *width = spec.shape[3];
    }
  } else if (spec.shape.size() == 3 && spec.layout == "nchw") {
    *batch = 1;
    *channels = spec.shape[0];
    *height = spec.shape[1];
    *width = spec.shape[2];
  } else {
    return false;
  }
  return *batch > 0 && *channels > 0 && *height > 0 && *width > 0 && *channels <= 3;
}

bool StoreValue(uint8_t* destination, ai::inference::TensorType type, float value) {
  if (destination == nullptr || !std::isfinite(value)) return false;
  switch (type) {
    case ai::inference::TensorType::kUInt8: {
      const float clamped = std::clamp(value, 0.0F, 255.0F);
      *destination = static_cast<uint8_t>(std::lround(clamped));
      return true;
    }
    case ai::inference::TensorType::kFloat16: {
      const uint16_t half = FloatToHalf(value);
      std::memcpy(destination, &half, sizeof(half));
      return true;
    }
    case ai::inference::TensorType::kFloat32:
      std::memcpy(destination, &value, sizeof(value));
      return true;
    case ai::inference::TensorType::kInt32: {
      const int32_t integer = static_cast<int32_t>(std::llround(value));
      std::memcpy(destination, &integer, sizeof(integer));
      return true;
    }
    case ai::inference::TensorType::kInt64: {
      const int64_t integer = static_cast<int64_t>(std::llround(value));
      std::memcpy(destination, &integer, sizeof(integer));
      return true;
    }
  }
  return false;
}

bool MakeImageInput(const ModelSpec& spec, const std::string& image_path,
                    ai::inference::Tensor* tensor, std::string* error) {
  if (tensor == nullptr) return false;
  int64_t batch = 0;
  int64_t channels = 0;
  int64_t height = 0;
  int64_t width = 0;
  if (!ImageDimensions(spec, &batch, &channels, &height, &width)) {
    if (error != nullptr) *error = "image model must have NCHW/NHWC shape with 1 or 3 channels";
    return false;
  }
  cv::Mat image = cv::imread(image_path, channels == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
  if (image.empty()) {
    if (error != nullptr) *error = "cannot read image: " + image_path;
    return false;
  }
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(static_cast<int>(width), static_cast<int>(height)));
  if (channels == 3) cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  resized.convertTo(resized, CV_32F, static_cast<double>(spec.scale));

  tensor->type = spec.type;
  tensor->shape = spec.shape;
  tensor->data.resize(tensor->ExpectedBytes());
  const size_t element_size = ai::inference::TensorTypeSize(spec.type);
  if (element_size == 0 || tensor->data.empty()) return false;
  const size_t plane = static_cast<size_t>(height) * static_cast<size_t>(width);
  const size_t per_batch = static_cast<size_t>(channels) * plane;
  for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (int64_t channel = 0; channel < channels; ++channel) {
      for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
          const float value = channels == 1
                                  ? resized.at<float>(static_cast<int>(y), static_cast<int>(x))
                                  : resized.at<cv::Vec3f>(static_cast<int>(y), static_cast<int>(x))[
                                        static_cast<int>(channel)];
          size_t element = 0;
          if (spec.layout == "nhwc") {
            element = static_cast<size_t>(batch_index) * per_batch +
                      (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) *
                          static_cast<size_t>(channels) + static_cast<size_t>(channel);
          } else {
            element = static_cast<size_t>(batch_index) * per_batch +
                      static_cast<size_t>(channel) * plane + static_cast<size_t>(y) * static_cast<size_t>(width) +
                      static_cast<size_t>(x);
          }
          if (!StoreValue(tensor->data.data() + element * element_size, spec.type, value)) return false;
        }
      }
    }
  }
  return tensor->IsValid();
}

bool LoadTensorInput(const ModelSpec& spec, const std::string& path,
                     ai::inference::Tensor* tensor, std::string* error) {
  if (tensor == nullptr) return false;
  tensor->type = spec.type;
  tensor->shape = spec.shape;
  const size_t expected = tensor->ExpectedBytes();
  if (expected == 0) {
    if (error != nullptr) *error = "invalid tensor shape or type for model " + spec.name;
    return false;
  }
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    if (error != nullptr) *error = "cannot read tensor input: " + path;
    return false;
  }
  const std::streamsize size = stream.tellg();
  if (size != static_cast<std::streamsize>(expected)) {
    if (error != nullptr) {
      *error = "tensor input has " + std::to_string(size) + " bytes; expected " + std::to_string(expected);
    }
    return false;
  }
  stream.seekg(0, std::ios::beg);
  tensor->data.resize(expected);
  if (!stream.read(reinterpret_cast<char*>(tensor->data.data()), size)) {
    if (error != nullptr) *error = "failed to read tensor input: " + path;
    return false;
  }
  return tensor->IsValid();
}

bool ParseJob(const std::string& value, JobArg* job) {
  if (job == nullptr) return false;
  const size_t separator = value.find('=');
  if (separator == std::string::npos || separator == 0 || separator + 1 >= value.size()) return false;
  job->model = value.substr(0, separator);
  job->input = value.substr(separator + 1);
  return true;
}

void PrintUsage() {
  std::cerr << "usage:\n"
            << "  trt_model_runner --list [--manifest FILE]\n"
            << "  trt_model_runner --manifest FILE --model NAME --input FILE [options]\n"
            << "  trt_model_runner --manifest FILE --job NAME=FILE [--job NAME=FILE ...] [options]\n\n"
            << "options: --workers N --requests N --device-id N --engine FILE --help\n"
            << "image jobs are resized and packed from the catalog; tensor jobs read raw bytes.\n";
}

bool ParseArguments(int argc, char** argv, Arguments* arguments) {
  if (arguments == nullptr) return false;
  std::string model;
  std::string input;
  std::string engine_override;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    const auto next = [&]() -> const char* {
      return index + 1 < argc ? argv[++index] : nullptr;
    };
    if (option == "--help" || option == "-h") {
      PrintUsage();
      return false;
    }
    if (option == "--list") {
      arguments->list = true;
    } else if (option == "--manifest") {
      const char* value = next();
      if (value == nullptr) return false;
      arguments->manifest = value;
    } else if (option == "--model") {
      const char* value = next();
      if (value == nullptr) return false;
      model = value;
    } else if (option == "--input") {
      const char* value = next();
      if (value == nullptr) return false;
      input = value;
    } else if (option == "--job") {
      const char* value = next();
      JobArg job;
      if (value == nullptr || !ParseJob(value, &job)) return false;
      arguments->jobs.emplace_back(std::move(job));
    } else if (option == "--engine") {
      const char* value = next();
      if (value == nullptr) return false;
      engine_override = value;
    } else if (option == "--workers") {
      const char* value = next();
      if (value == nullptr || !ParsePositive(value, &arguments->workers)) return false;
    } else if (option == "--requests") {
      const char* value = next();
      if (value == nullptr || !ParsePositive(value, &arguments->requests)) return false;
    } else if (option == "--device-id") {
      const char* value = next();
      if (value == nullptr || !ParseNonNegative(value, &arguments->device_id)) return false;
    } else {
      return false;
    }
  }
  if (!model.empty() || !input.empty()) {
    if (model.empty() || input.empty()) return false;
    arguments->jobs.push_back(JobArg{model, input, engine_override});
  } else if (!engine_override.empty()) {
    return false;
  }
  if (!engine_override.empty() && arguments->jobs.size() != 1) return false;
  return arguments->list || !arguments->jobs.empty();
}

struct PendingResult {
  std::string model;
  ai::inference::InferenceScheduler::Future future;
};

}  // namespace

int main(int argc, char** argv) {
  Arguments arguments;
  if (!ParseArguments(argc, argv, &arguments)) {
    if (argc <= 1) PrintUsage();
    return 2;
  }

  const fs::path manifest = FindManifest(arguments.manifest);
  ModelCatalog catalog;
  std::string error;
  if (!catalog.Load(manifest, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  if (arguments.list) {
    for (const auto& model : catalog.Models()) {
      const fs::path engine = ResolveEngine(manifest, model.engine);
      std::error_code filesystem_error;
      const bool present = fs::is_regular_file(engine, filesystem_error);
      std::cout << (present ? "ready" : "missing") << " model=" << model.name << " task=" << model.task
                << " input=" << model.input_kind << " shape=" << ShapeString(model.shape)
                << " type=" << TensorTypeName(model.type) << " engine=" << engine.string() << '\n';
    }
    if (arguments.jobs.empty()) return 0;
  }

  ai::inference::InferenceScheduler::Options options;
  options.workers = arguments.workers;
  options.backend.kind = ai::inference::BackendKind::kTensorRT;
  options.backend.device_id = arguments.device_id;
  ai::inference::InferenceScheduler scheduler(options);
  std::vector<PendingResult> pending;
  size_t submitted = 0;
  bool preparation_failed = false;
  for (const auto& job : arguments.jobs) {
    const ModelSpec* spec = catalog.Find(job.model);
    if (spec == nullptr) {
      std::cerr << "unknown model: " << job.model << '\n';
      preparation_failed = true;
      continue;
    }
    ai::inference::Tensor input_tensor;
    if (spec->input_kind == "image") {
      if (!MakeImageInput(*spec, job.input, &input_tensor, &error)) {
        std::cerr << "model=" << spec->name << " " << error << '\n';
        preparation_failed = true;
        continue;
      }
    } else if (!LoadTensorInput(*spec, job.input, &input_tensor, &error)) {
      std::cerr << "model=" << spec->name << " " << error << '\n';
      preparation_failed = true;
      continue;
    }
    ai::inference::BackendConfig backend = options.backend;
    backend.model_path = job.engine_override.empty() ? ResolveEngine(manifest, spec->engine).string()
                                                      : job.engine_override;
    for (size_t request_index = 0; request_index < arguments.requests; ++request_index) {
      ai::inference::InferenceScheduler::Request request;
      request.priority = static_cast<int64_t>(arguments.requests - request_index);
      request.backend = backend;
      request.inputs.push_back(input_tensor);
      pending.push_back(PendingResult{spec->name, scheduler.Submit(std::move(request))});
      ++submitted;
    }
  }

  size_t successful = 0;
  for (auto& item : pending) {
    const auto result = item.future.get();
    std::cout << "model=" << item.model << " request=" << result.id << " backend="
              << ai::inference::BackendKindName(result.backend) << " status="
              << ai::inference::BackendStatusName(result.status) << " detail=" << result.detail;
    if (!result.outputs.empty()) {
      std::cout << " outputs=" << result.outputs.size();
      for (const auto& output : result.outputs) {
        std::cout << " " << ShapeString(output.shape) << "/" << output.data.size() << "B";
      }
    }
    std::cout << '\n';
    if (result.status == ai::inference::BackendStatus::kSuccess) ++successful;
  }
  scheduler.Shutdown(true);
  std::cout << "trt_model_runner workers=" << arguments.workers << " requests=" << submitted
            << " successful=" << successful << '\n';
  return !preparation_failed && submitted > 0 && successful == submitted ? 0 : 1;
}
