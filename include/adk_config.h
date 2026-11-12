#ifndef ADK_CONFIG_H_
#define ADK_CONFIG_H_

#include <string>

#define ADK_VERSION_MAJOR 0
#define ADK_VERSION_MINOR 2
#define ADK_VERSION_PATCH 0

#define ADK_GET_VERSION(major, minor, patch) (((major) << 20) | ((minor) << 10) | (patch))
#define ADK_VERSION ADK_GET_VERSION(ADK_VERSION_MAJOR, ADK_VERSION_MINOR, ADK_VERSION_PATCH)

namespace ai {

/**
 * @brief Get ADK version string
 *
 * @return std::string version string
 */
inline static std::string Version() {
  // clang-format off
  return std::to_string(ADK_VERSION_MAJOR) + "." +
         std::to_string(ADK_VERSION_MINOR) + "." +
         std::to_string(ADK_VERSION_PATCH);
  // clang-format on
}

}  // namespace ai

#endif  // ADK_CONFIG_H_
