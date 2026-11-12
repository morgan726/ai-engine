#include "inference/backend.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  ai::inference::BackendKind requested = ai::inference::BackendKind::kAuto;
  if (argc > 1 && !ai::inference::ParseBackendKind(argv[1], &requested)) {
    std::cerr << "unknown backend: " << argv[1] << "\n";
    return 2;
  }

  const auto selected = ai::inference::BackendRegistry::Instance().Select(requested);
  std::cout << "requested=" << ai::inference::BackendKindName(requested)
            << " selected=" << ai::inference::BackendKindName(selected) << "\n";
  for (const auto& item : ai::inference::BackendRegistry::Instance().Discover()) {
    std::cout << ai::inference::BackendKindName(item.kind)
              << " runtime=" << (item.runtime_available ? "yes" : "no")
              << " factory=" << (item.factory_registered ? "yes" : "no")
              << " detail=" << item.detail << "\n";
  }

  std::string error;
  if (!ai::inference::RunBackendSmokeTest(&error)) {
    std::cerr << "backend smoke test failed: " << error << "\n";
    return 1;
  }
  return 0;
}
