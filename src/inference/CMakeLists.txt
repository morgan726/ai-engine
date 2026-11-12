add_library(ai_inference_backend STATIC backend.cpp)
target_include_directories(ai_inference_backend PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_link_libraries(ai_inference_backend PUBLIC ${CMAKE_DL_LIBS})

# InferenceScheduler uses the project's PriorityThreadPool.  Keep the
# backend registry usable on a minimal CPU host where glog is not installed,
# while enabling the actual scheduler in the Docker images (which install
# glog and build ai_runtime).
if(GLOG_FOUND)
  target_sources(ai_inference_backend PRIVATE scheduler.cpp)
  target_link_libraries(ai_inference_backend PUBLIC ai_runtime Threads::Threads)
  target_compile_definitions(ai_inference_backend PUBLIC AI_ENGINE_WITH_THREAD_POOL=1)
else()
  message(STATUS "InferenceScheduler disabled: glog development files are required")
endif()

# OpenVINO is optional.  Only add the adapter when the package exports a
# runtime target; this keeps CPU-only and NVIDIA-only builds free of vendor
# headers and link dependencies.
set(_ai_openvino_target "")
if(AI_ENABLE_OPENVINO)
  find_package(OpenVINO CONFIG QUIET)
  foreach(_candidate openvino::runtime OpenVINO::runtime openvino::runtime::runtime)
    if(TARGET ${_candidate})
      set(_ai_openvino_target ${_candidate})
      break()
    endif()
  endforeach()
endif()

if(_ai_openvino_target)
  target_sources(ai_inference_backend PRIVATE openvino_backend.cpp)
  target_compile_definitions(ai_inference_backend PRIVATE AI_ENGINE_WITH_OPENVINO=1)
  target_link_libraries(ai_inference_backend PRIVATE ${_ai_openvino_target})
  message(STATUS "OpenVINO backend: enabled (${_ai_openvino_target})")
else()
  message(STATUS "OpenVINO backend: disabled (runtime CMake target not found)")
endif()

if(AI_BUILD_TENSORRT_BACKEND)
  find_package(CUDAToolkit QUIET)
  find_path(_ai_tensorrt_include NAMES NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include)
  find_library(_ai_tensorrt_library NAMES nvinfer)
  if(CUDAToolkit_FOUND AND _ai_tensorrt_include AND _ai_tensorrt_library)
    target_sources(ai_inference_backend PRIVATE tensorrt_backend.cpp)
    target_compile_definitions(ai_inference_backend PRIVATE AI_ENGINE_WITH_TENSORRT=1)
    target_include_directories(ai_inference_backend PRIVATE ${_ai_tensorrt_include})
    target_link_libraries(ai_inference_backend PRIVATE ${_ai_tensorrt_library} CUDA::cudart)
    message(STATUS "TensorRT backend: enabled")
  else()
    message(STATUS "TensorRT backend: disabled (CUDA toolkit or TensorRT development files not found)")
  endif()
endif()

add_executable(backend_probe ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/backend_probe.cpp)
target_link_libraries(backend_probe PRIVATE ai_inference_backend)
target_compile_features(backend_probe PRIVATE cxx_std_17)
install(TARGETS backend_probe RUNTIME DESTINATION bin)

if(GLOG_FOUND)
  add_executable(scheduler_probe ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scheduler_probe.cpp)
  target_link_libraries(scheduler_probe PRIVATE ai_inference_backend)
  target_compile_features(scheduler_probe PRIVATE cxx_std_17)
  install(TARGETS scheduler_probe RUNTIME DESTINATION bin)
endif()
