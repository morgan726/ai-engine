cmake_minimum_required(VERSION 3.18)
project(ai_engine VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

option(ADK_BUILD_MSSD "Build the OpenCV MobileNet-SSD example" ON)
option(ADK_BUILD_TRT_YOLO "Build the CUDA/TensorRT YOLO backend" OFF)
option(ADK_BUILD_EXAMPLES "Build command-line examples" OFF)
option(ADK_BUILD_TESTS "Build legacy codec tests" OFF)
option(AI_ENABLE_OPENVINO "Enable OpenVINO backend probing" ON)
option(AI_ENABLE_TENSORRT "Alias for ADK_BUILD_TRT_YOLO" ${ADK_BUILD_TRT_YOLO})
option(AI_BUILD_MSSD "Alias for ADK_BUILD_MSSD" ${ADK_BUILD_MSSD})
option(AI_BUILD_TENSORRT_BACKEND "Build the TensorRT plan backend adapter" ${ADK_BUILD_TRT_YOLO})
if(AI_ENABLE_TENSORRT)
  set(ADK_BUILD_TRT_YOLO ON CACHE BOOL "" FORCE)
endif()
if(NOT AI_BUILD_MSSD)
  set(ADK_BUILD_MSSD OFF CACHE BOOL "" FORCE)
endif()
set(ADK_CUDA_ARCHITECTURES "89" CACHE STRING
    "CUDA architectures to compile (4060 Ti is 89; override for older toolkits)")

if(ADK_BUILD_TRT_YOLO)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "ADK_BUILD_TRT_YOLO=ON requires a CUDA compiler")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES "${ADK_CUDA_ARCHITECTURES}" CACHE STRING
      "CUDA architectures used by ADK targets" FORCE)
endif()

set(ADK_VERSION_MAJOR 0)
set(ADK_VERSION_MINOR 2)
set(ADK_VERSION_PATCH 0)
set(ADK_VERSION "${ADK_VERSION_MAJOR}.${ADK_VERSION_MINOR}.${ADK_VERSION_PATCH}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/adk_config.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/include/adk_config.h)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake
                      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sanitizers)
find_package(Sanitizers)
find_package(Threads REQUIRED)
find_package(Glog QUIET)

add_library(ai_runtime STATIC
  src/utils/thread_pool.cpp
  src/utils/timer.cpp
)
target_include_directories(ai_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(ai_runtime PUBLIC Threads::Threads)
if(GLOG_FOUND)
  target_include_directories(ai_runtime PUBLIC ${GLOG_INCLUDE_DIRS})
  target_link_libraries(ai_runtime PUBLIC ${GLOG_LIBRARIES})
else()
  message(STATUS "glog development files not found; scheduler runtime is disabled")
  set_target_properties(ai_runtime PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

add_subdirectory(src/inference)
add_subdirectory(src/model)
if(GLOG_FOUND)
  add_subdirectory(src/scheduler)
else()
  message(STATUS "Skipping scheduler: install libgoogle-glog-dev to build it")
endif()

if(ADK_BUILD_TESTS)
  add_subdirectory(3rdparty/googletest)
  add_subdirectory(unitest)
endif()

add_subdirectory(3rdparty/libyuv)
if(ADK_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/adk_config.h DESTINATION include)
