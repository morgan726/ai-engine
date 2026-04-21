cmake_minimum_required(VERSION 3.10)

project(ai_priority_scheduler)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV (optional but recommended)
find_package(OpenCV QUIET COMPONENTS core imgproc highgui dnn)
if(OpenCV_FOUND)
  add_definitions(-DHAVE_OPENCV)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# glog
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/FindGlog.cmake)
include_directories(${GLOG_INCLUDE_DIRS})

file(GLOB scheduler_srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/strategies/*.cpp
)

add_library(ai_priority STATIC ${scheduler_srcs})
target_include_directories(ai_priority PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(ai_priority
  ${GLOG_LIBRARIES}
  ${OpenCV_LIBS}
)

target_compile_options(ai_priority PRIVATE "-Wno-deprecated-declarations")

# Ensure scheduler sees utils/core sources when included as headers by examples
# Linkage of utils/core comes transitively from examples; keep library minimal here
