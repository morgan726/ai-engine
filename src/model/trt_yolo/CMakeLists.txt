set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wno-write-strings")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB2 REQUIRED glib-2.0)
include_directories(${GLIB2_INCLUDE_DIRS})
link_directories(${GLIB2_LIBRARY_DIRS})

find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include
)
find_library(TENSORRT_LIBRARY NAMES nvinfer REQUIRED)
find_library(TENSORRT_PLUGIN_LIBRARY NAMES nvinfer_plugin REQUIRED)
# nvparsers was removed from newer TensorRT packages.  It is not used by this
# backend, but link it when an older TensorRT installation provides it.
find_library(TENSORRT_PARSERS_LIBRARY NAMES nvparsers nvonnxparser)

if(NOT TENSORRT_INCLUDE_DIR)
  message(FATAL_ERROR "TensorRT headers were not found (NvInfer.h)")
endif()

file(GLOB_RECURSE sources ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
add_library(engine SHARED ${sources})
target_include_directories(engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/
  ${CMAKE_CURRENT_SOURCE_DIR}/
  ${OpenCV_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(engine PRIVATE
  ${TENSORRT_LIBRARY}
  ${TENSORRT_PLUGIN_LIBRARY}
  ${OpenCV_LIBS}
  ${GLIB2_LIBRARIES}
  CUDA::cudart
  "stdc++fs"
)
if(TENSORRT_PARSERS_LIBRARY)
  target_link_libraries(engine PRIVATE ${TENSORRT_PARSERS_LIBRARY})
endif()

add_executable(trt_yolo ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_include_directories(trt_yolo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/
  ${OpenCV_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(trt_yolo PRIVATE engine ${OpenCV_LIBS} ${GLIB2_LIBRARIES})
