# Set the root directory for the ADK project
set(ADK_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# ---[ Google-gflags
find_package(GFlags REQUIRED)
list(APPEND Examples_INCLUDE_DIRS ${GFLAGS_INCLUDE_DIRS})
list(APPEND Examples_LINK_LIBS ${GFLAGS_LIBRARIES})

# ---[ OpenCV
find_package(OpenCV REQUIRED)
list(APPEND Examples_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND Examples_LINK_LIBS ${OpenCV_LIBS})

# ---[ FFmpeg
find_package(FFmpeg REQUIRED)
if (FFMPEG_FOUND)
  list(APPEND Examples_INCLUDE_DIRS ${FFMPEG_INCLUDE_DIR})
  list(APPEND Examples_LINK_LIBS ${FFMPEG_LIBRARIES})
  set(HAVE_FFMPEG true)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_FFMPEG")
  if (WITH_FFMPEG_AVDEVICE)
    add_definitions(-DHAVE_FFMPEG_AVDEVICE)
  endif ()
else ()
  message(FATAL_ERROR "ffmpeg not found!")
endif ()

# ---[ LibYuv
list(APPEND Examples_INCLUDE_DIRS ${ADK_ROOT_DIR}/3rdparty/libyuv/include)
list(APPEND Examples_LINK_LIBS yuv)

# ---[ Threads and other system libraries
list(APPEND Examples_LINK_LIBS pthread dl)

option(BUILD_DETECTION "Build detection example that pulls Cambricon-specific pieces" OFF)
message(STATUS "@@@@@@@@@@@ Target : detection (BUILD_DETECTION=${BUILD_DETECTION})")

# ---[ Detection executable
if(BUILD_DETECTION)
  file(GLOB detection_srcs "${CMAKE_CURRENT_SOURCE_DIR}/detection.cpp")
  file(GLOB_RECURSE detection_srcs_sub "${CMAKE_CURRENT_SOURCE_DIR}/../../src/**/*.cpp")
  list(APPEND detection_srcs ${detection_srcs_sub})
  add_executable(detection ${detection_srcs})
  add_sanitizers(detection)
  target_compile_options(detection PRIVATE ${COMPILE_FLAGS})
  target_include_directories(detection PRIVATE ${Examples_INCLUDE_DIRS})
  target_link_libraries(detection ${Examples_LINK_LIBS})
  install(TARGETS detection RUNTIME DESTINATION bin)
endif()

message(STATUS "@@@@@@@@@@@ Target : priority_demo")
file(GLOB priority_demo_srcs "${CMAKE_CURRENT_SOURCE_DIR}/priority_demo.cpp")
file(GLOB_RECURSE priority_demo_srcs_sub "${CMAKE_CURRENT_SOURCE_DIR}/../../src/**/*.cpp")
list(APPEND priority_demo_srcs ${priority_demo_srcs_sub})
add_executable(priority_demo ${priority_demo_srcs})
add_sanitizers(priority_demo)
target_compile_options(priority_demo PRIVATE ${COMPILE_FLAGS})
target_include_directories(priority_demo PRIVATE ${Examples_INCLUDE_DIRS})
target_link_libraries(priority_demo ${Examples_LINK_LIBS} ai_priority)
install(TARGETS priority_demo RUNTIME DESTINATION bin)
