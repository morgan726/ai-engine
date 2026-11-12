cmake_minimum_required(VERSION 3.10)
find_package(OpenCV REQUIRED)
set(SOURCE_FILES SSDModel.cpp)
add_executable(mssd ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(mssd PRIVATE ${OpenCV_LIBS})

# Docker images install glog and use the project's priority thread pool for
# concurrent MSSD requests.  Keep the single-request CPU fallback available
# when building on a minimal host without glog.
if(GLOG_FOUND AND TARGET ai_runtime)
  target_compile_definitions(mssd PRIVATE AI_MSSD_USE_THREAD_POOL=1)
  target_link_libraries(mssd PRIVATE ai_runtime)
endif()
