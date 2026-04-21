cmake_minimum_required(VERSION 3.10)
set(ADK_VERSION_MAJOR 0)
set(ADK_VERSION_MINOR 1)
set(ADK_VERSION_PATCH 0)
set(ADK_VERSION "${ADK_VERSION_MAJOR}.${ADK_VERSION_MINOR}.${ADK_VERSION_PATCH}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/adk_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/adk_config.h)
install(FILES adk_config.h DESTINATION include)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin/)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(COMPILE_FLAGS -Wno-uninitialized)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(COMPILE_FLAGS -Wno-maybe-uninitialized)
else()
  message(FATAL_ERROR "Unsupported Compiler!")
endif()

add_subdirectory(src/model)

# Priority scheduler (producer-consumer, strategy/command, observer, singleton)
add_subdirectory(src/scheduler)

include_directories("${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/include")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sanitizers)
find_package(Sanitizers)

add_subdirectory(3rdparty/googletest)
add_subdirectory(unitest)

add_subdirectory(3rdparty/libyuv)
add_subdirectory(examples)
