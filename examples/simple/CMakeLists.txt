find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs highgui videoio)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/FindGFlags.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/FindGlog.cmake)

if(OpenCV_FOUND AND GFLAGS_FOUND AND GLOG_FOUND AND TARGET ai_priority)
  add_executable(priority_demo priority_demo.cpp)
  target_include_directories(priority_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS} ${GFLAGS_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
  target_link_libraries(priority_demo PRIVATE
    ai_priority ai_runtime ${OpenCV_LIBS} ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES}
    ${CMAKE_DL_LIBS})
  install(TARGETS priority_demo RUNTIME DESTINATION bin)
else()
  message(STATUS "priority_demo disabled: OpenCV, gflags and glog development files are required")
endif()
