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

if(OpenCV_FOUND AND GLOG_FOUND AND TARGET ai_inference_backend)
  add_executable(trt_scheduler_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/trt_scheduler_demo.cpp)
  target_include_directories(trt_scheduler_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
  target_link_libraries(trt_scheduler_demo PRIVATE
    ai_inference_backend ai_runtime ${OpenCV_LIBS} ${GLOG_LIBRARIES}
    ${CMAKE_DL_LIBS})
  target_compile_features(trt_scheduler_demo PRIVATE cxx_std_17)
  install(TARGETS trt_scheduler_demo RUNTIME DESTINATION bin)

  add_executable(trt_model_runner
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/trt_model_runner.cpp)
  target_include_directories(trt_model_runner PRIVATE
    ${OpenCV_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
  target_link_libraries(trt_model_runner PRIVATE
    ai_inference_backend ai_runtime ${OpenCV_LIBS} ${GLOG_LIBRARIES}
    ${CMAKE_DL_LIBS})
  target_compile_features(trt_model_runner PRIVATE cxx_std_17)
  install(TARGETS trt_model_runner RUNTIME DESTINATION bin)
else()
  message(STATUS "trt_scheduler_demo disabled: OpenCV, glog and inference backend are required")
endif()
