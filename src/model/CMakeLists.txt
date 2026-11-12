# Keep the model selection in one place.  AI_* names are the public options;
# ADK_* aliases preserve compatibility with older out-of-tree build scripts.
if(DEFINED AI_BUILD_MSSD)
  set(_ai_build_mssd ${AI_BUILD_MSSD})
elseif(DEFINED ADK_BUILD_MSSD)
  set(_ai_build_mssd ${ADK_BUILD_MSSD})
else()
  set(_ai_build_mssd ON)
endif()

if(DEFINED AI_ENABLE_TENSORRT)
  set(_ai_enable_tensorrt ${AI_ENABLE_TENSORRT})
elseif(DEFINED ADK_BUILD_TRT_YOLO)
  set(_ai_enable_tensorrt ${ADK_BUILD_TRT_YOLO})
else()
  set(_ai_enable_tensorrt ON)
endif()

if(_ai_build_mssd)
  add_subdirectory(mssd)
endif()

if(_ai_enable_tensorrt)
  add_subdirectory(trt_yolo)
endif()
