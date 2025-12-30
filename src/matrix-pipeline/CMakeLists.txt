set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Boost REQUIRED COMPONENTS stacktrace_backtrace)
find_package(CUDAToolkit)
find_package(protobuf CONFIG REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
find_package(cxxopts REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Protobuf REQUIRED)
find_package(cpr CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(Drogon CONFIG REQUIRED)
find_package(Microsoft.GSL CONFIG REQUIRED)
# --- Find TensorRT ---
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS $ENV{TRT_HOME}/include /usr/include /usr/local/cuda/include)
find_library(NVINFER_LIB nvinfer
        HINTS $ENV{TRT_HOME}/lib /usr/lib64 /usr/local/cuda/lib64)
find_library(NVONNX_LIB nvonnxparser
        HINTS $ENV{TRT_HOME}/lib /usr/lib64 /usr/local/cuda/lib64)

if (TENSORRT_INCLUDE_DIR AND NVINFER_LIB AND NVONNX_LIB)
    message(STATUS "Found TensorRT. TRT_HOME: $ENV{TRT_HOME}, NVINFER_LIB: ${NVINFER_LIB}, CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "TensorRT NOT found. $TRT_HOME: $ENV{TRT_HOME}")
endif ()

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/frame_msg.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})


add_subdirectory(utils)
add_subdirectory(synchronous_processing_units)
add_subdirectory(asynchronous_processing_units)

add_library(utils
        utils.cpp utils.h
)
add_library(video_feed_manager
        video_feed_manager.cpp video_feed_manager.h
)
target_link_libraries(video_feed_manager
        PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json
        PRIVATE utils asynchronous_processing_unit)
target_link_libraries(video_feed_manager
        PRIVATE
        ${OpenCV_LIBS} rt ssl crypto fmt::fmt
        Drogon::Drogon
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        CUDA::nvjpeg
        cppzmq cppzmq-static
        protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite
        Microsoft.GSL::GSL
)

add_executable(mp
        main.cpp
        global_vars.cpp
        ${PROTO_SRCS} ${PROTO_HDRS}
)
target_link_libraries(mp
        PUBLIC
        Boost::stacktrace_backtrace
        PRIVATE
        utils
        video_feed_manager
        spdlog::spdlog
)

add_custom_command(
        TARGET mp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mp> ../../
)

# So that Python bindings will always be generated, even if it is not used
add_custom_target(GeneratePythonBindings DEPENDS ${PROTO_PY})
add_dependencies(mp GeneratePythonBindings)
