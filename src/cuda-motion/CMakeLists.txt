set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#include_directories(http_service/)

find_package(OpenCV REQUIRED)
#find_package(OpenCV REQUIRED PATHS "$ENV{HOME}/opt/opencv-4.12.0/lib64/cmake/opencv4" NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(protobuf CONFIG REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
find_package(cxxopts REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Protobuf REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(Drogon CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/frame_msg.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})

add_executable(cm
        main.cpp
        device_manager.cpp
        global_vars.cpp
        utils.cpp
        ${PROTO_SRCS} ${PROTO_HDRS}
        interfaces/i_synchronous_processing_unit.h
        synchronous_processing_units/rotate_frame.h
        entities/synchronous_processing_result.h
        entities/processing_metadata.h
        synchronous_processing_units/calculate_change_rate.h
        synchronous_processing_units/control_fps.h
        entities/processing_units_variant.h
        interfaces/i_asynchronous_processing_unit.h
        asynchronous_processing_units/video_writer.h
        asynchronous_processing_units/http_service.h
        synchronous_processing_units/resize_frame.h
        asynchronous_processing_units/zeromq_publisher.h
)

target_link_libraries(cm
        ${OpenCV_LIBS} rt ssl crypto fmt::fmt
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
        CUDA::nvjpeg
        Drogon::Drogon
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        cppzmq cppzmq-static
        protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite
)

add_custom_command(
        TARGET cm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cm> ../../
)

# So that Python bindings will always be generated, even if it is not used
add_custom_target(GeneratePythonBindings DEPENDS ${PROTO_PY})
add_dependencies(cm GeneratePythonBindings)
