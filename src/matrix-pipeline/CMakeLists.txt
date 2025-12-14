set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
add_library(device_manager
        device_manager.cpp device_manager.h
)
target_link_libraries(device_manager
        PRIVATE
        utils
        matrix_notifier
        debug_output
        rotate_frame
        rtsp_producer
        resize_frame
        detect_objects
        prune_object_detection_results
        collect_stats
        overlay_info
        overlay_bounding_boxes
        measure_latency
        asynchronous_processing_unit)
target_link_libraries(device_manager
        PRIVATE
        ${OpenCV_LIBS} rt ssl crypto fmt::fmt
        Drogon::Drogon
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        CUDA::nvjpeg
        cppzmq cppzmq-static
        protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite
)

add_executable(mp
        main.cpp
        global_vars.cpp
        ${PROTO_SRCS} ${PROTO_HDRS}
)
target_link_libraries(mp
        PRIVATE
        utils
        device_manager
        spdlog::spdlog
)

add_custom_command(
        TARGET mp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mp> ../../
)

# So that Python bindings will always be generated, even if it is not used
add_custom_target(GeneratePythonBindings DEPENDS ${PROTO_PY})
add_dependencies(mp GeneratePythonBindings)
