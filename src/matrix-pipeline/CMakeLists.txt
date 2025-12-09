set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV found at: ${OpenCV_DIR}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

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

add_library(rotate_frame
        synchronous_processing_units/rotate_frame.cpp
        interfaces/i_synchronous_processing_unit.h
)

add_library(resize_frame
        synchronous_processing_units/resize_frame.cpp
        interfaces/i_synchronous_processing_unit.h
)

add_library(debug_output
        synchronous_processing_units/debug_oputput.cpp
        interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(debug_output spdlog::spdlog)

add_library(matrix_notifier
        asynchronous_processing_units/matrix_notifier.cpp
        interfaces/i_asynchronous_processing_unit.h
)
target_link_libraries(matrix_notifier nvjpeg_encoder ram_video_buffer matrix_sender)
target_link_libraries(matrix_notifier ${OpenCV_LIBS} CUDA::nvjpeg spdlog::spdlog cpr::cpr nlohmann_json::nlohmann_json)

add_library(detect_objects
        synchronous_processing_units/detect_objects.cpp
        interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(detect_objects spdlog::spdlog)

add_library(overlay_bounding_boxes
        synchronous_processing_units/overlay_bounding_boxes.cpp
        interfaces/i_synchronous_processing_unit.h
)

add_library(asynchronous_processing_unit
        asynchronous_processing_units/asynchronous_processing_unit.cpp
        interfaces/i_asynchronous_processing_unit.h
)
target_link_libraries(asynchronous_processing_unit CUDA::nvjpeg)

add_executable(mp
        main.cpp
        device_manager.cpp
        global_vars.cpp
        utils.cpp
        ${PROTO_SRCS} ${PROTO_HDRS}
)

target_link_libraries(mp
        PRIVATE
        matrix_notifier
        debug_output
        rotate_frame
        resize_frame
        detect_objects
        overlay_bounding_boxes
        asynchronous_processing_unit
)
target_link_libraries(mp
        PRIVATE
        ${OpenCV_LIBS} rt ssl crypto fmt::fmt
        Drogon::Drogon
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        CUDA::nvjpeg
        cppzmq cppzmq-static
        protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite
)

add_custom_command(
        TARGET mp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mp> ../../
)

# So that Python bindings will always be generated, even if it is not used
add_custom_target(GeneratePythonBindings DEPENDS ${PROTO_PY})
add_dependencies(mp GeneratePythonBindings)
