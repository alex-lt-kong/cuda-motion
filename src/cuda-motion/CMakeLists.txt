set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#include_directories(http_service/)

find_package(OpenCV REQUIRED)
#find_package(OpenCV REQUIRED PATHS "$ENV{HOME}/opt/opencv-4.12.0/lib64/cmake/opencv4" NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV found at: ${OpenCV_DIR}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

find_package(protobuf CONFIG REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
find_package(cxxopts REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Protobuf REQUIRED)
find_package(cpr CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(Drogon CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/frame_msg.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})

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
target_link_libraries(matrix_notifier spdlog::spdlog CUDA::nvjpeg cpr::cpr nlohmann_json::nlohmann_json)

add_library(detect_objects
        synchronous_processing_units/detect_objects.cpp
        interfaces/i_synchronous_processing_unit.h
)
add_library(overlay_bounding_boxes
        synchronous_processing_units/overlay_bounding_boxes.cpp
        interfaces/i_synchronous_processing_unit.h
)

add_library(asynchronous_processing_unit
        asynchronous_processing_units/asynchronous_processing_unit.cpp
        interfaces/i_asynchronous_processing_unit.h
)
target_link_libraries(asynchronous_processing_unit CUDA::nvjpeg)

add_executable(cm
        main.cpp
        device_manager.cpp
        global_vars.cpp
        utils.cpp
        ${PROTO_SRCS} ${PROTO_HDRS}
)

target_link_libraries(cm
        PRIVATE
        matrix_notifier
        debug_output
        rotate_frame
        resize_frame
        detect_objects
        overlay_bounding_boxes
        asynchronous_processing_unit
)
target_link_libraries(cm
        PRIVATE
        ${OpenCV_LIBS} rt ssl crypto fmt::fmt
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
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
