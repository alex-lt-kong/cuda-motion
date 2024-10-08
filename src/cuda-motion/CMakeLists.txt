set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(http_service/)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# find_library() has better compatibility when finding ZeroMQ (cppzmq to be specific)...
# find_package(ZeroMQ    4.3.1      REQUIRED)
# find_package(libzmq   4.3.1      REQUIRED)
find_library(ZMQ_LIB zmq)

if(NOT ZMQ_LIB)
  message(
    FATAL_ERROR
    "libzmp library not found, install it with 'apt install libzmq3-dev'"
  )
endif()

find_package(spdlog REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Drogon CONFIG REQUIRED)

if(Protobuf_FOUND)
  message(STATUS "Protobuf version: ${Protobuf_VERSION}")
  message(STATUS "Protobuf libraries: ${Protobuf_LIBRARIES}")
  message(STATUS "Protobuf lite libraries: ${Protobuf_LITE_LIBRARIES}")
  message(STATUS "Protobuf protoc: ${Protobuf_PROTOC_EXECUTABLE}")
else()
  message(WARNING "Protobuf package not found")
endif()

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/snapshot.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})

add_executable(cm
  main.cpp
  device_manager.cpp
  pc_queue_vw.cpp
  ipc.cpp
  pc_queue_ipc.cpp
  frame_handler.cpp
  global_vars.cpp
  ipc.cpp
  utils.cpp
  ${PROTO_SRCS} ${PROTO_HDRS}
)

# add_subdirectory(http_service/)

target_link_libraries(cm
  ${OpenCV_LIBS} rt ssl crypto spdlog fmt zmq
  ${Protobuf_LIBRARIES}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  Drogon::Drogon
)

add_custom_command(
  TARGET cm POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cm> ../../
)

# So that Python bindings will always be generated, even if it is not used
add_custom_target(GeneratePythonBindings DEPENDS ${PROTO_PY})
add_dependencies(cm GeneratePythonBindings)
