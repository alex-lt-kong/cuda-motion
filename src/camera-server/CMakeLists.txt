add_subdirectory(http_service/)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(http_service/)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(cppzmq)
find_package(spdlog REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(cs main.cpp)
add_library(device_manager device_manager.cpp)
add_library(frame_handler frame_handler.cpp)
add_library(global_vars global_vars.cpp)
add_library(ipc ipc.cpp)
add_library(utils utils.cpp)


target_link_libraries(cs
  http_service global_vars device_manager ipc frame_handler utils
  ${OpenCV_LIBS} rt ssl crypto spdlog fmt zmq
)

add_custom_command(
  TARGET cs POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cs> ../../
)
