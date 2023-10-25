set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#include_directories(/usr/local/include/opencv4)


find_package(OpenCV REQUIRED)
find_package(Crow REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(-DCROW_ENABLE_SSL)
add_executable(cs main.cpp)
add_library(device_manager device_manager.cpp)
add_library(frame_handler frame_handler.cpp)
add_library(global_vars global_vars.cpp)
add_library(utils utils.cpp)


target_link_libraries(cs
  global_vars device_manager frame_handler utils
  ${OpenCV_LIBS} rt ssl crypto spdlog fmt zmq
)

add_custom_command(
  TARGET cs POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cs> ../../
)
