set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(/usr/local/include/opencv4)

find_package(OpenCV REQUIRED)
find_package(Crow REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(-DCROW_ENABLE_SSL)
add_executable(cs main.cpp)
add_library(deviceManager deviceManager.cpp)
add_library(utils utils.cpp)


target_link_libraries(cs deviceManager utils ${OpenCV_LIBS} pthread rt ssl crypto spdlog fmt)

add_custom_command(
  TARGET cs POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cs> ../../
)
