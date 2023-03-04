set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(/usr/local/include/opencv4)

if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()

add_executable(cs main.cpp)
add_library(deviceManager deviceManager.cpp)
add_library(logger logger.cpp)

target_link_libraries(cs deviceManager logger deviceManager opencv_highgui opencv_imgcodecs opencv_video opencv_videoio opencv_imgproc opencv_core pthread spdlog::spdlog)

add_custom_command(
  TARGET cs POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cs> ../../
)
