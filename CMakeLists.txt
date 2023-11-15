cmake_minimum_required(VERSION 3.16.3)

project(CAMERA_SERVER)
add_definitions(-DPROJECT_NAME="${PROJECT_NAME}")

set (CMAKE_CXX_STANDARD 20)

# Always generate compile_commands.json for clangd, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


include(cmake/compiler-flags.cmake)
include(cmake/sanitizers.cmake)


add_subdirectory(src/camera-server)
