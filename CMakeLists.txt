cmake_minimum_required(VERSION 3.20.3)

project(cuda-motion)
add_definitions(-DPROJECT_NAME="${PROJECT_NAME}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Always generate compile_commands.json for clangd, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/compiler-flags.cmake)
include(cmake/sanitizers.cmake)
include(cmake/git.cmake)


add_subdirectory(src/cuda-motion)
