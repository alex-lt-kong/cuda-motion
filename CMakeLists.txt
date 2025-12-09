cmake_minimum_required(VERSION 3.26)

project(matrix-pipeline)
add_definitions(-DPROJECT_NAME="${PROJECT_NAME}")

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Always generate compile_commands.json for clangd, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/compiler-flags.cmake)
include(cmake/sanitizers.cmake)
include(cmake/git.cmake)


add_subdirectory(src/matrix-pipeline)