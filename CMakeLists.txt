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

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV found at: ${OpenCV_DIR}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

# disable Link Time Optimization
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG OFF)
# Force -O0 for Debug configurations
# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

add_subdirectory(src/matrix-pipeline)
add_subdirectory(src/tools)