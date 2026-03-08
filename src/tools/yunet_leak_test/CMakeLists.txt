cmake_minimum_required(VERSION 3.10)
project(YuNetLeakTest CXX)

# Require C++17 (if your parent project hasn't already enforced it)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV (You can specify components if you want to be strict,
# but finding the whole package is usually easiest)
find_package(OpenCV REQUIRED COMPONENTS
        core
        videoio
        highgui
        imgproc
        dnn
        objdetect
        cudawarping
)

# Define the executable
add_executable(yunet_leak_test yunet_leak_test.cpp)

# Link OpenCV
target_include_directories(yunet_leak_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yunet_leak_test PRIVATE ${OpenCV_LIBS})

# Optional: Add some basic optimization flags if not inherited from parent
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(yunet_leak_test PRIVATE -O3 -Wall)
endif ()