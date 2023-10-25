add_library(rest-lib
        controller/StaticController.hpp
        AppComponent.hpp
        error_handler.cpp
        error_handler.hpp)

## include directories
include_directories(/usr/local/include/torch/)

## link libs
find_package(oatpp          1.3.0 REQUIRED)
find_package(OpenCV REQUIRED)

target_link_libraries(rest-lib
        # Oat++
        PUBLIC oatpp::oatpp
        spdlog
        opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs
)


if(CMAKE_SYSTEM_NAME MATCHES Linux)
    find_package(Threads REQUIRED)
    target_link_libraries(rest-lib INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
endif()

add_library(http_service oatpp_entry.cpp)
target_link_libraries(http_service rest-lib)
