find_package(oatpp          1.3.0 REQUIRED)
find_package(oatpp-openssl  1.3.0 REQUIRED)
find_package(OpenSSL        1.1   REQUIRED)
find_package(OpenCV               REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(http_service
        oatpp_entry.cpp
        controller/static_controller.hpp
        app_component.hpp
        error_handler.cpp
        error_handler.hpp)

target_link_libraries(http_service
        # Oat++
        PUBLIC oatpp::oatpp
        PUBLIC oatpp::oatpp-openssl
        spdlog
)


if(CMAKE_SYSTEM_NAME MATCHES Linux)
    find_package(Threads REQUIRED)
    target_link_libraries(http_service INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
endif()