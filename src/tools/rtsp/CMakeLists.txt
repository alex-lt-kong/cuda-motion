add_executable(rtsp
        rtsp.cpp
)
target_link_libraries(rtsp
        PRIVATE
        ${OpenCV_LIBS}
        #spdlog::spdlog
)
