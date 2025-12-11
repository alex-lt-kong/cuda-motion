add_library(rtsp_producer
        rtsp_producer.cpp rtsp_producer.h
        ../interfaces/i_asynchronous_processing_unit.h
)
target_link_libraries(rtsp_producer spdlog::spdlog)