add_library(rtsp_producer
        rtsp_producer.cpp rtsp_producer.h
        ../interfaces/i_asynchronous_processing_unit.h
)
target_link_libraries(rtsp_producer spdlog::spdlog)

add_library(matrix_notifier
        matrix_notifier.cpp matrix_notifier.h
        ../interfaces/i_asynchronous_processing_unit.h
)
target_link_libraries(matrix_notifier nvjpeg_encoder ram_video_buffer matrix_sender)
target_link_libraries(matrix_notifier ${OpenCV_LIBS} CUDA::nvjpeg spdlog::spdlog cpr::cpr nlohmann_json::nlohmann_json)

add_library(asynchronous_processing_unit
        asynchronous_processing_unit.cpp asynchronous_processing_unit.h
        ../interfaces/i_asynchronous_processing_unit.h
)
target_link_libraries(asynchronous_processing_unit CUDA::nvjpeg)
