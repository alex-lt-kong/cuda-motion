add_library(nvjpeg_encoder
        nvjpeg_encoder.cpp
        nvjpeg_encoder.h
)
target_link_libraries(nvjpeg_encoder
        PUBLIC
        CUDA::cudart
        CUDA::nvjpeg
        PRIVATE
        spdlog::spdlog)


add_library(ram_video_buffer
        ram_video_buffer.cpp
        ram_video_buffer.h
)


add_library(matrix_sender
        matrix_sender.cpp
        matrix_sender.h
)
target_link_libraries(matrix_sender cpr::cpr spdlog::spdlog)


add_library(cuda_helper
        cuda_helper.cpp
        cuda_helper.h
)
target_include_directories(cuda_helper PUBLIC
        "$ENV{TRT_HOME}/include"
)
target_link_libraries(cuda_helper
        PUBLIC ${NVINFER_LIB} ${NVONNX_LIB} CUDA::cudart
        PRIVATE spdlog::spdlog)