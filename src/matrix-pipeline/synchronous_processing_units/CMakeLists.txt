add_library(rotate_frame
        rotate_frame.cpp rotate_frame.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(rotate_frame
        PUBLIC nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)


add_library(resize_frame
        resize_frame.cpp resize_frame.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(resize_frame
        PUBLIC nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)


add_library(measure_latency
        measure_latency.cpp measure_latency.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(measure_latency
        PUBLIC nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)


add_library(debug_output
        debug_output.cpp debug_output.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(debug_output
        PUBLIC nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)


add_library(yolo_detect
        yolo_detect.cpp yolo_detect.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(yolo_detect
        PUBLIC ${NVINFER_LIB} ${NVONNX_LIB} CUDA::cudart
        PRIVATE spdlog::spdlog cuda_helper
)


add_library(yolo_prune_detection_results
        yolo_prune_detection_results.cpp yolo_prune_detection_results.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(yolo_prune_detection_results
        PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)


add_library(collect_stats
        collect_stats.cpp collect_stats.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(collect_stats
        PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)


add_library(yolo_overlay_bounding_boxes
        yolo_overlay_bounding_boxes.cpp yolo_overlay_bounding_boxes.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(yolo_overlay_bounding_boxes
        PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)

add_library(overlay_info
        overlay_info.cpp overlay_info.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(overlay_info
        PUBLIC nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)


add_library(yunet_detect
        yunet_detect.cpp yunet_detect.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(yunet_detect
        PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)


add_library(yunet_overlay_landmarks
        yunet_overlay_landmarks.cpp yunet_overlay_landmarks.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(yunet_overlay_landmarks
        PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)
