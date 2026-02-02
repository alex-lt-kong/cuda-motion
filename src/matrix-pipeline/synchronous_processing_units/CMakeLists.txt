add_library(rotate_and_flip
        rotate_and_flip.cpp rotate_and_flip.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(rotate_and_flip
        PUBLIC nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)


add_library(resize
        resize.cpp resize.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(resize
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
        PUBLIC
        cuda_helper
        ${NVINFER_LIB} ${NVONNX_LIB} CUDA::cudart
        PRIVATE spdlog::spdlog
)


add_library(sface_detect
        sface_detect.cpp sface_detect.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(sface_detect
        PUBLIC
        cuda_helper
        ${NVINFER_LIB} ${NVONNX_LIB} CUDA::cudart
        PRIVATE spdlog::spdlog ${OpenCV_LIBS}
)


add_library(yolo_prune_detection_results
        yolo_prune_detection_results.cpp yolo_prune_detection_results.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(yolo_prune_detection_results
        PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog yolo_detect)

add_library(collect_stats
        collect_stats.cpp collect_stats.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(collect_stats
        PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)


add_library(yolo_overlay
        yolo_overlay.cpp yolo_overlay.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(yolo_overlay
        PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog yolo_detect)

add_library(overlay_text
        overlay_text.cpp overlay_text.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(overlay_text
        PUBLIC nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)


add_library(yunet_detect
        yunet_detect.cpp yunet_detect.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(yunet_detect
        PUBLIC
        cuda_helper
        ${OpenCV_LIBS} nlohmann_json::nlohmann_json ${NVINFER_LIB} ${NVONNX_LIB} CUDA::cudart
        PRIVATE spdlog::spdlog)

add_library(yunet_overlay_landmarks
        yunet_overlay_landmarks.cpp yunet_overlay_landmarks.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(yunet_overlay_landmarks
        PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)

add_library(sface_overlay
        sface_overlay.cpp sface_overlay.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(sface_overlay
        PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)

add_library(auto_zoom
        auto_zoom.cpp auto_zoom.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(auto_zoom
        PUBLIC cuda_helper ${OpenCV_LIBS} nlohmann_json::nlohmann_json ${NVINFER_LIB} ${NVONNX_LIB} CUDA::cudart
        PRIVATE spdlog::spdlog)


add_library(yolo_publish_mqtt
        yolo_publish_mqtt.cpp yolo_publish_mqtt.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_include_directories(yolo_publish_mqtt PRIVATE ${MOSQUITTO_INCLUDE_DIRS})
target_link_libraries(yolo_publish_mqtt
        PUBLIC nlohmann_json::nlohmann_json ${MOSQUITTO_LIBRARIES}
        PRIVATE spdlog::spdlog)
