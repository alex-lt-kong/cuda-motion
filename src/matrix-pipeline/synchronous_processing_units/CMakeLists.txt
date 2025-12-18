add_library(rotate_frame
        rotate_frame.cpp rotate_frame.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(rotate_frame spdlog::spdlog)


add_library(resize_frame
        resize_frame.cpp resize_frame.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(resize_frame spdlog::spdlog)

add_library(measure_latency
        measure_latency.cpp measure_latency.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(measure_latency spdlog::spdlog)

add_library(debug_output
        debug_output.cpp debug_output.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(debug_output spdlog::spdlog)

add_library(yolo_detect
        yolo_detect.cpp yolo_detect.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(yolo_detect spdlog::spdlog)

add_library(yolo_prune_detection_results
        yolo_prune_detection_results.cpp yolo_prune_detection_results.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(yolo_prune_detection_results spdlog::spdlog ${OpenCV_LIBS})

add_library(collect_stats
        collect_stats.cpp collect_stats.h
        ../interfaces/i_synchronous_processing_unit.h
        overlay_info.cpp
)
target_link_libraries(collect_stats spdlog::spdlog ${OpenCV_LIBS})

add_library(yolo_overlay_bounding_boxes
        yolo_overlay_bounding_boxes.cpp yolo_overlay_bounding_boxes.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(yolo_overlay_bounding_boxes spdlog::spdlog)

add_library(overlay_info
        overlay_info.cpp overlay_info.h
        ../interfaces/i_synchronous_processing_unit.h
)
