add_library(rotate_frame
        rotate_frame.cpp
        ../interfaces/i_synchronous_processing_unit.h
)

add_library(resize_frame
        resize_frame.cpp
        ../interfaces/i_synchronous_processing_unit.h
)

add_library(measure_latency
        measure_latency.cpp measure_latency.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(measure_latency spdlog::spdlog)

add_library(debug_output
        debug_oputput.cpp
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(debug_output spdlog::spdlog)

add_library(detect_objects
        detect_objects.cpp detect_objects.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(detect_objects spdlog::spdlog)

add_library(prune_object_detection_results
        prune_object_detection_results.cpp prune_object_detection_results.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(prune_object_detection_results spdlog::spdlog ${OpenCV_LIBS})

add_library(collect_stats
        collect_stats.cpp collect_stats.h
        ../interfaces/i_synchronous_processing_unit.h
)
target_link_libraries(collect_stats spdlog::spdlog ${OpenCV_LIBS})

add_library(overlay_bounding_boxes
        overlay_bounding_boxes.cpp overlay_bounding_boxes.h
        ../interfaces/i_synchronous_processing_unit.h
)


add_library(pipeline_executor
        pipeline_executor.cpp pipeline_executor.h
)
