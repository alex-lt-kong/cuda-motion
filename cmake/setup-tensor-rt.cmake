# cmake/SetupTensorRT.cmake

# Allow user to specify path via -DTENSORRT_PATH
set(TENSORRT_PATH "/apps/TensorRT-10.14.1.48" CACHE PATH "Path to TensorRT root")

find_path(TRT_INCLUDE_DIR NvInfer.h
        PATHS ${TENSORRT_PATH}/include
        NO_DEFAULT_PATH
)

find_library(TRT_NVINFER_LIB nvinfer
        PATHS ${TENSORRT_PATH}/lib
        NO_DEFAULT_PATH
)

find_library(TRT_ONNX_LIB nvonnxparser
        PATHS ${TENSORRT_PATH}/lib
        NO_DEFAULT_PATH
)

if (TRT_INCLUDE_DIR AND TRT_NVINFER_LIB AND TRT_ONNX_LIB)
    message(STATUS "Found TensorRT: ${TENSORRT_PATH}")

    # GLOBAL is the key to fixing your "header not found" in subdirectories
    add_library(TensorRT::Runtime INTERFACE IMPORTED GLOBAL)

    set_target_properties(TensorRT::Runtime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TRT_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${TRT_NVINFER_LIB};${TRT_ONNX_LIB}"
    )

    # Automatically "bake" the library path into the executable
    set(CMAKE_INSTALL_RPATH "${TENSORRT_PATH}/lib" PARENT_SCOPE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE PARENT_SCOPE)
else ()
    message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_PATH to the TRT root.")
endif ()