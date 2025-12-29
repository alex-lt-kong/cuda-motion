#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

// TensorRT Core Header
#include <NvInfer.h>
// ONNX Parser Header (Required to build the engine from .onnx at runtime)
#include <NvOnnxParser.h>
// CUDA Runtime Header (For cudaMalloc, cudaFree, cudaStream, etc.)
#include <cuda_runtime_api.h>
#include <opencv2/core/cuda.hpp>

namespace MatrixPipeline::ProcessingUnit {

using namespace std::chrono_literals;

class YoloDetect final : public ISynchronousProcessingUnit {
private:
  // --- TensorRT Smart Pointers ---
  // We use unique_ptrs with custom deleters or shared_ptrs for TRT objects
  std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
  std::unique_ptr<nvinfer1::IExecutionContext> m_context;

  // --- GPU Memory Management ---
  void *m_output_buffer_gpu = nullptr; // Raw pointer for TRT binding
  std::vector<float> m_output_cpu;     // For downloading inference results
  size_t m_output_count = 0;           // Total floats in the output tensor

  cudaStream_t m_cuda_stream = nullptr; // Async execution stream
  int m_output_dimensions;
  int m_output_rows;
  // --- OpenCV GpuMat Buffers (Reuse these to avoid re-allocation) ---
  cv::cuda::GpuMat m_resized_gpu;
  cv::cuda::GpuMat m_normalized_gpu;

  // Non-TRT-related
  std::string m_model_path;
  cv::Size m_model_input_size = {640, 640}; // Default YOLO size
  cv::dnn::Net m_net;
  float m_confidence_threshold = 0.5f;
  float m_nms_thres = 0.45f;
  int m_frame_interval = 10;
  std::chrono::milliseconds m_inference_interval = 100ms;
  std::chrono::time_point<std::chrono::steady_clock> m_last_inference_time;
  YoloContext m_prev_yolo_ctx;

  void post_process_yolo(const cv::cuda::GpuMat &frame,
                         PipelineContext &ctx) const;

public:
  explicit YoloDetect(const std::string &unit_path)
      : ISynchronousProcessingUnit(unit_path + "/YoloDetect") {}
  bool init(const njson &config) override;

  ~YoloDetect() override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;
};

} // namespace MatrixPipeline::ProcessingUnit