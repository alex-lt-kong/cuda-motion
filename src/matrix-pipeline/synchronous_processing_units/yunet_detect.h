#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"
#include "../utils/cuda_helper.h"

#include <NvInfer.h> // TensorRT Core Header
#include <NvOnnxParser.h> // ONNX Parser Header (Required to build the engine from .onnx at runtime)
#include <nlohmann/json.hpp>
#include <opencv2/core/cuda.hpp>

#include <chrono>
#include <map>
#include <memory>

namespace MatrixPipeline::ProcessingUnit {

using njson = nlohmann::json;

class YuNetDetect final : public ISynchronousProcessingUnit {
private:
  void decode_heads(const cv::cuda::GpuMat &frame, PipelineContext &ctx);

  // Config
  cv::Size m_model_input_size{1920, 1080};
  float m_score_threshold = 0.9f;
  float m_nms_threshold = 0.3f;
  std::chrono::milliseconds m_inference_interval{0};
  std::chrono::steady_clock::time_point m_last_inference_at;
  std::vector<FaceDetection> m_prev_yunet_ctx;

  // TensorRT
  std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
  std::unique_ptr<nvinfer1::IExecutionContext> m_context;

  // Multi-Head Buffers
  std::unique_ptr<float, Utils::CudaDeleter> m_input_buffer_gpu;
  std::map<std::string, std::unique_ptr<float, Utils::CudaDeleter>>
      m_gpu_buffers;
  std::map<std::string, std::vector<float>> m_cpu_buffers;

  // CUDA Stream
  cudaStream_t m_cuda_stream = nullptr;
  cv::cuda::Stream m_cv_stream;

  // GPU Scratch
  cv::cuda::GpuMat m_resized_gpu, m_rgb, m_normalized_gpu;

public:
  explicit YuNetDetect(const std::string &unit_path)
      : ISynchronousProcessingUnit(unit_path + "/YuNetDetect") {}

  ~YuNetDetect() override;

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;
};

} // namespace MatrixPipeline::ProcessingUnit