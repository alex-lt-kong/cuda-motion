#pragma once

#include "../entities/processing_context.h"
#include "../entities/synchronous_processing_result.h"

#include <nlohmann/json.hpp>
#include <opencv2/cudawarping.hpp>

using njson = nlohmann::json;

namespace MatrixPipeline::ProcessingUnit {

class ISynchronousProcessingUnit {
public:
  virtual ~ISynchronousProcessingUnit() = default;

  virtual bool init(const njson &config) = 0;

  ///
  /// @param frame the frame to be processed
  /// @return
  virtual SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                              PipelineContext &ctx) = 0;
};

class RotateFrame final : public ISynchronousProcessingUnit {
private:
  int m_angle{0};

public:
  RotateFrame() = default;
  ~RotateFrame() override = default;

  SynchronousProcessingResult
  process(cv::cuda::GpuMat &frame,
          [[maybe_unused]] PipelineContext &meta_data) override;

  bool init(const njson &config) override;
};

class ResizeFrame final : public ISynchronousProcessingUnit {
private:
  int m_target_width{0};
  int m_target_height{0};
  double m_scale_factor{0.0};
  int m_interpolation{cv::INTER_LINEAR};

public:
  ResizeFrame() = default;
  ~ResizeFrame() override = default;

  /**
   * @brief Initializes resize parameters from JSON.
   * * Supported Config Modes:
   * 1. Absolute: { "width": 1920, "height": 1080 }
   * 2. Scaling:  { "scale": 0.5 }
   * * Optional: { "interpolation": "nearest" | "linear" | "cubic" | "area" }
   * Default interpolation is Linear.
   */
  bool init(const njson &config) override;

  [[nodiscard]] SynchronousProcessingResult
  process(cv::cuda::GpuMat &frame, PipelineContext &ctx) override;
};



class DebugOutput final : public ISynchronousProcessingUnit {
  std::string m_custom_text;
public:
  DebugOutput() = default;
  ~DebugOutput() override = default;

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;
};
} // namespace MatrixPipeline::ProcessingUnit