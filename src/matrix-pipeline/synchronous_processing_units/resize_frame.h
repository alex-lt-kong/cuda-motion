#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <nlohmann/json.hpp>
#include <opencv2/cudawarping.hpp>

using njson = nlohmann::json;

namespace MatrixPipeline::ProcessingUnit {


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


} // namespace MatrixPipeline::ProcessingUnit