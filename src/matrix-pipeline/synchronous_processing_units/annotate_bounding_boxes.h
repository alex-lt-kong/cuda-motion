#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

namespace MatrixPipeline::ProcessingUnit {

class AnnotateBoundingBoxes final : public ISynchronousProcessingUnit {
public:
  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;

private:
  struct Range {
    // Make the range slightly larger to handle suspected precision is
    double min_val = -0.001;
    double max_val = 1.001;
  };

  Range m_left_constraint;
  Range m_right_constraint;
  Range m_top_constraint;
  Range m_bottom_constraint;

  bool m_is_overlay_enabled = false;
  cv::cuda::GpuMat m_overlay_buffer; // Reuse to avoid reallocation

  // Helper to parse a specific edge constraint from JSON
  static Range parse_constraint(const njson &constraints,
                                const std::string &key);
};

} // namespace MatrixPipeline::ProcessingUnit