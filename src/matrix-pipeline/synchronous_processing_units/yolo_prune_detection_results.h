#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <nlohmann/json.hpp>
#include <unordered_set>

namespace MatrixPipeline::ProcessingUnit {

class YoloPruneDetectionResults final : public ISynchronousProcessingUnit {
public:
  explicit YoloPruneDetectionResults(const std::string &unit_path) : ISynchronousProcessingUnit(unit_path  + "/YoloPruneDetectionResults") {}
  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;

private:
  struct Range {
    // Make the range slightly wider to handle suspected precision is
    double min_val = -0.01;
    double max_val = 1.01;
  };

  // Edge Constraints
  Range m_left_constraint;
  Range m_right_constraint;
  Range m_top_constraint;
  Range m_bottom_constraint;

  // New: Size Constraints
  enum class SizeMode { NONE, MIN_RATIO, MAX_RATIO };
  SizeMode m_size_limit_mode = SizeMode::NONE;
  double m_size_limit_val = 0.0;

  // 0 means disabled
  float m_debug_overlay_alpha = 0.0;
  cv::cuda::GpuMat m_overlay_buffer; // Reuse to avoid reallocation

  std::unordered_set<int> m_class_ids_of_interest;

  // Helper to parse a specific edge constraint from JSON
  static Range parse_constraint(const njson &constraints,
                                const std::string &key);

};

} // namespace MatrixPipeline::ProcessingUnit