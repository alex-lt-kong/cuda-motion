#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"
#include "yolo_detect.h"

#include <opencv2/opencv.hpp>

namespace MatrixPipeline::ProcessingUnit {

class AutoZoom : public ISynchronousProcessingUnit {
public:
  explicit AutoZoom(const std::string &unit_path)
      : ISynchronousProcessingUnit(unit_path + "/AutoZoom") {}
  ~AutoZoom() override = default;

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;

private:
  // Factor of the original frame size to set the output resolution.
  // Example: 0.6 on 1080p input results in a fixed 648p output stream.
  float m_output_scale_factor = 0.5f;
  // Smoothness: Pixels per frame the crop box can move/resize.
  float m_smooth_step_pixel = 2.0f;
  bool m_dimensions_set{false};
  std::optional<BoundingBoxScaleParams> m_bounding_box_scale_params;
  cv::Rect2f m_current_roi;
  cv::Size target_output_size_; // The fixed resolution downstream expects
  bool initialized_ = false;

  // Calculates the ideal crop box based on bounding boxes
  [[nodiscard]] cv::Rect calculate_target_roi(const cv::Size &input_size,
                                              const PipelineContext &ctx) const;

  // Expands a rect to match the aspect ratio of the output size
  static cv::Rect fix_aspect_ratio(const cv::Rect &input,
                                   const cv::Size &output_size,
                                   const cv::Size &input_limit);

  // Smoothly steps m_current_roi toward target_roi
  void update_current_roi(const cv::Rect &target_roi);
};

} // namespace MatrixPipeline::ProcessingUnit