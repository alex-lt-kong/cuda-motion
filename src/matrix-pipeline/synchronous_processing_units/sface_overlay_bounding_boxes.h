#pragma once

#include "../entities/processing_context.h"
#include "../interfaces/i_synchronous_processing_unit.h"

#include <nlohmann/json.hpp>

namespace MatrixPipeline::ProcessingUnit {

using njson = nlohmann::json;

class SFaceOverlayBoundingBoxes : public ISynchronousProcessingUnit {
public:
  SFaceOverlayBoundingBoxes(const std::string &unit_path)
      : ISynchronousProcessingUnit(unit_path + "/YoloOverlayBoundingBoxes") {}
  ~SFaceOverlayBoundingBoxes() override = default;

  /**
   * @brief Initializes the overlay unit.
   * All configuration parameters are optional.
   */
  bool init(const njson &config) override;

  /**
   * @brief Draws bounding boxes and identity labels on the frame.
   * Requires: valid SFace results and YuNet geometry in the context.
   */
  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;

private:
  // --- Configuration Defaults ---
  // These values are used if the config object is missing keys.

  // Default: Lime Green (BGR: 0, 255, 0)
  cv::Scalar m_box_color_bgr{0, 255, 0};

  // Default: White (BGR: 255, 255, 255) for high contrast text
  cv::Scalar m_text_color_bgr{255, 255, 255};

  // Default: 2 pixels (Standard visibility)
  int m_thickness = 2;

  // Default: 0.6 (Legible on standard 720p/1080p streams)
  double m_label_font_scale = 0.6;

  // Default: 1 (Standard font weight)
  int m_font_thickness = 1;
};

} // namespace MatrixPipeline::ProcessingUnit