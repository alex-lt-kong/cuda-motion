#pragma once

#include "../entities/processing_context.h"
#include "../interfaces/i_synchronous_processing_unit.h"

#include <nlohmann/json.hpp>
#include <opencv2/core/cuda.hpp>

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
  cv::Mat m_frame_cpu;
  cv::cuda::HostMem m_pinned_mem_for_cpu_frame;
  std::unordered_map<IdentityCategory, cv::Scalar> identity_to_box_color_bgr;
  const cv::Scalar m_text_color_bgr{255, 255, 255}; // white
  int m_bounding_box_border_thickness = 2;
  // Default: 0.6 (Legible on standard 720p/1080p streams)
  double m_label_font_scale = 0.6;
  int m_label_font_thickness = 1;
};

} // namespace MatrixPipeline::ProcessingUnit