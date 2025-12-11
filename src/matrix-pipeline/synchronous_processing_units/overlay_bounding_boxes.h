#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

namespace MatrixPipeline::ProcessingUnit {

class OverlayBoundingBoxes final : public ISynchronousProcessingUnit {
public:
  OverlayBoundingBoxes() = default;
  ~OverlayBoundingBoxes() override = default;

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;

private:
  // --- State ---
  std::vector<std::string> m_class_names;
  std::vector<cv::Scalar> m_colors;

  // --- Reusable Buffers (Avoid re-allocation) ---
  cv::Mat h_overlay_canvas;          // Host (CPU) Canvas
  cv::cuda::GpuMat d_overlay_canvas; // Device (GPU) Canvas
  cv::cuda::GpuMat d_overlay_gray;   // Intermediate Gray for masking
  cv::cuda::GpuMat d_overlay_mask;   // Final Mask
};


}