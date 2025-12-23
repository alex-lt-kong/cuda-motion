#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <opencv2/cudaimgproc.hpp>

namespace MatrixPipeline::ProcessingUnit {

class YuNetOverlayLandmarks : public ISynchronousProcessingUnit {
public:
  using ISynchronousProcessingUnit::ISynchronousProcessingUnit;

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;

private:
  cv::Scalar m_landmark_color{0, 255, 0}; // Default Green
  int m_radius = 2;
  int m_thickness = -1; // Filled
};

} // namespace MatrixPipeline::ProcessingUnit