#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/objdetect.hpp>

namespace MatrixPipeline::ProcessingUnit {

class YuNetDetect : public ISynchronousProcessingUnit {
private:
  cv::Ptr<cv::FaceDetectorYN> m_detector;

  // Internal variables follow snake_case
  float m_score_threshold = 0.9f;
  float m_nms_threshold = 0.3f;
  int m_top_k = 5000;
  bool m_disabled{false};

public:
  using ISynchronousProcessingUnit::ISynchronousProcessingUnit;

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;
};

} // namespace MatrixPipeline::ProcessingUnit