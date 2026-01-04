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
  cv::cuda::HostMem m_pinned_buffer;
  bool m_disabled{false};
  std::chrono::milliseconds m_inference_interval{100};
  std::chrono::time_point<std::chrono::steady_clock> m_last_inference_at;
  YuNetContext m_prev_yunet_ctx;

public:
  explicit YuNetDetect(const std::string &unit_path)
      : ISynchronousProcessingUnit(unit_path + "/YuNetDetect") {}

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;
};

} // namespace MatrixPipeline::ProcessingUnit