#include "rotate_and_flip.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

SynchronousProcessingResult
RotateAndFlip::process(cv::cuda::GpuMat &frame,
                       [[maybe_unused]] PipelineContext &meta_data) {
  if (frame.empty()) {
    return failure_and_stop;
  }
  if (m_angle.has_value()) {
    switch (m_angle.value()) {
    case 90:
      cv::cuda::rotate(frame.clone(), frame,
                       cv::Size(frame.size().height, frame.size().width),
                       m_angle.value(), 0, frame.size().width);
      break;
    case 180:
      cv::cuda::rotate(frame.clone(), frame, frame.size(), m_angle.value(),
                       frame.size().width, frame.size().height);
      break;
    case 270:
      cv::cuda::rotate(frame.clone(), frame,
                       cv::Size(frame.size().height, frame.size().width),
                       m_angle.value(), frame.size().height, 0);
      break;
    default: {
    }
    }
  }
  if (m_flip_code.has_value())
    cv::cuda::flip(frame, frame, m_flip_code.value());
  return success_and_continue;
}

bool RotateAndFlip::init(const njson &config) {
  m_angle = config.value("angle", m_angle);
  m_flip_code = config.value("flipCode", m_flip_code);
  SPDLOG_INFO("angle: {}, flip_code: {}",
              m_angle.has_value() ? std::to_string(m_angle.value()) : "nullopt",
              m_flip_code.has_value() ? std::to_string(m_flip_code.value())
                                      : "nullopt");
  return true;
}

} // namespace MatrixPipeline::ProcessingUnit
