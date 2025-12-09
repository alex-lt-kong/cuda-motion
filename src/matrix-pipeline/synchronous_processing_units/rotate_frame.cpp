#include "../interfaces/i_synchronous_processing_unit.h"

#include <opencv2/cudawarping.hpp>

namespace CudaMotion::ProcessingUnit {

SynchronousProcessingResult
RotateFrame::process(cv::cuda::GpuMat &frame,
                     [[maybe_unused]] PipelineContext &meta_data) {
  switch (m_angle) {
  case 90:
    cv::cuda::rotate(frame.clone(), frame,
                     cv::Size(frame.size().height, frame.size().width), m_angle,
                     0, frame.size().width);
    break;
  case 180:
    cv::cuda::rotate(frame.clone(), frame, frame.size(), m_angle,
                     frame.size().width, frame.size().height);
    break;
  case 270:
    cv::cuda::rotate(frame.clone(), frame,
                     cv::Size(frame.size().height, frame.size().width), m_angle,
                     frame.size().height, 0);
    break;
  default:
    return SynchronousProcessingResult::failure_and_continue;
  }
  return SynchronousProcessingResult::success_and_continue;
}

bool RotateFrame::init(const njson &config) {
  m_angle = config["angle"];
  return true;
};
} // namespace CudaMotion::ProcessingUnit
