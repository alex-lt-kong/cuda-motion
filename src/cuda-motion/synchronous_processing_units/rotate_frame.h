#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

namespace CudaMotion::ProcessingUnit {

class RotateFrame final : public ISynchronousProcessingUnit {
private:
  int m_angle{0};

public:
  [[nodiscard]] SynchronousProcessingResult
  process(cv::cuda::GpuMat &frame, ProcessingMetaData& meta_data) override {
    switch (m_angle) {
    case 90:
      cuda::rotate(frame.clone(), frame,
                   Size( frame.size().height, frame.size().width),
                   m_angle, 0, frame.size().width);
      break;
    case 180:
      cuda::rotate(frame.clone(), frame, frame.size(), m_angle,
                   frame.size().width, frame.size().height);
      break;
    case 270:
      cuda::rotate(frame.clone(), frame,
                   Size(frame.size().height, frame.size().width),
                   m_angle, frame.size().height, 0);
      break;
    default:
    return SynchronousProcessingResult::failure_and_continue;;
    }
    return SynchronousProcessingResult::success_and_continue;
  }

  bool init(const njson &config) override {
    m_angle = config["angle"];
    return true;
  };
  RotateFrame() {};
  ~RotateFrame() override = default;
};
} // namespace CudaMotion::ProcessingUnit