#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

namespace MatrixPipeline::ProcessingUnit {

class RotateFrame final : public ISynchronousProcessingUnit {
private:
  int m_angle{0};

public:
  RotateFrame() = default;
  ~RotateFrame() override = default;

  SynchronousProcessingResult
  process(cv::cuda::GpuMat &frame,
          [[maybe_unused]] PipelineContext &meta_data) override;

  bool init(const njson &config) override;
};

} // namespace MatrixPipeline::ProcessingUnit