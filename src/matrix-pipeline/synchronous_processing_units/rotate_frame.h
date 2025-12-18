#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

namespace MatrixPipeline::ProcessingUnit {

class RotateFrame final : public ISynchronousProcessingUnit {
private:
  int m_angle{0};

public:
  explicit RotateFrame(const std::string &unit_path) : ISynchronousProcessingUnit(unit_path  + "/RotateFrame") {}
  ~RotateFrame() override = default;

  SynchronousProcessingResult
  process(cv::cuda::GpuMat &frame,
          [[maybe_unused]] PipelineContext &meta_data) override;

  bool init(const njson &config) override;
};

} // namespace MatrixPipeline::ProcessingUnit