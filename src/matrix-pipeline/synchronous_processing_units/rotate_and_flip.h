#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

namespace MatrixPipeline::ProcessingUnit {

class RotateAndFlip final : public ISynchronousProcessingUnit {
private:
  std::optional<int> m_angle{std::nullopt};
  std::optional<int> m_flip_code{std::nullopt};

public:
  explicit RotateAndFlip(const std::string &unit_path)
      : ISynchronousProcessingUnit(unit_path + "/RotateAndFlip") {}
  ~RotateAndFlip() override = default;

  SynchronousProcessingResult
  process(cv::cuda::GpuMat &frame,
          [[maybe_unused]] PipelineContext &meta_data) override;

  bool init(const njson &config) override;
};

} // namespace MatrixPipeline::ProcessingUnit