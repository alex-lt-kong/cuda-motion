#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

namespace MatrixPipeline::ProcessingUnit {

class DebugOutput final : public ISynchronousProcessingUnit {
  std::string m_custom_text;
public:
  explicit DebugOutput(const std::string &unit_path) : ISynchronousProcessingUnit(unit_path  + "/DebugOutput") {}
  ~DebugOutput() override = default;

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;
};

}