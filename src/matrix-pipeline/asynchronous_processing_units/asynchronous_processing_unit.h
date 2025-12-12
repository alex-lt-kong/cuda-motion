#pragma once

#include "../entities/processing_units_variant.h"
#include "../interfaces/i_asynchronous_processing_unit.h"
// #include "../synchronous_processing_units/pipeline_executor.h"

namespace MatrixPipeline::ProcessingUnit {

class AsynchronousProcessingUnit final : public IAsynchronousProcessingUnit {
  std::vector<ProcessingUnitVariant> m_processing_units;
  // std::mutex m_mutex;

public:
  bool init(const njson &config) override;
  void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) override;
};

}