#pragma once

#include "../interfaces/i_asynchronous_processing_unit.h"
#include "../synchronous_processing_units/pipeline_executor.h"

namespace MatrixPipeline::ProcessingUnit {

class AsynchronousProcessingUnit final : public IAsynchronousProcessingUnit {
  PipelineExecutor m_exe;

public:
  bool init(const njson &config) override;
  void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) override;
};

}