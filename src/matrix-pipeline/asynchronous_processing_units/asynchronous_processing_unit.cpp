#include "asynchronous_processing_unit.h"

#include <iostream>

namespace MatrixPipeline::ProcessingUnit {

bool AsynchronousProcessingUnit::init(const njson &config) {
  // m_exe = std::make_unique<PipelineExecutor>();
  SPDLOG_INFO("{}", config.dump());
  return m_exe.init(config);
}

void AsynchronousProcessingUnit::on_frame_ready(cv::cuda::GpuMat &frame,
                                                PipelineContext &ctx) {
  m_exe.process(frame, ctx);
}

} // namespace MatrixPipeline::ProcessingUnit