#include "../interfaces/i_asynchronous_processing_unit.h"
#include "../pipeline_executor.h"

namespace MatrixPipeline::ProcessingUnit {

bool AsynchronousProcessingUnit::init(const njson &config) {
  m_exe = std::make_unique<PipelineExecutor>();
  return m_exe->init(config["pipeline"]);
}

void AsynchronousProcessingUnit::on_frame_ready(cv::cuda::GpuMat &frame,
                                                PipelineContext &ctx) {
  m_exe->on_frame_ready(frame, ctx);
}

} // namespace MatrixPipeline::ProcessingUnit