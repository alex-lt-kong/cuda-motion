#include "debug_output.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

[[nodiscard]] SynchronousProcessingResult
DebugOutput::process([[maybe_unused]] cv::cuda::GpuMat &frame,
                     PipelineContext &ctx) {
  if (!ctx.yunet_sface.results.empty())
    SPDLOG_INFO("frame_seq_num: {}, ctx.yolo.indices.size(): {}, "
                "ctx.yunet_sface.size(): {}",
                ctx.frame_seq_num, ctx.yolo.indices.size(),
                ctx.yunet_sface.results.size());
  return success_and_continue;
}

bool DebugOutput::init([[maybe_unused]] const njson &config) {
  m_custom_text = config.value("customText", "");
  return true;
}

} // namespace MatrixPipeline::ProcessingUnit