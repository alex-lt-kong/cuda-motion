#include "debug_output.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <chrono>

namespace MatrixPipeline::ProcessingUnit {

[[nodiscard]] SynchronousProcessingResult
DebugOutput::process([[maybe_unused]] cv::cuda::GpuMat &frame,
                     PipelineContext &ctx) {
  const auto latency = std::chrono::steady_clock::now() - ctx.capture_timestamp;
  SPDLOG_INFO("frame_seq_num: {}, latency: {}ms, "
              "ctx.yolo.indices.size(): {}, custom_text: {}",
              ctx.frame_seq_num,latency.count(),
              ctx.yolo.indices.size(), m_custom_text);
  return success_and_continue;
}

bool DebugOutput::init([[maybe_unused]] const njson &config) {
  m_custom_text = config.value("customText", "");
  return true;
}

} // namespace MatrixPipeline::ProcessingUnit