#include "../interfaces/i_synchronous_processing_unit.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace CudaMotion::ProcessingUnit {

[[nodiscard]] SynchronousProcessingResult
DebugOutput::process([[maybe_unused]] cv::cuda::GpuMat &frame,
             PipelineContext &ctx) {
  const auto latency_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count() -
      ctx.capture_timestamp_ms;
  SPDLOG_INFO("frame_seq_num: {}, capture_timestamp_ms: {}, latency: {}ms, ctx.yolo.indices.size(): {}",
              ctx.frame_seq_num, ctx.capture_timestamp_ms, latency_ms, ctx.yolo.indices.size());
  return success_and_continue;
}

bool DebugOutput::init([[maybe_unused]] const njson &config) { return true; }

}