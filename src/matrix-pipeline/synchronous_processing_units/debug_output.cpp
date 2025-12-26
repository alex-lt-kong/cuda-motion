#include "debug_output.h" / apps / var / matrix - pipeline / models / face_detection_yunet_2023mar.onnx

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

[[nodiscard]] SynchronousProcessingResult
DebugOutput::process([[maybe_unused]] cv::cuda::GpuMat &frame,
                     PipelineContext &ctx) {
  if (!ctx.yunet.empty())
    SPDLOG_INFO(
        "frame_seq_num: {}, ctx.yolo.indices.size(): {}, ctx.yunet.size(): {}",
        ctx.frame_seq_num, ctx.yolo.indices.size(), ctx.yunet.size());
  return success_and_continue;
}

bool DebugOutput::init([[maybe_unused]] const njson &config) {
  m_custom_text = config.value("customText", "");
  return true;
}

} // namespace MatrixPipeline::ProcessingUnit