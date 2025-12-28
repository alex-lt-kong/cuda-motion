#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

namespace MatrixPipeline::ProcessingUnit {

using namespace std::chrono_literals;

class YoloDetect final : public ISynchronousProcessingUnit {
private:
  std::string m_model_path;
  cv::Size m_model_input_size = {640, 640}; // Default YOLO size
  cv::dnn::Net m_net;
  float m_confidence_threshold = 0.5f;
  float m_nms_thres = 0.45f;
  int m_frame_interval = 10;
  std::chrono::milliseconds m_inference_interval = 100ms;
  std::chrono::time_point<std::chrono::steady_clock> m_last_inference_time;
  YoloContext m_prev_yolo_ctx;

  void post_process_yolo(const cv::cuda::GpuMat &frame,
                         PipelineContext &ctx) const;

public:
  explicit YoloDetect(const std::string &unit_path)
      : ISynchronousProcessingUnit(unit_path + "/YoloDetect") {}
  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;
};

} // namespace MatrixPipeline::ProcessingUnit