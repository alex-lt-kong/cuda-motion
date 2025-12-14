#pragma once

#include "../interfaces/i_asynchronous_processing_unit.h"

namespace MatrixPipeline::ProcessingUnit {

class MatrixNotifier final : public IAsynchronousProcessingUnit {
  std::unique_ptr<Utils::NvJpegEncoder> m_gpu_encoder{nullptr};
  std::unique_ptr<Utils::MatrixSender> m_sender{nullptr};
  std::string m_matrix_homeserver;
  std::string m_matrix_room_id;
  std::string m_matrix_access_token;
  int m_notification_interval_frame{300};
  bool m_is_send_image_enabled{true};
  bool m_is_send_video_enabled{true};
  std::chrono::seconds m_video_max_length{60};
  std::chrono::seconds  m_video_max_length_without_detection{10};

  std::chrono::time_point<std::chrono::steady_clock> m_current_video_start_at;
  size_t m_current_video_frame_count{0};
  std::chrono::time_point<std::chrono::steady_clock>  m_current_video_without_detection_since;
  float m_max_roi_score{0.0f};
  cv::cuda::GpuMat m_max_roi_score_frame{-1};
  const double m_target_fps{25.0};
  // 0-51, lower is better, effectively acts like CRF.
  // A value of ~25-30 is usually good for NVENC H.264.
  uint8_t m_target_quality{30};
  cv::Ptr<cv::cudacodec::VideoWriter> m_writer{nullptr};
  std::unique_ptr<Utils::RamVideoBuffer> m_ram_buf{nullptr};
  Utils::VideoRecordingState m_state{Utils::IDLE};

  static bool check_if_people_detected(const PipelineContext &ctx);

  void handle_image(const cv::cuda::GpuMat &frame,
                    [[maybe_unused]] const PipelineContext &ctx,
                    const bool is_people_detected) const;
  void handle_video(const cv::cuda::GpuMat &frame,
                    [[maybe_unused]] const PipelineContext &ctx,
                    const bool is_people_detected);

  static float calculate_roi_score(const YoloContext &yolo);

public:
  bool init(const njson &config) override;
  void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) override;
};

} // namespace MatrixPipeline::ProcessingUnit