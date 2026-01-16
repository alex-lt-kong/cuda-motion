#pragma once

#include "../entities/video_recording_state.h"
#include "../interfaces/i_asynchronous_processing_unit.h"
#include "../utils/nvjpeg_encoder.h"
#include "../utils/ram_video_buffer.h"

#include <boost/uuid/uuid.hpp> // main uuid family
#include <boost/uuid/uuid_generators.hpp>
#include <opencv2/cudacodec.hpp>

namespace MatrixPipeline::Utils {
struct RamVideoBuffer;
}
namespace MatrixPipeline::ProcessingUnit {

class MatrixNotifier final : public IAsynchronousProcessingUnit,
                             std::enable_shared_from_this<MatrixNotifier> {
  boost::uuids::random_generator m_uuid_generator;
  std::unique_ptr<Utils::NvJpegEncoder> m_gpu_encoder{nullptr};
  std::unique_ptr<Utils::MatrixSender> m_sender{nullptr};
  std::string m_matrix_homeserver;
  std::string m_matrix_room_id;
  std::string m_matrix_access_token;
  double m_activation_min_frame_change_rate{0.1};
  double m_maintenance_min_frame_change_rate{0.01};
  std::chrono::seconds m_video_max_length{60};
  std::chrono::seconds m_video_max_length_without_detection{10};
  std::chrono::seconds m_video_precapture{3};

  std::chrono::time_point<std::chrono::steady_clock> m_current_video_start_at;
  size_t m_current_video_frame_count{0};
  std::chrono::time_point<std::chrono::steady_clock>
      m_current_video_without_detection_since;
  float m_max_roi_score{0.0f};
  cv::cuda::GpuMat m_max_roi_score_frame{-1};
  double m_fps{25.0};
  // 0-51, lower is better, effectively acts like CRF.
  // A value of ~25-30 is usually good for NVENC H.264.
  uint8_t m_target_quality{30};
  cv::Ptr<cv::cudacodec::VideoWriter> m_writer{nullptr};
  std::unique_ptr<Utils::RamVideoBuffer> m_ram_buf{nullptr};
  Utils::VideoRecordingState m_state{Utils::IDLE};
  std::queue<AsyncPayload> m_frames_queue;

  static bool look_for_interesting_detection(const PipelineContext &ctx);

  void handle_video(const cv::cuda::GpuMat &frame,
                    [[maybe_unused]] const PipelineContext &ctx,
                    const bool is_detection_interesting);

  static float calculate_roi_score(const YoloContext &yolo);
  static void
  finalize_video_then_send_out(const std::unique_ptr<Utils::RamVideoBuffer> &,
                               const std::shared_ptr<MatrixNotifier> &);
  static std::unique_ptr<Utils::RamVideoBuffer> trim_video_ram(
      const std::string &input_sympath, // <--- Input is now just the path
      int frames_to_remove, boost::uuids::random_generator &uuid_gen);

public:
  explicit MatrixNotifier(const std::string &unit_path)
      : IAsynchronousProcessingUnit(unit_path + "/MatrixNotifier") {}
  bool init(const njson &config) override;
  void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) override;
};

} // namespace MatrixPipeline::ProcessingUnit