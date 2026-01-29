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

class MatrixNotifier final
    : public IAsynchronousProcessingUnit,
      public std::enable_shared_from_this<MatrixNotifier> {
  boost::uuids::random_generator m_uuid_generator;
  std::unique_ptr<Utils::NvJpegEncoder> m_gpu_encoder{nullptr};
  std::unique_ptr<Utils::MatrixSender> m_sender{nullptr};
  std::string m_matrix_homeserver;
  std::string m_matrix_room_id;
  std::string m_matrix_access_token;
  double m_activation_min_frame_change_rate{0.1};
  double m_maintenance_min_frame_change_rate{0.01};
  double m_detection_recognition_weight_ratio{3.14};
  std::chrono::seconds m_video_max_length{60};
  static inline std::unordered_map<IdentityCategory, float>
      m_identity_to_weight_map = {{IdentityCategory::Unknown, 1.0},
                                  {IdentityCategory::Unauthorized, 1.414},
                                  {IdentityCategory::Authorized, 1.732}};
  // using second as unit is more user-friendly but impose significant
  // difficulty on implementation. The problem is that there are two FPSes, one
  // from the video feed, which is dynamic; the other from videoWriter, which is
  // static. If we want to trim the back of the final video, two FPSes cause
  // confusion
  int m_video_precapture_frames{45};
  int m_detections_gap_tolerance_frames{120};
  int m_video_postcapture_frames{45};

  std::chrono::time_point<std::chrono::steady_clock> m_current_video_start_at;
  size_t m_current_video_frame_count{0};
  int m_current_video_without_detection_frames{-1};
  float m_max_roi_score{0.0f};
  cv::cuda::GpuMat m_max_roi_score_frame{-1};
  double m_fps{25.0};
  // 0-51, lower is better, effectively acts like CRF.
  // A value of ~25-30 is usually good for NVENC H.264.
  uint8_t m_target_quality{30};
  cv::Ptr<cv::cudacodec::VideoWriter> m_writer{nullptr};
  // std::unique_ptr<Utils::RamVideoBuffer> m_ram_buf{nullptr};
  std::string m_temp_video_path;
  Utils::VideoRecordingState m_state{Utils::IDLE};
  std::queue<AsyncPayload> m_frames_queue;

  static bool look_for_interesting_detection(const PipelineContext &ctx);

  void handle_video(const cv::cuda::GpuMat &frame,
                    [[maybe_unused]] const PipelineContext &ctx,
                    bool is_detection_interesting);

  double calculate_roi_score(const PipelineContext &ctx) const;
  static void
  finalize_video_then_send_out(std::string,
                               const std::shared_ptr<MatrixNotifier>);
  std::optional<std::string> trim_video(const std::string &input_video_path,
                                        int frames_to_remove);

public:
  explicit MatrixNotifier(const std::string &unit_path)
      : IAsynchronousProcessingUnit(unit_path + "/MatrixNotifier") {}
  ~MatrixNotifier() override = default;
  bool init(const njson &config) override;
  void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) override;
};

} // namespace MatrixPipeline::ProcessingUnit