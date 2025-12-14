#include "matrix_notifier.h"
#include "../utils/nvjpeg_encoder.h"

#include <fmt/chrono.h>
#include <nlohmann/json.hpp>

#
#include <chrono>
#include <future>
#include <string>
#include <vector>
namespace MatrixPipeline::ProcessingUnit {

bool MatrixNotifier::check_if_people_detected(const PipelineContext &ctx) {
  bool person_detected = false;
  for (const auto idx : ctx.yolo.indices) {
    if (ctx.yolo.class_ids[idx] == 0 && ctx.yolo.is_in_roi[idx]) {
      person_detected = true;
      break;
    }
  }
  return person_detected;
}

void MatrixNotifier::handle_image(const cv::cuda::GpuMat &frame,
                                  [[maybe_unused]] const PipelineContext &ctx,
                                  const bool is_people_detected) const {
  if (!m_is_send_image_enabled)
    return;
  if (!is_people_detected)
    return;
  if (ctx.frame_seq_num % m_notification_interval_frame != 0)
    return;
  std::string jpeg_bytes;
  if (const bool success = m_gpu_encoder->encode(frame, jpeg_bytes, 90);
      !success) {
    SPDLOG_ERROR("m_gpu_encoder->encode() failed");
    return;
  }
  m_sender->send_jpeg(jpeg_bytes, frame.cols, frame.rows,
                      fmt::format("{:%Y-%m-%dT%H:%M:%S}.jpg",
                                  std::chrono::system_clock::now()));
}

void MatrixNotifier::handle_video(const cv::cuda::GpuMat &frame,
                                  [[maybe_unused]] const PipelineContext &ctx,
                                  const bool is_people_detected) {
  auto now_sec = [] {
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch())
        .count();
  };

  if (!m_is_send_video_enabled)
    return;

  if (!is_people_detected && m_state == Utils::VideoRecordingState::IDLE)
    return;

  if (m_state == Utils::VideoRecordingState::IDLE) {
    std::string symlink_path;
    try {
      m_ram_buf = std::make_unique<Utils::RamVideoBuffer>();
      if (m_ram_buf == nullptr)
        throw std::bad_alloc();
      // cv::cudacodec::createVideoWriter expects a .mp4 file path, we must
      // emulate this behavior...
      symlink_path =
          "/tmp/nvenc_buffer_" + std::to_string(m_ram_buf->fd) + ".mp4";
      if (symlink(m_ram_buf->m_virtual_path.c_str(), symlink_path.c_str()) !=
          0) {
        throw std::runtime_error("symlink() failed");
      }

      cv::cudacodec::EncoderParams params;
      // 2. Set Rate Control to Variable Bitrate
      params.rateControlMode = cv::cudacodec::ENC_PARAMS_RC_VBR;
      params.targetQuality = m_target_quality;

      m_writer = cv::cudacodec::createVideoWriter(
          symlink_path, frame.size(), cv::cudacodec::Codec::H264, m_target_fps,
          cv::cudacodec::ColorFormat::BGR, params);

      m_current_video_start_at_sec = now_sec();
      m_current_video_without_detection_since_sec = now_sec();
      m_max_roi_score = -1;
      m_state = Utils::VideoRecordingState::RECORDING;
      SPDLOG_INFO("Start video recording for matrix message, saved file to "
                  "symlink_path {}, video_length_in_frame: {}",
                  symlink_path, m_video_max_length_in_sec);
    } catch (const cv::Exception &e) {
      SPDLOG_ERROR("cv::cudacodec::VideoWriter({}) failed: {}", symlink_path,
                   e.what());
      m_writer.release();
      m_writer = nullptr;
      m_ram_buf = nullptr;
      m_is_send_video_enabled = false;
      unlink(symlink_path.c_str());
      SPDLOG_WARN("is_send_video_enabled turned off to avoid log flooding");
      return;
    }
    // The Writer has already opened the file descriptor.
    // Unlinking the name won't stop the writing, but it keeps /tmp clean.
    unlink(symlink_path.c_str());
  }
  const auto is_max_length_reached =
      now_sec() - m_current_video_start_at_sec >= m_video_max_length_in_sec;
  const auto is_max_length_without_detection_reached =
      now_sec() - m_current_video_without_detection_since_sec >=
      m_video_max_length_without_detection_in_sec;

  if (is_max_length_reached || is_max_length_without_detection_reached) {

    m_writer.release();
    const std::shared_ptr ram_buf = std::move(m_ram_buf);
    std::thread([&, ram_buf] {
      if (!ram_buf->lock_and_map()) {
        SPDLOG_ERROR("ram_buf->lock_and_map() failed");
        return;
      }
      const std::string data(static_cast<const char *>(ram_buf->m_data_ptr),
                             ram_buf->size);

      std::string jpeg_data;
      if (!m_gpu_encoder->encode(m_max_roi_score_frame, jpeg_data, 90)) {
        SPDLOG_ERROR("m_gpu_encoder->encode() failed");
      }

      const auto video_length_sec = now_sec() - m_current_video_start_at_sec;
      SPDLOG_INFO(
          "Matrix video recording stopped (is_max_length_reached: {}, "
          "is_max_length_without_detection_reached: {}), "
          "video size: {}KB + thumbnail size {}KB, video_length_sec: {}",
          is_max_length_reached, is_max_length_without_detection_reached,
          ram_buf->size / 1024, jpeg_data.size() / 1024, video_length_sec);
      m_sender->send_video_from_memory(
          data,
          fmt::format("{:%Y-%m-%dT%H:%M:%S}.mp4",
                      std::chrono::system_clock::now()),
          video_length_sec * 1000, jpeg_data,
          m_max_roi_score_frame.size().width,
          m_max_roi_score_frame.size().height);
    }).detach();
    m_state = Utils::VideoRecordingState::IDLE;
    return;
  }
  if (const auto roi_value = calculate_roi_score(ctx.yolo);
      roi_value > m_max_roi_score) {
    m_max_roi_score_frame = frame;
    m_max_roi_score = roi_value;
  }

  m_writer->write(frame);
  if (is_people_detected)
    m_current_video_without_detection_since_sec = now_sec();
}

float MatrixNotifier::calculate_roi_score(const YoloContext &yolo) {
  float roi_value = 0.0;
  for (const auto idx : yolo.indices) {
    if (yolo.class_ids[idx] == 0 && yolo.is_in_roi[idx]) {
      roi_value += yolo.boxes[idx].area() * yolo.confidences[idx] *
                   pow(yolo.indices.size(), 0.5);
    }
  }
  return roi_value;
}

bool MatrixNotifier::init(const njson &config) {
  if (!config.contains("matrixHomeServer") ||
      !config.contains("matrixRoomId") ||
      !config.contains("matrixAccessToken")) {
    SPDLOG_ERROR("Missing matrix credentials");
    return false;
  }
  m_matrix_homeserver = config["matrixHomeServer"];
  m_matrix_room_id = config["matrixRoomId"];
  m_matrix_access_token = config["matrixAccessToken"];
  m_notification_interval_frame =
      config.value("notificationIntervalFrame", m_notification_interval_frame);

  m_is_send_image_enabled =
      config.value("isSendImageEnabled", m_is_send_image_enabled);
  m_is_send_video_enabled =
      config.value("isSendVideoEnabled", m_is_send_video_enabled);
  m_video_max_length_in_sec =
      config.value("videoMaxLengthInSeconds", m_video_max_length_in_sec);
  m_target_quality = config.value("videoTargetQuality", m_target_quality);
  m_video_max_length_without_detection_in_sec =
      config.value("videoMaxLengthWithoutPeopleDetectedInSeconds",
                   m_video_max_length_without_detection_in_sec);
  SPDLOG_INFO("matrix_homeserver: {}, matrix_room_id: {}, matrix_access_token: "
              "{}, notification_interval_frame: {}",
              m_matrix_homeserver, m_matrix_room_id, m_matrix_access_token,
              m_notification_interval_frame);
  if (m_is_send_image_enabled)
    SPDLOG_INFO("is_send_image_enabled: {}, notification_interval_frame: {}",
                m_is_send_image_enabled, m_notification_interval_frame);
  if (m_is_send_video_enabled)
    SPDLOG_INFO("is_send_video_enabled: {}, video_length_in_frame: {}, "
                "target_quality: {} (0-51, lower is better)",
                m_is_send_video_enabled, m_video_max_length_in_sec,
                m_target_quality);
  m_sender = std::make_unique<Utils::MatrixSender>(
      m_matrix_homeserver, m_matrix_access_token, m_matrix_room_id);
  m_gpu_encoder = std::make_unique<Utils::NvJpegEncoder>();
  if (config.value("testMatrixConnectivity", false))
    m_sender->sendText("MatrixPipeline started");
  return true;
}

void MatrixNotifier::on_frame_ready(cv::cuda::GpuMat &frame,
                                    [[maybe_unused]] PipelineContext &ctx) {
  auto is_people_detected = check_if_people_detected(ctx);
  auto future_image =
      std::async(std::launch::async, &MatrixNotifier::handle_image, this, frame,
                 ctx, is_people_detected);
  auto future_video =
      std::async(std::launch::async, &MatrixNotifier::handle_video, this, frame,
                 ctx, is_people_detected);
}

} // namespace MatrixPipeline::ProcessingUnit