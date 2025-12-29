#include "matrix_notifier.h"

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <future>
#include <string>
#include <vector>

namespace MatrixPipeline::ProcessingUnit {

bool MatrixNotifier::look_for_interesting_detection(
    const PipelineContext &ctx) {
  for (const auto idx : ctx.yolo.indices) {
    if (ctx.yolo.is_detection_valid[idx]) {
      return true;
    }
  }
  return false;
}

void MatrixNotifier::handle_image(const cv::cuda::GpuMat &frame,
                                  [[maybe_unused]] const PipelineContext &ctx,
                                  const bool is_interesting) const {
  if (!is_interesting)
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
                                  const bool is_detection_interesting) {
  using namespace std::chrono;

  m_frames_queue.push({frame, ctx});
  // the below relies on a non-empty queue
  while (m_frames_queue.size() > 1 &&
         steady_clock::now() - m_frames_queue.front().ctx.capture_timestamp >
             m_video_precapture) {
    m_frames_queue.pop();
  }

  // is_frame_changing and is_detection_interesting rely on the current frame
  const auto is_frame_changing = ctx.change_rate >= m_min_frame_change_rate;
  if ((!is_detection_interesting || !is_frame_changing) &&
      m_state == Utils::VideoRecordingState::IDLE)
    return;

  if (m_state == Utils::VideoRecordingState::IDLE) {
    std::string symlink_path;
    try {
      m_ram_buf = std::make_unique<Utils::RamVideoBuffer>();
      if (m_ram_buf == nullptr)
        throw std::bad_alloc();
      // cv::cudacodec::createVideoWriter expects a .mp4 file path, we must
      // emulate this behavior...
      symlink_path = fmt::format("/tmp/nvenc_buffer_{}.mp4", m_ram_buf->fd);
      if (symlink(m_ram_buf->m_virtual_path.c_str(), symlink_path.c_str()) !=
          0) {
        SPDLOG_ERROR("symlink({}, {}) failed", m_ram_buf->m_virtual_path,
                     symlink_path);
        return;
      }

      cv::cudacodec::EncoderParams params;
      // 2. Set Rate Control to Variable Bitrate
      params.rateControlMode = cv::cudacodec::ENC_PARAMS_RC_VBR;
      params.targetQuality = m_target_quality;
      m_writer = cv::cudacodec::createVideoWriter(
          symlink_path, frame.size(), cv::cudacodec::Codec::HEVC,
          ctx.fps > 0 ? ctx.fps : m_fallback_fps,
          cv::cudacodec::ColorFormat::BGR, params);

      m_current_video_start_at = m_frames_queue.front().ctx.capture_timestamp;
      m_current_video_without_detection_since = steady_clock::now();
      m_current_video_frame_count = 0;
      m_max_roi_score = -1;
      m_state = Utils::VideoRecordingState::RECORDING;
      SPDLOG_INFO("Start video recording for matrix message, saved file to "
                  "symlink_path {}, video_max_length(sec): {}",
                  symlink_path, m_video_max_length);
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
      steady_clock::now() - m_current_video_start_at >= m_video_max_length;
  const auto is_max_length_without_detection_reached =
      steady_clock::now() - m_current_video_without_detection_since >=
      m_video_max_length_without_detection;

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

      const auto video_duration_ms =
          static_cast<long>(m_current_video_frame_count * 1000.0 /
                            (ctx.fps > 0 ? ctx.fps : m_fallback_fps));
      SPDLOG_INFO(
          "Matrix video recording stopped (is_max_length_reached: {}, "
          "is_max_length_without_detection_reached: {}), "
          "video size: {}KB + thumbnail size {}KB, video_length(sec): {}",
          is_max_length_reached, is_max_length_without_detection_reached,
          ram_buf->size / 1024, jpeg_data.size() / 1024,
          video_duration_ms / 1000);
      const auto send_video_start_at = steady_clock::now();
      m_sender->send_video_from_memory(
          data, fmt::format("{:%Y-%m-%dT%H:%M:%S}.mp4", system_clock::now()),
          video_duration_ms,
          fmt::format("{:%Y-%m-%dT%H:%M:%S}", system_clock::now()), jpeg_data,
          m_max_roi_score_frame.size().width,
          m_max_roi_score_frame.size().height);
      const auto send_video_end_at = steady_clock::now();
      SPDLOG_INFO(
          "send_video_from_memory() took {}ms",
          duration_cast<milliseconds>(send_video_end_at - send_video_start_at)
              .count());
    }).detach();
    m_state = Utils::VideoRecordingState::IDLE;
    return;
  }
  if (const auto roi_value =
          calculate_roi_score(m_frames_queue.front().ctx.yolo);
      roi_value > m_max_roi_score) {
    m_max_roi_score_frame = m_frames_queue.front().frame;
    m_max_roi_score = roi_value;
  }

  m_writer->write(m_frames_queue.front().frame);
  m_frames_queue.pop();
  ++m_current_video_frame_count;
  if (is_detection_interesting && is_frame_changing)
    m_current_video_without_detection_since = steady_clock::now();
}

float MatrixNotifier::calculate_roi_score(const YoloContext &yolo) {
  float roi_value = 0.0;
  for (const auto idx : yolo.indices) {
    if (yolo.class_ids[idx] == 0 && yolo.is_detection_valid[idx]) {
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
  m_video_max_length = std::chrono::seconds(
      config.value("videoMaxLengthInSeconds", m_video_max_length.count()));
  m_target_quality = config.value("videoTargetQuality", m_target_quality);
  m_min_frame_change_rate =
      config.value("minFrameChangeRate", m_min_frame_change_rate);
  m_video_max_length_without_detection = std::chrono::seconds(
      config.value("videoMaxLengthWithoutPeopleDetectedInSeconds",
                   m_video_max_length_without_detection.count()));
  m_video_precapture = std::chrono::seconds(
      config.value("videoPrecaptureSec", m_video_precapture.count()));
  SPDLOG_INFO("matrix_homeserver: {}, matrix_room_id: {}, matrix_access_token: "
              "{}, notification_interval_frame: {}",
              m_matrix_homeserver, m_matrix_room_id, m_matrix_access_token,
              m_notification_interval_frame);
  if (m_is_send_image_enabled)
    SPDLOG_INFO("is_send_image_enabled: {}, notification_interval_frame: {}",
                m_is_send_image_enabled, m_notification_interval_frame);
  if (m_is_send_video_enabled)
    SPDLOG_INFO("is_send_video_enabled: {}, video_max_length(sec): {}, "
                "video_max_length_without_detection(sec): {}, "
                "m_video_precapture(sec): {}"
                "target_quality: {} (0-51, lower is better)",
                m_is_send_video_enabled, m_video_max_length,
                m_video_max_length_without_detection.count(),
                m_video_precapture.count(), m_target_quality);
  m_sender = std::make_unique<Utils::MatrixSender>(
      m_matrix_homeserver, m_matrix_access_token, m_matrix_room_id);
  m_gpu_encoder = std::make_unique<Utils::NvJpegEncoder>();
  if (config.value("testMatrixConnectivity", false))
    m_sender->sendText("MatrixPipeline started");
  return true;
}

void MatrixNotifier::on_frame_ready(cv::cuda::GpuMat &frame,
                                    [[maybe_unused]] PipelineContext &ctx) {
  const auto is_detection_interesting = look_for_interesting_detection(ctx);
  if (m_is_send_image_enabled)
    handle_image(frame, ctx, is_detection_interesting);
  if (m_is_send_video_enabled)
    handle_video(frame, ctx, is_detection_interesting);
}

} // namespace MatrixPipeline::ProcessingUnit