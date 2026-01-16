#include "matrix_notifier.h"
#include "../global_vars.h"

#include <boost/uuid/uuid_io.hpp>
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

std::unique_ptr<Utils::RamVideoBuffer> MatrixNotifier::trim_video_ram(
    const std::string &input_sympath, // <--- Input is now just the path
    int frames_to_remove, boost::uuids::random_generator &uuid_gen) {
  return nullptr;
}

void MatrixNotifier::finalize_video_then_send_out(
    const std::unique_ptr<Utils::RamVideoBuffer> &ram_buf,
    const std::shared_ptr<MatrixNotifier> This /* MUST pass by val here*/) {
  using namespace std::chrono;
  if (!ram_buf->lock_and_map()) {
    SPDLOG_ERROR("ram_buf->lock_and_map() failed");
    return;
  }
  const std::string data(static_cast<const char *>(ram_buf->m_data_ptr),
                         ram_buf->size);

  std::string jpeg_data;
  if (!This->m_gpu_encoder->encode(This->m_max_roi_score_frame, jpeg_data,
                                   90)) {
    SPDLOG_ERROR("m_gpu_encoder->encode() failed");
  }

  const auto video_duration_ms = static_cast<long>(
      This->m_current_video_frame_count * 1000.0 / This->m_fps);
  const auto send_video_start_at = steady_clock::now();
  This->m_sender->send_video_from_memory(
      data, fmt::format("{:%Y-%m-%dT%H:%M:%S}.mp4", system_clock::now()),
      video_duration_ms,
      fmt::format("{:%Y-%m-%dT%H:%M:%S}", system_clock::now()), jpeg_data,
      This->m_max_roi_score_frame.size().width,
      This->m_max_roi_score_frame.size().height);
  const auto send_video_end_at = steady_clock::now();
  SPDLOG_INFO(
      "video size: {}KB + thumbnail size {}KB, video_length(sec): {}, "
      "send_video_from_memory() took {}ms",
      ram_buf->size / 1024, jpeg_data.size() / 1024, video_duration_ms / 1000,
      duration_cast<milliseconds>(send_video_end_at - send_video_start_at)
          .count());
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
  // const auto is_frame_changing = ctx.change_rate >=
  // m_activation_min_frame_change_rate;
  if ((!is_detection_interesting ||
       ctx.change_rate < m_activation_min_frame_change_rate) &&
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
      symlink_path = fmt::format("/tmp/nvenc_buffer_{}.mp4",
                                 boost::uuids::to_string(m_uuid_generator()));
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
          symlink_path, frame.size(), cv::cudacodec::Codec::HEVC, m_fps,
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
      disable();
      unlink(symlink_path.c_str());
      SPDLOG_WARN("is_send_video_enabled turned off to avoid log flooding");
      return;
    }
    // The Writer has already opened the file descriptor.
    // Unlinking the name won't stop the writing, but it keeps /tmp clean.
    unlink(symlink_path.c_str());
  }

  const auto is_max_length_reached_or_program_exiting =
      steady_clock::now() - m_current_video_start_at >= m_video_max_length ||
      ev_flag != 0;
  const auto is_max_length_without_detection_reached =
      steady_clock::now() - m_current_video_without_detection_since >=
      m_video_max_length_without_detection;

  if (is_max_length_reached_or_program_exiting ||
      is_max_length_without_detection_reached) {

    m_writer.release();
    SPDLOG_INFO("Matrix video recording stopped, "
                "is_max_length_reached_or_program_exiting: {}, "
                "is_max_length_without_detection_reached: {}",
                is_max_length_reached_or_program_exiting,
                is_max_length_without_detection_reached);
    m_state = Utils::VideoRecordingState::IDLE;
    // creating a shared_ptr from *this, s.t. the detach()'ed t will never
    // access *this after *this is deleted.
    auto self_ptr = shared_from_this();
    std::thread(finalize_video_then_send_out, std::move(m_ram_buf), self_ptr)
        .detach();
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
  if (is_detection_interesting &&
      ctx.change_rate > m_maintenance_min_frame_change_rate)
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

  m_video_max_length = std::chrono::seconds(
      config.value("videoMaxLengthInSeconds", m_video_max_length.count()));
  m_target_quality = config.value("videoTargetQuality", m_target_quality);
  m_activation_min_frame_change_rate = config.value(
      "activationMinFrameChangeRate", m_activation_min_frame_change_rate);
  m_maintenance_min_frame_change_rate = config.value(
      "maintenanceMinFrameChangeRate", m_maintenance_min_frame_change_rate);
  m_fps = config.value("fps", m_fps);
  m_video_max_length_without_detection = std::chrono::seconds(
      config.value("videoMaxLengthWithoutPeopleDetectedInSeconds",
                   m_video_max_length_without_detection.count()));
  m_video_precapture = std::chrono::seconds(
      config.value("videoPrecaptureSec", m_video_precapture.count()));
  SPDLOG_INFO("matrix_homeserver: {}, matrix_room_id: {}, matrix_access_token: "
              "{}",
              m_matrix_homeserver, m_matrix_room_id, m_matrix_access_token);

  SPDLOG_INFO(
      "video_max_length(sec): {}, video_max_length_without_detection(sec): "
      "{}, "
      "video_precapture(sec): {}, fps: {}, activation_min_frame_change_rate: "
      "{}, maintenance_min_frame_change_rate: {}, target_quality: {} (0-51, "
      "lower is better)",
      m_video_max_length, m_video_max_length_without_detection.count(),
      m_video_precapture.count(), m_fps, m_activation_min_frame_change_rate,
      m_maintenance_min_frame_change_rate, m_target_quality);
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

  handle_video(frame, ctx, is_detection_interesting);
}

} // namespace MatrixPipeline::ProcessingUnit