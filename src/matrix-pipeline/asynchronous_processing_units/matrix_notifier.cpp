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
    if (ctx.yolo.is_detection_interesting[idx]) {
      return true;
    }
  }
  return false;
}

std::optional<std::string>
MatrixNotifier::trim_video(const std::string &input_video_path,
                           int frames_to_remove) {
  std::string temp_path = fmt::format(
      "/tmp/nvenc_buffer_{}.mp4", boost::uuids::to_string(m_uuid_generator()));

  try {
    // 2. Probe Metadata (Standard VideoCapture for header info)
    cv::VideoCapture cap(input_video_path);
    if (!cap.isOpened()) {
      SPDLOG_ERROR("!cap.isOpened()");
      return std::nullopt;
    }

    const auto total_frames =
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const cv::Size frame_size(
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cap.release();

    const int frames_after_trim = total_frames - frames_to_remove;
    if (frames_after_trim < 0) {
      SPDLOG_ERROR(
          "total_frames: {}, frames_to_remove: {}, frames_after_trim: {} < 0",
          total_frames, frames_to_remove, frames_after_trim);
      return std::nullopt;
    }

    cv::Ptr<cv::cudacodec::VideoReader> reader =
        cv::cudacodec::createVideoReader(input_video_path);
    reader->set(cv::cudacodec::ColorFormat::BGR);

    cv::cudacodec::EncoderParams params;
    // 2. Set Rate Control to Variable Bitrate
    params.rateControlMode = cv::cudacodec::ENC_PARAMS_RC_VBR;
    params.targetQuality = m_target_quality;
    const auto writer = cv::cudacodec::createVideoWriter(
        temp_path, frame_size, cv::cudacodec::Codec::HEVC, m_fps,
        cv::cudacodec::ColorFormat::BGR, params);

    cv::cuda::GpuMat d_frame;
    int current_frame = 0;

    while (current_frame < frames_after_trim && reader->nextFrame(d_frame)) {
      if (d_frame.empty()) {
        SPDLOG_ERROR("!d_frame.empty()");
        break;
      }
      writer->write(d_frame);
      current_frame++;
    }

    writer->release();

    std::error_code ec;
    auto size = std::filesystem::file_size(temp_path, ec);
    std::ifstream file(temp_path, std::ios::binary);
    if (ec || !file) {
      SPDLOG_ERROR("IO error");
      return std::nullopt;
    }
    std::string buffer(size, '\0'); // 1. Pre-allocate exact size
    if (!file.read(buffer.data(), size)) {
      SPDLOG_ERROR("IO error");
      return std::nullopt;
    }
    return buffer;

  } catch (const std::exception &e) {
    if (std::filesystem::exists(temp_path)) {
      std::filesystem::remove(temp_path);
    }
    SPDLOG_ERROR("e.what(): {}", e.what());
    return std::nullopt;
  }
}

void MatrixNotifier::finalize_video_then_send_out(
    std::string temp_video_path,
    const std::shared_ptr<MatrixNotifier> This /* MUST pass by val here*/) {
  using namespace std::chrono;

  std::string jpeg_data;
  if (!This->m_gpu_encoder->encode(This->m_max_roi_score_frame, jpeg_data,
                                   90)) {
    SPDLOG_ERROR("m_gpu_encoder->encode() failed");
  }

  const auto video_duration_ms = static_cast<long>(
      This->m_current_video_frame_count * 1000.0 / This->m_fps);
  const auto send_video_start_at = steady_clock::now();
  int frames_to_remove = This->m_detections_gap_tolerance_frames -
                         This->m_video_postcapture_frames -
                         This->m_video_precapture_frames;
  if (frames_to_remove < 0)
    frames_to_remove = 0;
  if (const auto video = This->trim_video(temp_video_path, frames_to_remove);
      video.has_value()) {
    This->m_sender->send_video_from_memory(
        video.value(),
        fmt::format("{:%Y-%m-%dT%H:%M:%S}.mp4", system_clock::now()),
        video_duration_ms,
        fmt::format("{:%Y-%m-%dT%H:%M:%S}", system_clock::now()), jpeg_data,
        This->m_max_roi_score_frame.size().width,
        This->m_max_roi_score_frame.size().height);
    const auto send_video_end_at = steady_clock::now();
    SPDLOG_INFO(
        "video size: {}KB + thumbnail size {}KB, video_length(sec): {}, "
        "send_video_from_memory() took {}ms({}KB/sec)",
        video.value().size() / 1024, jpeg_data.size() / 1024,
        video_duration_ms / 1000,
        duration_cast<milliseconds>(send_video_end_at - send_video_start_at)
            .count(),
        (video.value().size() + jpeg_data.size()) / 1024 /
            (video_duration_ms / 1000));
  } else {
    SPDLOG_ERROR("trim_video() failed");
  }
  try {
    if (!std::filesystem::remove(temp_video_path)) {
      SPDLOG_ERROR("std::filesystem::remove({}) returns false",
                   temp_video_path);
    }
  } catch (const std::filesystem::filesystem_error &e) {
    SPDLOG_ERROR("std::filesystem::remove({}) exception, e.what(): {}",
                 temp_video_path, e.what());
  }
}

void MatrixNotifier::handle_video(const cv::cuda::GpuMat &frame,
                                  [[maybe_unused]] const PipelineContext &ctx,
                                  const bool is_detection_interesting) {
  using namespace std::chrono;

  m_frames_queue.push({frame, ctx});
  while (m_frames_queue.size() >
         static_cast<size_t>(m_video_precapture_frames)) {
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
    try {
      // m_ram_buf = std::make_unique<Utils::RamVideoBuffer>();
      // cv::cudacodec::createVideoWriter expects a .mp4 file path, we must
      // emulate this behavior...
      m_temp_video_path =
          fmt::format("/tmp/nvenc_buffer_{}.mp4",
                      boost::uuids::to_string(m_uuid_generator()));

      cv::cudacodec::EncoderParams params;
      // 2. Set Rate Control to Variable Bitrate
      params.rateControlMode = cv::cudacodec::ENC_PARAMS_RC_VBR;
      params.targetQuality = m_target_quality;
      m_writer = cv::cudacodec::createVideoWriter(
          m_temp_video_path, frame.size(), cv::cudacodec::Codec::HEVC, m_fps,
          cv::cudacodec::ColorFormat::BGR, params);

      m_current_video_start_at = m_frames_queue.front().ctx.capture_timestamp;
      m_current_video_without_detection_frames = 0;
      m_current_video_frame_count = 0;
      m_max_roi_score = -1;
      SPDLOG_INFO("Start video recording for matrix message, saved file to "
                  "temp_video_path {}, video_max_length(sec): {}",
                  m_temp_video_path, m_video_max_length);
      m_state = Utils::VideoRecordingState::RECORDING;
    } catch (const cv::Exception &e) {
      SPDLOG_ERROR("cv::cudacodec::VideoWriter({}) failed: {}",
                   m_temp_video_path, e.what());
      m_writer.release();
      m_writer = nullptr;
      disable();
      unlink(m_temp_video_path.c_str());
      SPDLOG_WARN("{} turned off", m_unit_path);
      return;
    }
  }

  const auto is_max_length_reached_or_program_exiting =
      steady_clock::now() - m_current_video_start_at >= m_video_max_length ||
      ev_flag != 0;
  const auto is_detection_gap_tolerance_reached =
      m_current_video_without_detection_frames++ >=
      m_detections_gap_tolerance_frames;

  if (is_max_length_reached_or_program_exiting ||
      is_detection_gap_tolerance_reached) {

    m_writer.release();
    SPDLOG_INFO("video stopped, "
                "is_max_length_reached_or_program_exiting: {}, "
                "is_detection_gap_tolerance_reached: {}",
                is_max_length_reached_or_program_exiting,
                is_detection_gap_tolerance_reached);
    m_state = Utils::VideoRecordingState::IDLE;
    // creating a shared_ptr from *this, s.t. the detach()'ed t will never
    // access *this after *this is deleted.
    auto self_ptr = shared_from_this();
    std::thread(finalize_video_then_send_out, m_temp_video_path, self_ptr)
        .detach();
    return;
  }
  if (const auto roi_value = calculate_roi_score(m_frames_queue.front().ctx);
      roi_value > m_max_roi_score) {
    m_max_roi_score_frame = m_frames_queue.front().frame;
    m_max_roi_score = roi_value;
  }

  m_writer->write(m_frames_queue.front().frame);
  m_frames_queue.pop();
  ++m_current_video_frame_count;
  if (is_detection_interesting &&
      ctx.change_rate > m_maintenance_min_frame_change_rate)
    m_current_video_without_detection_frames = 0;
}

float MatrixNotifier::calculate_roi_score(const PipelineContext &ctx) {
  const auto yolo = ctx.yolo;
  float roi_value = 0.0;
  for (const auto idx : yolo.indices) {
    if (yolo.class_ids[idx] == 0 && yolo.is_detection_interesting[idx]) {
      roi_value += yolo.boxes[idx].area() * yolo.confidences[idx] *
                   pow(yolo.indices.size(), 0.5);
    }
  }

  for (size_t i = 0; i < ctx.sface.results.size(); ++i) {
    roi_value += ctx.yunet[i].bbox.area() *
                 ctx.sface.results[i].similarity_score *
                 m_identity_to_weight_map.at(ctx.sface.results[i].category);
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
  m_detections_gap_tolerance_frames = config.value(
      "detectionsGapToleranceFrames", m_detections_gap_tolerance_frames);
  m_video_precapture_frames =
      config.value("videoPrecaptureFrames", m_video_precapture_frames);
  m_video_postcapture_frames =
      config.value("videoPostcaptureFrames", m_video_postcapture_frames);
  SPDLOG_INFO("matrix_homeserver: {}, matrix_room_id: {}, matrix_access_token: "
              "{}",
              m_matrix_homeserver, m_matrix_room_id, m_matrix_access_token);

  SPDLOG_INFO(
      "video_max_length(sec): {}, m_detections_gap_tolerance_frames: {}, "
      "video_precapture_frames: {}, video_postcapture_frames: {}, fps: {}, "
      "activation_min_frame_change_rate: {}, "
      "maintenance_min_frame_change_rate: {}, target_quality: {} (0-51, "
      "lower is better)",
      m_video_max_length, m_detections_gap_tolerance_frames,
      m_video_precapture_frames, m_video_postcapture_frames, m_fps,
      m_activation_min_frame_change_rate, m_maintenance_min_frame_change_rate,
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
  const auto is_detection_interesting = look_for_interesting_detection(ctx);

  handle_video(frame, ctx, is_detection_interesting);
}

} // namespace MatrixPipeline::ProcessingUnit