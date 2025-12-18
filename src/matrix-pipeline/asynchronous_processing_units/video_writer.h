#pragma once

#include "../interfaces/i_asynchronous_processing_unit.h"
#include "../entities/video_recording_state.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/fmt/chrono.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <ctime>
#include <deque>
#include <regex>

namespace MatrixPipeline::ProcessingUnit {

struct VideoWriterConfig {
  std::string m_file_path_template;
  double m_change_rate_threshold = 0.0;
  int m_cool_off_sec = 30;
  int m_max_length_sec = 60;
  double m_target_fps = 30.0;
  int m_codec_fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');

  // N frames to keep
  int m_pre_record_frames = 30 * 5;
};

class VideoWriter final : public IAsynchronousProcessingUnit {
public:

  explicit VideoWriter(const std::string &unit_path) : IAsynchronousProcessingUnit(unit_path  + "/VideoWriter") {}
  ~VideoWriter() override {
    if (m_writer) {
      m_writer.release();
      SPDLOG_INFO("GPU Video writer released in destructor.");
    }
  }

  bool init(const njson &config) override {
    try {
      m_config.m_file_path_template =
          config.value("filePath", m_config.m_file_path_template);
      m_config.m_change_rate_threshold =
          config.value("changeRateThreshold", m_config.m_change_rate_threshold);
      m_config.m_cool_off_sec =
          config.value("coolOffSec", m_config.m_cool_off_sec);
      m_config.m_max_length_sec =
          config.value("maxLengthSec", m_config.m_max_length_sec);
      m_config.m_target_fps = config.value("targetFps", m_config.m_target_fps);

      m_config.m_pre_record_frames = config.value("preRecordFrames", 0);

      SPDLOG_INFO("change_rate_threshold: {}, pre_record_frames: {} frames, "
                  "cool_off_sec: {}, max_length_sec: {}",
                  m_config.m_change_rate_threshold,
                  m_config.m_pre_record_frames, m_config.m_cool_off_sec,
                  m_config.m_max_length_sec);
      return true;
    } catch (const njson::exception &e) {
      SPDLOG_ERROR("Failed to parse JSON config: {}", e.what());
      return false;
    }
  }

  static std::string
  generate_filename(const std::string &path_template,
                    std::chrono::system_clock::time_point timestamp) {
    std::string filename = path_template;
    static const std::regex placeholder_regex(
        R"(\{videoStartTime(?::([^}]+))?\})");
    std::smatch match;
    if (std::regex_search(filename, match, placeholder_regex)) {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    timestamp.time_since_epoch()) %
                1000;
      auto t_c = std::chrono::system_clock::to_time_t(timestamp);
      std::tm tm = *std::localtime(&t_c);
      std::string time_format = "{:%Y%m%d_%H%M%S}";
      if (match[1].matched) {
        std::string user_fmt = match[1].str();
        size_t f_pos = user_fmt.find("%f");
        if (f_pos != std::string::npos)
          user_fmt.replace(f_pos, 2, fmt::format("{:03d}", ms.count()));
        time_format = "{:" + user_fmt + "}";
      }
      try {
        std::string formatted_time = fmt::format(fmt::runtime(time_format), tm);
        filename.replace(match.position(), match.length(), formatted_time);
      } catch (const std::exception &e) {
        return "error.mp4";
      }
    } else {
      auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        timestamp.time_since_epoch())
                        .count();
      if (filename.find(".mp4") == std::string::npos)
        filename += "_" + std::to_string(now_ms) + ".mp4";
      else
        filename.insert(filename.rfind('.'), "_" + std::to_string(now_ms));
    }
    return filename;
  }

protected:
  // Renamed from process() to on_frame_ready()
  void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) override {
    if (frame.empty() || m_state == Utils::VideoRecordingState::DISABLED)
      return;

    const double change_rate = ctx.change_rate;
    const auto now = std::chrono::steady_clock::now();

    // --- IDLE STATE ---
    if (m_state == Utils::VideoRecordingState::IDLE) {

      // 1. Maintain Pre-Roll Buffer
      if (m_config.m_pre_record_frames > 0) {
        // Just push 'frame'. GpuMat acts like a shared_ptr/ref-counted object.
        m_pre_roll_buffer.push_back(frame);

        // Remove old frames
        while (m_pre_roll_buffer.size() >
               static_cast<size_t>(m_config.m_pre_record_frames)) {
          m_pre_roll_buffer.pop_front();
        }
      }

      // 2. Check Trigger
      if (change_rate >= m_config.m_change_rate_threshold &&
          ctx.captured_from_real_device &&
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch())
                      .count() -
                  ctx.capture_from_this_device_since_ms >
              10000) {
        if (!start_recording(frame.size()))
          return;

        // 3. FLUSH Pre-Roll Buffer to Writer
        if (!m_pre_roll_buffer.empty()) {
          SPDLOG_INFO("Flushing {} pre-roll frames.", m_pre_roll_buffer.size());
          while (!m_pre_roll_buffer.empty()) {
            write_frame(m_pre_roll_buffer.front());
            m_pre_roll_buffer.pop_front();
          }
        }

        m_state = Utils::VideoRecordingState::RECORDING;
        m_record_start_time = now;
        SPDLOG_INFO("Recording started (change_rate_threshold({}) vs "
                    "change_rate({:.3})), writing video to {}",
                    m_config.m_change_rate_threshold, change_rate, m_file_path);
      }
    }

    // --- RECORDING STATE ---
    if (m_state == Utils::VideoRecordingState::RECORDING) {
      const bool threshold_met =
          change_rate >= m_config.m_change_rate_threshold;

      if (!threshold_met) {
        if (m_last_below_threshold_time ==
            std::chrono::steady_clock::time_point{}) {
          m_last_below_threshold_time = now;
        }
      } else {
        m_last_below_threshold_time = std::chrono::steady_clock::time_point{};
      }

      const auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
                                   now - m_record_start_time)
                                   .count();
      const auto cooldown_sec =
          std::chrono::duration_cast<std::chrono::seconds>(
              now - m_last_below_threshold_time)
              .count();

      if (elapsed_sec >= m_config.m_max_length_sec ||
          (!threshold_met && cooldown_sec >= m_config.m_cool_off_sec)) {
        stop_recording();
        m_state = Utils::VideoRecordingState::IDLE;
        return;
      }

      write_frame(frame);
    }
  }

private:


  Utils::VideoRecordingState m_state = Utils::VideoRecordingState::IDLE;
  VideoWriterConfig m_config;
  cv::Ptr<cv::cudacodec::VideoWriter> m_writer;

  // Stores GpuMat references.
  std::deque<cv::cuda::GpuMat> m_pre_roll_buffer;

  std::chrono::steady_clock::time_point m_record_start_time;
  std::chrono::steady_clock::time_point m_last_below_threshold_time;
  std::string m_file_path;

  [[nodiscard]] std::string generate_filename() const {
    return generate_filename(m_config.m_file_path_template,
                             std::chrono::system_clock::now());
  }

  bool start_recording(const cv::Size frame_size) {
    if (m_writer)
      m_writer.release();
    m_file_path = generate_filename();
    try {
      SPDLOG_INFO("cv::cudacodec::createVideoWriter({})ing", m_file_path);
      m_writer = cv::cudacodec::createVideoWriter(
          m_file_path, frame_size, cv::cudacodec::Codec::H264,
          m_config.m_target_fps, cv::cudacodec::ColorFormat::BGR);
    } catch (const cv::Exception &e) {
      SPDLOG_ERROR("cv::cudacodec::createVideoWriter({}) failed: {}",
                   m_file_path, e.what());
      m_writer.release();
      m_state = Utils::VideoRecordingState::DISABLED;
      SPDLOG_WARN("Disabling videoWriter uni");
      return false;
    }
    return m_writer != nullptr;
  }

  void stop_recording() {
    if (m_writer) {
      m_writer.release();
      SPDLOG_INFO("Recording stopped, video written to {}", m_file_path);
    }
    m_record_start_time = {};
    m_last_below_threshold_time = {};
    m_pre_roll_buffer.clear();
  }

  void write_frame(const cv::cuda::GpuMat &frame) const {
    if (!m_writer)
      return;
    m_writer->write(frame);
  }
};

} // namespace MatrixPipeline::ProcessingUnit