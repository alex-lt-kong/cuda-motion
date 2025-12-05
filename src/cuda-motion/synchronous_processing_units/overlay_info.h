#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <nlohmann/json.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <algorithm>

using njson = nlohmann::json;

namespace CudaMotion::ProcessingUnit {

class OverlayInfo final : public ISynchronousProcessingUnit {
private:
  // --- Configuration Flags ---
  bool m_show_stats{true};
  bool m_show_device_name{true};
  bool m_show_date_time{true};
  bool m_show_outline{true};

  // --- Visual Settings ---
  // If > 0, calculates size based on frame height %.
  // Default 0.015 = 1.5% of frame height.
  float m_text_height_ratio{0.015f};

  // Absolute size in PIXELS. Used if m_text_height_ratio is 0.
  // Default to 24 pixels (readable on most 1080p screens).
  int m_target_font_size_px{24};

  // Calculated per-frame for OpenCV drawing
  float m_current_opencv_scale{1.0f};
  int m_current_thickness{2};

  cv::Scalar m_text_color{255, 255, 255};
  cv::Scalar m_glow_color{2, 2, 2};

  // --- Data: Device ---
  std::string m_device_name{"Unknown Device"};

  // --- Reusable Buffers ---
  cv::Mat h_text_strip;
  cv::cuda::GpuMat d_text_strip;
  cv::cuda::GpuMat d_strip_gray;
  cv::cuda::GpuMat d_mask;

  // Constant: The base pixel height of FONT_HERSHEY_DUPLEX at scale 1.0
  static constexpr float BASE_FONT_HEIGHT_PX = 22.0f;

public:
  inline OverlayInfo() = default;
  inline ~OverlayInfo() override = default;

  /**
   * @brief Configures overlay.
   * JSON Keys:
   * - fontSizePx: (int) Height in Pixels (e.g., 32). Disables dynamic ratio.
   * - textHeightRatio: (float) Height relative to frame (e.g., 0.02).
   */
  bool init(const njson &config) override {
    try {
      // 1. Load Flags
      if (config.contains("showStats")) m_show_stats = config["showStats"].get<bool>();
      if (config.contains("showDeviceName")) m_show_device_name = config["showDeviceName"].get<bool>();
      if (config.contains("showDateTime")) m_show_date_time = config["showDateTime"].get<bool>();
      if (config.contains("showOutline")) m_show_outline = config["showOutline"].get<bool>();

      // 2. Load Visual Config

      // OPTION A: Absolute Pixels
      if (config.contains("fontSizePx")) {
          m_target_font_size_px = config["fontSizePx"].get<int>();
          // Safety: Don't allow invisible text
          if (m_target_font_size_px < 8) m_target_font_size_px = 8;

          // Disable dynamic ratio since user requested specific pixel size
          m_text_height_ratio = 0.0f;
      }

      // OPTION B: Relative Ratio (Takes precedence if both provided)
      if (config.contains("textHeightRatio")) {
          m_text_height_ratio = config["textHeightRatio"].get<float>();
      }

      // Manual thickness override (optional)
      if (config.contains("thickness")) {
          // If user hardcodes thickness, we might store it,
          // but usually dynamic thickness is better.
          // We'll treat this as a "base" but calculation below might override.
          m_current_thickness = config["thickness"].get<int>();
      }

      // 3. Load Device Data
      if (config.contains("deviceName")) m_device_name = config["deviceName"].get<std::string>();
      SPDLOG_INFO("textHeightRatio: {}, fontSizePx: {}", m_text_height_ratio, m_target_font_size_px);
      return true;
    } catch (const std::exception &e) {
      SPDLOG_ERROR("OverlayInfo::init: {}", e.what());
      return false;
    }
  }

  [[nodiscard]] SynchronousProcessingResult process(cv::cuda::GpuMat &frame, ProcessingMetaData& meta_data) override {
    if (frame.empty()) return failure_and_continue;

    // --- 1. Determine Target Pixel Height ---
    float final_px_height = 0.0f;

    if (m_text_height_ratio > 0.0f) {
        // Dynamic Mode
        final_px_height = frame.rows * m_text_height_ratio;
    } else {
        // Absolute Mode
        final_px_height = static_cast<float>(m_target_font_size_px);
    }

    // Clamp minimum readable size (approx 10px)
    if (final_px_height < 10.0f) final_px_height = 10.0f;

    // --- 2. Convert Pixels -> OpenCV Scale ---
    // Scale = TargetPixels / BasePixels
    m_current_opencv_scale = final_px_height / BASE_FONT_HEIGHT_PX;

    // --- 3. Calculate Thickness ---
    // Heuristic: Thicker font for larger text.
    // Approx 1 thickness per 20px height looks decent for Duplex.
    m_current_thickness = std::max(1, static_cast<int>(final_px_height / 20.0f));

    // --- 4. Prepare Strip ---
    // Strip height is Font Height + Padding (1.5x)
    int stripHeight = static_cast<int>(final_px_height * 1.5f);
    if (stripHeight > frame.rows) stripHeight = frame.rows;

    cv::Size stripSize(frame.cols, stripHeight);
    if (h_text_strip.size() != stripSize || h_text_strip.type() != frame.type()) {
      h_text_strip.create(stripSize, frame.type());
    }

    // Center text vertically in the strip
    // Baseline offset approx same as font height for simple centering
    int textY = (stripHeight + static_cast<int>(final_px_height)) / 2;
    // Slight adjustment for baseline rendering
    textY -= static_cast<int>(final_px_height * 0.1f);

    // --- Draw Bottom Strip ---
    if (m_show_stats || m_show_device_name) {
      h_text_strip.setTo(cv::Scalar::all(0));

      if (m_show_stats) {
        char buff[128];
        float rate = (meta_data.change_rate >= 0) ? meta_data.change_rate * 100.0f : 0.0f;
        snprintf(buff, sizeof(buff) - 1, "%.2f%%, %.1ffps", rate, meta_data.fps);
        drawText(h_text_strip, buff, cv::Point(10, textY));
      }

      if (m_show_device_name) {
        int bl = 0;
        cv::Size textSize = cv::getTextSize(m_device_name, cv::FONT_HERSHEY_DUPLEX, m_current_opencv_scale, m_current_thickness, &bl);
        int textX = frame.cols - textSize.width - 10;
        drawText(h_text_strip, m_device_name, cv::Point(textX, textY));
      }

      uploadAndOverlay(frame, cv::Rect(0, frame.rows - stripHeight, frame.cols, stripHeight));
    }

    // --- Draw Top Strip ---
    if (m_show_date_time) {
      h_text_strip.setTo(cv::Scalar::all(0));

      std::time_t now_c;
      if (meta_data.capture_timestamp_ms > 0) {
          auto duration = std::chrono::milliseconds(meta_data.capture_timestamp_ms);
          auto time_point = std::chrono::system_clock::time_point(duration);
          now_c = std::chrono::system_clock::to_time_t(time_point);
      } else {
          now_c = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
      }

      std::tm now_tm;
      localtime_r(&now_c, &now_tm);

      std::stringstream ss;
      ss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");

      drawText(h_text_strip, ss.str(), cv::Point(10, textY));

      uploadAndOverlay(frame, cv::Rect(0, 0, frame.cols, stripHeight));
    }

    return success_and_continue;
  }

private:
  void drawText(cv::Mat &img, const std::string &text, cv::Point org) {
    if (m_show_outline) {
      cv::putText(img, text, org, cv::FONT_HERSHEY_DUPLEX, m_current_opencv_scale,
                  m_glow_color, m_current_thickness * 5, cv::LINE_8, false);
    }
    cv::putText(img, text, org, cv::FONT_HERSHEY_DUPLEX, m_current_opencv_scale,
                m_text_color, m_current_thickness, cv::LINE_8, false);
  }

  void uploadAndOverlay(cv::cuda::GpuMat &frame, cv::Rect roiRect) {
    d_text_strip.upload(h_text_strip);
    if (d_text_strip.channels() > 1) {
      cv::cuda::cvtColor(d_text_strip, d_strip_gray, cv::COLOR_BGR2GRAY);
    } else {
      d_strip_gray = d_text_strip;
    }
    cv::cuda::threshold(d_strip_gray, d_mask, 0, 255, cv::THRESH_BINARY);
    cv::cuda::GpuMat roi = frame(roiRect);
    d_text_strip.copyTo(roi, d_mask);
  }
};

} // namespace CudaMotion::ProcessingUnit