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
#include <algorithm> // For std::max

using njson = nlohmann::json;

namespace CudaMotion::ProcessingUnit {

class OverlayInfo final : public ISynchronousProcessingUnit {
private:
  // --- Configuration Flags ---
  bool m_show_stats{true};
  bool m_show_device_name{true};
  bool m_show_date_time{true};
  bool m_show_outline{true}; // Controls the black glow

  // --- Visual Settings ---
  // If m_text_height_ratio > 0, font scale is calculated dynamically based on frame height.
  // Default is 0.015 (1.5%) for less intrusive text.
  float m_text_height_ratio{0.015f};
  float m_font_scale{1.0f};
  int m_font_thickness{2};

  cv::Scalar m_text_color{255, 255, 255}; // White default
  // We use (2,2,2) instead of (0,0,0) because (0,0,0) is used as the "transparent" key
  // in the masking logic. (2,2,2) is visually indistinguishable from black.
  cv::Scalar m_glow_color{2, 2, 2};

  // --- Data: Stats (Stored locally except change_rate and fps) ---
  // Note: change_rate and fps come from ProcessingMetaData passed to process()
  int m_cd{0};
  long long int m_video_frame_count{0};
  uint32_t m_max_frames_per_video{0};

  // --- Data: Device ---
  std::string m_device_name{"Unknown Device"};

  // --- Reusable Buffers (CPU & GPU) ---
  cv::Mat h_text_strip;           // Host buffer for drawing text strips
  cv::cuda::GpuMat d_text_strip;  // Device buffer for upload
  cv::cuda::GpuMat d_strip_gray;  // Intermediate gray for mask generation
  cv::cuda::GpuMat d_mask;        // Binary mask for transparency

public:
  inline OverlayInfo() = default;
  inline ~OverlayInfo() override = default;

  /**
   * @brief Initializes stats, device info, and display flags from JSON.
   * Expected JSON structure (camelCase keys):
   * {
   * "showStats": true,
   * "showDeviceName": true,
   * "showDateTime": true,
   * "showOutline": true,
   * "deviceName": "Camera 1",
   * "textHeightRatio": 0.015, // 1.5% of image height
   * ...
   * }
   */
  bool init(const njson &config) override {
    try {
      // 1. Load Flags
      if (config.contains("showStats")) m_show_stats = config["showStats"].get<bool>();
      if (config.contains("showDeviceName")) m_show_device_name = config["showDeviceName"].get<bool>();
      if (config.contains("showDateTime")) m_show_date_time = config["showDateTime"].get<bool>();
      if (config.contains("showOutline")) m_show_outline = config["showOutline"].get<bool>();

      // 2. Load Visual Config
      // We prioritize textHeightRatio. If user explicitly sets fontScale, we might disable dynamic ratio by setting it to 0.
      if (config.contains("textHeightRatio")) m_text_height_ratio = config["textHeightRatio"].get<float>();

      // Fallback/Legacy: If specific scale/thickness provided, they are base values,
      // but ratio logic usually overrides scale.
      if (config.contains("fontScale")) m_font_scale = config["fontScale"].get<float>();
      if (config.contains("thickness")) m_font_thickness = config["thickness"].get<int>();

      // 3. Load Device Data
      if (config.contains("deviceName")) m_device_name = config["deviceName"].get<std::string>();

      // 4. Load Initial Stats (changeRate and currentFps removed as they are dynamic)
      if (config.contains("cd")) m_cd = config["cd"].get<int>();
      if (config.contains("videoFrameCount")) m_video_frame_count = config["videoFrameCount"].get<long long int>();
      if (config.contains("maxFramesPerVideo")) m_max_frames_per_video = config["maxFramesPerVideo"].get<uint32_t>();

      return true;
    } catch (const std::exception &e) {
      return false;
    }
  }

  [[nodiscard]] SynchronousProcessingResult process(cv::cuda::GpuMat &frame, ProcessingMetaData& meta_data) override {
    if (frame.empty()) return failure_and_continue;

    // --- Dynamic Font Calculation ---
    // Calculate font scale based on image height to ensure legibility.
    // Base height of FONT_HERSHEY_DUPLEX is approx 22 pixels at scale 1.0.
    if (m_text_height_ratio > 0.0f) {
        float target_pixel_height = frame.rows * m_text_height_ratio;
        // Simple heuristic: scale = target_px / 22.0
        m_font_scale = target_pixel_height / 22.0f;
        // Ensure scale isn't too small (lowered clamp to 0.3 to allow smaller text on low-res)
        if (m_font_scale < 0.3f) m_font_scale = 0.3f;

        // Dynamically adjust thickness to match scale (thinner for small text, thicker for large)
        m_font_thickness = std::max(1, static_cast<int>(m_font_scale * 2));
    }

    // Define strip height with some padding based on the calculated font scale
    // 22px base height * scale + padding
    int stripHeight = static_cast<int>((22.0f * m_font_scale) * 1.5f);
    if (stripHeight > frame.rows) stripHeight = frame.rows;

    // Ensure reusable buffers are allocated correctly
    cv::Size stripSize(frame.cols, stripHeight);
    if (h_text_strip.size() != stripSize || h_text_strip.type() != frame.type()) {
      h_text_strip.create(stripSize, frame.type());
    }

    // --- 1. Bottom Strip: Stats (Left) & Device Name (Right) ---
    if (m_show_stats || m_show_device_name) {
      h_text_strip.setTo(cv::Scalar::all(0)); // Clear buffer (black background)

      // Calculate vertical center for text
      int baseline = 0;
      // We use a dummy getTextSize just to get baseline info if needed, or rely on centering logic
      int textY = (stripHeight + static_cast<int>(22.0f * m_font_scale)) / 2;

      // Draw Stats (Left Aligned)
      if (m_show_stats) {
        char buff[128];
        float rate = (meta_data.change_rate >= 0) ? meta_data.change_rate * 100.0f : 0.0f;

        // Use meta_data.fps directly
        snprintf(buff, sizeof(buff) - 1, "%.2f%%, %.1ffps (%d, %lld)",
                 rate, meta_data.fps, m_cd, m_max_frames_per_video - m_video_frame_count);

        drawText(h_text_strip, buff, cv::Point(5, textY));
      }

      // Draw Device Name (Right Aligned)
      if (m_show_device_name) {
        int bl = 0;
        cv::Size textSize = cv::getTextSize(m_device_name, cv::FONT_HERSHEY_DUPLEX, m_font_scale, m_font_thickness, &bl);
        // Align to right edge with a margin related to font size
        int textX = frame.cols - static_cast<int>(textSize.width + (10.0f * m_font_scale));
        drawText(h_text_strip, m_device_name, cv::Point(textX, textY));
      }

      // Upload strip to GPU and overlay it onto the bottom of the frame
      uploadAndOverlay(frame, cv::Rect(0, frame.rows - stripHeight, frame.cols, stripHeight));
    }

    // --- 2. Top Strip: ISO DateTime (Left) ---
    if (m_show_date_time) {
      h_text_strip.setTo(cv::Scalar::all(0)); // Reuse/Clear buffer for top strip

      std::time_t now_c;
      // Use metadata timestamp if available (>0), otherwise fallback to system clock
      if (meta_data.capture_timestamp_ms > 0) {
          auto duration = std::chrono::milliseconds(meta_data.capture_timestamp_ms);
          auto time_point = std::chrono::system_clock::time_point(duration);
          now_c = std::chrono::system_clock::to_time_t(time_point);
      } else {
          now_c = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
      }

      // Generate ISO Timestamp (Linux/POSIX Thread Safe)
      std::tm now_tm;
      localtime_r(&now_c, &now_tm);

      std::stringstream ss;
      ss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");

      int textY = (stripHeight + static_cast<int>(22.0f * m_font_scale)) / 2;

      // Draw DateTime (Left Aligned)
      drawText(h_text_strip, ss.str(), cv::Point(5, textY));

      // Upload strip to GPU and overlay it onto the top of the frame
      uploadAndOverlay(frame, cv::Rect(0, 0, frame.cols, stripHeight));
    }

    return success_and_continue;
  }

  // Helper to update stats dynamically from C++ logic
  // Note: change_rate and fps are excluded here as they are passed via process() metadata
  void updateStats(int cd, long long frameCount) {
    m_cd = cd;
    m_video_frame_count = frameCount;
  }

private:
  // Internal helper to draw text with a black "glow" (thick outline)
  void drawText(cv::Mat &img, const std::string &text, cv::Point org) {
    // 1. Draw Glow/Outline (Black, Thicker)
    if (m_show_outline) {
      // Thickness * 6 ensures the black bleeds out significantly from under the white text
      cv::putText(img, text, org, cv::FONT_HERSHEY_DUPLEX, m_font_scale,
                  m_glow_color, m_font_thickness * 6, cv::LINE_8, false);
    }

    // 2. Draw Foreground (White, Normal Thickness)
    cv::putText(img, text, org, cv::FONT_HERSHEY_DUPLEX, m_font_scale,
                m_text_color, m_font_thickness, cv::LINE_8, false);
  }

  void uploadAndOverlay(cv::cuda::GpuMat &frame, cv::Rect roiRect) {
    // 1. Upload CPU strip to GPU memory
    d_text_strip.upload(h_text_strip);

    // 2. Generate Mask
    if (d_text_strip.channels() > 1) {
      cv::cuda::cvtColor(d_text_strip, d_strip_gray, cv::COLOR_BGR2GRAY);
    } else {
      d_strip_gray = d_text_strip;
    }

    // Create binary mask: Any non-zero pixel (text or glow) becomes white in mask.
    // We use threshold 0 because (0,0,0) is our transparent key, and our glow is (2,2,2).
    cv::cuda::threshold(d_strip_gray, d_mask, 0, 255, cv::THRESH_BINARY);

    // 3. Copy with Mask
    cv::cuda::GpuMat roi = frame(roiRect);
    d_text_strip.copyTo(roi, d_mask);
  }
};

} // namespace CudaMotion::ProcessingUnit