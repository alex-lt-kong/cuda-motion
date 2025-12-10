#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>

using njson = nlohmann::json;

namespace MatrixPipeline::ProcessingUnit {

class OverlayInfo final : public ISynchronousProcessingUnit {
private:
  // --- Configuration ---
  std::string m_format_template;

  // --- Visual Settings ---
  float m_text_height_ratio{0.01f}; // 1% of frame height
  int m_margin_x{5};
  int m_margin_y{5};

  // Calculated per-frame
  float m_current_opencv_scale{1.0f};
  int m_current_thickness{2};
  int m_line_height_px{0};

  cv::Scalar m_text_color{255, 255, 255};
  cv::Scalar m_glow_color{2, 2, 2};
  // 0.0 = disable. 0.25 = border is 25% of font height.
  float m_outline_ratio{0.25f};
  float m_current_outline_thickness;

  // --- Buffers ---
  cv::Mat h_text_strip;
  cv::cuda::GpuMat d_text_strip;
  cv::cuda::GpuMat d_strip_gray;
  cv::cuda::GpuMat d_mask;

  static constexpr float BASE_FONT_HEIGHT_PX = 22.0f;

public:
  inline OverlayInfo() = default;
  inline ~OverlayInfo() override = default;

  bool init(const njson &config) override {
    try {

      m_text_height_ratio = config.value("textHeightRatio", m_text_height_ratio);
      m_outline_ratio = config.value("outlineRatio", m_outline_ratio);

      m_format_template = config.value(
          "text", "{deviceName},\nChg: {changeRate:.2f}, FPS: "
                  "{fps:.1f}\n{frameCaptureTime:%Y-%m-%d %H:%M:%S}");

      SPDLOG_INFO("outline_ratio: {}, text_height_ratio: {}, format_template: {}", m_outline_ratio, m_text_height_ratio, m_format_template);
      return true;
    } catch (const std::exception &e) {
      SPDLOG_ERROR("error: {}", e.what());
      return false;
    }
  }

  [[nodiscard]] SynchronousProcessingResult
  process(cv::cuda::GpuMat &frame, PipelineContext &ctx) override {
    if (frame.empty())
      return success_and_continue;

    // --- 1. Prepare Data & Format String ---
    // A. Resolve Timestamp
    auto now_tp = (ctx.capture_timestamp_ms > 0)
                      ? std::chrono::system_clock::time_point(
                            std::chrono::milliseconds(ctx.capture_timestamp_ms))
                      : std::chrono::system_clock::now();

    // B. Convert to Local Time (std::tm)
    // This ensures {dateTimeNow:%Y-%m-%d} works correctly and uses Local Time
    // zone.
    std::time_t now_c = std::chrono::system_clock::to_time_t(now_tp);
    std::tm now_tm;
    // Note: localtime_r is the thread-safe version on Linux/POSIX.
    // If you are on Windows, use localtime_s(&now_tm, &now_c);
    localtime_r(&now_c, &now_tm);

    std::string full_text;
    try {
      full_text = fmt::format(fmt::runtime(m_format_template),
                              fmt::arg("deviceName", ctx.device_info.name),
                              fmt::arg("fps", ctx.fps),
                              fmt::arg("changeRate", ctx.change_rate),
                              fmt::arg("frameCaptureTime", now_tm));
    } catch (const std::exception &e) {
      full_text = std::string("FMT Error: ") + e.what();
    }

    if (full_text.empty())
      return success_and_continue;

    // --- 2. Split into Lines ---
    std::vector<std::string> lines;
    std::stringstream ss(full_text);
    std::string line;
    while (std::getline(ss, line, '\n')) {
      lines.push_back(line);
    }

    if (lines.empty())
      return success_and_continue;

    // --- 3. Update Font Metrics ---
    updateFontMetrics(frame.rows);

    // --- 4. Prepare Rendering Surface ---
    int stripHeight =
        (static_cast<int>(lines.size()) * m_line_height_px) + (2 * m_margin_y);
    if (stripHeight > frame.rows)
      stripHeight = frame.rows;

    if (h_text_strip.cols != frame.cols || h_text_strip.rows != stripHeight ||
        h_text_strip.type() != CV_8UC3) {
      h_text_strip.create(stripHeight, frame.cols, CV_8UC3);
    }

    h_text_strip.setTo(cv::Scalar::all(0));

    // --- 5. Draw Lines (Left Aligned) ---
    int currentY = m_margin_y + static_cast<int>(BASE_FONT_HEIGHT_PX *
                                                 m_current_opencv_scale);

    // CHANGED: Simpler X calculation for Left Alignment
    int x = m_margin_x;

    for (const auto &txt : lines) {
      if (txt.empty()) {
        currentY += m_line_height_px;
        continue;
      }

      // We no longer need to measure text width for alignment,
      // but we still call getTextSize inside putText internally by OpenCV.
      // For left alignment, X is constant.
      cv::Point org(x, currentY);

      // CHANGED: Use outlineRatio logic
      if (m_outline_ratio > 0.0f) {
        // Draw thicker outline behind
        cv::putText(h_text_strip, txt, org, cv::FONT_HERSHEY_DUPLEX, m_current_opencv_scale,
                    m_glow_color, m_current_outline_thickness, cv::LINE_AA);
      }
      // Draw main text
      cv::putText(h_text_strip, txt, org, cv::FONT_HERSHEY_DUPLEX,
                  m_current_opencv_scale, m_text_color, m_current_thickness,
                  cv::LINE_AA);

      currentY += m_line_height_px;
    }

    // --- 6. Upload and Overlay ---
    uploadAndOverlay(frame, cv::Rect(0, 0, frame.cols, stripHeight));

    return success_and_continue;
  }

private:
  void updateFontMetrics(int frameRows) {
    // 1. Calculate base text height
    float final_px_height = std::max(frameRows * m_text_height_ratio, 6.0f);

    m_current_opencv_scale = final_px_height / BASE_FONT_HEIGHT_PX;
    m_current_thickness = std::max(1, static_cast<int>(final_px_height / 20.0f));

    // 2. Calculate Border Size
    int border_px = 0;
    if (m_outline_ratio > 0.0f) {
      border_px = static_cast<int>(final_px_height * m_outline_ratio);
      // Safety: Ensure at least 1px if ratio is set
      if (border_px < 1) border_px = 1;

      m_current_outline_thickness = m_current_thickness + (2 * border_px);
    } else {
      m_current_outline_thickness = 0;
    }

    // 3. Fix Line Height
    // Logic: (Font Height + Standard Spacing) + (Top Border + Bottom Border)
    m_line_height_px = static_cast<int>(final_px_height * 1.2f) + (2 * border_px);
  }

  void uploadAndOverlay(cv::cuda::GpuMat &frame, cv::Rect roiRect) {
    cv::Rect validRoi = roiRect & cv::Rect(0, 0, frame.cols, frame.rows);
    if (validRoi.empty())
      return;

    cv::Mat cpuSrc =
        h_text_strip(cv::Rect(0, 0, validRoi.width, validRoi.height));
    d_text_strip.upload(cpuSrc);

    cv::cuda::cvtColor(d_text_strip, d_strip_gray, cv::COLOR_BGR2GRAY);
    cv::cuda::threshold(d_strip_gray, d_mask, 1, 255, cv::THRESH_BINARY);

    cv::cuda::GpuMat roi = frame(validRoi);
    d_text_strip.copyTo(roi, d_mask);
  }
};

} // namespace MatrixPipeline::ProcessingUnit