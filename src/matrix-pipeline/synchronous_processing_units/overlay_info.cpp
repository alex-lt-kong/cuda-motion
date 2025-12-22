#include "overlay_info.h"
#include "../utils.h"

#include <fmt/chrono.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <sstream>
#include <vector>

namespace MatrixPipeline::ProcessingUnit {

OverlayInfo::~OverlayInfo() = default;

bool OverlayInfo::init(const nlohmann::json &config) {
  try {
    m_text_height_ratio = config.value("textHeightRatio", m_text_height_ratio);
    m_outline_ratio = config.value("outlineRatio", m_outline_ratio);

    m_format_template =
        config.value("text", "{deviceName},\nChg: {changeRate:.2f}, FPS: "
                             "{fps:.1f}\n{frameCaptureTime:%Y-%m-%d %H:%M:%S}");

    SPDLOG_INFO("outline_ratio: {}, text_height_ratio: {}, format_template: {:?}",
                m_outline_ratio, m_text_height_ratio, m_format_template);
    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("OverlayInfo init error: {}", e.what());
    return false;
  }
}

SynchronousProcessingResult OverlayInfo::process(cv::cuda::GpuMat &frame,
                                                 PipelineContext &ctx) {
  if (frame.empty())
    return success_and_continue;
  using namespace std::chrono_literals;
  if (std::chrono::steady_clock::now() - m_last_info_update_time > 1s) {
    m_last_info_update_time = std::chrono::steady_clock::now();
    // --- 1. Prepare Data & Format String ---
    const auto now_tp = Utils::steady_clock_to_system_time(ctx.capture_timestamp);
    // const auto now_tp = std::chrono::clock_cast<std::chrono::system_clock>(ctx.capture_timestamp);

    // Convert to Local Time
    std::time_t now_c = std::chrono::system_clock::to_time_t(now_tp);
    std::tm now_tm;
    // POSIX thread-safe localtime
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
    update_font_metrics(frame.rows);

    // --- 4. Prepare Rendering Surface ---
    m_stripHeight =
        (static_cast<int>(lines.size()) * m_line_height_px) + (2 * m_margin_y);
    if (m_stripHeight > frame.rows)
      m_stripHeight = frame.rows;

    if (m_h_text_strip.cols != frame.cols || m_h_text_strip.rows != m_stripHeight ||
        m_h_text_strip.type() != CV_8UC3) {
      m_h_text_strip.create(m_stripHeight, frame.cols, CV_8UC3);
    }

    m_h_text_strip.setTo(cv::Scalar::all(0));

    // --- 5. Draw Lines (Left Aligned) ---
    int currentY = m_margin_y + static_cast<int>(BASE_FONT_HEIGHT_PX *
                                                 m_current_opencv_scale);
    int x = m_margin_x;

    for (const auto &txt : lines) {
      if (txt.empty()) {
        currentY += m_line_height_px;
        continue;
      }

      cv::Point org(x, currentY);

      if (m_outline_ratio > 0.0f) {
        // Draw thicker outline behind
        cv::putText(m_h_text_strip, txt, org, cv::FONT_HERSHEY_DUPLEX,
                    m_current_opencv_scale, m_glow_color,
                    m_current_outline_thickness, cv::LINE_AA);
      }
      // Draw main text
      cv::putText(m_h_text_strip, txt, org, cv::FONT_HERSHEY_DUPLEX,
                  m_current_opencv_scale, m_text_color, m_current_thickness,
                  cv::LINE_AA);

      currentY += m_line_height_px;
    }
  }
  // --- 6. Upload and Overlay ---
  uploadAndOverlay(frame, cv::Rect(0, 0, frame.cols, m_stripHeight));

  return success_and_continue;
}

void OverlayInfo::update_font_metrics(const int frameRows) {
  // 1. Calculate base text height
  float final_px_height = std::max(frameRows * m_text_height_ratio, 6.0f);

  m_current_opencv_scale = final_px_height / BASE_FONT_HEIGHT_PX;
  m_current_thickness = std::max(1, static_cast<int>(final_px_height / 20.0f));

  // 2. Calculate Border Size
  int border_px = 0;
  if (m_outline_ratio > 0.0f) {
    border_px = static_cast<int>(final_px_height * m_outline_ratio);
    // Safety: Ensure at least 1px if ratio is set
    if (border_px < 1)
      border_px = 1;

    m_current_outline_thickness = m_current_thickness + (2 * border_px);
  } else {
    m_current_outline_thickness = 0;
  }

  // 3. Fix Line Height
  // Logic: (Font Height + Standard Spacing) + (Top Border + Bottom Border)
  m_line_height_px = static_cast<int>(final_px_height * 1.2f) + (2 * border_px);
}

void OverlayInfo::uploadAndOverlay(cv::cuda::GpuMat &frame, cv::Rect roiRect) {
  cv::Rect validRoi = roiRect & cv::Rect(0, 0, frame.cols, frame.rows);
  if (validRoi.empty())
    return;

  cv::Mat cpuSrc =
      m_h_text_strip(cv::Rect(0, 0, validRoi.width, validRoi.height));
  d_text_strip.upload(cpuSrc);

  cv::cuda::cvtColor(d_text_strip, d_strip_gray, cv::COLOR_BGR2GRAY);
  cv::cuda::threshold(d_strip_gray, d_mask, 1, 255, cv::THRESH_BINARY);

  cv::cuda::GpuMat roi = frame(validRoi);
  d_text_strip.copyTo(roi, d_mask);
}

} // namespace MatrixPipeline::ProcessingUnit