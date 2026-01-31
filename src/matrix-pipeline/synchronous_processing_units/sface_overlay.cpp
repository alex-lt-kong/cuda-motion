#include "sface_overlay.h"

#include <fmt/format.h> // Assuming fmt is available given modern C++ usage, or use std::format
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

bool SFaceOverlay::init(const njson &config) {
  identity_to_box_color_bgr[IdentityCategory::Unknown] =
      cv::Scalar(127, 127, 127);
  identity_to_box_color_bgr[IdentityCategory::Unauthorized] =
      cv::Scalar(0, 204, 255);
  identity_to_box_color_bgr[IdentityCategory::Authorized] =
      cv::Scalar(0, 181, 0);

  try {
    m_bounding_box_border_thickness = config.value(
        "boundingBoxBorderThickness", m_bounding_box_border_thickness);
    m_label_font_scale = config.value("labelFontScale", m_label_font_scale);
    m_label_font_thickness =
        config.value("labelFontThickness", m_label_font_thickness);

    SPDLOG_INFO("bounding_box_border_thickness: {}, label_font_scale: {}, "
                "label_font_thickness: {}",
                m_bounding_box_border_thickness, m_label_font_scale,
                m_label_font_thickness);
    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("Failed to init SFaceOverlay: {}", e.what());
    return false;
  }
}

SynchronousProcessingResult SFaceOverlay::process(cv::cuda::GpuMat &frame,
                                                  PipelineContext &ctx) {

  std::string yunet_text_to_overlay = "YuNet: ";
  std::string sface_text_to_overlay = "SFace: ";
  // ctx.text_to_overlay += "SFace: ";
  // We also check YuNet because SFace results rely on YuNet geometry.
  if (ctx.yunet_sface.results.empty()) {
    ctx.text_to_overlay +=
        fmt::format("{}: <None>\n{}: <None>\n", yunet_text_to_overlay,
                    sface_text_to_overlay);
    return failure_and_continue;
  }

  frame.download(m_pinned_mem_for_cpu_frame);
  // point the cv::Mat directly to the pinned memory
  m_frame_cpu = m_pinned_mem_for_cpu_frame.createMatHeader();

  // std::string text_to_overlay;
  for (const auto &[detection, recognition] : ctx.yunet_sface.results) {

    yunet_text_to_overlay +=
        fmt::format("conf: {:.2f}, ", detection.face_score);
    // Convert Rect2f (float) to Rect (int) for cleaner pixel drawing
    const cv::Rect bounding_box = detection.bounding_box;

    // Draw the bounding box
    cv::rectangle(m_frame_cpu, bounding_box,
                  identity_to_box_color_bgr[recognition.category],
                  m_bounding_box_border_thickness);

    sface_text_to_overlay +=
        fmt::format("{}: {{cos: {:.2f}, L2: {:.1f}}}, ", recognition.identity,
                    recognition.cosine_score, recognition.l2_norm);
    std::string bounding_box_label_text =
        recognition.category != IdentityCategory::Unknown
            ? fmt::format("{}", recognition.identity)
            : "?";

    // Calculate text position (put it slightly above the box)
    int baseLine;
    cv::Size label_size =
        cv::getTextSize(bounding_box_label_text, cv::FONT_HERSHEY_SIMPLEX,
                        m_label_font_scale, m_label_font_thickness, &baseLine);

    const auto label_x = bounding_box.x;
    // Ensure it doesn't go off-screen top
    const auto label_y = std::max(bounding_box.y - 5, label_size.height);

    // Draw label box
    cv::rectangle(m_frame_cpu, cv::Point(label_x, label_y - label_size.height),
                  cv::Point(label_x + label_size.width, label_y + baseLine),
                  identity_to_box_color_bgr[recognition.category], cv::FILLED);
    // Draw label text
    cv::putText(m_frame_cpu, bounding_box_label_text,
                cv::Point(label_x, label_y), cv::FONT_HERSHEY_SIMPLEX,
                m_label_font_scale, m_text_color_bgr, m_label_font_thickness);
  }

  if (yunet_text_to_overlay.length() >= 2) {
    yunet_text_to_overlay.resize(yunet_text_to_overlay.length() - 2);
  }
  ctx.text_to_overlay += yunet_text_to_overlay + "\n";

  if (sface_text_to_overlay.length() >= 2) {
    sface_text_to_overlay.resize(sface_text_to_overlay.length() - 2);
  }
  ctx.text_to_overlay += sface_text_to_overlay + "\n";

  frame.upload(m_frame_cpu);

  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit