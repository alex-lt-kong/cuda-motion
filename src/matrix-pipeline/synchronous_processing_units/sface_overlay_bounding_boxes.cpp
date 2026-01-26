#include "sface_overlay_bounding_boxes.h"

#include <fmt/format.h> // Assuming fmt is available given modern C++ usage, or use std::format
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

bool SFaceOverlayBoundingBoxes::init(const njson &config) {
  identity_to_box_color_bgr[IdentityCategory::Unknown] = cv::Scalar(0, 0, 255);
  identity_to_box_color_bgr[IdentityCategory::Unauthorized] =
      cv::Scalar(0, 204, 255);
  identity_to_box_color_bgr[IdentityCategory::Authorized] =
      cv::Scalar(102, 204, 0);

  try {
    m_thickness = config.value("thickness", m_thickness);
    m_label_font_scale = config.value("labelFontScale", m_label_font_scale);
    m_font_thickness = config.value("fontThickness", m_font_thickness);

    SPDLOG_INFO("thickness: {}, label_font_scale: {}", m_thickness,
                m_label_font_scale);
    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("Failed to init SFaceOverlayBoundingBoxes: {}", e.what());
    return false;
  }
}

SynchronousProcessingResult
SFaceOverlayBoundingBoxes::process(cv::cuda::GpuMat &frame,
                                   PipelineContext &ctx) {
  // We also check YuNet because SFace results rely on YuNet geometry.
  if (ctx.sface.results.empty() || ctx.yunet.empty()) {
    return failure_and_continue;
  }

  cv::Mat cpu_frame;
  frame.download(cpu_frame);

  // 3. Iterate via index to access both Identity (SFace) and Geometry (YuNet)
  // We use the smaller size to prevent out-of-bounds access if synchronization
  // failed
  size_t count = std::min(ctx.sface.results.size(), ctx.yunet.size());

  for (size_t i = 0; i < count; ++i) {
    const auto &recognition = ctx.sface.results[i];
    const auto &detection = ctx.yunet[i];

    // Convert Rect2f (float) to Rect (int) for cleaner pixel drawing
    cv::Rect box = detection.bbox;
    // Draw the bounding box
    cv::rectangle(cpu_frame, box,
                  identity_to_box_color_bgr[recognition.category], m_thickness);

    std::string label = fmt::format(
        "{} ({:.2f})",
        recognition.category != IdentityCategory::Unknown ? recognition.identity
                                                          : "?",
        recognition.similarity_score);

    // Calculate text position (put it slightly above the box)
    int baseLine;
    cv::Size label_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, m_label_font_scale,
                        m_font_thickness, &baseLine);

    int text_x = box.x;
    // Ensure it doesn't go off-screen top
    int text_y = std::max(box.y - 5, label_size.height);

    // Draw label box
    cv::rectangle(cpu_frame, cv::Point(text_x, text_y - label_size.height),
                  cv::Point(text_x + label_size.width, text_y + baseLine),
                  identity_to_box_color_bgr[recognition.category], cv::FILLED);
    // Draw Label text
    cv::putText(cpu_frame, label, cv::Point(text_x, text_y),
                cv::FONT_HERSHEY_SIMPLEX, m_label_font_scale, m_text_color_bgr,
                m_font_thickness);
  }

  frame.upload(cpu_frame);

  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit