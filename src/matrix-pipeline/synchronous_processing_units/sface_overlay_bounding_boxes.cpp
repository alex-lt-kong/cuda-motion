#include "sface_overlay_bounding_boxes.h"

#include <fmt/format.h> // Assuming fmt is available given modern C++ usage, or use std::format
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

bool SFaceOverlayBoundingBoxes::init(const njson &config) {
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
    SPDLOG_ERROR("Failed to init SFaceOverlayBoundingBoxes: {}", e.what());
    return false;
  }
}

SynchronousProcessingResult
SFaceOverlayBoundingBoxes::process(cv::cuda::GpuMat &frame,
                                   PipelineContext &ctx) {
  // We also check YuNet because SFace results rely on YuNet geometry.
  if (ctx.yunet_sface.results.empty()) {
    return failure_and_continue;
  }

  frame.download(m_pinned_mem_for_cpu_frame);
  // point the cv::Mat directly to the pinned memory
  m_frame_cpu = m_pinned_mem_for_cpu_frame.createMatHeader();

  for (size_t i = 0; i < ctx.yunet_sface.results.size(); ++i) {
    if (!ctx.yunet_sface.results[i].recognition.has_value())
      return failure_and_continue;
    const auto &recognition = ctx.yunet_sface.results[i].recognition.value();
    const auto &detection = ctx.yunet_sface.results[i].detection;

    // Convert Rect2f (float) to Rect (int) for cleaner pixel drawing
    const cv::Rect bounding_box = detection.bounding_box;
    // Draw the bounding box
    cv::rectangle(m_frame_cpu, bounding_box,
                  identity_to_box_color_bgr[recognition.category],
                  m_bounding_box_border_thickness);

    {
      // Draw label box + text
      std::string label = recognition.category != IdentityCategory::Unknown
                              ? fmt::format("{} ({:.2f})", recognition.identity,
                                            recognition.similarity_score)
                              : "?";

      // Calculate text position (put it slightly above the box)
      int baseLine;
      cv::Size label_size =
          cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, m_label_font_scale,
                          m_label_font_thickness, &baseLine);

      const auto label_x = bounding_box.x;
      // Ensure it doesn't go off-screen top
      const auto label_y = std::max(bounding_box.y - 5, label_size.height);

      // Draw label box
      cv::rectangle(
          m_frame_cpu, cv::Point(label_x, label_y - label_size.height),
          cv::Point(label_x + label_size.width, label_y + baseLine),
          identity_to_box_color_bgr[recognition.category], cv::FILLED);
      // Draw label text
      cv::putText(m_frame_cpu, label, cv::Point(label_x, label_y),
                  cv::FONT_HERSHEY_SIMPLEX, m_label_font_scale,
                  m_text_color_bgr, m_label_font_thickness);
    }
  }

  frame.upload(m_frame_cpu);

  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit