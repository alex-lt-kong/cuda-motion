#include "sface_overlay_bounding_boxes.h"

#include <fmt/format.h> // Assuming fmt is available given modern C++ usage, or use std::format
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

bool SFaceOverlayBoundingBoxes::init(const njson &config) {
  try {
    std::srand(time(nullptr));
    // we want the color to be on the dark side
    m_box_color_bgr =
        cv::Scalar(std::rand() % 127, std::rand() % 127, std::rand() % 127);
    // This is white
    m_text_color_bgr = cv::Scalar(255, 255, 255);

    // 3. Parse geometric/font properties
    m_thickness = config.value("thickness", 2);
    m_label_font_scale = config.value("labelFontScale", 0.5);
    m_font_thickness = config.value("fontThickness", 1);

    SPDLOG_INFO("SFaceOverlay init - BoxColor: {{{},{},{}}}, Thickness: {}",
                m_box_color_bgr[0], m_box_color_bgr[1], m_box_color_bgr[2],
                m_thickness);
    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("Failed to init SFaceOverlayBoundingBoxes: {}", e.what());
    return false;
  }
}

SynchronousProcessingResult
SFaceOverlayBoundingBoxes::process(cv::cuda::GpuMat &frame,
                                   PipelineContext &ctx) {
  // 1. Check if SFace results exist.
  // We also check YuNet because SFace results rely on YuNet geometry.
  if (ctx.sface.results.empty() || ctx.yunet.empty()) {
    return SynchronousProcessingResult::failure_and_continue;
  }

  // 2. Download to CPU for drawing
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
    cv::rectangle(cpu_frame, box, m_box_color_bgr, m_thickness);

    // Prepare label: "Name (Score)"
    // Using simple snprintf style or string concatenation if fmt isn't
    // available, but assuming C++20/fmt based on your profile context.
    std::string label = fmt::format("{} ({:.2f})", recognition.identity,
                                    recognition.similarity_score);

    // Calculate text position (put it slightly above the box)
    int baseLine;
    cv::Size label_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, m_label_font_scale,
                        m_font_thickness, &baseLine);

    int text_x = box.x;
    int text_y = std::max(
        box.y - 5, label_size.height); // Ensure it doesn't go off-screen top

    cv::rectangle(cpu_frame, cv::Point(text_x, text_y - label_size.height),
                  cv::Point(text_x + label_size.width, text_y + baseLine),
                  m_box_color_bgr, cv::FILLED);
    // Draw Label Text (White)
    cv::putText(cpu_frame, label, cv::Point(text_x, text_y),
                cv::FONT_HERSHEY_SIMPLEX, m_label_font_scale, m_text_color_bgr,
                m_font_thickness);
  }

  // 4. Upload back to GPU
  frame.upload(cpu_frame);

  return SynchronousProcessingResult::success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit