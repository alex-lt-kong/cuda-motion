#include "sface_overlay.h"
#include "../utils/misc.h"

#include <fmt/format.h> // Assuming fmt is available given modern C++ usage, or use std::format
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

bool SFaceOverlay::init(const njson &config) {
  identity_to_box_color_bgr[IdentityCategory::Unknown] =
      cv::Scalar(127, 127, 127);
  identity_to_box_color_bgr[IdentityCategory::Unauthorized] =
      cv::Scalar(0, 179, 255);
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

  // std::string yunet_text_to_overlay = "YuNet: ";
  njson yunet_jsons;
  njson sface_jsons;
  // std::string sface_text_to_overlay = "SFace: ";
  // ctx.text_to_overlay += "SFace: ";
  // We also check YuNet because SFace results rely on YuNet geometry.
  if (ctx.yunet_sface.results.empty()) {
    ctx.text_to_overlay += fmt::format("YuNet: []\nSFace: []\n");
    return failure_and_continue;
  }

  frame.download(m_pinned_mem_for_cpu_frame);
  // point the cv::Mat directly to the pinned memory
  m_frame_cpu = m_pinned_mem_for_cpu_frame.createMatHeader();

  // std::string text_to_overlay;
  for (const auto &[detection, recognition] : ctx.yunet_sface.results) {

    njson yunet_json;
    yunet_json["conf"] = fmt::format("{:.2f}, ", detection.face_score);
    yunet_jsons.emplace_back(yunet_json);
    // Convert Rect2f (float) to Rect (int) for cleaner pixel drawing
    const cv::Rect bounding_box = detection.bounding_box;

    // Draw the bounding box
    cv::rectangle(m_frame_cpu, bounding_box,
                  identity_to_box_color_bgr[recognition.category],
                  m_bounding_box_border_thickness);
    njson sface_json;
    sface_json["ID"] = recognition.identity;
    sface_json["cos"] = fmt::format("{:.2f}", recognition.cosine_score);
    sface_json["L2"] = fmt::format("{:.2f}", recognition.l2_norm);
    sface_jsons.emplace_back(sface_json);
    if (!recognition.l2_norm_threshold_crossed ||
        !recognition.cosine_score_threshold_crossed)
      continue;

    const auto bounding_box_label_text =
        recognition.category != IdentityCategory::Unknown
            ? fmt::format("{}", recognition.identity)
            : "?";

    // --- POSITIONING LOGIC FIX START ---

    int baseLine;
    cv::Size label_size =
        cv::getTextSize(bounding_box_label_text, cv::FONT_HERSHEY_SIMPLEX,
                        m_label_font_scale, m_label_font_thickness, &baseLine);

    const auto label_x = bounding_box.x;
    int label_y = 0;
    const int margin = 5;

    // 1. Calculate the Y position if we were to place it ABOVE the face
    //    (We want the bottom of the text to be 'margin' pixels above the box)
    int y_above = bounding_box.y - margin;

    // 2. Check if the top of the text (y_above - height) would go off-screen (<
    // 0)
    bool fits_above = (y_above - label_size.height) >= 0;

    if (fits_above) {
      // Standard: Place above
      label_y = y_above;
    } else {
      // Fallback: Place BELOW the face
      // We calculate the baseline such that the top of the text starts after
      // the box math: (FaceBottom) + (Margin) + (TextHeight)
      label_y =
          (bounding_box.y + bounding_box.height) + margin + label_size.height;
    }

    // Draw label background box
    // We explicitly calculate the rectangle coordinates to ensure they match
    // the text
    cv::Point box_top_left(label_x, label_y - label_size.height);
    cv::Point box_bottom_right(label_x + label_size.width, label_y + baseLine);

    cv::rectangle(m_frame_cpu, box_top_left, box_bottom_right,
                  identity_to_box_color_bgr[recognition.category], cv::FILLED);

    // Draw label text
    cv::putText(m_frame_cpu, bounding_box_label_text,
                cv::Point(label_x, label_y), cv::FONT_HERSHEY_SIMPLEX,
                m_label_font_scale, m_text_color_bgr, m_label_font_thickness);

    // --- POSITIONING LOGIC FIX END ---
  }

  ctx.text_to_overlay +=
      fmt::format("YuNet: {}\n", Utils::hybrid_njson_array_dump(yunet_jsons));
  ctx.text_to_overlay +=
      fmt::format("SFace: {}\n", Utils::hybrid_njson_array_dump(sface_jsons));

  frame.upload(m_frame_cpu);

  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit