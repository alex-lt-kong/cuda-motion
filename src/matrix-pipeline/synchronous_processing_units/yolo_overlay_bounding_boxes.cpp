#include "yolo_overlay_bounding_boxes.h"
#include "yolo_detect.h"

#include <fmt/ranges.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>

namespace MatrixPipeline::ProcessingUnit {

bool YoloOverlayBoundingBoxes::init([[maybe_unused]] const njson &config) {
  m_class_names = config.value("classNames", m_class_names);
  m_label_font_scale = config.value("labelFontScale", m_label_font_scale);
  std::srand(4); // we want deterministic coloring
  SPDLOG_INFO("label_font_scale: {}, class_names: {}", m_label_font_scale,
              fmt::join(m_class_names, ", "));
  return true;
}

SynchronousProcessingResult
YoloOverlayBoundingBoxes::process(cv::cuda::GpuMat &frame,
                                  PipelineContext &ctx) {
  if (frame.empty() || ctx.yolo.indices.empty()) {
    return success_and_continue;
  }
  if (!m_scaling_params.has_value())
    m_scaling_params = YoloDetect::get_bounding_box_scale(frame, ctx);
  try {

    // Calculate the padding that was added to the model input

    // ---------------------------------------------------------
    // 2. Prepare CPU Canvas
    // ---------------------------------------------------------
    if (m_h_overlay_canvas.size() != frame.size() ||
        m_h_overlay_canvas.type() != frame.type()) {
      m_h_overlay_canvas.create(frame.size(), frame.type());
    }
    // Clear canvas to black (transparent key)
    m_h_overlay_canvas.setTo(cv::Scalar::all(0));

    // ---------------------------------------------------------
    // 3. Draw Detections
    // ---------------------------------------------------------
    for (const auto idx : ctx.yolo.indices) {
      const auto class_id = ctx.yolo.class_ids[idx];
      const auto &orig_box =
          ctx.yolo.bounding_boxes[idx]; // Box in 640x640 space
      float conf = ctx.yolo.confidences[idx];
      auto drawn_box = YoloDetect::get_scaled_bounding_box_coordinates(
          orig_box, m_scaling_params.value());
      // clip the bounding box so that it stays strictly within the image
      // boundaries.
      drawn_box &= cv::Rect(0, 0, frame.cols, frame.rows);

      std::string label = "Undefined";
      if (class_id < m_class_names.size()) {
        label = m_class_names[class_id];
      }

      std::string label_text = fmt::format(
          "{}{} {:.2f} ", !ctx.yolo.is_detection_interesting[idx] ? "(!)" : "",
          label, conf);

      // Determine Color
      cv::Scalar color;
      if (!ctx.yolo.is_detection_interesting[idx])
        color = cv::Scalar(127, 127, 127);
      else {
        while (m_colors.size() <= class_id) {
          m_colors.emplace_back(std::rand() % 127, std::rand() % 127,
                                std::rand() % 127);
        }
        color = m_colors[class_id];
      }

      // Draw Bounding Box (Use drawn_box, NOT raw_box)
      cv::rectangle(m_h_overlay_canvas, drawn_box, color, 2);

      // Draw Label Background
      int baseLine;
      cv::Size labelSize = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX,
                                           m_label_font_scale, 1, &baseLine);
      int top = std::max(drawn_box.y, labelSize.height);

      cv::rectangle(m_h_overlay_canvas,
                    cv::Point(drawn_box.x, top - labelSize.height),
                    cv::Point(drawn_box.x + labelSize.width, top + baseLine),
                    color, cv::FILLED);

      // Draw Label Text
      cv::putText(m_h_overlay_canvas, label_text, cv::Point(drawn_box.x, top),
                  cv::FONT_HERSHEY_SIMPLEX, m_label_font_scale,
                  cv::Scalar(255, 255, 255), 1);
    }

    // ---------------------------------------------------------
    // 4. Upload & Stamp
    // ---------------------------------------------------------
    m_d_overlay_canvas.upload(m_h_overlay_canvas);

    if (m_d_overlay_canvas.channels() > 1) {
      cv::cuda::cvtColor(m_d_overlay_canvas, m_d_overlay_gray,
                         cv::COLOR_BGR2GRAY);
    } else {
      m_d_overlay_gray = m_d_overlay_canvas;
    }

    cv::cuda::threshold(m_d_overlay_gray, d_overlay_mask, 1, 255,
                        cv::THRESH_BINARY);
    m_d_overlay_canvas.copyTo(frame, d_overlay_mask);

    return success_and_continue;

  } catch (const cv::Exception &e) {
    SPDLOG_ERROR("YoloOverlayBoundingBoxes OpenCV Error: {}", e.what());
    return failure_and_continue;
  }
}

} // namespace MatrixPipeline::ProcessingUnit