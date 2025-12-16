#include "yolo_overlay_bounding_boxes.h"

#include <fmt/ranges.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <algorithm>

namespace MatrixPipeline::ProcessingUnit {

bool YoloOverlayBoundingBoxes::init([[maybe_unused]] const njson &config) {
  m_class_names = config.value("classNames", m_class_names);
  std::srand(4); // we want deterministic coloring
  m_colors.clear();
  for (size_t i = 0; i < m_class_names.size(); ++i) {
    m_colors.emplace_back(std::rand() % 255, std::rand() % 255, std::rand() % 255);
  }
  SPDLOG_INFO("class_names: {}", fmt::join(m_class_names, ", "));
  return true;
}

SynchronousProcessingResult
YoloOverlayBoundingBoxes::process(cv::cuda::GpuMat &frame,
                                  PipelineContext &ctx) {
  // Fast exit if no detections or invalid frame
  if (frame.empty() || ctx.yolo.indices.empty()) {
    return success_and_continue;
  }

  try {
    // 1. Prepare CPU Canvas
    // Ensure CPU canvas matches the GPU frame size/type
    if (h_overlay_canvas.size() != frame.size() ||
        h_overlay_canvas.type() != frame.type()) {
      h_overlay_canvas.create(frame.size(), frame.type());
    }

    // Clear canvas to black (0,0,0) - this is our transparent key
    h_overlay_canvas.setTo(cv::Scalar::all(0));

    // 2. Draw Detections on CPU Canvas
    for (const auto idx : ctx.yolo.indices) {
      auto class_id = ctx.yolo.class_ids[idx];

      const auto &box = ctx.yolo.boxes[idx];
      float conf = ctx.yolo.confidences[idx];

      std::string label;
      std::string label_text;
      if (class_id >= m_class_names.size()) {
        label = "Undefined";
      } else {
        label = m_class_names[class_id];
      }

      label_text = fmt::format(
          "{}{} {:.2f} ", !ctx.yolo.is_in_roi[idx] ? "(!)" : "", label, conf);

      cv::Scalar color;
      if (!ctx.yolo.is_in_roi[idx])
        color = cv::Scalar(127, 127, 127);
      else
        color = (static_cast<size_t>(class_id) < m_colors.size())
                    ? m_colors[class_id]
                    : cv::Scalar(0, 255, 0);

      // Draw Bounding Box
      cv::rectangle(h_overlay_canvas, box, color, 2);

      // Draw Label Background
      int baseLine;
      cv::Size labelSize = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX,
                                           0.5, 1, &baseLine);
      int top = std::max(box.y, labelSize.height);

      cv::rectangle(h_overlay_canvas, cv::Point(box.x, top - labelSize.height),
                    cv::Point(box.x + labelSize.width, top + baseLine), color,
                    cv::FILLED);

      // Draw Label Text (White)
      cv::putText(h_overlay_canvas, label_text, cv::Point(box.x, top),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // 3. Upload Canvas to GPU
    d_overlay_canvas.upload(h_overlay_canvas);

    // 4. Generate Mask from Canvas
    // Convert to grayscale to check for non-black pixels
    if (d_overlay_canvas.channels() > 1) {
      cv::cuda::cvtColor(d_overlay_canvas, d_overlay_gray, cv::COLOR_BGR2GRAY);
    } else {
      d_overlay_gray = d_overlay_canvas;
    }

    // Threshold: 1 = Drawing Present (Keep Canvas), 0 = Black (Keep Original
    // Frame)
    cv::cuda::threshold(d_overlay_gray, d_overlay_mask, 1, 255,
                        cv::THRESH_BINARY);

    // 5. Stamp Overlay onto Original Frame
    // This copies pixels from d_overlay_canvas into 'frame' ONLY where mask is
    // non-zero.
    d_overlay_canvas.copyTo(frame, d_overlay_mask);

    return success_and_continue;

  } catch (const cv::Exception &e) {
    SPDLOG_ERROR("YoloOverlayBoundingBoxes OpenCV Error: {}", e.what());
    return failure_and_continue;
  }
}

} // namespace MatrixPipeline::ProcessingUnit