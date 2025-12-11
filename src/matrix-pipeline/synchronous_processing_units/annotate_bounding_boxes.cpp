#include "annotate_bounding_boxes.h"

#include <opencv2/cudaarithm.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

bool AnnotateBoundingBoxes::init(const njson &config) {
  // 1. Parse Constraints
  if (config.contains("validityConstraints")) {
    const auto &constraints = config["validityConstraints"];
    m_left_constraint = parse_constraint(constraints, "leftEdge");
    m_right_constraint = parse_constraint(constraints, "rightEdge");
    m_top_constraint = parse_constraint(constraints, "topEdge");
    m_bottom_constraint = parse_constraint(constraints, "bottomEdge");
  }

  // 2. Parse Visualization Config (Simplified)
  m_is_overlay_enabled = config.value("isDebugOverlayEnabled", false);

  SPDLOG_INFO("Allowed bounding boxes interval: L:[{:.2f}, {:.2f}], R:[{:.2f}, "
              "{:.2f}], T:[{:.2f}, {:.2f}], B:[{:.2f}, {:.2f}] | "
              "is_overlay_enabled: {}",
              m_left_constraint.min_val, m_left_constraint.max_val,
              m_right_constraint.min_val, m_right_constraint.max_val,
              m_top_constraint.min_val, m_top_constraint.max_val,
              m_bottom_constraint.min_val, m_bottom_constraint.max_val,
              m_is_overlay_enabled ? "ON" : "OFF");

  return true;
}

AnnotateBoundingBoxes::Range
AnnotateBoundingBoxes::parse_constraint(const njson &constraints,
                                        const std::string &key) {
  Range range; // Defaults to 0.0 - 1.0
  if (constraints.contains(key)) {
    auto obj = constraints[key];
    if (obj.contains("min"))
      range.min_val = obj["min"].get<float>();
    if (obj.contains("max"))
      range.max_val = obj["max"].get<float>();
  }
  return range;
}

SynchronousProcessingResult
AnnotateBoundingBoxes::process(cv::cuda::GpuMat &frame, PipelineContext &ctx) {
  int img_w = frame.cols;
  int img_h = frame.rows;

  if (img_w == 0 || img_h == 0)
    return failure_and_continue;

  // ---------------------------------------------------------
  // 1. VISUALIZATION (ROI Based)
  // ---------------------------------------------------------
  if (m_is_overlay_enabled) {
    // We only need a small 1x1 buffer to hold the color,
    // or we can create a temporary ROI on the fly.
    // Ideally, we create a buffer matching the frame SIZE just once to be safe,
    // but we only WRITE to the ROI parts.

    if (m_overlay_buffer.size() != frame.size() ||
        m_overlay_buffer.type() != frame.type()) {
      m_overlay_buffer.create(frame.size(), frame.type());
      // Pre-fill with the target color (Red) so any ROI we pick is already Red.
      // We don't need to clear it to 0 every frame because we only use the
      // parts we blend.
      m_overlay_buffer.setTo(cv::Scalar(0, 255, 0));
    }

    constexpr double alpha = 0.05;

    auto is_restrictive = [](const Range &r) {
      return r.min_val > 0.001f || r.max_val < 0.999f;
    };

    // Helper: Define ROI and Blend ONLY that area
    auto draw_corridor_roi = [&](const Range &r, bool is_vertical) {
      if (!is_restrictive(r))
        return;

      int x = 0, y = 0, w = img_w, h = img_h;

      if (is_vertical) {
        // Vertical Strip (Left/Right constraints)
        x = static_cast<int>(r.min_val * img_w);
        int x2 = static_cast<int>(r.max_val * img_w);
        w = x2 - x;
      } else {
        // Horizontal Strip (Top/Bottom constraints)
        y = static_cast<int>(r.min_val * img_h);
        int y2 = static_cast<int>(r.max_val * img_h);
        h = y2 - y;
      }

      // Sanity checks
      if (w <= 0 || h <= 0)
        return;
      if (x + w > img_w)
        w = img_w - x;
      if (y + h > img_h)
        h = img_h - y;

      cv::Rect rect(x, y, w, h);

      // Create "Windows" into GPU memory (No data copy)
      cv::cuda::GpuMat frame_roi = frame(rect);
      cv::cuda::GpuMat color_roi = m_overlay_buffer(rect);

      // Blend: Frame_ROI = Frame_ROI * 0.7 + Color_ROI * 0.3
      // This ONLY modifies the pixels inside 'rect'
      cv::cuda::addWeighted(frame_roi, 1.0 - alpha, color_roi, alpha, 0.0,
                            frame_roi);
    };

    // Apply overlays
    draw_corridor_roi(m_left_constraint, true);
    draw_corridor_roi(m_right_constraint, true);
    draw_corridor_roi(m_top_constraint, false);
    draw_corridor_roi(m_bottom_constraint, false);
  }

  // ---------------------------------------------------------
  // 2. LOGIC (Filter Boxes) - Unchanged
  // ---------------------------------------------------------
  size_t box_count = ctx.yolo.boxes.size();
  if (ctx.yolo.is_in_roi.size() != box_count) {
    ctx.yolo.is_in_roi.resize(box_count);
  }
  // ... rest of logic code ...

  if (box_count == 0)
    return success_and_continue;

  float f_w = static_cast<float>(img_w);
  float f_h = static_cast<float>(img_h);

  for (size_t i = 0; i < box_count; ++i) {
    const cv::Rect &box = ctx.yolo.boxes[i];
    // ... (your existing validation logic)

    float box_left = box.x / f_w;
    float box_right = (box.x + box.width) / f_w;
    float box_top = box.y / f_h;
    float box_bottom = (box.y + box.height) / f_h;

    bool valid_left = (box_left >= m_left_constraint.min_val &&
                       box_left <= m_left_constraint.max_val);
    bool valid_right = (box_right >= m_right_constraint.min_val &&
                        box_right <= m_right_constraint.max_val);
    bool valid_top = (box_top >= m_top_constraint.min_val &&
                      box_top <= m_top_constraint.max_val);
    bool valid_bottom = (box_bottom >= m_bottom_constraint.min_val &&
                         box_bottom <= m_bottom_constraint.max_val);

    ctx.yolo.is_in_roi[i] =
        (valid_left && valid_right && valid_top && valid_bottom);
  }

  return success_and_continue;
}
} // namespace MatrixPipeline::ProcessingUnit