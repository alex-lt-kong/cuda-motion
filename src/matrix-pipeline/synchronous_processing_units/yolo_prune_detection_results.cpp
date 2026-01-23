#include "yolo_prune_detection_results.h"

#include <fmt/ranges.h>
#include <opencv2/cudaarithm.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

bool YoloPruneDetectionResults::init(const njson &config) {
  try {
    if (config.contains("edgeConstraints")) {
      const auto &constraints = config["edgeConstraints"];
      m_left_constraint = parse_constraint(constraints, "left");
      m_right_constraint = parse_constraint(constraints, "right");
      m_top_constraint = parse_constraint(constraints, "top");
      m_bottom_constraint = parse_constraint(constraints, "bottom");
    }

    if (config.contains("sizeConstraint")) {

      if (const auto &constraints = config["sizeConstraint"];
          constraints.contains("minAreaRatio")) {
        m_size_limit_mode = SizeMode::MIN_RATIO;
        m_size_limit_val = constraints["minAreaRatio"].get<double>();
        SPDLOG_INFO("Constraint: Box must be larger than {:.2f}% of frame",
                    m_size_limit_val * 100.0);
      } else if (constraints.contains("maxAreaRatio")) {
        m_size_limit_mode = SizeMode::MAX_RATIO;
        m_size_limit_val = constraints["maxAreaRatio"].get<double>();
        SPDLOG_INFO("Constraint: Box must be smaller than {:.2f}% of frame",
                    m_size_limit_val * 100.0);
      }
    }

    // 2. Parse Visualization Config
    m_debug_overlay_alpha =
        config.value("debugOverlayAlpha", m_debug_overlay_alpha);

    const auto class_ids_of_interest_vector =
        config.value("classIdsOfInterest", std::vector<int>{});
    if (!class_ids_of_interest_vector.empty())
      for (const auto &idx : class_ids_of_interest_vector)
        m_class_ids_of_interest.insert(idx);
    else
      for (int i = 0; i < 80; ++i)
        m_class_ids_of_interest.insert(i);
    SPDLOG_INFO("region_constraint: L:[{:.2f}, {:.2f}], R:[{:.2f}, "
                "{:.2f}], T:[{:.2f}, {:.2f}], B:[{:.2f}, {:.2f}], "
                "debug_overlay_alpha: {}",
                m_left_constraint.min_val, m_left_constraint.max_val,
                m_right_constraint.min_val, m_right_constraint.max_val,
                m_top_constraint.min_val, m_top_constraint.max_val,
                m_bottom_constraint.min_val, m_bottom_constraint.max_val,
                m_debug_overlay_alpha);
    SPDLOG_INFO("size_limit_val: {}, size_limit_mode: {}", m_size_limit_val,
                static_cast<int>(m_size_limit_mode));
    SPDLOG_INFO("class_ids_of_interest: {}",
                fmt::join(class_ids_of_interest_vector, ", "));
    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("init() failed, e.what(): {}", e.what());
    return false;
  }
}

YoloPruneDetectionResults::Range
YoloPruneDetectionResults::parse_constraint(const njson &constraints,
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
YoloPruneDetectionResults::process(cv::cuda::GpuMat &frame,
                                   PipelineContext &ctx) {
  int img_w = frame.cols;
  int img_h = frame.rows;

  if (img_w == 0 || img_h == 0)
    return failure_and_continue;

  // ---------------------------------------------------------
  // 1. VISUALIZATION (ROI Based)
  // ---------------------------------------------------------
  if (m_debug_overlay_alpha > 0) {
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

    auto is_restrictive = [](const Range &r) {
      return r.min_val > 0.001f || r.max_val < 0.999f;
    };

    // Helper: Define ROI and Blend ONLY that area
    auto draw_corridor_roi = [&](const Range &r, const bool is_vertical) {
      if (!is_restrictive(r))
        return;

      int x = 0, y = 0, w = img_w, h = img_h;

      if (is_vertical) {
        // Vertical Strip (Left/Right constraints)
        x = std::max(static_cast<int>(r.min_val * img_w), 0);
        int x2 = static_cast<int>(r.max_val * img_w);
        w = x2 - x;
      } else {
        // Horizontal Strip (Top/Bottom constraints)
        y = std::max(static_cast<int>(r.min_val * img_h), 0);
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
      cv::cuda::addWeighted(frame_roi, 1.0 - m_debug_overlay_alpha, color_roi,
                            m_debug_overlay_alpha, 0.0, frame_roi);
    };

    // Apply overlays
    draw_corridor_roi(m_left_constraint, true);
    draw_corridor_roi(m_right_constraint, true);
    draw_corridor_roi(m_top_constraint, false);
    draw_corridor_roi(m_bottom_constraint, false);
  }

  // ---------------------------------------------------------
  // 2. LOGIC (Filter Boxes) - Updated with Size Check
  // ---------------------------------------------------------
  const size_t box_count = ctx.yolo.boxes.size();
  if (ctx.yolo.is_detection_interesting.size() != box_count) {
    ctx.yolo.is_detection_interesting = std::vector<short>(box_count);
  }

  if (box_count == 0)
    return success_and_continue;

  const auto f_w = static_cast<double>(img_w);
  const auto f_h = static_cast<double>(img_h);

  // Pre-calculate frame area for ratio checks
  double frame_area = static_cast<double>(img_w) * static_cast<double>(img_h);

  for (const auto idx : ctx.yolo.indices) {
    const cv::Rect &box = ctx.yolo.boxes[idx];

    // ---------------------------------------------------------
    // 1. Calculate Inverse Letterbox Parameters
    // ---------------------------------------------------------
    // These must match the logic used in letterbox_resize@yolo_detect.cpp
    const auto scale_x =
        static_cast<double>(ctx.yolo.inference_input_size.width) /
        static_cast<double>(frame.cols);
    const auto scale_y =
        static_cast<double>(ctx.yolo.inference_input_size.height) /
        static_cast<double>(frame.rows);
    const auto scale = std::min(scale_x, scale_y);
    // Calculate the padding that was added to the model input
    const int x_offset =
        (ctx.yolo.inference_input_size.width -
         static_cast<int>(static_cast<double>(frame.cols) * scale)) /
        2;
    const int y_offset =
        (ctx.yolo.inference_input_size.height -
         static_cast<int>(static_cast<double>(frame.rows) * scale)) /
        2;

    // --- A. Edge Logic (Corrected) ---

    // 1. Transform from Letterbox Space -> Original Image Pixel Space
    //    Formula: x_orig = (x_model - offset) / scale
    double left_px = (box.x - x_offset) / scale;
    double right_px = (box.x + box.width - x_offset) / scale;
    double top_px = (box.y - y_offset) / scale;
    double bottom_px = (box.y + box.height - y_offset) / scale;

    // 2. Clamp values to be safe (prevent boxes slightly outside image due to
    // rounding)
    left_px = std::max(0.0, left_px);
    right_px = std::min(f_w, right_px);
    top_px = std::max(0.0, top_px);
    bottom_px = std::min(f_h, bottom_px);

    // 3. Normalize (0.0 to 1.0)
    const auto box_left = left_px / f_w;
    const auto box_right = right_px / f_w;
    const auto box_top = top_px / f_h;
    const auto box_bottom = bottom_px / f_h;

    bool valid_left = box_left >= m_left_constraint.min_val &&
                      box_left <= m_left_constraint.max_val;
    bool valid_right = box_right >= m_right_constraint.min_val &&
                       box_right <= m_right_constraint.max_val;
    bool valid_top = box_top >= m_top_constraint.min_val &&
                     box_top <= m_top_constraint.max_val;
    bool valid_bottom = box_bottom >= m_bottom_constraint.min_val &&
                        box_bottom <= m_bottom_constraint.max_val;

    // --- B. Size Logic ---
    bool valid_size = true;
    if (m_size_limit_mode != SizeMode::NONE) {
      const auto box_area = static_cast<double>(box.width * box.height);
      const auto ratio = box_area / frame_area;

      if (m_size_limit_mode == SizeMode::MIN_RATIO) {
        // Criterion 1: Must be BIGGER than ratio (e.g., > 10%)
        valid_size = ratio >= m_size_limit_val;
      } else {
        // Criterion 2: Must be SMALLER than ratio (e.g., < 80%)
        valid_size = ratio <= m_size_limit_val;
      }
    }

    // Combine all checks
    ctx.yolo.is_detection_interesting[idx] =
        valid_left && valid_right && valid_top && valid_bottom && valid_size &&
        m_class_ids_of_interest.contains(
            static_cast<int>(ctx.yolo.class_ids[idx]));
  }

  return success_and_continue;
}
} // namespace MatrixPipeline::ProcessingUnit