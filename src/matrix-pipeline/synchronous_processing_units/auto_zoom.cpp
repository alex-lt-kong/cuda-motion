#include "auto_zoom.h"

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

namespace MatrixPipeline::ProcessingUnit {

// 2. Initialization phase
bool AutoZoom::init(const njson &config) {
  m_output_scale_factor =
      config.value("outputScaleFactor", m_output_scale_factor);
  m_smooth_step_pixel = config.value("smoothStepPixel", m_smooth_step_pixel);
  SPDLOG_INFO("output_scale_factor: {}, smooth_step_pixel: {}",
              m_output_scale_factor, m_smooth_step_pixel);
  return true;
}

SynchronousProcessingResult AutoZoom::process(cv::cuda::GpuMat &frame,
                                              PipelineContext &ctx) {

  cv::Size input_size = frame.size();
  if (!m_bounding_box_scale_params.has_value())
    m_bounding_box_scale_params =
        YoloDetect::get_bounding_box_scale(frame, ctx);
  // 3. Lazy Initialization of Resolution-Dependent values (On First Frame)
  if (!m_dimensions_set) {
    target_output_size_.width =
        static_cast<int>(input_size.width * m_output_scale_factor);
    target_output_size_.height =
        static_cast<int>(input_size.height * m_output_scale_factor);

    // Start with full frame
    m_current_roi = cv::Rect2f(0, 0, static_cast<float>(input_size.width),
                               static_cast<float>(input_size.height));

    m_dimensions_set = true;
  }

  // --- Standard Processing Loop ---

  // Calculate Target
  const cv::Rect target_roi = calculate_target_roi(input_size, ctx);

  // Apply Smoothing
  update_current_roi(target_roi);

  // Convert to Integer Rect
  cv::Rect crop_rect = m_current_roi;

  // Clamp to input bounds
  crop_rect.x = std::clamp(crop_rect.x, 0, input_size.width - 1);
  crop_rect.y = std::clamp(crop_rect.y, 0, input_size.height - 1);
  crop_rect.width =
      std::clamp(crop_rect.width, 1, input_size.width - crop_rect.x);
  crop_rect.height =
      std::clamp(crop_rect.height, 1, input_size.height - crop_rect.y);

  // Crop and Resize
  cv::cuda::GpuMat cropped_frame = frame(crop_rect);
  cv::cuda::GpuMat resized_frame;

  cv::cuda::resize(cropped_frame, resized_frame, target_output_size_, 0, 0,
                   cv::INTER_LINEAR);

  frame = std::move(resized_frame);

  return success_and_continue;
}

cv::Rect AutoZoom::calculate_target_roi(const cv::Size &input_size,
                                        const PipelineContext &ctx) const {
  cv::Rect union_rect(0, 0, input_size.width, input_size.height);
  bool valid_box_found = false;

  if (!ctx.yolo.boxes.empty()) {
    int min_x = input_size.width, min_y = input_size.height;
    int max_x = 0, max_y = 0;

    for (const auto idx : ctx.yolo.indices) {
      if (!ctx.yolo.is_detection_interesting[idx])
        continue;
      const auto scaled_box = YoloDetect::get_scaled_bounding_box_coordinates(
          ctx.yolo.boxes[idx], m_bounding_box_scale_params.value());

      if (scaled_box.width <= 0 || scaled_box.height <= 0)
        continue;

      // We found at least one good box!
      valid_box_found = true;

      min_x = std::min(min_x, scaled_box.x);
      min_y = std::min(min_y, scaled_box.y);
      max_x = std::max(max_x, scaled_box.x + scaled_box.width);
      max_y = std::max(max_y, scaled_box.y + scaled_box.height);
    }

    // Only update union_rect if we actually calculated valid min/max values
    if (valid_box_found) {
      int pad_x = (max_x - min_x) * 0.1;
      int pad_y = (max_y - min_y) * 0.1;

      union_rect =
          cv::Rect(min_x - pad_x, min_y - pad_y, (max_x - min_x) + 2 * pad_x,
                   (max_y - min_y) + 2 * pad_y);
    }
  }

  // Min Crop Size Logic
  int min_allowed_w =
      static_cast<int>(input_size.width * m_output_scale_factor);
  int min_allowed_h =
      static_cast<int>(input_size.height * m_output_scale_factor);

  if (union_rect.width < min_allowed_w) {
    int diff = min_allowed_w - union_rect.width;
    union_rect.x -= diff / 2;
    union_rect.width = min_allowed_w;
  }
  if (union_rect.height < min_allowed_h) {
    int diff = min_allowed_h - union_rect.height;
    union_rect.y -= diff / 2;
    union_rect.height = min_allowed_h;
  }

  return fix_aspect_ratio(union_rect, target_output_size_, input_size);
}

cv::Rect AutoZoom::fix_aspect_ratio(const cv::Rect &input,
                                    const cv::Size &output_size,
                                    const cv::Size &input_limit) {
  // 1. Determine Target Aspect Ratio
  double target_ar = (double)output_size.width / output_size.height;

  // 2. Expand Input to match Aspect Ratio
  // We create a floating point center/size to calculate accurately
  double w = input.width;
  double h = input.height;
  double cx = input.x + w / 2.0;
  double cy = input.y + h / 2.0;

  double input_ar = w / h;

  if (input_ar > target_ar) {
    // Too wide: Increase height
    h = w / target_ar;
  } else {
    // Too tall: Increase width
    w = h * target_ar;
  }

  // 3. Handle "Too Big for Frame" (Zoomed out too far)
  // If the expanded box is larger than the input image, we must shrink it
  // while KEEPING the aspect ratio.
  if (w > input_limit.width) {
    w = input_limit.width;
    h = w / target_ar;
  }
  if (h > input_limit.height) {
    h = input_limit.height;
    w = h * target_ar;
  }

  // 4. Re-center and Clamp
  // Convert back to top-left
  double x = cx - w / 2.0;
  double y = cy - h / 2.0;

  // Slide into bounds (Clamping)
  if (x < 0)
    x = 0;
  if (y < 0)
    y = 0;
  if (x + w > input_limit.width)
    x = input_limit.width - w;
  if (y + h > input_limit.height)
    y = input_limit.height - h;

  return cv::Rect((int)x, (int)y, (int)w, (int)h);
}

void AutoZoom::update_current_roi(const cv::Rect &target_roi) {
  // 1. Calculate Target Properties (Center & Width)
  float target_w = (float)target_roi.width;
  float target_cx = target_roi.x + target_roi.width / 2.0f;
  float target_cy = target_roi.y + target_roi.height / 2.0f;

  // 2. Calculate Current Properties
  float current_w = m_current_roi.width;
  float current_cx = m_current_roi.x + m_current_roi.width / 2.0f;
  float current_cy = m_current_roi.y + m_current_roi.height / 2.0f;

  // 3. Smooth Step for Width
  if (std::abs(target_w - current_w) <= m_smooth_step_pixel) {
    current_w = target_w;
  } else {
    current_w +=
        (target_w > current_w) ? m_smooth_step_pixel : -m_smooth_step_pixel;
  }

  // 4. Smooth Step for Center (X and Y)
  auto move_val = [&](float &current, float target) {
    if (std::abs(target - current) <= m_smooth_step_pixel) {
      current = target;
    } else {
      current +=
          (target > current) ? m_smooth_step_pixel : -m_smooth_step_pixel;
    }
  };
  move_val(current_cx, target_cx);
  move_val(current_cy, target_cy);

  // 5. DERIVE Height from Width (Enforce Aspect Ratio)
  // This is the key fix. We effectively ignore the "current_h" state and
  // recalculate it every frame so it matches the output shape perfectly.
  float target_ar =
      (float)target_output_size_.width / target_output_size_.height;
  float current_h = current_w / target_ar;

  // 6. Reconstruct Rect
  m_current_roi.x = current_cx - current_w / 2.0f;
  m_current_roi.y = current_cy - current_h / 2.0f;
  m_current_roi.width = current_w;
  m_current_roi.height = current_h;

  // 7. Sanity Clamp (Keep inside frame)
  // Since we moved center and width independently, we might drift slightly
  // off-edge. We simply slide the box back in without changing its size/AR.

  // Slide X
  if (m_current_roi.x < 0)
    m_current_roi.x = 0;
  // Note: target_output_size_ isn't the limit, the input frame size is.
  // Assuming you have access to input_frame_size here (or store it in the
  // class) If not, you might need to pass it or store it in `init/process`. For
  // now, assuming standard clamping logic:
  /*
  if (m_current_roi.x + m_current_roi.width > input_width) {
       m_current_roi.x = input_width - m_current_roi.width;
  }
  if (m_current_roi.y < 0) m_current_roi.y = 0;
  if (m_current_roi.y + m_current_roi.height > input_height) {
       m_current_roi.y = input_height - m_current_roi.height;
  }
  */
}
} // namespace MatrixPipeline::ProcessingUnit