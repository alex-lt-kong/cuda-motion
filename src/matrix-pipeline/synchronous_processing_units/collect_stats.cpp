#include "collect_stats.h"
#include "../utils/misc.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

CollectStats::~CollectStats() = default;

bool CollectStats::init(const njson &config) {
  try {

    m_overlay_text_template = config.value(
        "overlayTextTemplate", "{deviceName},\nChg: {changeRatePct:.1f}%, FPS: "
                               "{fps:.1f}\n{timestamp:%Y-%m-%d %H:%M:%S}\n");
    m_change_rate_threshold_per_pixel =
        config.value("/changeRate/thresholdPerPixel"_json_pointer,
                     m_change_rate_threshold_per_pixel);

    m_change_rate_frame_compare_interval = std::chrono::milliseconds(
        config.value("/changeRate/frameCompareIntervalMs"_json_pointer,
                     m_change_rate_frame_compare_interval.count()));

    if (m_change_rate_frame_compare_interval < 0ms)
      m_change_rate_frame_compare_interval = 0ms;

    // Create the filter
    m_blur_filter = cv::cuda::createGaussianFilter(
        CV_8UC1, CV_8UC1,
        cv::Size(m_change_rate_kernel_size, m_change_rate_kernel_size), 0);

    m_fps_sliding_window_length = std::chrono::milliseconds(
        config.value("/fps/slidingWindowLengthMs"_json_pointer,
                     m_fps_sliding_window_length.count()));

    SPDLOG_INFO(
        "change_rate_threshold_per_pixel: {}, "
        "change_rate_frame_compare_interval(ms): {}, "
        "fps_sliding_window_length(ms): {}, append_info_to_overlay_text: {}, "
        "overlay_text_template: {:?}",
        m_change_rate_threshold_per_pixel,
        m_change_rate_frame_compare_interval.count(),
        m_fps_sliding_window_length.count(), m_append_info_to_overlay_text,
        m_overlay_text_template);

    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("CollectStats init failed: {}", e.what());
    return false;
  }
}

SynchronousProcessingResult CollectStats::process(cv::cuda::GpuMat &frame,
                                                  PipelineContext &ctx) {
  if (frame.empty())
    return failure_and_continue;

  // =========================================================
  // 1. Calculate FPS
  // =========================================================
  {
    const auto capture_timestamp = ctx.capture_timestamp;

    m_frame_timestamps.push_back(capture_timestamp);

    // Remove frames older than window
    while (!m_frame_timestamps.empty() &&
           (capture_timestamp - m_frame_timestamps.front() >
            m_fps_sliding_window_length)) {
      m_frame_timestamps.pop_front();
    }

    // 1. Calculate the actual time span in the window
    // duration_cast to seconds<float> handles the / 1000.0f logic automatically
    const auto actual_duration = capture_timestamp - m_frame_timestamps.front();
    // 2. Determine the divisor (clamped by the max window length)
    // We use duration<float> to keep decimal precision
    auto window_duration_float =
        std::chrono::duration<float>(m_fps_sliding_window_length);

    if (actual_duration < m_fps_sliding_window_length &&
        actual_duration > std::chrono::steady_clock::duration::zero()) {
      window_duration_float = std::chrono::duration<float>(actual_duration);
    }

    // 3. Calculate FPS
    float seconds = window_duration_float.count();
    size_t count = m_frame_timestamps.size();

    if (count > 1 && seconds > 0.0f) {
      // We use count - 1 because it takes two points to define one interval
      ctx.fps = static_cast<float>(count - 1) / seconds;
    } else {
      ctx.fps = 0.0f;
    }
  }

  // =========================================================
  // 2. Calculate Change Rate
  // =========================================================

  // Resize
  cv::Size small_size(static_cast<int>(frame.cols * m_scale_factor),
                      static_cast<int>(frame.rows * m_scale_factor));

  if (d_small.size() != small_size) {
    d_small.create(small_size, frame.type());
    m_history_buffer.clear(); // Reset history on size change
  }

  cv::cuda::resize(frame, d_small, small_size);

  // Convert to Grayscale
  if (d_small.channels() > 1) {
    cv::cuda::cvtColor(d_small, d_current, cv::COLOR_BGR2GRAY);
  } else {
    d_small.copyTo(d_current);
  }

  // Blur
  m_blur_filter->apply(d_current, d_current);

  // Handle First Frame
  if (m_history_buffer.empty()) {
    m_history_buffer.push_back({ctx.capture_timestamp, d_current.clone()});
    ctx.change_rate = 0.0f;
    return success_and_continue;
  }

  const auto capture_timestamp = ctx.capture_timestamp;

  // Prune history to maintain correct interval
  while (m_history_buffer.size() > 1) {
    const auto next_oldest_time = m_history_buffer[1].first;
    if (capture_timestamp - next_oldest_time >=
        m_change_rate_frame_compare_interval) {
      m_history_buffer.pop_front();
    } else {
      break;
    }
  }

  // Compare with oldest valid reference
  const auto &reference = m_history_buffer.front();

  if (capture_timestamp - reference.first >=
      m_change_rate_frame_compare_interval) {
    cv::cuda::absdiff(d_current, reference.second, d_diff);
    cv::cuda::threshold(d_diff, d_mask, m_change_rate_threshold_per_pixel, 255,
                        cv::THRESH_BINARY);

    int non_zero = cv::cuda::countNonZero(d_mask);
    int total_pixels = d_mask.cols * d_mask.rows;

    if (total_pixels > 0) {
      ctx.change_rate =
          static_cast<float>(non_zero) / static_cast<float>(total_pixels);
    } else {
      ctx.change_rate = 0.0f;
    }
  } else {
    ctx.change_rate = 0.0f;
  }

  // Store current frame
  m_history_buffer.emplace_back(capture_timestamp, d_current.clone());

  if (m_append_info_to_overlay_text) {
    if (const auto full_text =
            Utils::evaluate_text_template(m_overlay_text_template, ctx);
        full_text.has_value())
      ctx.text_to_overlay += full_text.value();
  }
  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit