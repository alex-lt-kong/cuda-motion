#include "collect_stats.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

CollectStats::CollectStats() = default;

CollectStats::~CollectStats() = default;

bool CollectStats::init(const njson &config) {
  try {
    m_change_rate_threshold_per_pixel = config.value(
        "/changeRate/thresholdPerPixel"_json_pointer, m_change_rate_threshold_per_pixel);
    
    m_change_rate_frame_compare_interval_ms = config.value(
        "/changeRate/frameCompareIntervalMs"_json_pointer, m_change_rate_frame_compare_interval_ms);

    if (m_change_rate_frame_compare_interval_ms < 0)
      m_change_rate_frame_compare_interval_ms = 0;

    // Create the filter
    m_blur_filter = cv::cuda::createGaussianFilter(
        CV_8UC1, CV_8UC1,
        cv::Size(m_change_rate_kernel_size, m_change_rate_kernel_size), 0);

    m_fps_sliding_window_length_ms = config.value(
        "/fps/slidingWindowLengthMs"_json_pointer, m_fps_sliding_window_length_ms);

    SPDLOG_INFO("ChangeRate: threshold_per_pixel: {}, compare_interval_ms: {}",
                m_change_rate_threshold_per_pixel,
                m_change_rate_frame_compare_interval_ms);
    SPDLOG_INFO("Fps: sliding_window_length_ms: {}",
                m_fps_sliding_window_length_ms);

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
    int64_t current_time = ctx.capture_timestamp_ms;

    m_frame_timestamps.push_back(current_time);

    // Remove frames older than window
    while (!m_frame_timestamps.empty() &&
           (current_time - m_frame_timestamps.front() >
            m_fps_sliding_window_length_ms)) {
      m_frame_timestamps.pop_front();
    }

    float seconds =
        static_cast<float>(m_fps_sliding_window_length_ms) / 1000.0f;

    // Adjust seconds if window is not full yet
    int64_t actual_duration = current_time - m_frame_timestamps.front();
    if (actual_duration < m_fps_sliding_window_length_ms &&
        actual_duration > 0) {
      seconds = static_cast<float>(actual_duration) / 1000.0f;
    }

    size_t count = m_frame_timestamps.size();
    if (count > 1 && seconds > 0) {
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
    m_history_buffer.push_back({ctx.capture_timestamp_ms, d_current.clone()});
    ctx.change_rate = 0.0f;
    return success_and_continue;
  }

  int64_t current_time = ctx.capture_timestamp_ms;

  // Prune history to maintain correct interval
  while (m_history_buffer.size() > 1) {
    int64_t next_oldest_time = m_history_buffer[1].first;
    if (current_time - next_oldest_time >=
        m_change_rate_frame_compare_interval_ms) {
      m_history_buffer.pop_front();
    } else {
      break;
    }
  }

  // Compare with oldest valid reference
  const auto &reference = m_history_buffer.front();

  if (current_time - reference.first >=
      m_change_rate_frame_compare_interval_ms) {
    
    cv::cuda::absdiff(d_current, reference.second, d_diff);
    cv::cuda::threshold(d_diff, d_mask, m_change_rate_threshold_per_pixel,
                        255, cv::THRESH_BINARY);

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
  m_history_buffer.push_back({current_time, d_current.clone()});

  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit