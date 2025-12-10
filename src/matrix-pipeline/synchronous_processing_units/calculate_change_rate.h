#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <deque>
#include <utility>

namespace MatrixPipeline::ProcessingUnit {

class CalculateChangeRate final : public ISynchronousProcessingUnit {
private:
  // --- Configuration ---
  double m_scale_factor{0.25};
  double m_threshold_per_pixel{25.0};
  int m_kernel_size{5}; // the dimension of the Gaussian kernel. A kernel is a tiny matrix to be applied to an image, this operation is called convolution
  int64_t m_frame_compare_interval_ms{1000}; // Minimum age of the reference frame

  // --- State ---
  // History buffer storing {timestamp_ms, processed_frame}
  // processed_frame is the small, gray, blurred copy.
  std::deque<std::pair<int64_t, cv::cuda::GpuMat>> m_history_buffer;

  // --- Reusable GPU Buffers ---
  cv::cuda::GpuMat d_small;      // Resized current
  cv::cuda::GpuMat d_current;    // Grayscale + Blurred current
  cv::cuda::GpuMat d_diff;       // Absolute difference
  cv::cuda::GpuMat d_mask;       // Binary threshold mask

  cv::Ptr<cv::cuda::Filter> m_blur_filter;

public:
  inline CalculateChangeRate() = default;

  // Destructor handles GpuMat cleanup automatically
  inline ~CalculateChangeRate() override = default;

  /**
   * @brief Init configuration.
   * JSON: { "scale": 0.25, "threshold": 25.0, "kernelSize": 5, "frameCompareIntervalMs": 10000 }
   */
  bool init(const njson &config) override {
    try {
      if (config.contains("scale")) m_scale_factor = config["scale"].get<double>();
      if (config.contains("threshold")) m_threshold_per_pixel = config["thresholdPerPixel"].get<double>();
      if (config.contains("kernelSize")) m_kernel_size = config["kernelSize"].get<int>();
      if (config.contains("frameCompareIntervalMs")) m_frame_compare_interval_ms = config["frameCompareIntervalMs"].get<int64_t>();

      if (m_scale_factor <= 0.0 || m_scale_factor > 1.0) m_scale_factor = 0.25;
      if (m_kernel_size % 2 == 0) m_kernel_size++;
      if (m_frame_compare_interval_ms < 0) m_frame_compare_interval_ms = 0;

      m_blur_filter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(m_kernel_size, m_kernel_size), 0);
      SPDLOG_INFO("scale_factor: {}, threshold_per_pixel: {}, compare_interval_ms: {}, kernel_size: {}", m_scale_factor, m_threshold_per_pixel, m_frame_compare_interval_ms, m_kernel_size);

      return true;
    } catch (...) {
      return false;
    }
  }

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame, PipelineContext& meta_data) override {
    if (frame.empty()) return failure_and_continue;

    // 1. Resize
    cv::Size small_size(static_cast<int>(frame.cols * m_scale_factor),
                        static_cast<int>(frame.rows * m_scale_factor));

    if (d_small.size() != small_size) {
        d_small.create(small_size, frame.type());
        // Resolution changed, invalidate history to prevent size mismatch errors
        m_history_buffer.clear();
    }

    cv::cuda::resize(frame, d_small, small_size);

    // 2. Convert to Grayscale
    if (d_small.channels() > 1) {
        cv::cuda::cvtColor(d_small, d_current, cv::COLOR_BGR2GRAY);
    } else {
        d_small.copyTo(d_current);
    }

    // 3. Blur (Noise Reduction)
    m_blur_filter->apply(d_current, d_current);

    // 4. Handle Empty History (First Frame)
    if (m_history_buffer.empty()) {
        // We must Clone because d_current is reused in the next iteration
        m_history_buffer.push_back({meta_data.capture_timestamp_ms, d_current.clone()});
        meta_data.change_rate = 0.0f;
        return success_and_continue;
    }

    int64_t current_time = meta_data.capture_timestamp_ms;

    // 5. Sliding Window Maintenance (Pruning)
    // We want the OLDEST frame that is NOT "too old" relative to the sliding window target.
    // Ideally, we want RefFrame such that: (CurrentTime - RefTime) >= Interval.
    // If the *next* oldest frame also satisfies this condition, then the current oldest
    // is unnecessarily old (it represents a window larger than requested). We discard it.
    // This keeps the buffer size roughly constant (~ Interval * FPS).
    while (m_history_buffer.size() > 1) {
        int64_t next_oldest_time = m_history_buffer[1].first;
        if (current_time - next_oldest_time >= m_frame_compare_interval_ms) {
            // Next frame is also a valid reference (it satisfies the frameCompareIntervalMs gap),
            // so the current front is stale.
            m_history_buffer.pop_front();
        } else {
            break;
        }
    }

    // 6. Comparison
    // Check if the reference frame satisfies the minimum frameCompareIntervalMs
    auto& reference = m_history_buffer.front();

    if (current_time - reference.first >= m_frame_compare_interval_ms) {
        // absdiff(current, ref_frame)
        cv::cuda::absdiff(d_current, reference.second, d_diff);

        cv::cuda::threshold(d_diff, d_mask, m_threshold_per_pixel, 255, cv::THRESH_BINARY);

        int non_zero = cv::cuda::countNonZero(d_mask);
        int total_pixels = d_mask.cols * d_mask.rows;

        if (total_pixels > 0) {
            meta_data.change_rate = static_cast<float>(non_zero) / static_cast<float>(total_pixels);
        } else {
            meta_data.change_rate = 0.0f;
        }
    } else {
        // Not enough history accumulated yet to satisfy the frameCompareIntervalMs requirement
        meta_data.change_rate = 0.0f;
    }

    // 7. Store Current Frame
    // Push at the end. Note: We clone because d_current is a reused scratch buffer.
    m_history_buffer.push_back({current_time, d_current.clone()});

    return success_and_continue;
  }
};

} // namespace MatrixPipeline::ProcessingUnit