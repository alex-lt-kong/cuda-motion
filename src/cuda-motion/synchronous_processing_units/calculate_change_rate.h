#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

namespace CudaMotion::ProcessingUnit {

class CalculateChangeRate final : public ISynchronousProcessingUnit {
private:
  // --- Configuration ---
  double m_scale_factor{0.25};   // Downscale frame for faster processing (0.25 = 1/16th area)
  double m_threshold{25.0};      // Intensity difference to consider a pixel "changed"
  int m_blur_size{5};            // Gaussian blur kernel size (must be odd)

  // --- State ---
  bool m_is_first_frame{true};

  // --- Reusable GPU Buffers ---
  cv::cuda::GpuMat d_small;      // Resized current frame
  cv::cuda::GpuMat d_current;    // Grayscale + Blurred current
  cv::cuda::GpuMat d_prev;       // Grayscale + Blurred previous
  cv::cuda::GpuMat d_diff;       // Absolute difference
  cv::cuda::GpuMat d_mask;       // Binary threshold mask

  // Filter pointer (created once)
  cv::Ptr<cv::cuda::Filter> m_blur_filter;

public:
  inline CalculateChangeRate() = default;
  inline ~CalculateChangeRate() override = default;

  /**
   * @brief Init configuration.
   * Expected JSON:
   * {
   * "scale": 0.25,      // Resize factor (0.1 to 1.0)
   * "threshold": 25.0,  // Pixel intensity diff threshold
   * "blurSize": 5       // Kernel size for noise reduction
   * }
   */
  bool init(const njson &config) override {
    try {
      if (config.contains("scale")) m_scale_factor = config["scale"].get<double>();
      if (config.contains("threshold")) m_threshold = config["threshold"].get<double>();
      if (config.contains("blurSize")) m_blur_size = config["blurSize"].get<int>();

      // Validate
      if (m_scale_factor <= 0.0 || m_scale_factor > 1.0) m_scale_factor = 0.25;
      if (m_blur_size % 2 == 0) m_blur_size++; // Ensure odd kernel size

      // Create filter once to save init time during process
      m_blur_filter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(m_blur_size, m_blur_size), 0);

      return true;
    } catch (...) {
      return false;
    }
  }

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame, ProcessingMetaData& meta_data) override {
    if (frame.empty()) return failure_and_continue;

    // 1. Resize (Optimization: Work on small image)
    // Create destination size if needed
    cv::Size small_size(static_cast<int>(frame.cols * m_scale_factor),
                        static_cast<int>(frame.rows * m_scale_factor));

    if (d_small.size() != small_size) {
        d_small.create(small_size, frame.type());
        m_is_first_frame = true; // Reset logic if resolution changes drastically
    }

    cv::cuda::resize(frame, d_small, small_size);

    // 2. Convert to Grayscale
    // We allocate d_current here. If d_small is 3 channels, we need 1 channel output.
    if (d_small.channels() > 1) {
        cv::cuda::cvtColor(d_small, d_current, cv::COLOR_BGR2GRAY);
    } else {
        d_small.copyTo(d_current);
    }

    // 3. Blur (Noise Reduction)
    // Apply in-place or to aux buffer. Here we filter d_current in-place.
    m_blur_filter->apply(d_current, d_current);

    // 4. Handle First Frame Logic
    if (m_is_first_frame) {
      d_current.copyTo(d_prev); // Save current as previous
      m_is_first_frame = false;
      meta_data.change_rate = 0.0f; // No change on first frame
      return success_and_continue;
    }

    // 5. Compute Difference
    // absdiff(current, prev) -> diff
    cv::cuda::absdiff(d_current, d_prev, d_diff);

    // 6. Threshold
    // If diff > threshold -> 255 (white), else 0 (black)
    cv::cuda::threshold(d_diff, d_mask, m_threshold, 255, cv::THRESH_BINARY);

    // 7. Count Changed Pixels
    // This downloads a single int from GPU, which is fast.
    int non_zero = cv::cuda::countNonZero(d_mask);
    int total_pixels = d_mask.cols * d_mask.rows;

    // 8. Update Metadata
    if (total_pixels > 0) {
        meta_data.change_rate = static_cast<float>(non_zero) / static_cast<float>(total_pixels);
    } else {
        meta_data.change_rate = 0.0f;
    }

    // 9. Swap Buffers for next iteration
    // We swap d_prev and d_current so d_current becomes d_prev for next frame.
    // This is a pointer swap (O(1)), no deep copy needed.
    d_prev.swap(d_current);

    return success_and_continue;
  }
};

} // namespace CudaMotion::ProcessingUnit