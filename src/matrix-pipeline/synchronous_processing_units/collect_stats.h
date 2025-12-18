#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <deque>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <utility>

namespace MatrixPipeline::ProcessingUnit {

class CollectStats final : public ISynchronousProcessingUnit {
public:
  explicit CollectStats(const std::string &unit_path) : ISynchronousProcessingUnit(unit_path  + "/CollectStats") {}
  ~CollectStats() override;

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;

private:
  // --- Configuration ---
  const double m_scale_factor{0.25};
  double m_change_rate_threshold_per_pixel{25.0};
  const int m_change_rate_kernel_size{5};
  int64_t m_change_rate_frame_compare_interval_ms{1000};
  int64_t m_fps_sliding_window_length_ms{10000};

  // --- State ---
  std::deque<int64_t> m_frame_timestamps;

  // History buffer storing {timestamp_ms, processed_frame}
  std::deque<std::pair<int64_t, cv::cuda::GpuMat>> m_history_buffer;

  // --- Reusable GPU Buffers ---
  cv::cuda::GpuMat d_small;   // Resized current
  cv::cuda::GpuMat d_current; // Grayscale + Blurred current
  cv::cuda::GpuMat d_diff;    // Absolute difference
  cv::cuda::GpuMat d_mask;    // Binary threshold mask

  cv::Ptr<cv::cuda::Filter> m_blur_filter;
};

} // namespace MatrixPipeline::ProcessingUnit