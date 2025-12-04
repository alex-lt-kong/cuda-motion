#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"
#include <nlohmann/json.hpp>

namespace CudaMotion::ProcessingUnit {

class CalculateFps final : public ISynchronousProcessingUnit {
private:
  // --- Configuration ---
  int64_t m_update_interval_ms{1000}; // Update FPS every 1 second by default

  // --- State ---
  int64_t m_last_calculation_time{0};
  uint32_t m_frames_since_last_calc{0};
  float m_cached_fps{0.0f};

public:
  inline CalculateFps() = default;
  inline ~CalculateFps() override = default;

  /**
   * @brief Init configuration.
   * Expected JSON:
   * {
   * "interval": 1000  // Time in ms to average FPS over (default 1000)
   * }
   */
  bool init(const njson &config) override {
    try {
      if (config.contains("interval")) {
        m_update_interval_ms = config["interval"].get<int64_t>();
      }
      // Sanity check to prevent divide by zero or negative intervals
      if (m_update_interval_ms <= 0) m_update_interval_ms = 1000;
      
      return true;
    } catch (...) {
      return false;
    }
  }

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame, ProcessingMetaData& meta_data) override {
    // Note: We don't check frame.empty() here because FPS calculation 
    // might be relevant even if the frame is invalid but the "cycle" happened.
    // However, if strict frame processing is required, uncomment below:
    // if (frame.empty()) return failure_and_continue;

    int64_t current_time = meta_data.capture_timestamp_ms;

    // Handle first frame or invalid timestamps
    if (m_last_calculation_time == 0 || current_time < m_last_calculation_time) {
      m_last_calculation_time = current_time;
      m_frames_since_last_calc = 0;
      meta_data.fps = 0.0f;
      return success_and_continue;
    }

    m_frames_since_last_calc++;

    int64_t diff = current_time - m_last_calculation_time;

    // Only recalculate if the time interval has passed
    if (diff >= m_update_interval_ms) {
      // Calculate FPS: (Frames / Time_in_Seconds)
      // diff is in ms, so we divide by 1000.0 to get seconds.
      m_cached_fps = static_cast<float>(m_frames_since_last_calc) * 1000.0f / static_cast<float>(diff);

      // Reset counters
      m_last_calculation_time = current_time;
      m_frames_since_last_calc = 0;
    }

    // Always populate metadata with the latest calculated value
    meta_data.fps = m_cached_fps;

    return success_and_continue;
  }
};

} // namespace CudaMotion::ProcessingUnit