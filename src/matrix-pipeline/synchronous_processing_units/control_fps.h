#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"
#include <deque>
#include <nlohmann/json.hpp>

namespace MatrixPipeline::ProcessingUnit {

class ControlFps final : public ISynchronousProcessingUnit {
private:
  // --- Configuration ---
  int64_t m_sliding_window_length_ms{10000}; // The size of the lookback window
  float m_fps_cap{30};

  // --- State ---
  // Stores the timestamp of every frame currently inside the window
  std::deque<int64_t> m_frame_timestamps;

public:
  inline ControlFps() = default;
  inline ~ControlFps() override = default;

  bool init(const njson &config) override {
    try {
      if (config.contains("slidingWindowLengthMs")) {
        m_sliding_window_length_ms =
            config["slidingWindowLengthMs"].get<int64_t>();
      }
      if (config.contains("fpsCap")) {
        m_fps_cap = config["fpsCap"].get<float>();
      }
      if (m_sliding_window_length_ms <= 1000)
        m_sliding_window_length_ms = 1000;
      return true;
    } catch (...) {
      return false;
    }
  }

  SynchronousProcessingResult process([[maybe_unused]] cv::cuda::GpuMat &frame,
                                      PipelineContext &meta_data) override {
    int64_t current_time = meta_data.capture_timestamp_ms;

    // 1. Add current frame to history
    m_frame_timestamps.push_back(current_time);

    // 2. Remove frames that are older than the window size
    // Example: If current is 15000 and window is 1000, remove anything < 14000
    while (!m_frame_timestamps.empty() &&
           (current_time - m_frame_timestamps.front() >
            m_sliding_window_length_ms)) {
      m_frame_timestamps.pop_front();
    }

    // 3. Calculate FPS based on how many frames are actually in the deque
    // We avoid divide by zero if window is extremely small, though init guards
    // it.
    float seconds = static_cast<float>(m_sliding_window_length_ms) / 1000.0f;

    // If we haven't filled the window yet (e.g., first 2 seconds of a 10s
    // window), we should calculate based on elapsed time, not full window size,
    // otherwise FPS will look like it's ramping up slowly from 0.
    int64_t actual_duration = current_time - m_frame_timestamps.front();
    if (actual_duration < m_sliding_window_length_ms && actual_duration > 0) {
      seconds = static_cast<float>(actual_duration) / 1000.0f;
    }

    // -1 because the duration covers from Frame 1 to Frame N.
    // The number of *intervals* is N-1.
    size_t count = m_frame_timestamps.size();
    if (count > 1 && seconds > 0) {
      meta_data.fps = static_cast<float>(count - 1) / seconds;
    } else {
      meta_data.fps = 0.0f;
    }

    if (meta_data.fps > m_fps_cap) {
      m_frame_timestamps.pop_back();
      return success_and_stop;
    }
    return success_and_continue;
  }
};

} // namespace MatrixPipeline::ProcessingUnit