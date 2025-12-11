#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <chrono>
#include <deque>
#include <map>
#include <vector>
#include <string>

namespace MatrixPipeline::ProcessingUnit {

class MeasureLatency final : public ISynchronousProcessingUnit {
public:
  enum class Position { START, END };

  MeasureLatency() = default;
  ~MeasureLatency() override = default;

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;

  /**
   * @brief Returns the calculated latencies for the configured percentiles
   * based on the time-based rolling window.
   */
  [[nodiscard]] std::map<double, long long> get_percentile_stats() const;

  static std::string to_string(const std::map<double, long long>& stats);

private:
  std::chrono::steady_clock::time_point m_last_log_time{std::chrono::steady_clock::now()};
  Position m_position{Position::START};
  std::vector<double> m_target_percentiles{0.5, 0.9, 0.99};

  std::string m_label;
  double m_window_duration_sec{5.0};

  // History stores: <Time of measurement, Latency in microseconds>
  std::deque<std::pair<std::chrono::steady_clock::time_point, long long>> m_history;
};

} // namespace MatrixPipeline::ProcessingUnit