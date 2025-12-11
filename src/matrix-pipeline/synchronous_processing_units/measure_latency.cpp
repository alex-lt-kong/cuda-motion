#include "measure_latency.h"

#include <algorithm>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <sstream>

namespace MatrixPipeline::ProcessingUnit {

bool MeasureLatency::init(const njson &config) {
  // 1. Parse Position (using string intermediate default)
  std::string pos_str = config.value("position", "start");

  if (pos_str == "start") {
    m_position = Position::START;
  } else if (pos_str == "end") {
    m_position = Position::END;
  } else {
    SPDLOG_ERROR("Invalid position '{}'. Use 'start' or 'end'.", pos_str);
    return false;
  }

  m_target_percentiles = config.value("percentiles", m_target_percentiles);
  m_window_duration_sec = config.value("rollingWindowSec", m_window_duration_sec);
  m_label = config.value("label", m_label);

  if (m_window_duration_sec <= 0.0) {
    SPDLOG_WARN("rollingWindowSec must be > 0. Resetting to default 5.0s");
    m_window_duration_sec = 5.0;
  } else {
    SPDLOG_INFO("window_duration_sec {:.1f}.", m_window_duration_sec);
  }

  return true;
}

SynchronousProcessingResult
MeasureLatency::process([[maybe_unused]] cv::cuda::GpuMat &frame,
                        PipelineContext &ctx) {
  if (m_position == Position::START) {
    ctx.latency_start_time = std::chrono::steady_clock::now();
  } else {
    // Position == END
    auto now = std::chrono::steady_clock::now();

    // Safety check
    if (ctx.latency_start_time.time_since_epoch().count() == 0) {
      return failure_and_continue;
    }

    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           now - ctx.latency_start_time)
                           .count();

    // Add new sample
    m_history.emplace_back(now, duration_us);

    // Prune old samples
    auto window_duration =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(m_window_duration_sec));

    auto cutoff_time = now - window_duration;

    while (!m_history.empty() && m_history.front().first < cutoff_time) {
      m_history.pop_front();
    }

    // --- NEW: Check if enough time has passed to log ---
    if ((now - m_last_log_time) >= window_duration) {
      SPDLOG_INFO("{}: {}", m_label, this->to_string(get_percentile_stats()));
      m_last_log_time = now;
    }
  }

  return success_and_continue;
}

std::map<double, long long> MeasureLatency::get_percentile_stats() const {
  if (m_history.empty()) {
    return {};
  }

  // Extract values for sorting
  // We reserve space to avoid reallocations during extraction
  std::vector<long long> values;
  values.reserve(m_history.size());

  for (const auto &pair : m_history) {
    values.push_back(pair.second);
  }

  // Sort to determine percentiles
  // So there is no escape, the Gemini 3Pro's version also needs sorting
  std::sort(values.begin(), values.end());
  size_t count = values.size();

  std::map<double, long long> results;
  for (double p : m_target_percentiles) {
    double valid_p = std::max(0.0, std::min(1.0, p));
    size_t index = static_cast<size_t>(valid_p * (count - 1));
    results[p] = values[index];
  }

  return results;
}

std::string
MeasureLatency::to_string(const std::map<double, long long> &stats) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2);
  bool first = true;

  for (const auto &[percentile, latency] : stats) {
    if (!first)
      ss << ", ";

    if (latency > 1000) {
      ss << "P" << (percentile * 100) << ": " << (latency / 1000.0) << "ms";
    } else {
      ss << "P" << (percentile * 100) << ": " << latency << "us";
    }
    first = false;
  }

  if (stats.empty())
    return "No Data";

  return ss.str();
}

} // namespace MatrixPipeline::ProcessingUnit