#pragma once

#include <cpr/cpr.h>
#include <drogon/drogon.h>
#include <nlohmann/json.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <string>
#include <thread>

namespace MatrixPipeline::Utils {
using njson = nlohmann::json;
typedef auto (*signal_handler_callback)(int) -> void;

void install_signal_handler(signal_handler_callback cb);

// a temporary solution, we should not need it after C++23 is fully implemented
template <typename Duration>
auto steady_clock_to_system_time(
    std::chrono::time_point<std::chrono::steady_clock, Duration> steady_tp) {
  using namespace std::chrono;

  // We calculate the delta between clocks at the moment of conversion.
  // This is the manual version of what clock_cast is supposed to do.
  const auto sys_now = system_clock::now();
  const auto steady_now = steady_clock::now();
  const auto drift = sys_now.time_since_epoch() - steady_now.time_since_epoch();

  // Reconstruct the point in the system_clock timeline
  return system_clock::time_point(duration_cast<system_clock::duration>(
      steady_tp.time_since_epoch() + drift));
}

} // namespace MatrixPipeline::Utils