#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <nlohmann/json.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

using njson = nlohmann::json;
namespace CudaMotion::ProcessingUnit {

class Debug final : public ISynchronousProcessingUnit {

public:
  inline Debug() = default;
  inline ~Debug() override = default;

  bool init([[maybe_unused]] const njson &config) override { return true; }

  [[nodiscard]] SynchronousProcessingResult
  process([[maybe_unused]] cv::cuda::GpuMat &frame,
          [[maybe_unused]] ProcessingMetaData &meta_data) override {
    SPDLOG_INFO("Debug processing unit: {}, latency: {}ms",
                meta_data.capture_timestamp_ms,
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count() -
                    meta_data.capture_timestamp_ms);
    return success_and_continue;
  }
};

} // namespace CudaMotion::ProcessingUnit